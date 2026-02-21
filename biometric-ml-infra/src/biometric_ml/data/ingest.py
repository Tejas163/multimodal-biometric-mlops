"""
ingest.py
---------
Ingestion pipeline — stores raw normalised pixel arrays in Parquet.
Matches the Kaggle notebook approach exactly (no HOG, raw images).

Per subject (45 total):
    Fingerprint: 20 BMPs → each stored as separate row (RGB 128x128 → 49152 floats)
    Iris left  : 10 BMPs → paired with iris right by index
    Iris right : 10 BMPs → paired with iris left by index
    Result     : 10 rows per subject × 45 subjects = 450 rows total
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from biometric_ml.data.schema import (
    FINGERPRINT_DIM,
    FUSED_SCHEMA,
    IRIS_LEFT_DIM,
    IRIS_RIGHT_DIM,
)

log = logging.getLogger(__name__)

FINGERPRINT_SIZE = (128, 128)
IRIS_SIZE        = (64,  64)


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    """Load image as RGB float32 [0,1], shape (H,W,3)."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB").resize(size)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, size).astype(np.float32) / 255.0
    except Exception:
        pass
    return None


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    """Load image as grayscale float32 [0,1], shape (H,W,1)."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize(size)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[:, :, np.newaxis]   # H,W → H,W,1
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size).astype(np.float32) / 255.0
            return img[:, :, np.newaxis]
    except Exception:
        pass
    return None


def _get_bmps(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() == ".bmp" and "desktop" not in p.name.lower()
    )


@ray.remote
def process_subject(
    subject_id: int,
    subject_dir: str,
    split: str,
) -> list[dict[str, Any]]:
    """
    Produce one row per iris sample pair.
    Matches notebook: load_image() loads files[0] but we use all for more data.
    Fingerprint is averaged across all 20 images (normalised) per row.
    """
    subj = Path(subject_dir)

    finger_paths = _get_bmps(subj / "Fingerprint")
    left_paths   = _get_bmps(subj / "left")
    right_paths  = _get_bmps(subj / "right")

    n_samples = max(len(left_paths), len(right_paths))
    if n_samples == 0:
        return []

    # Average all fingerprint images → one representative vector per subject
    finger_vecs = []
    for p in finger_paths:
        img = _load_rgb(p, FINGERPRINT_SIZE)
        if img is not None:
            finger_vecs.append(img.flatten())
    if finger_vecs:
        finger_flat = np.mean(finger_vecs, axis=0).astype(np.float32)
    else:
        finger_flat = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for i in range(n_samples):
        # Load left iris
        if i < len(left_paths):
            left_img = _load_gray(left_paths[i], IRIS_SIZE)
            left_flat = left_img.flatten().astype(np.float32) if left_img is not None \
                        else np.zeros(IRIS_LEFT_DIM, dtype=np.float32)
        else:
            left_flat = np.zeros(IRIS_LEFT_DIM, dtype=np.float32)

        # Load right iris
        if i < len(right_paths):
            right_img = _load_gray(right_paths[i], IRIS_SIZE)
            right_flat = right_img.flatten().astype(np.float32) if right_img is not None \
                         else np.zeros(IRIS_RIGHT_DIM, dtype=np.float32)
        else:
            right_flat = np.zeros(IRIS_RIGHT_DIM, dtype=np.float32)

        rows.append({
            "subject_id":  subject_id,
            "sample_id":   f"subj{subject_id:04d}_s{i:02d}",
            "fingerprint": finger_flat.tolist(),
            "iris_left":   left_flat.tolist(),
            "iris_right":  right_flat.tolist(),
            "split":       split,
        })

    return rows


def run_ingestion(
    raw_dir: str | Path,
    parquet_dir: str | Path,
    num_subjects: int,
    active_modalities: list[str],
    splits: dict[str, float],
    num_cpus: int | None = None,
) -> None:
    raw_dir     = Path(raw_dir)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = raw_dir / "IRIS and FINGERPRINT DATASET"
    if not dataset_root.exists():
        dataset_root = raw_dir

    subject_ids = sorted(
        int(d.name) for d in dataset_root.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    log.info("Found %d subjects", len(subject_ids))

    random.seed(42)
    shuffled = subject_ids.copy()
    random.shuffle(shuffled)

    split_assignments: dict[int, str] = {}
    cum = 0
    for split_name, frac in splits.items():
        n = max(1, int(frac * len(shuffled)))
        for sid in shuffled[cum: cum + n]:
            split_assignments[sid] = split_name
        cum += n
    for sid in shuffled[cum:]:
        split_assignments[sid] = "train"

    dist = {k: sum(1 for v in split_assignments.values() if v == k) for k in splits}
    log.info("Subject split (subjects): %s → rows: %s", dist, {k: v*10 for k, v in dist.items()})

    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)

    futures = [
        process_subject.remote(
            subject_id=sid,
            subject_dir=str(dataset_root / str(sid)),
            split=split_assignments[sid],
        )
        for sid in subject_ids
    ]

    all_rows: list[dict[str, Any]] = []
    for result in ray.get(futures):
        all_rows.extend(result)

    log.info("Total rows: %d", len(all_rows))
    _write_parquet(all_rows, parquet_dir)
    log.info("Done → %s", parquet_dir)


def _write_parquet(rows: list[dict[str, Any]], parquet_dir: Path) -> None:
    splits_seen: dict[str, list] = {}
    for row in rows:
        splits_seen.setdefault(row["split"], []).append(row)
    for split_name, split_rows in splits_seen.items():
        table    = pa.Table.from_pylist(split_rows, schema=FUSED_SCHEMA)
        out_path = parquet_dir / f"{split_name}.parquet"
        pq.write_table(table, out_path, compression="snappy", write_statistics=True)
        log.info("  %s → %d rows", out_path, len(split_rows))
