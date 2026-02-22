"""
ingest.py — Split by SAMPLE within each subject, not by subject.

Why: With only 45 subjects and 5 samples each, holding out entire subjects for
val means the model never sees those identities during training → accuracy=0 forever.

Strategy:
  - Every subject appears in BOTH train and val
  - Val = 1 original sample per subject (the last one, index 4)
  - Train = remaining 4 originals + N_AUG augmented copies of each
  - Test = same 1 sample as val (we have no spare data)

Result: 45 subjects × (4 orig + 4×N_AUG aug) = 45 × 36 = 1620 train rows
        45 subjects × 1 = 45 val rows (one clean sample per identity)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from biometric_ml.data.schema import (
    FINGERPRINT_SHAPE,
    FUSED_SCHEMA,
    IRIS_SHAPE,
)

log = logging.getLogger(__name__)

FINGERPRINT_SIZE = (FINGERPRINT_SHAPE[2], FINGERPRINT_SHAPE[1])  # (128, 128)
IRIS_SIZE        = (IRIS_SHAPE[2],        IRIS_SHAPE[1])         # (64, 64)
N_AUG = 8  # augmented copies per train sample


def _get_bmps(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    seen, out = set(), []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() == ".bmp" and "desktop" not in p.name.lower():
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


def _load_rgb(path: Path, size: tuple) -> np.ndarray | None:
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB").resize(size),
                        dtype=np.float32) / 255.0
    except Exception:
        return None


def _load_gray(path: Path, size: tuple) -> np.ndarray | None:
    try:
        from PIL import Image
        img = np.array(Image.open(path).convert("L").resize(size),
                       dtype=np.float32) / 255.0
        return img[:, :, np.newaxis]
    except Exception:
        return None


def _augment(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    arr = arr * rng.uniform(0.8, 1.2)
    if rng.random() < 0.5:
        arr = arr[:, ::-1, :]
    arr = arr + np.random.randn(*arr.shape).astype(np.float32) * 0.02
    return np.clip(arr, 0.0, 1.0)


def _make_row(subject_id, label, sample_id, fp, li, ri, split):
    return {
        "subject_id":  subject_id,
        "label":       label,
        "sample_id":   sample_id,
        "fingerprint": fp.flatten().astype(np.float32).tolist(),
        "iris_left":   li.flatten().astype(np.float32).tolist(),
        "iris_right":  ri.flatten().astype(np.float32).tolist(),
        "split":       split,
    }


def process_subject(subject_dir: Path, subject_id: int, label: int) -> dict[str, list]:
    """
    Returns train_rows and val_rows for one subject.
    Val = last sample (index 4), clean/original only.
    Train = samples 0-3 (original) + N_AUG augmented copies of each.
    """
    subj = Path(subject_dir)
    finger_paths = _get_bmps(subj / "Fingerprint")
    left_paths   = _get_bmps(subj / "left")
    right_paths  = _get_bmps(subj / "right")

    n = min(len(left_paths), len(right_paths))
    if n == 0:
        return {"train": [], "val": [], "test": []}

    # Load all fingerprint images for variety in augmentation
    finger_imgs = []
    for p in finger_paths:
        img = _load_rgb(p, FINGERPRINT_SIZE)
        if img is not None:
            finger_imgs.append(img)
    if not finger_imgs:
        finger_imgs = [np.zeros((*FINGERPRINT_SIZE, 3), dtype=np.float32)]
    finger_mean = np.mean(finger_imgs, axis=0)

    rng = random.Random(subject_id * 1000)
    train_rows, val_rows, test_rows = [], [], []

    for i in range(n):
        li = _load_gray(left_paths[i],  IRIS_SIZE)
        ri = _load_gray(right_paths[i], IRIS_SIZE)
        if li is None: li = np.zeros((*IRIS_SIZE, 1), dtype=np.float32)
        if ri is None: ri = np.zeros((*IRIS_SIZE, 1), dtype=np.float32)

        sid = f"s{subject_id:04d}_i{i:02d}"

        if i == n - 1:
            # Last sample → val AND test (we have no spare data)
            row = _make_row(subject_id, label, sid + "_orig", finger_mean, li, ri, "val")
            val_rows.append(row)
            test_rows.append({**row, "split": "test"})
        else:
            # Samples 0..n-2 → train (original + augmented)
            train_rows.append(
                _make_row(subject_id, label, sid + "_orig", finger_mean, li, ri, "train")
            )
            for aug_i in range(N_AUG):
                fp_src = rng.choice(finger_imgs)
                train_rows.append(_make_row(
                    subject_id, label, f"{sid}_a{aug_i:02d}",
                    _augment(fp_src.copy(), rng),
                    _augment(li.copy(), rng),
                    _augment(ri.copy(), rng),
                    "train",
                ))

    return {"train": train_rows, "val": val_rows, "test": test_rows}


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
    if num_subjects > 0:
        subject_ids = subject_ids[:num_subjects]

    # Global label map — consistent across all splits
    id_map = {sid: i for i, sid in enumerate(subject_ids)}
    log.info("Subjects: %d | N_AUG=%d | Strategy: split-by-sample (all subjects in train+val)",
             len(subject_ids), N_AUG)

    all_rows: dict[str, list] = {"train": [], "val": [], "test": []}

    for sid in subject_ids:
        label    = id_map[sid]
        subj_dir = dataset_root / str(sid)
        rows     = process_subject(subj_dir, sid, label)
        for split, r in rows.items():
            all_rows[split].extend(r)

    # Shuffle train rows
    rng = random.Random(42)
    rng.shuffle(all_rows["train"])

    for split_name, rows in all_rows.items():
        if not rows:
            continue
        table = pa.Table.from_pylist(rows, schema=FUSED_SCHEMA)
        out   = parquet_dir / f"{split_name}.parquet"
        pq.write_table(table, out)
        # Count unique subjects in this split
        labels = set(r["label"] for r in rows)
        log.info("  %s → %d rows, %d subjects", out, len(rows), len(labels))

    log.info("Done. Train subjects: %d (all) | Val subjects: %d (all) | Val rows: %d",
             len(subject_ids), len(subject_ids), len(all_rows["val"]))
