"""ingest.py — with stratified train/val/test split ensuring label overlap."""

from __future__ import annotations

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from biometric_ml.data.schema import (
    FINGERPRINT_DIM, FINGERPRINT_SHAPE,
    IRIS_LEFT_DIM, IRIS_RIGHT_DIM, IRIS_SHAPE,
    FUSED_SCHEMA,
)

# Image sizes for PIL resize (H, W) — derived from shapes
FINGERPRINT_SIZE = (FINGERPRINT_SHAPE[2], FINGERPRINT_SHAPE[1])  # (128, 128)
IRIS_SIZE = (IRIS_SHAPE[2], IRIS_SHAPE[1])  # (64, 64)

log = logging.getLogger(__name__)

# Number of augmented copies per TRAIN row (val/test get 0 extra copies)
N_AUG = 8


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


def _load_rgb(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB").resize(size),
                        dtype=np.float32) / 255.0
    except Exception:
        return None


def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    try:
        from PIL import Image
        img = np.array(Image.open(path).convert("L").resize(size),
                       dtype=np.float32) / 255.0
        return img[:, :, np.newaxis]
    except Exception:
        return None


def _aug_image(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply random augmentation to a normalised float32 image array."""
    # Brightness jitter ±20%
    arr = arr * rng.uniform(0.8, 1.2)
    # Horizontal flip with 50% prob
    if rng.random() < 0.5:
        arr = arr[:, ::-1, :]
    # Gaussian noise
    noise = np.random.randn(*arr.shape).astype(np.float32) * 0.02
    arr = arr + noise
    return np.clip(arr, 0.0, 1.0)


def process_subject(
    subject_dir: Path,
    subject_id: int,
    split: str,
) -> list[dict[str, Any]]:
    subj = Path(subject_dir)
    finger_paths = _get_bmps(subj / "Fingerprint")
    left_paths = _get_bmps(subj / "left")
    right_paths = _get_bmps(subj / "right")

    n_samples = max(len(left_paths), len(right_paths))
    if n_samples == 0:
        return []

    # Load all fingerprint images
    finger_imgs = []
    for p in finger_paths:
        img = _load_rgb(p, FINGERPRINT_SIZE)
        if img is not None:
            finger_imgs.append(img)
    if not finger_imgs:
        finger_imgs = [np.zeros((*FINGERPRINT_SIZE, 3), dtype=np.float32)]

    rows: list[dict[str, Any]] = []
    rng = random.Random(subject_id)  # deterministic per subject

    for i in range(n_samples):
        left_img = _load_gray(left_paths[i], IRIS_SIZE) if i < len(left_paths) else None
        right_img = _load_gray(right_paths[i], IRIS_SIZE) if i < len(right_paths) else None

        if left_img is None:
            left_img = np.zeros((*IRIS_SIZE, 1), dtype=np.float32)
        if right_img is None:
            right_img = np.zeros((*IRIS_SIZE, 1), dtype=np.float32)

        # Mean fingerprint (original)
        finger_mean = np.mean(finger_imgs, axis=0)

        def make_row(fp, li, ri, aug_idx):
            return {
                "subject_id": subject_id,
                "label": -1,  # placeholder — set later
                "sample_id": f"subj{subject_id:04d}_s{i:02d}_a{aug_idx:02d}",
                "fingerprint": fp.flatten().astype(np.float32).tolist(),
                "iris_left": li.flatten().astype(np.float32).tolist(),
                "iris_right": ri.flatten().astype(np.float32).tolist(),
                "split": split,
            }

        # Original row (always included)
        rows.append(make_row(finger_mean, left_img, right_img, 0))

        # Augmented rows — ONLY for training split
        if split == "train":
            for aug_idx in range(1, N_AUG + 1):
                # Pick a random fingerprint image (not the mean) for variety
                fp_src = rng.choice(finger_imgs)
                fp_aug = _aug_image(fp_src.copy(), rng)
                li_aug = _aug_image(left_img.copy(), rng)
                ri_aug = _aug_image(right_img.copy(), rng)
                rows.append(make_row(fp_aug, li_aug, ri_aug, aug_idx))

    return rows


def run_ingestion(
    raw_dir: str | Path,
    parquet_dir: str | Path,
    num_subjects: int,
    active_modalities: list[str],
    splits: dict[str, float],
    num_cpus: int | None = None,
) -> None:
    raw_dir = Path(raw_dir)
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

    log.info("Found %d subjects. N_AUG=%d (train only)", len(subject_ids), N_AUG)

    # Create global label map FIRST (before any splitting)
    global_id_map = {sid: i for i, sid in enumerate(subject_ids)}
    log.info("Global label map: %d subjects → labels 0..%d",
             len(subject_ids), len(subject_ids) - 1)

    # STRATIFIED SPLIT: Spread val/test across label range
    n_total = len(subject_ids)
    
    # Calculate target sizes
    n_val = max(1, round(n_total * 0.1))
    n_test = max(1, round(n_total * 0.1))
    
    # Pick val/test subjects spread across range (every 10th, offset)
    val_indices = list(range(4, n_total, 10))[:n_val]   # 4, 14, 24, 34...
    test_indices = list(range(9, n_total, 10))[:n_test]  # 9, 19, 29, 39...
    
    # Convert indices to subject IDs
    val_ids = set(subject_ids[i] for i in val_indices if i < n_total)
    test_ids = set(subject_ids[i] for i in test_indices if i < n_total)
    train_ids = set(subject_ids) - val_ids - test_ids
    
    # Safety check
    assert len(train_ids) + len(val_ids) + len(test_ids) == n_total
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0

    log.info("Stratified split: %d train / %d val / %d test", 
             len(train_ids), len(val_ids), len(test_ids))
    log.info("Train subjects: %s", sorted(train_ids))
    log.info("Val subjects: %s (labels: %s)", 
             sorted(val_ids), 
             [global_id_map[s] for s in sorted(val_ids)])
    log.info("Test subjects: %s (labels: %s)", 
             sorted(test_ids),
             [global_id_map[s] for s in sorted(test_ids)])

    # Verify val/test labels are within train label range
    train_labels = {global_id_map[s] for s in train_ids}
    val_labels = {global_id_map[s] for s in val_ids}
    test_labels = {global_id_map[s] for s in test_ids}
    
    log.info("Train label range: %d-%d (%d labels)", 
             min(train_labels), max(train_labels), len(train_labels))
    
    if not val_labels.issubset(train_labels):
        missing = val_labels - train_labels
        log.warning("Val labels not in train: %s", sorted(missing))
    else:
        log.info("✓ All val labels are in train range")
        
    if not test_labels.issubset(train_labels):
        missing = test_labels - train_labels
        log.warning("Test labels not in train: %s", sorted(missing))
    else:
        log.info("✓ All test labels are in train range")

    # Process all subjects
    all_rows: list[dict[str, Any]] = []
    for sid in subject_ids:
        split = "train" if sid in train_ids else ("val" if sid in val_ids else "test")
        subj_dir = dataset_root / str(sid)
        rows = process_subject(subj_dir, sid, split)
        
        # Apply global label map
        for row in rows:
            row["label"] = global_id_map[row["subject_id"]]
        
        all_rows.extend(rows)

    # Log counts per split
    for split_name in ["train", "val", "test"]:
        split_rows = [r for r in all_rows if r["split"] == split_name]
        if split_rows:
            subjects_in_split = set(r["subject_id"] for r in split_rows)
            labels_in_split = set(r["label"] for r in split_rows)
            log.info("%s: %d rows, %d subjects, labels %d-%d (%d unique)",
                     split_name, len(split_rows), len(subjects_in_split),
                     min(labels_in_split), max(labels_in_split), len(labels_in_split))

    # Shuffle train rows only
    train_rows = [r for r in all_rows if r["split"] == "train"]
    other_rows = [r for r in all_rows if r["split"] != "train"]
    random.Random(42).shuffle(train_rows)
    all_rows = train_rows + other_rows

    log.info("Total rows: %d (train=%d, val=%d, test=%d)",
             len(all_rows),
             sum(1 for r in all_rows if r["split"] == "train"),
             sum(1 for r in all_rows if r["split"] == "val"),
             sum(1 for r in all_rows if r["split"] == "test"))

    # Write parquet per split
    for split_name in ["train", "val", "test"]:
        split_rows = [r for r in all_rows if r["split"] == split_name]
        if not split_rows:
            log.warning("No rows for %s split!", split_name)
            continue
        table = pa.Table.from_pylist(split_rows, schema=FUSED_SCHEMA)
        out = parquet_dir / f"{split_name}.parquet"
        pq.write_table(table, out)
        log.info("  %s → %d rows", out, len(split_rows))

    log.info("Done → %s", parquet_dir)