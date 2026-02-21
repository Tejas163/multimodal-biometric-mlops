"""
ingest.py
---------
Real-data ingestion for the Kaggle Multimodal Iris+Fingerprint dataset.

Dataset layout:
    data/data/
        iris/
            {id}_left/    fatmal1.bmp ... fatmal5.bmp   (5 samples, left eye)
            {id}_right/   fatmar1.bmp ... fatmar5.bmp   (5 samples, right eye)
        fingerprint/
            {id}/
                {id}__M_Left_index_finger.BMP
                {id}__M_Left_little_finger.BMP
                {id}__M_Left_middle_finger.BMP
                {id}__M_Left_ring_finger.BMP
                {id}__M_Left_thumb_finger.BMP
                {id}__M_Right_index_finger.BMP
                {id}__M_Right_little_finger.BMP
                {id}__M_Right_middle_finger.BMP
                {id}__M_Right_ring_finger.BMP
                {id}__M_Right_thumb_finger.BMP

Fusion strategy per subject:
    Iris:        5 samples × (left + right averaged) → 5 rows
    Fingerprint: 10 fingers averaged into 1 vector   → reused across all 5 rows
    Result:      5 fused rows per subject
"""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from biometric_ml.data.schema import (
    FINGERPRINT_DIM,
    FUSED_SCHEMA,
    IRIS_DIM,
)


def _load_kaggle_credentials() -> None:
    """
    Load Kaggle credentials from .env file into environment variables.

    The Kaggle CLI and kaggle-python SDK both read from:
        KAGGLE_USERNAME and KAGGLE_KEY environment variables
        OR ~/.kaggle/kaggle.json

    This function loads them from the project-level .env file so you
    never need to hardcode credentials or manage kaggle.json manually.

    .env file format (project root):
        KAGGLE_USERNAME=your_username
        KAGGLE_KEY=your_api_key
    """
    from pathlib import Path

    # Walk up from this file to find the project root .env
    env_path = Path(__file__).resolve()
    for _ in range(6):  # search up to 6 levels up
        env_path = env_path.parent
        candidate = env_path / ".env"
        if candidate.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(candidate, override=False)  # don't override existing env vars
                log.info("Loaded Kaggle credentials from %s", candidate)
            except ImportError:
                # dotenv not installed — parse manually
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            k, v = k.strip(), v.strip()
                            if k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
                                import os as _os
                                _os.environ.setdefault(k, v)
                log.info("Loaded credentials from %s (manual parse)", candidate)
            return
    log.warning(
        ".env file not found. Kaggle download will use existing env vars or ~/.kaggle/kaggle.json"
    )

log = logging.getLogger(__name__)

IRIS_IMG_SIZE        = (64, 64)
FINGERPRINT_IMG_SIZE = (96, 96)

FINGER_NAMES = [
    "Left_index_finger",
    "Left_little_finger",
    "Left_middle_finger",
    "Left_ring_finger",
    "Left_thumb_finger",
    "Right_index_finger",
    "Right_little_finger",
    "Right_middle_finger",
    "Right_ring_finger",
    "Right_thumb_finger",
]


# ── Image utilities ──────────────────────────────────────────────────────────

def _load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Load BMP as grayscale float32 array normalised to [0, 1]."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L").resize(size)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception:
        pass
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return cv2.resize(img, size).astype(np.float32) / 255.0
    except Exception:
        pass
    return np.zeros(size, dtype=np.float32)


def _hog(img: np.ndarray, target_dim: int) -> np.ndarray:
    """Extract HOG feature vector, pad/truncate to target_dim."""
    try:
        from skimage.feature import hog
        feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)
    except Exception:
        gy, gx = np.gradient(img)
        feat = np.sqrt(gx**2 + gy**2).flatten().astype(np.float32)

    if feat.shape[0] < target_dim:
        feat = np.pad(feat, (0, target_dim - feat.shape[0]))
    elif feat.shape[0] > target_dim:
        feat = feat[:target_dim]

    norm = np.linalg.norm(feat)
    return feat / (norm + 1e-8)


def _imgs_to_vec(paths: list[Path], size: tuple[int, int], dim: int) -> np.ndarray:
    """Load images, extract HOG per image, return L2-normalised mean vector."""
    if not paths:
        return np.zeros(dim, dtype=np.float32)
    vecs = [_hog(_load_gray(p, size), dim) for p in paths]
    mean = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(mean)
    return mean / (norm + 1e-8)


# ── Ray remote task ──────────────────────────────────────────────────────────

@ray.remote
def process_subject(
    subject_id: int,
    iris_root: str,
    fingerprint_root: str,
    split: str,
) -> list[dict[str, Any]]:
    """
    Build fused feature rows for one subject.

    Iris   → 5 samples, each = avg(left_sample_i, right_sample_i)
    Finger → 10 finger images averaged into 1 vector, reused for all 5 rows
    """
    iris_left_dir  = Path(iris_root)  / f"{subject_id}_left"
    iris_right_dir = Path(iris_root)  / f"{subject_id}_right"
    finger_dir     = Path(fingerprint_root) / str(subject_id)

    def bmps(d: Path) -> list[Path]:
        if not d.exists():
            return []
        return sorted(p for p in d.iterdir()
                      if p.suffix.upper() == ".BMP" and p.name != "desktop.ini")

    iris_left  = bmps(iris_left_dir)
    iris_right = bmps(iris_right_dir)
    n_samples  = max(len(iris_left), len(iris_right), 1)

    # ── Fingerprint: average ALL 10 fingers into one vector ────────────────
    finger_paths = []
    for fname in FINGER_NAMES:
        # Pattern: {id}__M_{finger_name}.BMP  (double underscore after id)
        candidate = finger_dir / f"{subject_id}__M_{fname}.BMP"
        if candidate.exists():
            finger_paths.append(candidate)
        else:
            # Try case-insensitive fallback
            matches = list(finger_dir.glob(f"*{fname}*")) if finger_dir.exists() else []
            if matches:
                finger_paths.append(matches[0])

    finger_vec = _imgs_to_vec(finger_paths, FINGERPRINT_IMG_SIZE, FINGERPRINT_DIM)

    # ── Iris: one vector per sample index ─────────────────────────────────
    rows: list[dict[str, Any]] = []
    for i in range(n_samples):
        left_imgs  = iris_left[i:i+1]   if i < len(iris_left)  else []
        right_imgs = iris_right[i:i+1]  if i < len(iris_right) else []

        left_vec  = _imgs_to_vec(left_imgs,  IRIS_IMG_SIZE, IRIS_DIM)
        right_vec = _imgs_to_vec(right_imgs, IRIS_IMG_SIZE, IRIS_DIM)

        iris_vec = ((left_vec + right_vec) / 2.0).astype(np.float32)
        norm = np.linalg.norm(iris_vec)
        iris_vec = iris_vec / (norm + 1e-8)

        rows.append({
            "subject_id":           subject_id,
            "sample_id":            f"subj{subject_id:04d}_s{i:02d}",
            "iris_features":        iris_vec.tolist(),
            "fingerprint_features": finger_vec.tolist(),
            "split":                split,
        })

    return rows


# ── Orchestrator ─────────────────────────────────────────────────────────────

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

    iris_root        = raw_dir / "iris"
    fingerprint_root = raw_dir / "fingerprint"

    if not iris_root.exists():
        raise FileNotFoundError(
            f"iris/ folder not found at {iris_root}. "
            f"Set data.raw_dir to the folder containing iris/ and fingerprint/. "
            f"Currently: {raw_dir}"
        )

    # ── Discover subject IDs from iris folder names ({id}_left / {id}_right)
    subject_ids: set[int] = set()
    for folder in iris_root.iterdir():
        if folder.is_dir():
            m = re.match(r"^(\d+)_(left|right)$", folder.name)
            if m:
                subject_ids.add(int(m.group(1)))

    if not subject_ids:
        raise ValueError(
            f"No subject folders found in {iris_root}. "
            "Expected names like '8_left', '8_right', '9_left', etc."
        )

    subject_list = sorted(subject_ids)
    log.info(
        "Discovered %d subjects (IDs %s ... %s)",
        len(subject_list), subject_list[:3], subject_list[-3:]
    )

    # ── Assign train/val/test splits deterministically ────────────────────
    random.seed(42)
    shuffled = subject_list.copy()
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

    log.info(
        "Split distribution: %s",
        {k: sum(1 for v in split_assignments.values() if v == k) for k in splits}
    )

    # ── Ray ───────────────────────────────────────────────────────────────
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
        log.info("Ray initialised (%s CPUs)", num_cpus or os.cpu_count())

    futures = [
        process_subject.remote(
            subject_id=sid,
            iris_root=str(iris_root),
            fingerprint_root=str(fingerprint_root),
            split=split_assignments[sid],
        )
        for sid in subject_list
    ]

    log.info("Processing %d subjects in parallel...", len(futures))
    all_rows: list[dict[str, Any]] = []
    for result in ray.get(futures):
        all_rows.extend(result)

    log.info("Total rows ingested: %d", len(all_rows))
    _write_parquet(all_rows, parquet_dir)
    log.info("Done → %s", parquet_dir)


def _write_parquet(rows: list[dict[str, Any]], parquet_dir: Path) -> None:
    splits_seen: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        splits_seen.setdefault(row["split"], []).append(row)

    for split_name, split_rows in splits_seen.items():
        table    = pa.Table.from_pylist(split_rows, schema=FUSED_SCHEMA)
        out_path = parquet_dir / f"{split_name}.parquet"
        pq.write_table(table, out_path, compression="snappy", write_statistics=True)
        log.info("  %s → %d rows", out_path, len(split_rows))

# ── Optional: download from Kaggle ───────────────────────────────────────────

def download_dataset(
    dataset_slug: str = "ninadmehendale/multimodal-iris-fingerprint-biometric-data",
    download_dir: str | Path = "data",
) -> Path:
    """
    Download and unzip the Kaggle dataset using credentials from .env.

    Args:
        dataset_slug: Kaggle dataset identifier (owner/dataset-name)
        download_dir: Local directory to download into

    Returns:
        Path to the extracted dataset root (contains iris/ and fingerprint/)

    Usage::

        from biometric_ml.data.ingest import download_dataset
        raw_dir = download_dataset()
        run_ingestion(raw_dir=raw_dir / "data", ...)
    """
    import os
    _load_kaggle_credentials()

    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")

    if not username or not key or "your_kaggle" in username:
        raise ValueError(
            "Kaggle credentials not set.\n"
            "Edit .env in the project root:\n"
            "  KAGGLE_USERNAME=your_actual_username\n"
            "  KAGGLE_KEY=your_actual_api_key\n"
            "Get your key from https://www.kaggle.com/settings → API"
        )

    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "kaggle package not installed. Run: pip install kaggle"
        )

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading dataset '%s' to %s ...", dataset_slug, download_dir)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset_slug,
        path=str(download_dir),
        unzip=True,
        quiet=False,
    )
    log.info("Download complete → %s", download_dir)
    return download_dir
