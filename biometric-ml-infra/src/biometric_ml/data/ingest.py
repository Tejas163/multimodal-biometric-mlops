"""
ingest.py
---------
Raw-data ingestion pipeline: reads per-subject files from ``raw_dir``,
applies preprocessing (normalisation, quality filtering), and writes
split-stratified Parquet files to ``parquet_dir``.

Parallelism strategy (Ray):
    Each *subject* is processed independently, making subject-level
    parallelism a natural fit.  Ray remote tasks are dispatched one per
    subject; Ray handles scheduling across all available CPU cores without
    requiring the caller to manage worker pools explicitly.

    For very large datasets (>1 M samples) replace the list of Ray futures
    with a Ray Dataset pipeline to enable streaming / out-of-core processing.

Usage (via CLI script):
    python scripts/ingest_data.py data.raw_dir=data/raw data.parquet_dir=data/parquet
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

from biometric_ml.data.schema import FUSED_SCHEMA, MODALITY_REGISTRY

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ray remote task — processes a single subject's raw data
# ---------------------------------------------------------------------------

@ray.remote
def process_subject(
    subject_id: int,
    subject_dir: Path,
    active_modalities: list[str],
    split: str,
) -> list[dict[str, Any]]:
    """
    Load, validate, and preprocess all samples for one subject.

    Returns a list of fused-row dicts ready to be written as Parquet rows.
    Each row contains feature vectors for every active modality, already
    L2-normalised (face) or z-score normalised (other modalities).

    NOTE: In a real deployment this function reads actual binary files
    (images, audio, etc.) and calls the appropriate feature extractor.
    Here we generate synthetic vectors so the infrastructure can be
    exercised without a proprietary dataset.
    """
    rng = np.random.default_rng(seed=subject_id)  # deterministic per subject
    samples_per_subject = int(rng.integers(5, 15))
    rows: list[dict[str, Any]] = []

    for i in range(samples_per_subject):
        sample_id = f"subj{subject_id:04d}_sample{i:03d}"
        row: dict[str, Any] = {
            "subject_id": subject_id,
            "sample_id": sample_id,
            "split": split,
            "face_embedding": None,
            "fingerprint_features": None,
            "voice_features": None,
            "gait_features": None,
        }

        for modality in active_modalities:
            _, field, dim = MODALITY_REGISTRY[modality]
            vec = rng.standard_normal(dim).astype(np.float32)

            if modality == "face":
                # L2-normalise face embeddings (common practice)
                norm = np.linalg.norm(vec)
                vec = vec / (norm + 1e-8)
                row["face_embedding"] = vec.tolist()
            elif modality == "fingerprint":
                row["fingerprint_features"] = vec.tolist()
            elif modality == "voice":
                row["voice_features"] = vec.tolist()
            elif modality == "gait":
                row["gait_features"] = vec.tolist()

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_ingestion(
    raw_dir: str | Path,
    parquet_dir: str | Path,
    num_subjects: int,
    active_modalities: list[str],
    splits: dict[str, float],
    num_cpus: int | None = None,
) -> None:
    """
    Orchestrate parallel ingestion across all subjects.

    Args:
        raw_dir:           Root directory containing per-subject subdirectories.
        parquet_dir:       Output directory for Parquet partition files.
        num_subjects:      Total number of unique subjects in the dataset.
        active_modalities: List of modality names to ingest (see schema.py).
        splits:            Dict mapping split name → fraction (must sum to 1.0).
        num_cpus:          CPUs to give Ray (None = auto-detect).
    """
    raw_dir = Path(raw_dir)
    parquet_dir = Path(parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    # Initialise Ray (no-op if already initialised, e.g. in tests)
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
        log.info("Ray initialised with %s CPUs", num_cpus or os.cpu_count())

    # Deterministic split assignment per subject
    subject_ids = list(range(num_subjects))
    random.seed(0)
    random.shuffle(subject_ids)

    split_assignments: dict[int, str] = {}
    cum = 0
    for split_name, frac in splits.items():
        n = int(frac * num_subjects)
        for sid in subject_ids[cum: cum + n]:
            split_assignments[sid] = split_name
        cum += n
    # Assign any remainder to train
    for sid in subject_ids[cum:]:
        split_assignments[sid] = "train"

    log.info(
        "Split distribution: %s",
        {k: sum(1 for v in split_assignments.values() if v == k) for k in splits},
    )

    # Dispatch one Ray task per subject
    futures = [
        process_subject.remote(
            subject_id=sid,
            subject_dir=raw_dir / f"subject_{sid:04d}",
            active_modalities=active_modalities,
            split=split_assignments[sid],
        )
        for sid in range(num_subjects)
    ]

    log.info("Dispatched %d Ray tasks, collecting results…", len(futures))
    all_rows: list[dict[str, Any]] = []
    for result in ray.get(futures):
        all_rows.extend(result)

    log.info("Total samples ingested: %d", len(all_rows))

    # Write split-partitioned Parquet files
    _write_parquet(all_rows, parquet_dir)

    log.info("Ingestion complete → %s", parquet_dir)


def _write_parquet(rows: list[dict[str, Any]], parquet_dir: Path) -> None:
    """Group rows by split and write one Parquet file per split."""
    splits_seen: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        splits_seen.setdefault(row["split"], []).append(row)

    for split_name, split_rows in splits_seen.items():
        table = pa.Table.from_pylist(split_rows, schema=FUSED_SCHEMA)
        out_path = parquet_dir / f"{split_name}.parquet"
        pq.write_table(
            table,
            out_path,
            compression="snappy",      # Fast read; good for ML workloads
            write_statistics=True,     # Enables predicate push-down
        )
        log.info("Wrote %d rows → %s", len(split_rows), out_path)
