"""
scripts/ingest_data.py
----------------------
CLI entry point for the Ray-parallel ingestion pipeline.

Reads raw biometric data from ``data.raw_dir``, processes it in parallel
using Ray, and writes split-stratified Parquet files to ``data.parquet_dir``.

Usage::

    # Default config
    python scripts/ingest_data.py

    # Override num_subjects and output dir
    python scripts/ingest_data.py num_subjects=500 data.parquet_dir=data/parquet_v2

    # Use a specific Hydra config group override
    python scripts/ingest_data.py data=biometric training=default
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Ensure the src package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biometric_ml.data.ingest import run_ingestion
from biometric_ml.utils.logging import setup_logging


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(level="INFO", json_format=False)
    log = logging.getLogger(__name__)

    active_modalities = [
        m for m, enabled in cfg.data.modalities.items() if enabled
    ]
    log.info("Active modalities: %s", active_modalities)

    run_ingestion(
        raw_dir=cfg.data.raw_dir,
        parquet_dir=cfg.data.parquet_dir,
        num_subjects=cfg.get("num_subjects", 200),  # override via CLI
        active_modalities=active_modalities,
        splits=dict(cfg.data.splits),
    )


if __name__ == "__main__":
    main()
