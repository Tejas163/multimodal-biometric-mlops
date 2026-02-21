"""
scripts/ingest_data.py
----------------------
CLI entry point for the Kaggle iris+fingerprint ingestion pipeline.

Usage::

    # Data already downloaded and extracted to data/data/
    python scripts/ingest_data.py

    # Override paths
    python scripts/ingest_data.py data.raw_dir=data/data data.parquet_dir=data/parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biometric_ml.data.ingest import run_ingestion
from biometric_ml.utils.logging import setup_logging


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(level="INFO", json_format=False)
    log = logging.getLogger(__name__)

    active_modalities = [m for m, enabled in cfg.data.modalities.items() if enabled]
    log.info("Active modalities : %s", active_modalities)
    log.info("Reading images from: %s", cfg.data.raw_dir)
    log.info("Writing Parquet to : %s", cfg.data.parquet_dir)

    run_ingestion(
        raw_dir=cfg.data.raw_dir,
        parquet_dir=cfg.data.parquet_dir,
        num_subjects=cfg.num_subjects,
        active_modalities=active_modalities,
        splits=dict(cfg.data.splits),
    )


if __name__ == "__main__":
    main()
