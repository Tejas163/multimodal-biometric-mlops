"""
scripts/ingest_data.py
----------------------
CLI entry point for the Ray-parallel ingestion pipeline.

Usage::

    # Default (200 subjects, paths from conf/)
    python scripts/ingest_data.py

    # Override subjects and output dir
    python scripts/ingest_data.py num_subjects=500 data.parquet_dir=data/parquet_v2
"""

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

    active_modalities = [
        m for m, enabled in cfg.data.modalities.items() if enabled
    ]
    log.info("Active modalities: %s", active_modalities)
    log.info("Ingesting %d subjects → %s", cfg.num_subjects, cfg.data.parquet_dir)

    run_ingestion(
        raw_dir=cfg.data.raw_dir,
        parquet_dir=cfg.data.parquet_dir,
        num_subjects=cfg.num_subjects,       # ← reads from config, overridable via CLI
        active_modalities=active_modalities,
        splits=dict(cfg.data.splits),
    )


if __name__ == "__main__":
    main()
