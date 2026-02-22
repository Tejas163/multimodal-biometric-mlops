"""
scripts/train.py
----------------
Hydra-powered CLI entry point for model training.

Supports both regular training and cross-validation via --cv flag.

Usage::

    # Regular training (uses train/val/test splits from parquet)
    python scripts/train.py

    # 5-fold cross-validation (uses only train.parquet, creates folds)
    python scripts/train.py --cv

    # Override specific hyperparameters
    python scripts/train.py training.epochs=100 training.learning_rate=5e-4

    # Change fusion method
    python scripts/train.py model.fusion.method=attention

    # Hyperparameter sweep (Hydra multirun)
    python scripts/train.py --multirun \\
        training.learning_rate=1e-3,5e-4,1e-4 \\
        model.encoder_hidden_dim=128,256

    # Use Azure MLflow tracking server
    python scripts/train.py mlflow.tracking_uri=https://my-mlflow.azureml.net
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biometric_ml.training.trainer import build_and_fit, build_and_fit_cv
from biometric_ml.utils.logging import setup_logging

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cv", action="store_true", 
                        help="Use 5-fold cross-validation instead of fixed splits")
    return parser.parse_known_args()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(level="DEBUG", json_format=False)
    log = logging.getLogger(__name__)
    
    # Check for --cv flag
    cv_args, _ = parse_args()
    
    if cv_args.cv:
        log.info("Starting 5-fold cross-validation training")
        log.info("Seed: %d | Deterministic: %s", 
                 cfg.training.seed, cfg.training.deterministic)
        build_and_fit_cv(cfg)
    else:
        log.info("Starting regular training run")
        log.info("Seed: %d | Deterministic: %s", 
                 cfg.training.seed, cfg.training.deterministic)
        build_and_fit(cfg)


if __name__ == "__main__":
    main()