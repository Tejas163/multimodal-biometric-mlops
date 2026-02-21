"""
scripts/train.py
----------------
Hydra-powered CLI entry point for model training.

Hydra handles:
  * Config composition from the conf/ directory tree.
  * CLI overrides (``training.learning_rate=5e-4 model.fusion.method=attention``).
  * Output directory management and config snapshot saving.
  * Multi-run sweeps (``--multirun training.learning_rate=1e-3,5e-4``).

Usage::

    # Default training run
    python scripts/train.py

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

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biometric_ml.training.trainer import build_and_fit
from biometric_ml.utils.logging import setup_logging


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging(level="DEBUG", json_format=False)
    log = logging.getLogger(__name__)
    log.info("Starting training run")
    log.info("Seed: %d | Deterministic: %s", cfg.training.seed, cfg.training.deterministic)
    build_and_fit(cfg)


if __name__ == "__main__":
    main()
