"""
datamodule.py
-------------
DataLoader factory that wires together BiometricDataset instances for each
split and returns configured DataLoaders ready for the training loop.

Keeping DataLoader construction here (rather than inside the trainer) makes
it easy to swap datasets, add samplers, or test data loading independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from biometric_ml.data.dataset import BiometricDataset


@dataclass
class DataConfig:
    """Mirrors the Hydra ``data`` config group."""

    parquet_dir: str
    active_modalities: list[str] = field(default_factory=lambda: ["face", "fingerprint", "voice"])
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    balance_classes: bool = False  # Use WeightedRandomSampler to handle imbalance
    train_transforms: Optional[dict[str, Any]] = None


class BiometricDataModule:
    """
    Instantiates train / val / test DataLoaders from Parquet splits.

    Example::

        dm = BiometricDataModule(cfg)
        train_loader = dm.train_dataloader()
        val_loader   = dm.val_dataloader()
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        parquet_dir = Path(cfg.parquet_dir)

        self.train_ds = BiometricDataset(
            parquet_dir / "train.parquet", cfg.active_modalities,transform=cfg.train_transforms
        )
        self.val_ds = BiometricDataset(
            parquet_dir / "val.parquet", cfg.active_modalities
        )
        self.test_ds = BiometricDataset(
            parquet_dir / "test.parquet", cfg.active_modalities
        )

    # ------------------------------------------------------------------
    # DataLoader builders
    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        sampler = None
        if self.cfg.balance_classes:
            sampler = self._make_weighted_sampler(self.train_ds)

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),  # mutually exclusive with sampler
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=False,
            drop_last=True,           # Keep all samples — dataset too small to drop any
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_weighted_sampler(dataset: BiometricDataset) -> WeightedRandomSampler:
        """
        Build a WeightedRandomSampler that up-samples minority classes so
        each class is seen equally often per epoch.
        """
        labels = dataset._labels
        class_counts = torch.bincount(torch.tensor(labels))
        # Weight for each sample = inverse frequency of its class
        weights = 1.0 / class_counts[torch.tensor(labels)].float()
        return WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

    @property
    def num_classes(self) -> int:
        return self.train_ds.num_classes

    @property
    def feature_dims(self) -> dict[str, int]:
        return self.train_ds.feature_dims
