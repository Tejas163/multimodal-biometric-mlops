"""
dataset.py
----------
PyTorch Dataset reading raw image tensors from Parquet.

CRITICAL FIX: Labels are now read directly from the Parquet 'label' column
which was assigned globally during ingestion. This ensures train/val/test
all share the same label space — subject 7 = label 6 in every split.

Previously each split built its own label_map independently, so:
  train: subject 7 → didn't exist (only train subjects mapped)
  val:   subject 7 → label 0  (only val subjects mapped)
This caused top1=0 and top5=0 forever since labels never matched.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from biometric_ml.data.schema import (
    FINGERPRINT_DIM,
    FINGERPRINT_SHAPE,
    IRIS_LEFT_DIM,
    IRIS_RIGHT_DIM,
    IRIS_SHAPE,
)


class BiometricDataset(Dataset):
    """
    Returns per-sample dict:
        fingerprint : Tensor (3, 128, 128)
        iris_left   : Tensor (1, 64, 64)
        iris_right  : Tensor (1, 64, 64)
        label       : int  ← global label from Parquet (consistent across splits)
    """

    def __init__(self, parquet_path: Path | str, active_modalities: list[str],
                 transform=None) -> None:
        self.active_modalities = active_modalities
        self.transform         = transform

        table = pq.read_table(
            Path(parquet_path),
            columns=["subject_id", "label", "sample_id",
                     "fingerprint", "iris_left", "iris_right"]
        )
        self._data = table.to_pydict()
        self._n    = len(self._data["subject_id"])

        # num_classes = max global label + 1 (consistent across splits)
        self._num_classes = max(self._data["label"]) + 1

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Tensor | int]:
        def to_tensor(raw, shape):
            if raw is None:
                return torch.zeros(shape, dtype=torch.float32)
            return torch.tensor(raw, dtype=torch.float32).reshape(shape)

        fp    = to_tensor(self._data["fingerprint"][idx], FINGERPRINT_SHAPE)
        left  = to_tensor(self._data["iris_left"][idx],   IRIS_SHAPE)
        right = to_tensor(self._data["iris_right"][idx],  IRIS_SHAPE)
        label = int(self._data["label"][idx])   # use global label directly

        if self.transform:
            if "fingerprint" in self.transform:
                fp = self.transform["fingerprint"](fp)
            if "iris" in self.transform:
                left  = self.transform["iris"](left)
                right = self.transform["iris"](right)

        return {
            "fingerprint": fp,
            "iris_left":   left,
            "iris_right":  right,
            "label":       label,
        }

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def feature_dims(self) -> dict[str, int]:
        return {
            "fingerprint": FINGERPRINT_DIM,
            "iris_left":   IRIS_LEFT_DIM,
            "iris_right":  IRIS_RIGHT_DIM,
        }
