"""
dataset.py
----------
PyTorch Dataset returning raw image tensors in CHW format.
Matches the Kaggle notebook's image loading approach.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from biometric_ml.data.schema import (
    FINGERPRINT_DIM, FINGERPRINT_SHAPE,
    IRIS_LEFT_DIM, IRIS_RIGHT_DIM, IRIS_SHAPE,
)


class BiometricDataset(Dataset):
    """
    Returns per-sample dict with keys:
        fingerprint : Tensor (3, 128, 128)  — RGB normalised [0,1]
        iris_left   : Tensor (1, 64, 64)    — grayscale normalised [0,1]
        iris_right  : Tensor (1, 64, 64)    — grayscale normalised [0,1]
        label       : int
    """

    def __init__(self, parquet_path: Path | str, active_modalities: list[str]) -> None:
        self.active_modalities = active_modalities
        table = pq.read_table(
            Path(parquet_path),
            columns=["subject_id", "sample_id", "fingerprint", "iris_left", "iris_right"]
        )
        self._data = table.to_pydict()

        unique_ids      = sorted(set(self._data["subject_id"]))
        self._label_map = {sid: i for i, sid in enumerate(unique_ids)}
        self._n         = len(self._data["subject_id"])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Tensor | int]:
        def to_tensor(raw, shape):
            if raw is None:
                return torch.zeros(shape, dtype=torch.float32)
            return torch.tensor(raw, dtype=torch.float32).reshape(shape)

        return {
            "fingerprint": to_tensor(self._data["fingerprint"][idx], FINGERPRINT_SHAPE),
            "iris_left":   to_tensor(self._data["iris_left"][idx],   IRIS_SHAPE),
            "iris_right":  to_tensor(self._data["iris_right"][idx],  IRIS_SHAPE),
            "label":       self._label_map[self._data["subject_id"][idx]],
        }

    @property
    def num_classes(self) -> int:
        return len(self._label_map)

    @property
    def feature_dims(self) -> dict[str, int]:
        return {
            "fingerprint": FINGERPRINT_DIM,
            "iris_left":   IRIS_LEFT_DIM,
            "iris_right":  IRIS_RIGHT_DIM,
        }
