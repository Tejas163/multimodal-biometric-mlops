"""
dataset.py
----------
PyTorch Dataset that reads iris + fingerprint features from Parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from biometric_ml.data.schema import FINGERPRINT_DIM, IRIS_DIM, MODALITY_REGISTRY

# Map modality name → (parquet column, expected dim)
_COLUMN_MAP = {
    "iris":        ("iris_features",        IRIS_DIM),
    "fingerprint": ("fingerprint_features", FINGERPRINT_DIM),
}


class BiometricDataset(Dataset):
    """
    Index-addressable PyTorch Dataset backed by a Parquet file.

    Args:
        parquet_path:      Path to a split Parquet file (train/val/test).
        active_modalities: Subset of ["iris", "fingerprint"] to load.
    """

    def __init__(self, parquet_path: Path | str, active_modalities: list[str]) -> None:
        self.parquet_path = Path(parquet_path)
        self.active_modalities = active_modalities

        columns = ["subject_id", "sample_id"] + [
            _COLUMN_MAP[m][0] for m in active_modalities if m in _COLUMN_MAP
        ]
        table = pq.read_table(self.parquet_path, columns=columns)
        self._data = table.to_pydict()

        # Build contiguous integer labels from subject_ids
        unique_ids = sorted(set(self._data["subject_id"]))
        self._label_map: dict[int, int] = {sid: i for i, sid in enumerate(unique_ids)}
        self._n = len(self._data["subject_id"])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, Tensor | int]:
        item: dict[str, Any] = {
            "label": self._label_map[self._data["subject_id"][idx]]
        }
        for modality in self.active_modalities:
            col, dim = _COLUMN_MAP[modality]
            raw = self._data[col][idx]
            if raw is None:
                item[modality] = torch.zeros(dim, dtype=torch.float32)
            else:
                item[modality] = torch.tensor(raw, dtype=torch.float32)
        return item

    @property
    def num_classes(self) -> int:
        return len(self._label_map)

    @property
    def feature_dims(self) -> dict[str, int]:
        return {m: _COLUMN_MAP[m][1] for m in self.active_modalities if m in _COLUMN_MAP}
