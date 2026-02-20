"""
dataset.py
----------
PyTorch Dataset abstraction for the multimodal biometric Parquet store.

Design notes:
    * PyArrow is used for I/O rather than pandas because it reads Parquet
      column-by-column without materialising unnecessary data.  Only the
      modality columns that are active in the current config are loaded —
      reducing memory footprint proportionally to the number of disabled
      modalities.

    * The dataset is fully index-addressable (__getitem__ by integer) so it
      integrates transparently with torch.utils.data.DataLoader, including
      distributed samplers for multi-GPU training.

    * Missing modality values (NULL in Parquet) are replaced with zero
      vectors so the model can handle partially labelled samples without
      requiring a separate masking pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset

from biometric_ml.data.schema import MODALITY_REGISTRY

# Map modality name → column name in the fused Parquet schema
_MODALITY_COL: dict[str, str] = {
    "face": "face_embedding",
    "fingerprint": "fingerprint_features",
    "voice": "voice_features",
    "gait": "gait_features",
}


class BiometricDataset(Dataset):
    """
    Reads a split Parquet file and returns per-sample tensors.

    Args:
        parquet_path:       Path to the split's ``.parquet`` file.
        active_modalities:  List of modality names to load.
        transform:          Optional callable applied to the sample dict
                            after tensor conversion (e.g., augmentation).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        active_modalities: list[str],
        transform: Any | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        self.active_modalities = active_modalities
        self.transform = transform

        # Eagerly load the table into memory.
        # For datasets > available RAM, replace with pq.ParquetFile and
        # read row groups lazily in __getitem__.
        columns = ["subject_id"] + [
            _MODALITY_COL[m] for m in active_modalities if m in _MODALITY_COL
        ]
        self._table = pq.read_table(self.parquet_path, columns=columns)

        # Build a contiguous label array (subject_id → class index)
        subject_ids = self._table["subject_id"].to_pylist()
        unique_ids = sorted(set(subject_ids))
        self._id_to_class: dict[int, int] = {
            sid: idx for idx, sid in enumerate(unique_ids)
        }
        self._labels: list[int] = [self._id_to_class[sid] for sid in subject_ids]
        self.num_classes: int = len(unique_ids)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: int) -> dict[str, Tensor | int]:
        """
        Returns a dict with:
            * One float tensor per active modality keyed by modality name.
            * ``label``: integer class index.
        """
        row = self._table.slice(idx, 1)
        sample: dict[str, Tensor | int] = {"label": self._labels[idx]}

        for modality in self.active_modalities:
            col = _MODALITY_COL[modality]
            _, _, dim = MODALITY_REGISTRY[modality]
            value = row[col][0].as_py()

            if value is None:
                # Replace missing modality with a zero vector
                tensor = torch.zeros(dim, dtype=torch.float32)
            else:
                tensor = torch.tensor(value, dtype=torch.float32)

            sample[modality] = tensor

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def feature_dims(self) -> dict[str, int]:
        """Return {modality: dim} for all active modalities."""
        return {m: MODALITY_REGISTRY[m][2] for m in self.active_modalities}
