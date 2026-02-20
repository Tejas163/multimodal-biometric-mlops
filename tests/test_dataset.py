"""
tests/test_dataset.py
---------------------
Unit tests for the BiometricDataset and PyArrow schema layer.

These tests are fully self-contained — they generate synthetic Parquet
files in a temp directory and tear them down after each test, so they
run without any external data or network access (CI-friendly).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from biometric_ml.data.dataset import BiometricDataset
from biometric_ml.data.schema import FUSED_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_parquet(path: Path, n_subjects: int = 5, n_samples_each: int = 3) -> None:
    """Write a minimal valid fused Parquet file for testing."""
    rng = np.random.default_rng(0)
    rows = []
    for subj in range(n_subjects):
        for i in range(n_samples_each):
            rows.append(
                {
                    "subject_id": subj,
                    "sample_id": f"s{subj}_i{i}",
                    "face_embedding": rng.random(512).astype(np.float32).tolist(),
                    "fingerprint_features": rng.random(256).astype(np.float32).tolist(),
                    "voice_features": rng.random(128).astype(np.float32).tolist(),
                    "gait_features": None,
                    "split": "train",
                }
            )
    table = pa.Table.from_pylist(rows, schema=FUSED_SCHEMA)
    pq.write_table(table, path)


@pytest.fixture()
def parquet_path(tmp_path: Path) -> Path:
    p = tmp_path / "train.parquet"
    _make_parquet(p)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBiometricDataset:
    def test_length(self, parquet_path: Path) -> None:
        ds = BiometricDataset(parquet_path, active_modalities=["face", "fingerprint", "voice"])
        assert len(ds) == 15  # 5 subjects × 3 samples

    def test_item_keys(self, parquet_path: Path) -> None:
        ds = BiometricDataset(parquet_path, active_modalities=["face", "voice"])
        item = ds[0]
        assert "face" in item
        assert "voice" in item
        assert "label" in item
        assert "fingerprint" not in item

    def test_tensor_dtypes_and_shapes(self, parquet_path: Path) -> None:
        ds = BiometricDataset(parquet_path, active_modalities=["face", "fingerprint"])
        item = ds[0]
        assert item["face"].dtype == torch.float32
        assert item["face"].shape == (512,)
        assert item["fingerprint"].shape == (256,)

    def test_label_is_int(self, parquet_path: Path) -> None:
        ds = BiometricDataset(parquet_path, active_modalities=["face"])
        item = ds[0]
        assert isinstance(item["label"], int)

    def test_num_classes(self, parquet_path: Path) -> None:
        ds = BiometricDataset(parquet_path, active_modalities=["face"])
        assert ds.num_classes == 5

    def test_missing_modality_zero_vector(self, parquet_path: Path) -> None:
        """Gait is NULL in fixture — should return zeros, not crash."""
        ds = BiometricDataset(parquet_path, active_modalities=["face", "gait"])
        item = ds[0]
        assert item["gait"].shape == (64,)
        assert (item["gait"] == 0).all()

    def test_feature_dims_property(self, parquet_path: Path) -> None:
        ds = BiometricDataset(parquet_path, active_modalities=["face", "voice"])
        assert ds.feature_dims == {"face": 512, "voice": 128}

    def test_index_range(self, parquet_path: Path) -> None:
        ds = BiometricDataset(parquet_path, active_modalities=["face"])
        for i in [0, 7, 14]:
            item = ds[i]
            assert "face" in item

    def test_schema_enforcement(self, tmp_path: Path) -> None:
        """Writing with wrong column types should raise a specific ArrowInvalid error."""
        bad_rows = [{"subject_id": "not-an-int", "sample_id": "x"}]
        with pytest.raises(pa.lib.ArrowInvalid):
            pa.Table.from_pylist(bad_rows, schema=FUSED_SCHEMA)
