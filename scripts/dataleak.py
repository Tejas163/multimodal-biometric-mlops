"""Check for data leakage between train/val/test splits and model output diversity."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biometric_ml.models.fusion import BiometricFusionModel

# 1. Check label overlap
for split in ["train", "val", "test"]:
    table = pq.read_table(f"data/parquet/{split}.parquet", columns=["label"])
    labels = set(table["label"].to_pylist())
    print(f"{split}: {len(labels)} labels, {min(labels)}-{max(labels)}")

# 2. Check if model produces varying outputs
model = BiometricFusionModel(num_classes=45)
model.eval()

with torch.no_grad():
    inp1 = {
        "fingerprint": torch.rand(2, 3, 128, 128),
        "iris_left": torch.rand(2, 1, 64, 64),
        "iris_right": torch.rand(2, 1, 64, 64),
    }
    inp2 = {
        "fingerprint": torch.rand(2, 3, 128, 128),
        "iris_left": torch.rand(2, 1, 64, 64),
        "iris_right": torch.rand(2, 1, 64, 64),
    }

    out1 = model(inp1)
    out2 = model(inp2)

    print(f"Output 1: {out1[0, :5].tolist()}")
    print(f"Output 2: {out2[0, :5].tolist()}")
    print(f"Are they different? {not torch.allclose(out1, out2)}")
    print(f"Preds 1: {out1.argmax(dim=-1).tolist()}")
    print(f"Preds 2: {out2.argmax(dim=-1).tolist()}")
