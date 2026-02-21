"""
schema.py
---------
PyArrow schema for the Kaggle Multimodal Iris+Fingerprint dataset.

Each row stores raw pixel data as flattened float32 arrays:
    fingerprint : 128 x 128 x 3 (RGB)  = 49152 values
    iris_left   : 64  x 64  x 1 (gray) = 4096  values
    iris_right  : 64  x 64  x 1 (gray) = 4096  values
"""

from __future__ import annotations

import pyarrow as pa

FINGERPRINT_DIM = 128 * 128 * 3   # 49152 — flattened RGB image
IRIS_LEFT_DIM   = 64  * 64  * 1   # 4096  — flattened grayscale image
IRIS_RIGHT_DIM  = 64  * 64  * 1   # 4096  — flattened grayscale image

# Image shapes for reshaping inside the model
FINGERPRINT_SHAPE = (3,   128, 128)   # CHW for PyTorch
IRIS_SHAPE        = (1,   64,  64)    # CHW for PyTorch

FUSED_SCHEMA = pa.schema([
    pa.field("subject_id",      pa.int32(),  nullable=False),
    pa.field("label",           pa.int32(),  nullable=False),  # global 0-indexed label (consistent across all splits)
    pa.field("sample_id",       pa.string(), nullable=False),
    pa.field("fingerprint",     pa.list_(pa.float32(), FINGERPRINT_DIM), nullable=False),
    pa.field("iris_left",       pa.list_(pa.float32(), IRIS_LEFT_DIM),   nullable=False),
    pa.field("iris_right",      pa.list_(pa.float32(), IRIS_RIGHT_DIM),  nullable=False),
    pa.field("split",           pa.string(), nullable=False),
])

MODALITY_REGISTRY: dict[str, tuple[pa.Schema, str, int]] = {
    "fingerprint": (FUSED_SCHEMA, "fingerprint", FINGERPRINT_DIM),
    "iris_left":   (FUSED_SCHEMA, "iris_left",   IRIS_LEFT_DIM),
    "iris_right":  (FUSED_SCHEMA, "iris_right",  IRIS_RIGHT_DIM),
}
