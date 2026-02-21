"""
schema.py
---------
Canonical PyArrow schemas for the Kaggle Multimodal Iris+Fingerprint dataset.

Dataset structure (ninadmehendale/multimodal-iris-fingerprint-biometric-data):
    data/
      iris/
        {subject_id}_left/   fatmal1.bmp ... fatmal5.bmp
        {subject_id}_right/  fatmar1.bmp ... fatmar5.bmp
      fingerprint/
        {subject_id}_left/   ...
        {subject_id}_right/  ...

Feature extraction:
    Iris        → HOG descriptor on 64x64 grayscale BMP  → 1764-d vector
    Fingerprint → HOG descriptor on 96x96 grayscale BMP  → 1764-d vector

Both left+right samples are averaged into one vector per sample per modality,
giving one fused row per (subject, sample_index) pair.
"""

from __future__ import annotations

import pyarrow as pa

# Feature vector dimensions after HOG extraction
IRIS_DIM         = 1764   # HOG on 64x64, 9 orientations, 2x2 cells per block
FINGERPRINT_DIM  = 1764   # HOG on 96x96, same params — matches iris dim

IRIS_SCHEMA = pa.schema([
    pa.field("subject_id",  pa.int32(),  nullable=False),
    pa.field("sample_id",   pa.string(), nullable=False),
    pa.field("features",    pa.list_(pa.float32(), IRIS_DIM), nullable=False),
    pa.field("side",        pa.string()),   # "left" | "right" | "both"
    pa.field("quality",     pa.float32()),
])

FINGERPRINT_SCHEMA = pa.schema([
    pa.field("subject_id",  pa.int32(),  nullable=False),
    pa.field("sample_id",   pa.string(), nullable=False),
    pa.field("features",    pa.list_(pa.float32(), FINGERPRINT_DIM), nullable=False),
    pa.field("side",        pa.string()),
    pa.field("quality",     pa.float32()),
])

# Fused schema — one row per (subject, sample_index)
FUSED_SCHEMA = pa.schema([
    pa.field("subject_id",            pa.int32(),  nullable=False),
    pa.field("sample_id",             pa.string(), nullable=False),
    pa.field("iris_features",         pa.list_(pa.float32(), IRIS_DIM)),
    pa.field("fingerprint_features",  pa.list_(pa.float32(), FINGERPRINT_DIM)),
    pa.field("split",                 pa.string(), nullable=False),
])

# Registry: modality → (schema, feature_field, vector_dim)
MODALITY_REGISTRY: dict[str, tuple[pa.Schema, str, int]] = {
    "iris":        (IRIS_SCHEMA,        "features", IRIS_DIM),
    "fingerprint": (FINGERPRINT_SCHEMA, "features", FINGERPRINT_DIM),
}
