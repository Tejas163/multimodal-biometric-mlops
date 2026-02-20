"""
schema.py
---------
Canonical PyArrow schemas for each biometric modality.

Design note:
    All raw data is normalised to fixed-width floating-point feature vectors
    and stored as Parquet files keyed by subject_id and sample_id.  Using
    explicit schemas enforces contracts across ingestion, training, and
    inference — any upstream change that breaks the schema is caught at
    ingest time rather than silently corrupting model inputs.
"""

from __future__ import annotations

import pyarrow as pa

# ---------------------------------------------------------------------------
# Per-modality feature schemas
# ---------------------------------------------------------------------------

FACE_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int32(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        # 512-d L2-normalised face embedding (e.g., from ArcFace / FaceNet)
        pa.field("embedding", pa.list_(pa.float32(), 512), nullable=False),
        pa.field("source", pa.string()),          # camera / sensor identifier
        pa.field("quality_score", pa.float32()),  # optional liveness / quality
    ]
)

FINGERPRINT_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int32(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        # 256-d minutiae descriptor vector
        pa.field("features", pa.list_(pa.float32(), 256), nullable=False),
        pa.field("finger_id", pa.int8()),         # 0-9 for each finger
        pa.field("quality_score", pa.float32()),
    ]
)

VOICE_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int32(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        # 128-d MFCC / x-vector speaker embedding
        pa.field("features", pa.list_(pa.float32(), 128), nullable=False),
        pa.field("duration_sec", pa.float32()),
        pa.field("snr_db", pa.float32()),         # signal-to-noise ratio
    ]
)

GAIT_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int32(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        # 64-d gait cycle descriptor
        pa.field("features", pa.list_(pa.float32(), 64), nullable=False),
        pa.field("num_steps", pa.int16()),
    ]
)

# ---------------------------------------------------------------------------
# Fused (joined) sample schema — produced by the dataset class
# ---------------------------------------------------------------------------

FUSED_SCHEMA = pa.schema(
    [
        pa.field("subject_id", pa.int32(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("face_embedding", pa.list_(pa.float32(), 512)),
        pa.field("fingerprint_features", pa.list_(pa.float32(), 256)),
        pa.field("voice_features", pa.list_(pa.float32(), 128)),
        pa.field("gait_features", pa.list_(pa.float32(), 64)),
        pa.field("split", pa.string(), nullable=False),  # train/val/test
    ]
)

# Map modality name → (schema, feature field name, vector length)
MODALITY_REGISTRY: dict[str, tuple[pa.Schema, str, int]] = {
    "face": (FACE_SCHEMA, "embedding", 512),
    "fingerprint": (FINGERPRINT_SCHEMA, "features", 256),
    "voice": (VOICE_SCHEMA, "features", 128),
    "gait": (GAIT_SCHEMA, "features", 64),
}
