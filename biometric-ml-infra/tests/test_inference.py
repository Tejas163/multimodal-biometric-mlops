"""
tests/test_inference.py
-----------------------
Unit tests for the InferencePipeline (checkpoint loading path only;
MLflow Registry path requires a running MLflow server).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from biometric_ml.inference.pipeline import InferencePipeline, PredictionResult
from biometric_ml.models.fusion import BiometricFusionModel

ACTIVE = ["face", "fingerprint", "voice"]
FEATURE_DIMS = {"face": 512, "fingerprint": 256, "voice": 128}
NUM_CLASSES = 20


def _make_model() -> BiometricFusionModel:
    return BiometricFusionModel(
        active_modalities=ACTIVE,
        feature_dims=FEATURE_DIMS,
        encoder_hidden_dim=64,
        encoder_dropout=0.0,
        fusion_method="concat",
        fusion_hidden_dims=[64],
        fusion_dropout=0.0,
        num_classes=NUM_CLASSES,
    )


def _save_checkpoint(path: Path, model: BiometricFusionModel) -> None:
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "val_loss": 0.5,
        },
        path,
    )


@pytest.fixture()
def pipeline(tmp_path):
    model = _make_model()
    ckpt_path = tmp_path / "model.pt"
    _save_checkpoint(ckpt_path, model)

    def factory():
        return _make_model()

    return InferencePipeline.from_checkpoint(
        checkpoint_path=ckpt_path,
        model_factory=factory,
        active_modalities=ACTIVE,
        top_k=5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInferencePipeline:
    def test_predict_returns_result(self, pipeline):
        features = {m: torch.randn(FEATURE_DIMS[m]).tolist() for m in ACTIVE}
        result = pipeline.predict(features)
        assert isinstance(result, PredictionResult)

    def test_top_k_length(self, pipeline):
        features = {m: torch.randn(FEATURE_DIMS[m]).tolist() for m in ACTIVE}
        result = pipeline.predict(features)
        assert len(result.top_k_ids) == 5
        assert len(result.top_k_probs) == 5

    def test_probs_sum_to_one_approx(self, pipeline):
        """Top-k probs won't sum to 1 if k < num_classes; just check range."""
        features = {m: torch.randn(FEATURE_DIMS[m]).tolist() for m in ACTIVE}
        result = pipeline.predict(features)
        for p in result.top_k_probs:
            assert 0.0 <= p <= 1.0

    def test_tensor_input_accepted(self, pipeline):
        features = {m: torch.randn(FEATURE_DIMS[m]) for m in ACTIVE}
        result = pipeline.predict(features)
        assert isinstance(result, PredictionResult)

    def test_missing_modality_raises(self, pipeline):
        features = {"face": torch.randn(512).tolist()}  # missing fingerprint+voice
        with pytest.raises(ValueError, match="Missing modality"):
            pipeline.predict(features)

    def test_wrong_dim_raises(self, pipeline):
        features = {
            "face": torch.randn(128).tolist(),       # wrong dim (expected 512)
            "fingerprint": torch.randn(256).tolist(),
            "voice": torch.randn(128).tolist(),
        }
        with pytest.raises(ValueError, match="expected last dim"):
            pipeline.predict(features)

    def test_predict_batch(self, pipeline):
        feature_list = [
            {m: torch.randn(FEATURE_DIMS[m]).tolist() for m in ACTIVE}
            for _ in range(4)
        ]
        results = pipeline.predict_batch(feature_list)
        assert len(results) == 4
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_deterministic_in_eval_mode(self, pipeline):
        features = {m: torch.randn(FEATURE_DIMS[m]) for m in ACTIVE}
        r1 = pipeline.predict(features)
        r2 = pipeline.predict(features)
        assert r1.top_k_ids == r2.top_k_ids
        assert r1.top_k_probs == r2.top_k_probs
