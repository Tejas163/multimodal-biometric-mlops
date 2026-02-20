"""
tests/test_model.py
-------------------
Unit tests for the encoder and fusion model modules.

Tests are parameter-free (no Parquet, no MLflow) and run on CPU only,
making them fast and CI-safe.
"""

from __future__ import annotations

import pytest
import torch

from biometric_ml.models.encoders import ModalityEncoder, build_encoders
from biometric_ml.models.fusion import BiometricFusionModel, FusionMLP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH = 8
ACTIVE = ["face", "fingerprint", "voice"]
FEATURE_DIMS = {"face": 512, "fingerprint": 256, "voice": 128}
NUM_CLASSES = 50
HIDDEN_DIM = 64


def _make_inputs(batch_size: int = BATCH) -> dict[str, torch.Tensor]:
    return {
        m: torch.randn(batch_size, FEATURE_DIMS[m])
        for m in ACTIVE
    }


def _make_model(fusion_method: str = "concat") -> BiometricFusionModel:
    return BiometricFusionModel(
        active_modalities=ACTIVE,
        feature_dims=FEATURE_DIMS,
        encoder_hidden_dim=HIDDEN_DIM,
        encoder_dropout=0.0,
        fusion_method=fusion_method,
        fusion_hidden_dims=[128, 64],
        fusion_dropout=0.0,
        num_classes=NUM_CLASSES,
    )


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------


class TestModalityEncoder:
    def test_output_shape(self):
        enc = ModalityEncoder(input_dim=512, hidden_dim=HIDDEN_DIM)
        x = torch.randn(BATCH, 512)
        out = enc(x)
        assert out.shape == (BATCH, HIDDEN_DIM)

    def test_residual_projection_different_dims(self):
        enc = ModalityEncoder(input_dim=512, hidden_dim=HIDDEN_DIM)
        # Ensure residual Linear is used (not Identity)
        assert not isinstance(enc._residual, torch.nn.Identity)

    def test_residual_identity_same_dims(self):
        enc = ModalityEncoder(input_dim=HIDDEN_DIM, hidden_dim=HIDDEN_DIM)
        assert isinstance(enc._residual, torch.nn.Identity)

    def test_build_encoders_returns_module_dict(self):
        enc_dict = build_encoders(ACTIVE, FEATURE_DIMS, HIDDEN_DIM, dropout=0.0)
        assert isinstance(enc_dict, torch.nn.ModuleDict)
        assert set(enc_dict.keys()) == set(ACTIVE)


# ---------------------------------------------------------------------------
# Fusion model tests
# ---------------------------------------------------------------------------


class TestBiometricFusionModel:
    @pytest.mark.parametrize("method", ["concat", "attention", "mean"])
    def test_output_shape_all_fusion_methods(self, method):
        model = _make_model(fusion_method=method)
        inputs = _make_inputs()
        logits = model(inputs)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_invalid_fusion_method_raises(self):
        with pytest.raises(ValueError, match="Unknown fusion method"):
            _make_model(fusion_method="invalid")

    def test_no_nan_in_output(self):
        model = _make_model()
        inputs = _make_inputs()
        logits = model(inputs)
        assert not torch.isnan(logits).any()

    def test_zero_input_no_crash(self):
        """Model should handle zero vectors (missing modality stand-in)."""
        model = _make_model()
        inputs = {m: torch.zeros(BATCH, FEATURE_DIMS[m]) for m in ACTIVE}
        logits = model(inputs)
        assert logits.shape == (BATCH, NUM_CLASSES)

    def test_parameter_count_positive(self):
        model = _make_model()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0

    def test_eval_mode_no_dropout_effect(self):
        """In eval mode two identical forward passes must give identical output."""
        model = _make_model()
        model.eval()
        inputs = _make_inputs()
        with torch.no_grad():
            out1 = model(inputs)
            out2 = model(inputs)
        assert torch.allclose(out1, out2)

    def test_gradient_flows(self):
        model = _make_model()
        model.train()
        inputs = _make_inputs()
        logits = model(inputs)
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_from_config_factory(self):
        model_cfg = {
            "encoder_hidden_dim": HIDDEN_DIM,
            "encoder_dropout": 0.0,
            "fusion": {"method": "concat", "hidden_dims": [64], "dropout": 0.0},
        }
        data_cfg = {
            "modalities": {"face": True, "fingerprint": True, "voice": True, "gait": False},
        }
        model = BiometricFusionModel.from_config(
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            num_classes=NUM_CLASSES,
            feature_dims=FEATURE_DIMS,
        )
        out = model(_make_inputs())
        assert out.shape == (BATCH, NUM_CLASSES)


# ---------------------------------------------------------------------------
# FusionMLP tests
# ---------------------------------------------------------------------------


class TestFusionMLP:
    def test_output_shape(self):
        mlp = FusionMLP(input_dim=128, hidden_dims=[64, 32], num_classes=10)
        x = torch.randn(BATCH, 128)
        assert mlp(x).shape == (BATCH, 10)
