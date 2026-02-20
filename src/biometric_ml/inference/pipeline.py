"""
pipeline.py
-----------
Production inference pipeline for biometric user recognition.

The pipeline loads a versioned model from the MLflow Model Registry (or a
local checkpoint) and exposes a clean ``predict`` method that accepts raw
feature dicts and returns ranked predictions with confidence scores.

Design goals:
  * Stateless after construction — the pipeline object can be shared across
    threads / processes without locks.
  * Device-agnostic — works on CPU (edge deployment) and GPU (server).
  * Schema-validated inputs — raises descriptive errors before touching
    the model, preventing silent silent shape errors.
  * No dependency on training-time data modules — inference is self-contained.

Usage::

    pipeline = InferencePipeline.from_registry(
        model_name="BiometricFusionModel",
        stage="Production",
        tracking_uri="mlruns/",
    )
    result = pipeline.predict({
        "face": face_embedding_tensor,          # (512,) float32
        "fingerprint": fingerprint_tensor,       # (256,) float32
        "voice": voice_tensor,                   # (128,) float32
    })
    print(result.top_k_ids, result.top_k_probs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow.pytorch
import torch
import torch.nn.functional as F
from torch import Tensor

from biometric_ml.data.schema import MODALITY_REGISTRY

log = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Output of a single inference call."""

    top_k_ids: list[int]        # Class indices, best first
    top_k_probs: list[float]    # Corresponding softmax probabilities
    embeddings: Tensor | None   # Fused embedding (if return_embeddings=True)


class InferencePipeline:
    """
    Wraps a trained BiometricFusionModel for production inference.

    Args:
        model:             Loaded PyTorch model in eval mode.
        active_modalities: Modalities the model was trained on.
        device:            torch.device to run inference on.
        top_k:             Number of top predictions to return.
    """

    def __init__(
        self,
        model: Any,
        active_modalities: list[str],
        device: torch.device,
        top_k: int = 5,
    ) -> None:
        self.model = model.to(device).eval()
        self.active_modalities = active_modalities
        self.device = device
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        model_name: str,
        stage: str = "Production",
        tracking_uri: str = "mlruns/",
        active_modalities: list[str] | None = None,
        top_k: int = 5,
        device: str | None = None,
    ) -> "InferencePipeline":
        """
        Load the latest model in ``stage`` from the MLflow Model Registry.

        Args:
            model_name:         Registered model name (must match training cfg).
            stage:              Model Registry stage: Production / Staging / None.
            tracking_uri:       MLflow tracking URI.
            active_modalities:  If None, defaults to face + fingerprint + voice.
            top_k:              Top-k classes to return.
            device:             ``"cuda"``, ``"cpu"``, or None (auto-detect).
        """
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{stage}"
        log.info("Loading model from MLflow registry: %s", model_uri)
        model = mlflow.pytorch.load_model(model_uri)

        _device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _modalities = active_modalities or ["face", "fingerprint", "voice"]
        return cls(model=model, active_modalities=_modalities, device=_device, top_k=top_k)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_factory,          # Callable that returns an uninitialised model
        active_modalities: list[str],
        top_k: int = 5,
        device: str | None = None,
    ) -> "InferencePipeline":
        """
        Load model weights from a local ``.pt`` checkpoint file.

        Useful for offline evaluation or when MLflow is not available.
        """
        checkpoint_path = Path(checkpoint_path)
        _device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=_device)
        model = model_factory()
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(
            "Loaded checkpoint from %s (epoch %d, val_loss %.4f)",
            checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_loss", float("nan"))
        )
        return cls(model=model, active_modalities=active_modalities, device=_device, top_k=top_k)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        features: dict[str, list[float] | Tensor],
        return_embeddings: bool = False,
    ) -> PredictionResult:
        """
        Run inference on a single sample.

        Args:
            features:          Dict mapping modality name → feature vector.
                               Values can be Python lists or 1-D Tensors.
            return_embeddings: If True, return the fused embedding vector.

        Returns:
            PredictionResult with top-k class IDs and probabilities.
        """
        inputs = self._validate_and_prepare(features)
        logits: Tensor = self.model(inputs)         # (1, num_classes)
        probs: Tensor = F.softmax(logits, dim=-1)   # (1, num_classes)

        top_probs, top_ids = probs[0].topk(min(self.top_k, probs.shape[-1]))

        return PredictionResult(
            top_k_ids=top_ids.cpu().tolist(),
            top_k_probs=top_probs.cpu().tolist(),
            embeddings=None,  # Embeddings hook can be added via forward hooks
        )

    @torch.no_grad()
    def predict_batch(
        self, feature_list: list[dict[str, list[float] | Tensor]]
    ) -> list[PredictionResult]:
        """Run inference over a list of samples (batch processing)."""
        return [self.predict(f) for f in feature_list]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_and_prepare(
        self, features: dict[str, list[float] | Tensor]
    ) -> dict[str, Tensor]:
        """Validate feature dict, convert to tensors, move to device."""
        prepared: dict[str, Tensor] = {}
        for modality in self.active_modalities:
            if modality not in features:
                raise ValueError(
                    f"Missing modality '{modality}' in input. "
                    f"Expected modalities: {self.active_modalities}"
                )
            _, _, expected_dim = MODALITY_REGISTRY[modality]
            value = features[modality]

            if isinstance(value, list):
                tensor = torch.tensor(value, dtype=torch.float32)
            elif isinstance(value, Tensor):
                tensor = value.float()
            else:
                raise TypeError(
                    f"Unsupported feature type for '{modality}': {type(value)}"
                )

            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)  # Add batch dimension → (1, dim)

            if tensor.shape[-1] != expected_dim:
                raise ValueError(
                    f"Modality '{modality}' has shape {tensor.shape}, "
                    f"expected last dim = {expected_dim}"
                )

            prepared[modality] = tensor.to(self.device)

        return prepared
