"""
fusion.py
---------
Multimodal late-fusion model for biometric user recognition.

Architecture overview:
                ┌─────────────┐
    face ──────▶│ FaceEncoder │──┐
                └─────────────┘  │
                ┌──────────────┐ │   ┌─────────────────────────┐
    finger ────▶│ FingEncoder  │─┼──▶│  FusionMLP (concat/attn)│──▶ logits
                └──────────────┘ │   └─────────────────────────┘
                ┌─────────────┐  │
    voice ─────▶│ VoiceEncoder│──┘
                └─────────────┘

Fusion strategies (config.model.fusion.method):
    * ``concat``    — concatenate embeddings → MLP  (baseline, fast)
    * ``attention`` — learned cross-modality attention weights before concat
    * ``mean``      — simple average pooling (parameter-free ablation)

The forward pass accepts a dict of tensors rather than positional args so
that:
  (a) callers are agnostic to modality order,
  (b) missing modalities can be masked without changing the call signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from biometric_ml.models.encoders import build_encoders


class AttentionFusion(nn.Module):
    """
    Softmax-weighted fusion: learns a scalar importance weight per modality.

    Shape: (B, n_modalities, hidden_dim) → (B, hidden_dim)
    """

    def __init__(self, n_modalities: int, hidden_dim: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, embeddings: list[Tensor]) -> Tensor:
        stacked = torch.stack(embeddings, dim=1)          # (B, M, H)
        scores = self.attn(stacked).squeeze(-1)            # (B, M)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, M, 1)
        fused = (stacked * weights).sum(dim=1)             # (B, H)
        return fused


class FusionMLP(nn.Module):
    """
    MLP classifier head applied after embedding fusion.

    Args:
        input_dim:   Dimensionality of fused vector fed into the MLP.
        hidden_dims: List of hidden layer widths.
        num_classes: Number of output classes (unique users).
        dropout:     Dropout probability between hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BiometricFusionModel(nn.Module):
    """
    Full multimodal biometric recognition model.

    Args:
        active_modalities: List of modality names present in inputs.
        feature_dims:      {modality: raw_feature_dim} mapping.
        encoder_hidden_dim: Shared encoder output dimension.
        encoder_dropout:   Dropout inside encoders.
        fusion_method:     One of ``concat``, ``attention``, ``mean``.
        fusion_hidden_dims: MLP layer sizes after fusion.
        fusion_dropout:    Dropout inside the fusion MLP.
        num_classes:       Number of unique identities to recognise.
    """

    def __init__(
        self,
        active_modalities: list[str],
        feature_dims: dict[str, int],
        encoder_hidden_dim: int = 256,
        encoder_dropout: float = 0.3,
        fusion_method: str = "concat",
        fusion_hidden_dims: list[int] | None = None,
        fusion_dropout: float = 0.4,
        num_classes: int = 100,
    ) -> None:
        super().__init__()

        if fusion_hidden_dims is None:
            fusion_hidden_dims = [512, 256]

        self.active_modalities = active_modalities
        self.fusion_method = fusion_method

        # ── Per-modality encoders ──────────────────────────────────────────
        self.encoders = build_encoders(
            active_modalities, feature_dims, encoder_hidden_dim, encoder_dropout
        )

        # ── Fusion layer ───────────────────────────────────────────────────
        n = len(active_modalities)
        if fusion_method == "concat":
            fusion_input_dim = encoder_hidden_dim * n
        elif fusion_method in {"attention", "mean"}:
            fusion_input_dim = encoder_hidden_dim
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method!r}")

        if fusion_method == "attention":
            self.attention = AttentionFusion(n, encoder_hidden_dim)

        # ── Classification head ────────────────────────────────────────────
        self.classifier = FusionMLP(
            input_dim=fusion_input_dim,
            hidden_dims=fusion_hidden_dims,
            num_classes=num_classes,
            dropout=fusion_dropout,
        )

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict mapping modality name → (batch, feature_dim) tensor.
                    Keys must match ``self.active_modalities``.
        Returns:
            logits: (batch, num_classes) unnormalised class scores.
        """
        embeddings: list[Tensor] = [
            self.encoders[m](inputs[m]) for m in self.active_modalities
        ]

        if self.fusion_method == "concat":
            fused = torch.cat(embeddings, dim=-1)
        elif self.fusion_method == "attention":
            fused = self.attention(embeddings)
        elif self.fusion_method == "mean":
            fused = torch.stack(embeddings, dim=0).mean(dim=0)
        else:
            raise RuntimeError("Unreachable")

        return self.classifier(fused)

    # ------------------------------------------------------------------
    # Convenience: build from Hydra config dicts
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_cfg: dict,
        data_cfg: dict,
        num_classes: int,
        feature_dims: dict[str, int],
    ) -> "BiometricFusionModel":
        """Construct the model directly from Hydra OmegaConf dicts."""
        active = [m for m, enabled in data_cfg["modalities"].items() if enabled]
        return cls(
            active_modalities=active,
            feature_dims=feature_dims,
            encoder_hidden_dim=model_cfg["encoder_hidden_dim"],
            encoder_dropout=model_cfg["encoder_dropout"],
            fusion_method=model_cfg["fusion"]["method"],
            fusion_hidden_dims=list(model_cfg["fusion"]["hidden_dims"]),
            fusion_dropout=model_cfg["fusion"]["dropout"],
            num_classes=num_classes,
        )
