"""
encoders.py
-----------
Per-modality encoder modules.

Each encoder projects a raw feature vector (face embedding, fingerprint
descriptor, voice features, or gait descriptor) into a shared latent space
of dimension ``hidden_dim``.  Projecting to a shared dimension before fusion
lets the fusion MLP treat all modalities symmetrically regardless of their
original feature sizes.

Architecture per encoder:
    Linear(input_dim → hidden_dim) → LayerNorm → GELU → Dropout
    Linear(hidden_dim → hidden_dim)  ← residual if dims match

Design notes:
    * LayerNorm (rather than BatchNorm) is used because it is stable at
      small batch sizes and works correctly with missing modalities (zero
      vectors still produce well-normalised outputs).
    * We keep encoders intentionally shallow; deep per-modality towers tend
      to overfit on small biometric datasets.  Depth should live in the
      fusion head where all modalities interact.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ModalityEncoder(nn.Module):
    """
    Generic single-modality projection encoder.

    Args:
        input_dim:   Dimensionality of the raw feature vector.
        hidden_dim:  Output embedding dimensionality (shared across modalities).
        dropout:     Dropout probability applied after the activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.3) -> None:
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)

        # Residual projection only needed when dims differ
        self._residual = (
            nn.Linear(input_dim, hidden_dim, bias=False)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch, input_dim) float tensor.
        Returns:
            (batch, hidden_dim) embedding tensor.
        """
        residual = self._residual(x)
        out = self.proj(x)
        out = self.out(out + residual)
        return out  # (B, hidden_dim)


def build_encoders(
    active_modalities: list[str],
    feature_dims: dict[str, int],
    hidden_dim: int,
    dropout: float,
) -> nn.ModuleDict:
    """
    Construct one ModalityEncoder per active modality and return as a
    ModuleDict so parameters are properly registered with the parent module.

    Args:
        active_modalities: E.g. ["face", "fingerprint", "voice"].
        feature_dims:      {modality: raw_feature_dim}.
        hidden_dim:        Shared projection dimension.
        dropout:           Dropout rate for all encoders.

    Returns:
        nn.ModuleDict with keys matching ``active_modalities``.
    """
    return nn.ModuleDict(
        {
            modality: ModalityEncoder(
                input_dim=feature_dims[modality],
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for modality in active_modalities
        }
    )
