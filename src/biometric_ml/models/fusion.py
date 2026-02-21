"""
fusion.py
---------
PyTorch port of the Kaggle notebook multimodal biometric model.

Architecture:
    Fingerprint : MobileNetV2 pretrained + FROZEN → (B, 1280)
    Iris        : Shared CNN (left + right) → (B, 32) each
    Fusion      : concat → BN → Linear(256) → Dropout(0.5) → Linear(num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class IrisBranch(nn.Module):
    """
    Shared CNN for iris images (left and right use same weights).
    Input:  (B, 1, 64, 64)
    Output: (B, 32)
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=0), # Fewer filters, larger stride
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.5), # Ultra-high dropout
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 64), # Smaller output dimension (16 instead of 32/64)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
        
class GatedFusion(nn.Module):
    def __init__(self, fp_dim, iris_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(fp_dim + iris_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, fp, left, right):
        stacked = torch.cat([fp, left, right], dim=-1)
        weights = self.gate(stacked)
        return fp * weights[:, 0:1], left * weights[:, 1:2], right * weights[:, 2:3]

class BiometricFusionModel(nn.Module):
    """
    Full multimodal model.

    Inputs:
        fingerprint : (B, 3, 128, 128)
        iris_left   : (B, 1, 64, 64)
        iris_right  : (B, 1, 64, 64)
    Output:
        logits : (B, num_classes)
    """

    def __init__(self, num_classes: int = 45, freeze_mobilenet: bool = True) -> None:
        super().__init__()

        # ── Fingerprint: MobileNetV2, FROZEN ──────────────────────────────
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.fingerprint_branch = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),               # → (B, 1280)
        )
        self.set_mobilenet_frozen(freeze_mobilenet)
        
        for param in self.fingerprint_branch.parameters():
            param.requires_grad = False  # frozen — prevents overfitting on small dataset

        # ── Iris: shared CNN ──────────────────────────────────────────────
        self.iris_branch = IrisBranch()  # → (B, 64) per eye; called twice in forward
        # --- Fusion layer --------------------------------------------
        self.fusion_layer = GatedFusion(fp_dim=1280, iris_dim=64)
        # ── Classifier: 1280 + 64 + 64 = 1408 → num_classes ─────────────
        fusion_dim = 1280 + 64 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), # Reduced from 128 to prevent overfitting
            nn.ReLU(),
            nn.Dropout(0.5),           # High dropout to fight overfitting
            nn.Linear(256, num_classes),
        )
    
    def set_mobilenet_frozen(self, freeze: bool):
        for param in self.fingerprint_branch.parameters():
            param.requires_grad = not freeze

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        fp_feat    = self.fingerprint_branch(inputs["fingerprint"])  # (B, 1280)
        left_feat  = self.iris_branch(inputs["iris_left"])           # (B, 64)
        right_feat = self.iris_branch(inputs["iris_right"])          # (B, 64)
        w_fp, w_left, w_right = self.fusion_layer(fp_feat, left_feat, right_feat)
        fused = torch.cat([w_fp, w_left, w_right], dim=-1)
        return self.classifier(fused)

    @classmethod
    def from_config(cls, model_cfg, data_cfg, num_classes, feature_dims):
        return cls(num_classes=num_classes)
