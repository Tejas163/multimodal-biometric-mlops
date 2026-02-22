"""
fusion.py — Tiny trainable CNN for fingerprint + iris.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class IrisBranch(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FingerprintBranch(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        # Input: (B, 3, 128, 128)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BiometricFusionModel(nn.Module):
    def __init__(self, num_classes: int = 45):
        super().__init__()
        self.fingerprint_branch = FingerprintBranch(out_dim=64)
        self.iris_branch = IrisBranch(out_dim=32)   # shared for left/right

        # Classifier – small with moderate dropout
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),           # was 0.8 — caused mode collapse
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        fp = inputs["fingerprint"]          # (B, 3, 128, 128)
        left = inputs["iris_left"]           # (B, 1, 128, 128)
        right = inputs["iris_right"]         # (B, 1, 128, 128)

        fp_feat = self.fingerprint_branch(fp)   # (B, 64)
        left_feat = self.iris_branch(left)      # (B, 32)
        right_feat = self.iris_branch(right)    # (B, 32)

        combined = torch.cat([fp_feat, left_feat, right_feat], dim=1)
        return self.classifier(combined)

    @classmethod
    def from_config(cls, model_cfg, data_cfg, num_classes, feature_dims):
        return cls(num_classes=num_classes)
