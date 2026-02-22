"""
fusion.py — MobileNet FULLY FROZEN. Only iris branch + classifier trained.

With 1480 train rows (after 8x augmentation), we can safely train:
  - IrisBranch: ~5K params
  - Classifier: ~340K params
  Total: ~345K trainable params on 1480 rows = 4.3 params/sample (healthy)

MobileNet: 2.2M params FROZEN — pretrained ImageNet features used as-is.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class IrisBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BiometricFusionModel(nn.Module):
    def __init__(self, num_classes: int = 45) -> None:
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.fingerprint_branch = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),   # → (B, 1280)
        )
        # Fully frozen — don't touch pretrained weights
        for p in self.fingerprint_branch.parameters():
            p.requires_grad = False
            
        for m in self.fingerprint_branch.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.track_running_stats = False 

        self.iris_branch = IrisBranch()  # shared for left + right

        # 1280 + 32 + 32 = 1344
        self.classifier = nn.Sequential(
            nn.Linear(1344, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        fp    = self.fingerprint_branch(inputs["fingerprint"])
        left  = self.iris_branch(inputs["iris_left"])
        right = self.iris_branch(inputs["iris_right"])
        return self.classifier(torch.cat([fp, left, right], dim=-1))

    @classmethod
    def from_config(cls, model_cfg, data_cfg, num_classes, feature_dims):
        return cls(num_classes=num_classes)
