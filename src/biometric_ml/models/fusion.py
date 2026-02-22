"""
fusion.py — Frozen MobileNet, linear classifier only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class IrisBranch(nn.Module):
    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class BiometricFusionModel(nn.Module):
    def __init__(self, num_classes: int = 45) -> None:
        super().__init__()

        # Load pretrained MobileNet – fully frozen
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        mobilenet = models.mobilenet_v2(weights=weights)

        self.fingerprint_features = mobilenet.features
        for param in self.fingerprint_features.parameters():
            param.requires_grad = False
        self.fingerprint_features.eval()

        self.fp_pool = nn.AdaptiveAvgPool2d(1)
        self.fp_flatten = nn.Flatten()

        self.iris_branch = IrisBranch(output_dim=32)

        # Linear classifier on concatenated features
        self.dropout = nn.Dropout(0.8)          # strong dropout
        self.classifier = nn.Linear(1280 + 32 + 32, num_classes)

        # Initialise linear layer with small weights
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)
        nn.init.zeros_(self.classifier.bias)

        # ImageNet normalisation buffers
        self.register_buffer(
            '_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            '_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _prepare_fingerprint(self, x: Tensor) -> Tensor:
        """Input (B,3,128,128) in [0,1] → (B,3,224,224) normalised."""
        x = x.float()
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = torch.nn.functional.interpolate(
                x, size=(224, 224), mode='bilinear', align_corners=False
            )
        x = (x - self._mean.to(x.device)) / self._std.to(x.device)
        return x

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        # Fingerprint – frozen
        fp_raw = inputs["fingerprint"]
        fp_prep = self._prepare_fingerprint(fp_raw)
        with torch.no_grad():
            fp = self.fingerprint_features(fp_prep)
        fp = self.fp_pool(fp)
        fp = self.fp_flatten(fp)                     # (B, 1280)

        # Iris
        left = self.iris_branch(inputs["iris_left"])   # (B, 32)
        right = self.iris_branch(inputs["iris_right"]) # (B, 32)

        # Concatenate and classify
        features = torch.cat([fp, left, right], dim=1) # (B, 1344)
        features = self.dropout(features)
        return self.classifier(features)

    @classmethod
    def from_config(cls, model_cfg, data_cfg, num_classes, feature_dims):
        return cls(num_classes=num_classes)