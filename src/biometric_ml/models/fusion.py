"""
fusion.py
---------
PyTorch port of the Kaggle notebook multimodal biometric model.

Notebook architecture (TensorFlow) → PyTorch equivalent:

Fingerprint branch:
    MobileNetV2(pretrained=ImageNet, include_top=False, pooling='avg')
    → features: (B, 1280)    [MobileNetV2 output channels]
    Weights FROZEN (base_model.trainable = False)

Iris branch (shared weights for left + right, same as notebook):
    Conv2d(1→16, 3x3) → ReLU → MaxPool2d(2)
    Conv2d(16→32, 3x3) → ReLU → MaxPool2d(2)
    AdaptiveAvgPool2d(1) → Flatten
    → features: (B, 32)

Fusion (matches notebook Concatenate + Dense):
    concat([fingerprint_feat, left_feat, right_feat])  → (B, 1280+32+32) = (B, 1344)
    Linear(1344 → 128) → ReLU
    Dropout(0.5)
    Linear(128 → num_classes)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class IrisBranch(nn.Module):
    """
    CNN branch for processing iris images.
    Shared between left and right eye (same weights) — matches notebook.

    Input:  (B, 1, 64, 64)
    Output: (B, 32)
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Conv block 1 — matches: Conv2D(16, (3,3), activation='relu')
            nn.Conv2d(1, 16, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # MaxPooling2D()

            # Conv block 2 — matches: Conv2D(32, (3,3), activation='relu')
            nn.Conv2d(16, 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # MaxPooling2D()

            # Global average pooling — matches: GlobalAveragePooling2D()
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                       # → (B, 32)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

# In fusion.py -> BiometricFusionModel
class GatedFusion(nn.Module):
    def __init__(self, fp_dim, iris_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(fp_dim + iris_dim * 2, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, fp, left, right):
        stacked = torch.cat([fp, left, right], dim=-1)
        weights = self.gate(stacked) # (B, 3)
        # Weight each modality feature vector
        return fp * weights[:, 0:1], left * weights[:, 1:2], right * weights[:, 2:3]
        
class BiometricFusionModel(nn.Module):
    """
    Full multimodal biometric model — PyTorch port of the Kaggle notebook.

    Inputs (dict):
        fingerprint : (B, 3, 128, 128) — RGB normalised [0,1]
        iris_left   : (B, 1, 64,  64)  — grayscale normalised [0,1]
        iris_right  : (B, 1, 64,  64)  — grayscale normalised [0,1]

    Output:
        logits : (B, num_classes)
    """

    def __init__(self, num_classes: int = 45) -> None:
        super().__init__()

        # ── Fingerprint branch: MobileNetV2 pretrained on ImageNet ────────
        # Matches: MobileNetV2(include_top=False, weights='imagenet', pooling='avg')
        mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the classifier head, keep feature extractor
        self.fingerprint_branch = nn.Sequential(
            mobilenet.features,
            nn.AdaptiveAvgPool2d(1),    # global avg pool → (B, 1280, 1, 1)
            nn.Flatten(),               # → (B, 1280)
        )
        # Freeze pretrained weights — matches: base_model.trainable = False
        for param in self.fingerprint_branch.parameters():
            param.requires_grad = True

        # ── Iris branch: shared CNN (same weights left + right) ───────────
        # Matches: iris_processor = create_iris_branch(iris_shape)
        #          left_iris_features  = iris_processor(left_iris_input)
        #          right_iris_features = iris_processor(right_iris_input)
        self.iris_branch = IrisBranch()  # shared — called twice in forward

        # ── Fusion + classification head ─────────────────────────────────
        # Matches: Concatenate → Dense(128, relu) → Dropout(0.5) → Dense(num_classes, softmax)
        # 1280 (mobilenet) + 32 (left iris) + 32 (right iris) = 1344
        fusion_dim = 1280 + 32 + 32
        self.fusion_layer = GatedFusion(fp_dim=1280, iris_dim=32)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            # No softmax here — CrossEntropyLoss applies it internally
        )

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        fp_feat = self.fingerprint_branch(inputs["fingerprint"])
        left_feat = self.iris_branch(inputs["iris_left"])
        right_feat = self.iris_branch(inputs["iris_right"])

        # Use the gate instead of simple torch.cat
        w_fp, w_left, w_right = self.fusion_layer(fp_feat, left_feat, right_feat)
        
        fused = torch.cat([w_fp, w_left, w_right], dim=-1)
        return self.classifier(fused)
    @classmethod
    def from_config(cls, model_cfg: dict, data_cfg: dict,
                    num_classes: int, feature_dims: dict) -> "BiometricFusionModel":
        """Factory method called by build_and_fit()."""
        return cls(num_classes=num_classes)
