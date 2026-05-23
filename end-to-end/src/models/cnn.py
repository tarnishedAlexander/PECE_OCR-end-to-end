"""CNN model implementations."""

from __future__ import annotations

import warnings

import torch.nn as nn
from torchvision import models


def build_mobilenet_v2(
    num_classes: int,
    pretrained: bool = False,
    freeze_features: bool = True,
):
    """Return a MobileNetV2 instance with a custom head for `num_classes`.

    `pretrained=False` keeps the model fully offline-friendly. If set to True,
    torchvision will try to load ImageNet weights.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        cnn = models.mobilenet_v2(weights=weights)
    except Exception as exc:
        if pretrained:
            warnings.warn(f"Falling back to MobileNetV2 without pretrained weights: {exc}")
        cnn = models.mobilenet_v2(weights=None)
    if freeze_features:
        for p in cnn.features.parameters():
            p.requires_grad = False

    cnn.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(cnn.classifier[1].in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return cnn
