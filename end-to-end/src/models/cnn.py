"""CNN model implementations."""

from __future__ import annotations

import warnings

try:
    import torch.nn as nn
    from torchvision import models
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def build_mobilenet_v2(
    num_classes: int,
    pretrained: bool = False,
    freeze_features: bool = True,
    unfreeze_last_blocks: int = 0,
):
    """Return a MobileNetV2 instance with a custom head for `num_classes`.

    `pretrained=False` keeps the model fully offline-friendly. If set to True,
    torchvision will try to load ImageNet weights.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch and torchvision are required to build MobileNetV2, but could not be imported.")
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
        if unfreeze_last_blocks > 0:
            for block in cnn.features[-unfreeze_last_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    cnn.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(cnn.classifier[1].in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )
    return cnn
