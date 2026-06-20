from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Feature(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1280

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class MobileNetV2Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = MobileNetV2Feature(in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))
