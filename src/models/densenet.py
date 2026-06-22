from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights, DenseNet201_Weights


DENSENET_BUILDERS = {
    "densenet121": (models.densenet121, DenseNet121_Weights.IMAGENET1K_V1),
    "densenet169": (models.densenet169, DenseNet169_Weights.IMAGENET1K_V1),
    "densenet201": (models.densenet201, DenseNet201_Weights.IMAGENET1K_V1),
}


class DenseNetFeature(nn.Module):
    def __init__(self, model_name: str, in_channels: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if model_name not in DENSENET_BUILDERS:
            raise ValueError(f"Unsupported DenseNet model_name: {model_name}")
        builder, default_weights = DENSENET_BUILDERS[model_name]
        model = builder(weights=default_weights if pretrained else None, memory_efficient=True)
        if in_channels != 3:
            first_conv = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = model.classifier.in_features

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.pool(x)
        return x.flatten(1)


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = DenseNetFeature(model_name, in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))
