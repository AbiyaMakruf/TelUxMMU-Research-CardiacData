from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights


MOBILENET_BUILDERS = {
    "mobilenet_v2": (models.mobilenet_v2, MobileNet_V2_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": (models.mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
}


class MobileNetFeature(nn.Module):
    def __init__(self, model_name: str, in_channels: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if model_name not in MOBILENET_BUILDERS:
            raise ValueError(f"Unsupported MobileNet model_name: {model_name}")
        builder, default_weights = MOBILENET_BUILDERS[model_name]
        model = builder(weights=default_weights if pretrained else None)
        if in_channels != 3:
            first_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
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
        self.feature_dim = model.classifier[-1].in_features

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
        self.backbone = MobileNetFeature("mobilenet_v2", in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class MobileNetClassifier(nn.Module):
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
        self.backbone = MobileNetFeature(model_name, in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class MobileNetV2Feature(MobileNetFeature):
    def __init__(self, in_channels: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__("mobilenet_v2", in_channels, pretrained, freeze_backbone)
