from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, EfficientNet_V2_S_Weights


EFFICIENTNET_BUILDERS = {
    "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "efficientnet_b1": (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V2),
    "efficientnet_b2": (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "efficientnet_v2_s": (models.efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
}


class EfficientNetFeature(nn.Module):
    def __init__(self, model_name: str, in_channels: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if model_name not in EFFICIENTNET_BUILDERS:
            raise ValueError(f"Unsupported EfficientNet model_name: {model_name}")
        builder, default_weights = EFFICIENTNET_BUILDERS[model_name]
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
        self.pool = model.avgpool
        self.feature_dim = model.classifier[1].in_features

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class EfficientNetV2SClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = EfficientNetFeature("efficientnet_v2_s", in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class EfficientNetClassifier(nn.Module):
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
        self.backbone = EfficientNetFeature(model_name, in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))
