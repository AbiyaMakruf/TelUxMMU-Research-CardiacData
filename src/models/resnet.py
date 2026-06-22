from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights


RESNET_BUILDERS = {
    "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
}


class ResNetFeature(nn.Module):
    def __init__(self, model_name: str, in_channels: int = 3, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        if model_name not in RESNET_BUILDERS:
            raise ValueError(f"Unsupported ResNet model_name: {model_name}")
        builder, default_weights = RESNET_BUILDERS[model_name]
        model = builder(weights=default_weights if pretrained else None)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels,
                model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=False,
            )
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.pool = model.avgpool
        self.feature_dim = model.fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


class ResNet50Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = ResNetFeature("resnet50", in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class ResNetClassifier(nn.Module):
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
        self.backbone = ResNetFeature(model_name, in_channels, pretrained, freeze_backbone)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.feature_dim, num_classes))

    def forward(self, x):
        return self.classifier(self.backbone(x))
