from __future__ import annotations

from src.models.densenet import DENSENET_BUILDERS, DenseNetClassifier, DenseNetFeature
from src.models.efficientnet_v2 import EFFICIENTNET_BUILDERS, EfficientNetClassifier, EfficientNetFeature
from src.models.mobilenet_v2 import MOBILENET_BUILDERS, MobileNetClassifier, MobileNetFeature
from src.models.resnet import RESNET_BUILDERS, ResNetClassifier, ResNetFeature


def build_feature_extractor(model_name: str, in_channels: int, pretrained: bool, freeze_backbone: bool):
    if model_name in MOBILENET_BUILDERS:
        return MobileNetFeature(model_name, in_channels, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name in EFFICIENTNET_BUILDERS:
        return EfficientNetFeature(model_name, in_channels, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name in RESNET_BUILDERS:
        return ResNetFeature(model_name, in_channels, pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name in DENSENET_BUILDERS:
        return DenseNetFeature(model_name, in_channels, pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_classifier(model_name: str, num_classes: int, in_channels: int, pretrained: bool, freeze_backbone: bool, dropout: float):
    if model_name in MOBILENET_BUILDERS:
        return MobileNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    if model_name in EFFICIENTNET_BUILDERS:
        return EfficientNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    if model_name in RESNET_BUILDERS:
        return ResNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    if model_name in DENSENET_BUILDERS:
        return DenseNetClassifier(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")
