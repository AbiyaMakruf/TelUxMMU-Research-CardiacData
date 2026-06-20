from __future__ import annotations

import torch
import torch.nn as nn

from src.models.mobilenet_v2 import MobileNetV2Feature


class MultiBranchMobileNetV2(nn.Module):
    def __init__(
        self,
        branch_channels: list[int],
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [MobileNetV2Feature(ch, pretrained=pretrained, freeze_backbone=freeze_backbone) for ch in branch_channels]
        )
        feature_dim = sum(branch.feature_dim for branch in self.branches)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def forward(self, inputs):
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        return self.classifier(torch.cat(features, dim=1))


def build_multibranch_model(config: dict, branch_channels: list[int], num_classes: int):
    model_cfg = config["model"]
    if model_cfg["model_name"] != "mobilenet_v2":
        raise ValueError(f"Unsupported model_name: {model_cfg['model_name']}")
    return MultiBranchMobileNetV2(
        branch_channels=branch_channels,
        num_classes=num_classes,
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
        dropout=model_cfg.get("dropout", 0.3),
    )
