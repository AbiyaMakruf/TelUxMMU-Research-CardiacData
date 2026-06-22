from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbones import build_feature_extractor


class MultiBranchClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        branch_channels: list[int],
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                build_feature_extractor(model_name, ch, pretrained=pretrained, freeze_backbone=freeze_backbone)
                for ch in branch_channels
            ]
        )
        feature_dim = sum(branch.feature_dim for branch in self.branches)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, num_classes))

    def forward(self, inputs):
        features = [branch(x) for branch, x in zip(self.branches, inputs)]
        return self.classifier(torch.cat(features, dim=1))


class SharedBackboneMultiBranchClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        branch_channels: list[int],
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        shared_in_channels: int = 3,
        branch_projection_dim: int = 512,
        use_branch_heads: bool = True,
    ):
        super().__init__()
        self.num_branches = len(branch_channels)
        self.branch_channels = branch_channels
        self.shared_in_channels = shared_in_channels
        self.use_branch_heads = use_branch_heads
        self.input_adapters = nn.ModuleList(
            [
                nn.Identity()
                if channels == shared_in_channels
                else nn.Sequential(
                    nn.Conv2d(channels, shared_in_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(shared_in_channels),
                    nn.ReLU(inplace=True),
                )
                for channels in branch_channels
            ]
        )
        self.backbone = build_feature_extractor(
            model_name,
            shared_in_channels,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        if use_branch_heads:
            self.branch_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.backbone.feature_dim, branch_projection_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    )
                    for _ in branch_channels
                ]
            )
            fusion_dim = branch_projection_dim * self.num_branches
        else:
            self.branch_heads = nn.ModuleList([nn.Identity() for _ in branch_channels])
            fusion_dim = self.backbone.feature_dim * self.num_branches
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, inputs):
        if len(inputs) != self.num_branches:
            raise ValueError(f"Expected {self.num_branches} branches, got {len(inputs)}")
        features = [
            branch_head(self.backbone(input_adapter(x)))
            for input_adapter, branch_head, x in zip(self.input_adapters, self.branch_heads, inputs)
        ]
        return self.classifier(torch.cat(features, dim=1))


def build_multibranch_model(config: dict, branch_channels: list[int], num_classes: int):
    model_cfg = config["model"]
    model_name = model_cfg["model_name"]
    sharing = model_cfg.get("multibranch_backbone_sharing", model_cfg.get("share_branch_backbone", "shared"))
    if isinstance(sharing, bool):
        sharing = "shared" if sharing else "independent"
    if sharing not in {"shared", "independent"}:
        raise ValueError("model.multibranch_backbone_sharing must be 'shared' or 'independent'")

    common_kwargs = {
        "model_name": model_name,
        "branch_channels": branch_channels,
        "num_classes": num_classes,
        "pretrained": model_cfg["pretrained"],
        "freeze_backbone": model_cfg["freeze_backbone"],
        "dropout": model_cfg.get("dropout", 0.3),
    }
    if sharing == "independent":
        return MultiBranchClassifier(**common_kwargs)
    return SharedBackboneMultiBranchClassifier(
        **common_kwargs,
        shared_in_channels=model_cfg.get("multibranch_shared_in_channels", 3),
        branch_projection_dim=model_cfg.get("multibranch_branch_projection_dim", 512),
        use_branch_heads=model_cfg.get("multibranch_use_branch_heads", True),
    )
