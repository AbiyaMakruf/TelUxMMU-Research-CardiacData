import torch
import torch.nn as nn

from src.models.backbones import build_feature_extractor
from src.models.single_input import build_single_input_model


def build_stacked_model(config: dict, in_channels: int, num_classes: int):
    return build_single_input_model(config, in_channels, num_classes)


class LeadSequenceMobileNetV2(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_leads: int,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.backbone = build_feature_extractor(model_name, 3, pretrained=pretrained, freeze_backbone=freeze_backbone)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.feature_dim * num_leads, num_classes),
        )

    def forward(self, inputs: torch.Tensor):
        batch_size, num_leads, channels, height, width = inputs.shape
        if num_leads != self.num_leads:
            raise ValueError(f"Expected {self.num_leads} leads, got {num_leads}")
        features = [self.backbone(inputs[:, lead_index, :, :, :]) for lead_index in range(num_leads)]
        features = torch.cat(features, dim=1)
        return self.classifier(features)


def build_stacked_lead_sequence_model(config: dict, num_leads: int, num_classes: int):
    model_cfg = config["model"]
    return LeadSequenceMobileNetV2(
        model_name=model_cfg["model_name"],
        num_leads=num_leads,
        num_classes=num_classes,
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
        dropout=model_cfg.get("dropout", 0.3),
    )
