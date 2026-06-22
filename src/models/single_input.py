from src.models.backbones import build_classifier


def build_single_input_model(config: dict, in_channels: int, num_classes: int):
    model_cfg = config["model"]
    return build_classifier(
        model_name=model_cfg["model_name"],
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
        dropout=model_cfg.get("dropout", 0.3),
    )
