from src.models.mobilenet_v2 import MobileNetV2Classifier


def build_single_input_model(config: dict, in_channels: int, num_classes: int):
    model_cfg = config["model"]
    if model_cfg["model_name"] != "mobilenet_v2":
        raise ValueError(f"Unsupported model_name: {model_cfg['model_name']}")
    return MobileNetV2Classifier(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=model_cfg["pretrained"],
        freeze_backbone=model_cfg["freeze_backbone"],
        dropout=model_cfg.get("dropout", 0.3),
    )
