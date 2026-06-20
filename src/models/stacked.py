from src.models.single_input import build_single_input_model


def build_stacked_model(config: dict, in_channels: int, num_classes: int):
    return build_single_input_model(config, in_channels, num_classes)
