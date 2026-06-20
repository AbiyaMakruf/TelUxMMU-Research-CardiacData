from __future__ import annotations

from src.models.multibranch import build_multibranch_model
from src.models.single_input import build_single_input_model
from src.models.stacked import build_stacked_model


def infer_input_shape(input_scheme: str):
    if input_scheme in {"single_clean_image", "single_raw_image", "single_long_lead_ii"}:
        return "single", 3
    if input_scheme == "single_12_lead":
        return "single", 36
    if input_scheme in {"stacked_12lead_longlead", "stacked_6lead_6lead_longlead", "stacked_13lead_individual"}:
        return "stacked", 39
    if input_scheme == "multibranch_12lead_longlead":
        return "multibranch", [36, 3]
    if input_scheme == "multibranch_6lead_6lead_longlead":
        return "multibranch", [18, 18, 3]
    if input_scheme == "multibranch_13lead_individual":
        return "multibranch", [3] * 13
    raise ValueError(f"Unsupported input_scheme: {input_scheme}")


def build_model(config: dict):
    num_classes = config["data"]["num_classes"]
    input_scheme = config["data"]["input_scheme"]
    kind, shape = infer_input_shape(input_scheme)
    if kind == "single":
        return build_single_input_model(config, shape, num_classes)
    if kind == "stacked":
        return build_stacked_model(config, shape, num_classes)
    return build_multibranch_model(config, shape, num_classes)
