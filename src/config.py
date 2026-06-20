from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.utils.io import deep_update, read_yaml


def load_config(path: str | Path | None) -> dict:
    if path is None:
        path = "configs/default.yaml"
    config = read_yaml(path)
    base_path = config.pop("_base_", None)
    if base_path:
        base = load_config(base_path)
        config = deep_update(base, config)
    return config


def _set_nested(config: dict, dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "mode": "run.mode",
        "run_name": "run.run_name",
        "model_name": "model.model_name",
        "input_scheme": "data.input_scheme",
        "data_dir": "data.data_dir",
        "epochs": "training.epochs",
        "batch_size": "training.batch_size",
        "learning_rate": "training.learning_rate",
        "min_learning_rate": "training.min_learning_rate",
        "weight_decay": "training.weight_decay",
        "num_workers": "runtime.num_workers",
        "seed": "project.seed",
        "device": "runtime.device",
        "output_dir": "run.output_dir",
        "checkpoint_path": "run.checkpoint_path",
        "inference_file": "run.inference_file",
        "inference_dir": "run.inference_dir",
        "max_samples": "data.max_samples",
    }
    for arg_name, dotted_key in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            _set_nested(config, dotted_key, value)
    if getattr(args, "pretrained", None) is not None:
        _set_nested(config, "model.pretrained", args.pretrained)
    if getattr(args, "freeze_backbone", None) is not None:
        _set_nested(config, "model.freeze_backbone", args.freeze_backbone)
    if getattr(args, "early_stopping", None) is not None:
        _set_nested(config, "training.early_stopping", args.early_stopping)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECG disease classification pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train_eval", "train_only", "eval_only", "inference_only"])
    parser.add_argument("--run_name")
    parser.add_argument("--model_name")
    parser.add_argument("--input_scheme")
    parser.add_argument("--data_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--min_learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output_dir")
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--inference_file")
    parser.add_argument("--inference_dir")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run_discovery", action="store_true")
    return parser
