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
        "checkpoint_interval": "training.checkpoint_interval",
        "num_workers": "runtime.num_workers",
        "seed": "project.seed",
        "device": "runtime.device",
        "output_dir": "run.output_dir",
        "checkpoint_path": "run.checkpoint_path",
        "inference_file": "run.inference_file",
        "inference_dir": "run.inference_dir",
        "max_samples": "data.max_samples",
        "max_samples_percent": "data.max_samples_percent",
        "multibranch_backbone_sharing": "model.multibranch_backbone_sharing",
        "multibranch_shared_in_channels": "model.multibranch_shared_in_channels",
        "multibranch_branch_projection_dim": "model.multibranch_branch_projection_dim",
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
    if getattr(args, "multibranch_use_branch_heads", None) is not None:
        _set_nested(config, "model.multibranch_use_branch_heads", args.multibranch_use_branch_heads)
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECG disease classification pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train_eval", "train_only", "eval_only", "inference_only"])
    parser.add_argument("--run_name", "--run-name", dest="run_name")
    parser.add_argument("--model_name", "--model-name", dest="model_name")
    parser.add_argument("--input_scheme", "--input-scheme", dest="input_scheme")
    parser.add_argument("--data_dir", "--data-dir", dest="data_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float)
    parser.add_argument("--min_learning_rate", "--min-learning-rate", dest="min_learning_rate", type=float)
    parser.add_argument("--weight_decay", "--weight-decay", dest="weight_decay", type=float)
    parser.add_argument("--checkpoint_interval", "--checkpoint-interval", dest="checkpoint_interval", type=int)
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir")
    parser.add_argument("--checkpoint_path", "--checkpoint-path", dest="checkpoint_path")
    parser.add_argument("--inference_file", "--inference-file", dest="inference_file")
    parser.add_argument("--inference_dir", "--inference-dir", dest="inference_dir")
    parser.add_argument("--max_samples", "--max-samples", dest="max_samples", type=int)
    parser.add_argument("--max_samples_percent", "--max-samples-percent", dest="max_samples_percent", type=float)
    parser.add_argument("--multibranch_backbone_sharing", "--multibranch-backbone-sharing", dest="multibranch_backbone_sharing", choices=["shared", "independent"])
    parser.add_argument("--multibranch_shared_in_channels", "--multibranch-shared-in-channels", dest="multibranch_shared_in_channels", type=int)
    parser.add_argument("--multibranch_branch_projection_dim", "--multibranch-branch-projection-dim", dest="multibranch_branch_projection_dim", type=int)
    parser.add_argument("--multibranch_use_branch_heads", "--multibranch-use-branch-heads", dest="multibranch_use_branch_heads", action=argparse.BooleanOptionalAction)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run_discovery", "--dry-run-discovery", dest="dry_run_discovery", action="store_true")
    return parser
