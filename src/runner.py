from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import torch

from src.config import apply_cli_overrides, build_arg_parser, load_config
from src.data.data_discovery import build_manifest, discover_data, save_discovery_outputs
from src.data.dataloaders import build_dataloaders
from src.engine.evaluate import evaluate_model, save_evaluation_outputs
from src.engine.inference import run_inference
from src.engine.train import train_model
from src.logger import setup_logger, setup_training_logger
from src.models.build_model import build_model
from src.seed import seed_everything
from src.utils.environment import collect_environment
from src.utils.io import ensure_dir, write_json, write_yaml


def create_run_dir(config: dict) -> Path:
    run_cfg = config.setdefault("run", {})
    if run_cfg.get("output_dir"):
        return ensure_dir(run_cfg["output_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_cfg.get("run_name", "run")
    return ensure_dir(Path(config["project"]["output_root"]) / f"{timestamp}_{run_name}")


def select_device(config: dict):
    requested = config["runtime"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def count_model_parameters(model) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def apply_resume_architecture_compatibility(config: dict, run_dir: Path, logger) -> None:
    if not str(config["data"].get("input_scheme", "")).startswith("multibranch_"):
        return
    latest_checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    if not latest_checkpoint_path.exists():
        return
    try:
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(latest_checkpoint_path, map_location="cpu")
    except Exception as exc:
        logger.warning("resume_architecture_compatibility skipped checkpoint=%s error=%s", latest_checkpoint_path, exc)
        return

    state_keys = checkpoint.get("model_state", {}).keys()
    model_cfg = config.setdefault("model", {})
    if any(key.startswith("branches.") for key in state_keys):
        model_cfg["multibranch_backbone_sharing"] = "independent"
        logger.info("resume_architecture_compatibility=independent checkpoint=%s", latest_checkpoint_path)
    elif any(key.startswith("backbone.") for key in state_keys):
        model_cfg["multibranch_backbone_sharing"] = "shared"
        if not any(key.startswith("branch_heads.") for key in state_keys):
            model_cfg["multibranch_use_branch_heads"] = False
            logger.info("resume_architecture_compatibility=shared_legacy checkpoint=%s", latest_checkpoint_path)


def save_run_summary(run_dir: Path, config: dict, discovery: dict, train_summary: dict | None, test_summary: dict | None) -> None:
    lines = [
        "# Run Summary",
        "",
        "## Basic Information",
        f"- Run name: {config['run'].get('run_name')}",
        f"- Mode: {config['run'].get('mode')}",
        f"- Model: {config['model'].get('model_name')}",
        f"- Input scheme: {config['data'].get('input_scheme')}",
        f"- Multibranch backbone sharing: {config['model'].get('multibranch_backbone_sharing')}",
        f"- Number of classes: {config['data'].get('num_classes')}",
        f"- Class names: {', '.join(config['data'].get('class_names', []))}",
        "",
        "## Data",
        f"- Data directory: {config['data'].get('data_dir')}",
        f"- Detected structure: {discovery.get('structure_type')}",
        f"- Number of samples: {discovery.get('number_of_samples')}",
        f"- Split strategy: stratified group split by input hash, ratio {config['data'].get('train_ratio')}/{config['data'].get('val_ratio')}/{config['data'].get('test_ratio')}",
        "",
        "## Training Configuration",
        f"- Epochs: {config['training'].get('epochs')}",
        f"- Batch size: {config['training'].get('batch_size')}",
        f"- Learning rate: {config['training'].get('learning_rate')}",
        f"- Optimizer: {config['training'].get('optimizer')}",
        f"- Scheduler: {config['training'].get('scheduler')}",
        f"- Pretrained: {config['model'].get('pretrained')}",
        f"- Freeze backbone: {config['model'].get('freeze_backbone')}",
    ]
    if train_summary:
        lines.extend(
            [
                "",
                "## Best Validation Result",
                f"- Best epoch: {train_summary.get('best_epoch')}",
                f"- Validation macro F1-score: {train_summary.get('best_val_macro_f1')}",
            ]
        )
    if test_summary:
        lines.extend(
            [
                "",
                "## Test Result",
                f"- Test accuracy: {test_summary.get('accuracy')}",
                f"- Test balanced accuracy: {test_summary.get('balanced_accuracy')}",
                f"- Test macro precision: {test_summary.get('macro_precision')}",
                f"- Test macro recall: {test_summary.get('macro_recall')}",
                f"- Test macro F1-score: {test_summary.get('macro_f1')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "- Best checkpoint: checkpoints/best.pt",
            "- Last checkpoint: checkpoints/last.pt",
            "- Metrics: metrics/",
            "- Plots: plots/",
            "- Logs: logs/",
            "- Artifacts: artifacts/",
        ]
    )
    (run_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = apply_cli_overrides(load_config(args.config), args)
    config["run"]["mode"] = config["run"].get("mode") or "train_eval"

    seed_everything(config["project"]["seed"])
    run_dir = create_run_dir(config)
    for folder in ["artifacts", "checkpoints", "logs", "metrics", "plots"]:
        ensure_dir(run_dir / folder)
    logger = setup_logger(run_dir)
    training_logger = setup_training_logger(run_dir)

    try:
        apply_resume_architecture_compatibility(config, run_dir, logger)
        logger.info("run_dir=%s", run_dir)
        training_logger.info("latest_training_log=training.log")
        training_logger.info("run_dir=%s", run_dir)
        training_logger.info("run_name=%s mode=%s", config["run"].get("run_name"), config["run"].get("mode"))
        training_logger.info("training_log_start=%s", datetime.now().isoformat(timespec="seconds"))
        write_yaml(config, run_dir / "config.yaml")
        write_yaml(config, run_dir / "config_resolved.yaml")
        write_json(collect_environment(), run_dir / "artifacts" / "environment.json")

        data_dir = config["data"]["data_dir"]
        label_mapping = config["data"]["label_mapping"]
        training_logger.info("data_loader_phase=start data_dir=%s", data_dir)
        manifest = build_manifest(data_dir, label_mapping)
        discovery = discover_data(data_dir, label_mapping)
        save_discovery_outputs(discovery, manifest, run_dir / "artifacts")
        training_logger.info("data_loader_phase=discovery_complete samples=%s classes=%s", discovery["number_of_samples"], discovery["class_counts"])
        training_logger.info("input_scheme=%s image_size=%s", config["data"]["input_scheme"], config["data"]["image_size"])

        if args.dry_run_discovery:
            save_run_summary(run_dir, config, discovery, None, None)
            training_logger.info("dry_run_discovery_complete")
            training_logger.info("training_log_finish=%s", datetime.now().isoformat(timespec="seconds"))
            logger.info("dry_run_discovery complete")
            return 0

        device = select_device(config)
        logger.info("device=%s", device)
        mode = config["run"]["mode"]
        train_summary = None
        test_summary = None

        if mode == "inference_only":
            checkpoint_path = config["run"].get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError("--checkpoint_path is required for inference_only")
            rows = run_inference(config, checkpoint_path, run_dir / "metrics" / "predictions.csv", device)
            logger.info("inference complete samples=%s", len(rows))
            save_run_summary(run_dir, config, discovery, None, None)
            return 0

        train_loader, val_loader, test_loader, _dataset = build_dataloaders(manifest, config, run_dir)
        training_logger.info(
            "data_loader_phase=splits_ready train_batches=%s val_batches=%s test_batches=%s batch_size=%s num_workers=%s",
            len(train_loader),
            len(val_loader),
            len(test_loader),
            config["training"]["batch_size"],
            config["runtime"]["num_workers"],
        )
        model = build_model(config).to(device)
        (run_dir / "artifacts" / "model_summary.txt").write_text(str(model), encoding="utf-8")
        param_counts = count_model_parameters(model)
        training_logger.info(
            "model_ready name=%s architecture=%s multibranch_backbone_sharing=%s total_params=%s trainable_params=%s frozen_params=%s",
            config["model"]["model_name"],
            type(model).__name__,
            config["model"].get("multibranch_backbone_sharing"),
            param_counts["total"],
            param_counts["trainable"],
            param_counts["frozen"],
        )

        if mode in {"train_eval", "train_only"}:
            model, train_summary = train_model(model, train_loader, val_loader, config, run_dir, logger, training_logger)

        if mode == "eval_only":
            checkpoint_path = config["run"].get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError("--checkpoint_path is required for eval_only")
            training_logger.info("eval_only_load_checkpoint=%s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])

        if mode in {"train_eval", "eval_only"}:
            training_logger.info("evaluation_phase=start split=test")
            criterion = torch.nn.CrossEntropyLoss()
            use_mixed_precision = bool(config.get("runtime", {}).get("mixed_precision", False)) and device.type == "cuda"
            result = evaluate_model(model, test_loader, criterion, device, use_mixed_precision=use_mixed_precision)
            test_summary = save_evaluation_outputs(result, config["data"]["class_names"], run_dir, split_name="test")
            training_logger.info("evaluation_phase=finish test_metrics=%s", test_summary)

        save_run_summary(run_dir, config, discovery, train_summary, test_summary)
        training_logger.info("training_log_finish=%s", datetime.now().isoformat(timespec="seconds"))
        logger.info("run complete")
        return 0
    except Exception:
        training_logger.exception("pipeline_failed")
        logger.exception("pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
