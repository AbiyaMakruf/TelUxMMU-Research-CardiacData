from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.engine.evaluate import evaluate_model, move_inputs_to_device
from src.metrics.classification_metrics import compute_classification_metrics
from src.utils.io import ensure_dir, write_csv, write_json
from src.utils.plots import save_training_curves


def _torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_optimizer(model, config: dict):
    train_cfg = config["training"]
    if train_cfg.get("optimizer", "adamw").lower() == "adam":
        return Adam(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    return AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])


def _mixed_precision_enabled(config: dict, device) -> bool:
    return bool(config.get("runtime", {}).get("mixed_precision", False)) and getattr(device, "type", str(device)) == "cuda"


def _save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    config: dict,
    epoch: int,
    best_epoch: int,
    best_score: float,
    history: list[dict],
    no_improve: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": config,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "history": history,
            "no_improve": no_improve,
        },
        path,
    )


def _save_progress(path: Path, status: str, epoch: int, last_checkpoint_epoch: int, total_epochs: int, best_epoch: int, best_score: float) -> None:
    write_json(
        {
            "status": status,
            "completed_epoch": epoch,
            "last_checkpoint_epoch": last_checkpoint_epoch,
            "total_epochs": total_epochs,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixed_precision: bool = False, scaler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels, _sample_ids in loader:
        labels = labels.to(device)
        inputs = move_inputs_to_device(inputs, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_mixed_precision):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if scaler is not None and use_mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(1) == labels).sum().item()
        total += batch_size
    return total_loss / max(1, total), correct / max(1, total)


def train_model(model, train_loader, val_loader, config: dict, run_dir: str | Path, logger, training_logger=None):
    run_dir = Path(run_dir)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    metrics_dir = ensure_dir(run_dir / "metrics")
    artifacts_dir = ensure_dir(run_dir / "artifacts")
    class_names = config["data"]["class_names"]
    train_cfg = config["training"]
    progress_logger = training_logger or logger

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config)
    device = next(model.parameters()).device
    use_mixed_precision = _mixed_precision_enabled(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)
    scheduler = None
    if train_cfg.get("scheduler") == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, train_cfg["epochs"]),
            eta_min=train_cfg.get("min_learning_rate", 0.0),
        )

    best_score = -1.0
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    no_improve = 0
    history = []
    total_epochs = train_cfg["epochs"]
    start_epoch = 1
    last_checkpoint_epoch = 0
    checkpoint_interval = max(1, int(train_cfg.get("checkpoint_interval", 10)))
    latest_checkpoint_path = checkpoint_dir / "latest.pt"
    progress_path = artifacts_dir / "training_progress.json"

    if latest_checkpoint_path.exists():
        checkpoint = _torch_load(latest_checkpoint_path, next(model.parameters()).device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", best_score)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        no_improve = checkpoint.get("no_improve", no_improve)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        last_checkpoint_epoch = int(checkpoint.get("epoch", 0))
        progress_logger.info("resume_from_checkpoint=%s start_epoch=%s", latest_checkpoint_path, start_epoch)

    best_checkpoint_path = checkpoint_dir / "best.pt"
    if best_checkpoint_path.exists():
        best_checkpoint = _torch_load(best_checkpoint_path, next(model.parameters()).device)
        best_state = deepcopy(best_checkpoint.get("model_state", model.state_dict()))
        best_epoch = best_checkpoint.get("best_epoch", best_epoch)
        best_score = best_checkpoint.get("best_score", best_checkpoint.get("best_val_macro_f1", best_score))
    training_started_at = datetime.now()
    training_start_time = perf_counter()

    progress_logger.info("training_start=%s", training_started_at.isoformat(timespec="seconds"))
    progress_logger.info("total_epochs=%s", total_epochs)
    progress_logger.info("start_epoch=%s checkpoint_interval=%s", start_epoch, checkpoint_interval)
    progress_logger.info(
        "optimizer=%s learning_rate=%s min_learning_rate=%s weight_decay=%s",
        train_cfg.get("optimizer"),
        train_cfg["learning_rate"],
        train_cfg.get("min_learning_rate"),
        train_cfg["weight_decay"],
    )
    progress_logger.info("scheduler=%s early_stopping=%s patience=%s", train_cfg.get("scheduler"), train_cfg.get("early_stopping"), train_cfg.get("patience"))
    progress_logger.info("mixed_precision=%s device=%s", use_mixed_precision, device)

    if start_epoch > total_epochs:
        progress_logger.info("training_already_complete completed_epoch=%s total_epochs=%s", start_epoch - 1, total_epochs)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixed_precision, scaler)
        val_result = evaluate_model(model, val_loader, criterion, device, use_mixed_precision=use_mixed_precision)
        val_summary, _ = compute_classification_metrics(val_result["labels"], val_result["preds"], class_names)
        epoch_elapsed = perf_counter() - epoch_start_time
        total_elapsed = perf_counter() - training_start_time
        avg_epoch_seconds = total_elapsed / epoch
        eta_seconds = avg_epoch_seconds * max(0, total_epochs - epoch)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_result["loss"],
            "val_accuracy": val_summary["accuracy"],
            "val_macro_precision": val_summary["macro_precision"],
            "val_macro_recall": val_summary["macro_recall"],
            "val_macro_f1": val_summary["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_elapsed_seconds": epoch_elapsed,
            "total_elapsed_seconds": total_elapsed,
            "eta_seconds": eta_seconds,
        }
        history.append(row)
        logger.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f",
            epoch,
            train_loss,
            val_result["loss"],
            row["val_macro_f1"],
        )
        progress_logger.info(
            "epoch=%s/%s train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f "
            "val_macro_f1=%.4f epoch_elapsed_seconds=%.2f total_elapsed_seconds=%.2f eta_seconds=%.2f",
            epoch,
            total_epochs,
            train_loss,
            train_acc,
            val_result["loss"],
            row["val_accuracy"],
            row["val_macro_f1"],
            epoch_elapsed,
            total_elapsed,
            eta_seconds,
        )

        if row["val_macro_f1"] > best_score:
            best_score = row["val_macro_f1"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state": best_state,
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                },
                ensure_dir(checkpoint_dir) / "best.pt",
            )
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if epoch % checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            _save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_epoch,
                best_score,
                history,
                no_improve,
            )
            _save_training_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_epoch,
                best_score,
                history,
                no_improve,
            )
            last_checkpoint_epoch = epoch
            progress_logger.info("checkpoint_saved epoch=%s path=%s", epoch, checkpoint_path)

        _save_progress(progress_path, "running", epoch, last_checkpoint_epoch, total_epochs, best_epoch, best_score)

        if train_cfg.get("early_stopping") and no_improve >= train_cfg.get("patience", 10):
            logger.info("early stopping at epoch=%s", epoch)
            progress_logger.info("early_stopping_triggered epoch=%s no_improve=%s", epoch, no_improve)
            break

    final_epoch = history[-1]["epoch"] if history else start_epoch - 1
    _save_training_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizer,
        scheduler,
        config,
        final_epoch,
        best_epoch,
        best_score,
        history,
        no_improve,
    )
    _save_training_checkpoint(
        latest_checkpoint_path,
        model,
        optimizer,
        scheduler,
        config,
        final_epoch,
        best_epoch,
        best_score,
        history,
        no_improve,
    )
    last_checkpoint_epoch = final_epoch
    model.load_state_dict(best_state)

    write_csv(history, metrics_dir / "train_history.csv")
    write_json(history, metrics_dir / "train_history.json")
    save_training_curves(history, run_dir / "plots")
    total_time = perf_counter() - training_start_time
    training_finished_at = datetime.now()
    progress_logger.info("best_epoch=%s best_val_macro_f1=%.6f", best_epoch, best_score)
    progress_logger.info("training_finish=%s", training_finished_at.isoformat(timespec="seconds"))
    progress_logger.info("total_training_time_seconds=%.2f", total_time)
    _save_progress(progress_path, "complete", final_epoch, last_checkpoint_epoch, total_epochs, best_epoch, best_score)
    return model, {"best_epoch": best_epoch, "best_val_macro_f1": best_score, "history": history}
