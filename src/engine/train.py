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


def _build_optimizer(model, config: dict):
    train_cfg = config["training"]
    if train_cfg.get("optimizer", "adamw").lower() == "adam":
        return Adam(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    return AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels, _sample_ids in loader:
        labels = labels.to(device)
        inputs = move_inputs_to_device(inputs, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
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
    class_names = config["data"]["class_names"]
    train_cfg = config["training"]
    progress_logger = training_logger or logger

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config)
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
    training_started_at = datetime.now()
    training_start_time = perf_counter()

    progress_logger.info("training_start=%s", training_started_at.isoformat(timespec="seconds"))
    progress_logger.info("total_epochs=%s", total_epochs)
    progress_logger.info(
        "optimizer=%s learning_rate=%s min_learning_rate=%s weight_decay=%s",
        train_cfg.get("optimizer"),
        train_cfg["learning_rate"],
        train_cfg.get("min_learning_rate"),
        train_cfg["weight_decay"],
    )
    progress_logger.info("scheduler=%s early_stopping=%s patience=%s", train_cfg.get("scheduler"), train_cfg.get("early_stopping"), train_cfg.get("patience"))

    for epoch in range(1, total_epochs + 1):
        epoch_start_time = perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, next(model.parameters()).device)
        val_result = evaluate_model(model, val_loader, criterion, next(model.parameters()).device)
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
            torch.save({"model_state": best_state, "config": config, "best_epoch": best_epoch}, checkpoint_dir / "best.pt")
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            scheduler.step()

        if train_cfg.get("early_stopping") and no_improve >= train_cfg.get("patience", 10):
            logger.info("early stopping at epoch=%s", epoch)
            progress_logger.info("early_stopping_triggered epoch=%s no_improve=%s", epoch, no_improve)
            break

    torch.save({"model_state": model.state_dict(), "config": config, "last_epoch": history[-1]["epoch"]}, checkpoint_dir / "last.pt")
    model.load_state_dict(best_state)

    write_csv(history, metrics_dir / "train_history.csv")
    write_json(history, metrics_dir / "train_history.json")
    save_training_curves(history, run_dir / "plots")
    total_time = perf_counter() - training_start_time
    training_finished_at = datetime.now()
    progress_logger.info("best_epoch=%s best_val_macro_f1=%.6f", best_epoch, best_score)
    progress_logger.info("training_finish=%s", training_finished_at.isoformat(timespec="seconds"))
    progress_logger.info("total_training_time_seconds=%.2f", total_time)
    return model, {"best_epoch": best_epoch, "best_val_macro_f1": best_score, "history": history}
