from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.metrics.classification_metrics import compute_classification_metrics
from src.metrics.confusion_matrix import save_confusion_matrix
from src.utils.io import write_csv, write_json


def move_inputs_to_device(inputs, device):
    if isinstance(inputs, (list, tuple)):
        return tuple(item.to(device) for item in inputs)
    return inputs.to(device)


def evaluate_model(model, loader, criterion, device, use_mixed_precision: bool = False):
    model.eval()
    total_loss = 0.0
    total = 0
    all_probs = []
    all_labels = []
    all_sample_ids = []

    with torch.no_grad():
        for inputs, labels, sample_ids in loader:
            labels = labels.to(device)
            inputs = move_inputs_to_device(inputs, device)
            with torch.amp.autocast(device_type="cuda", enabled=use_mixed_precision and getattr(device, "type", str(device)) == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                probs = torch.softmax(outputs, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_sample_ids.extend(list(sample_ids))

    probs_np = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0))
    preds = probs_np.argmax(axis=1).tolist() if len(probs_np) else []
    return {
        "loss": total_loss / max(1, total),
        "labels": all_labels,
        "preds": preds,
        "probs": probs_np,
        "sample_ids": all_sample_ids,
    }


def save_evaluation_outputs(result: dict, class_names: list[str], run_dir: str | Path, split_name: str = "test") -> dict:
    run_dir = Path(run_dir)
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"

    summary, per_class = compute_classification_metrics(result["labels"], result["preds"], class_names)
    summary[f"{split_name}_loss"] = float(result["loss"])
    write_json(summary, metrics_dir / f"{split_name}_metrics.json")
    write_csv([summary], metrics_dir / f"{split_name}_metrics.csv")
    write_csv(per_class, metrics_dir / "per_class_metrics.csv")
    save_confusion_matrix(
        result["labels"],
        result["preds"],
        class_names,
        metrics_dir / "confusion_matrix.csv",
        plots_dir / "confusion_matrix.png",
    )

    probs = result["probs"]
    prediction_rows = []
    for i, sample_id in enumerate(result["sample_ids"]):
        row = {
            "sample_id": sample_id,
            "true_label": class_names[result["labels"][i]],
            "predicted_label": class_names[result["preds"][i]],
            "confidence": float(probs[i].max()) if len(probs) else 0.0,
            "correct": bool(result["labels"][i] == result["preds"][i]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probs[i][class_index]) if len(probs) else 0.0
        prediction_rows.append(row)
    write_csv(prediction_rows, metrics_dir / "predictions.csv")
    return summary
