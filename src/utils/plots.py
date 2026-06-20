from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.io import ensure_dir


def save_training_curves(history: list[dict], plots_dir: str | Path) -> None:
    plots_dir = ensure_dir(plots_dir)
    if not history:
        return

    def _plot(keys: list[str], filename: str, ylabel: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = [row["epoch"] for row in history]
        for key in keys:
            ax.plot(epochs, [row.get(key, 0.0) for row in history], label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=200)
        plt.close(fig)

    _plot(["train_loss", "val_loss"], "training_curve_loss.png", "Loss")
    _plot(["train_accuracy", "val_accuracy"], "training_curve_accuracy.png", "Accuracy")
    _plot(["val_macro_f1"], "training_curve_f1.png", "Macro F1")
