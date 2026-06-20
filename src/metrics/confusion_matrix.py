from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.utils.io import ensure_dir


def save_confusion_matrix(y_true, y_pred, class_names: list[str], csv_path: str | Path, png_path: str | Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    ensure_dir(Path(csv_path).parent)
    np.savetxt(csv_path, cm, delimiter=",", fmt="%d")

    ensure_dir(Path(png_path).parent)
    fig, ax = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
