from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(run_dir: str | Path, name: str = "runner") -> logging.Logger:
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(logs_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def setup_training_logger(run_dir: str | Path, root_log_path: str | Path = "training.log") -> logging.Logger:
    """Create a live training log that is overwritten for each new run.

    The root-level `training.log` is intentionally opened in write mode so users
    can tail one stable file for the latest run. A per-run copy is also saved in
    `runs/.../logs/training.log` for reproducibility.
    """
    run_dir = Path(run_dir)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for path, mode in [(Path(root_log_path), "w"), (logs_dir / "training.log", "a")]:
        file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
