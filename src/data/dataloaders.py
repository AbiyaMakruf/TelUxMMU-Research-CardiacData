from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.datasets import ECGManifestDataset
from src.data.lead_parser import ALL_13_LEADS, GRID_13_LEADS, LONG_LEAD, SHORT_12_LEADS, lead_column_name
from src.utils.io import ensure_dir, write_csv, write_json


def _subset_rows(rows: list[dict], max_samples: int | None = None, max_samples_percent: float | None = None) -> list[dict]:
    if max_samples_percent:
        max_samples = max(1, ceil(len(rows) * max_samples_percent / 100.0))
    if not max_samples:
        return rows
    by_label = {}
    for row in rows:
        by_label.setdefault(row["target_label"], []).append(row)
    selected = []
    per_label = max(1, max_samples // max(1, len(by_label)))
    for label in sorted(by_label):
        selected.extend(by_label[label][:per_label])
    return selected[:max_samples]


def _file_md5(path: str) -> str:
    if not path:
        return "missing"
    file_path = Path(path)
    if not file_path.exists():
        return "missing"
    digest = hashlib.md5()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_group_key(row: dict, input_scheme: str) -> str:
    if input_scheme == "single_raw_image":
        parts = ["raw", _file_md5(row.get("raw_image_path", ""))]
    elif input_scheme == "single_clean_image":
        parts = ["clean", _file_md5(row.get("clean_image_path", ""))]
    elif input_scheme == "single_long_lead_ii":
        parts = ["long", _file_md5(row.get(lead_column_name(LONG_LEAD), ""))]
    elif input_scheme == "single_12_lead":
        parts = ["12lead"] + [_file_md5(row.get(lead_column_name(prefix), "")) for prefix in SHORT_12_LEADS]
    elif input_scheme == "stacked_12lead_longlead":
        parts = ["stacked_12plus1_grid"] + [_file_md5(row.get(lead_column_name(prefix), "")) for prefix in GRID_13_LEADS]
    elif input_scheme == "stacked_6lead_6lead_longlead":
        parts = ["stacked_6plus6plus1_anatomical"] + [
            _file_md5(row.get(lead_column_name(prefix), "")) for prefix in ALL_13_LEADS
        ]
    elif input_scheme == "stacked_13lead_individual":
        parts = ["stacked_13lead_sequence"] + [_file_md5(row.get(lead_column_name(prefix), "")) for prefix in GRID_13_LEADS]
    elif input_scheme == "multibranch_12lead_longlead":
        parts = ["12plus1"] + [_file_md5(row.get(lead_column_name(prefix), "")) for prefix in SHORT_12_LEADS + [LONG_LEAD]]
    elif input_scheme == "multibranch_6lead_6lead_longlead":
        parts = ["6plus6plus1"] + [_file_md5(row.get(lead_column_name(prefix), "")) for prefix in ALL_13_LEADS]
    elif input_scheme == "multibranch_13lead_individual":
        parts = ["13branch"] + [_file_md5(row.get(lead_column_name(prefix), "")) for prefix in ALL_13_LEADS]
    else:
        parts = ["sample", row["sample_id"]]
    return hashlib.md5(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def _stratify_labels_or_none(labels: list[str], test_size):
    label_counts = Counter(labels)
    if len(label_counts) < 2 or min(label_counts.values()) < 2:
        return None
    if isinstance(test_size, float):
        test_count = ceil(len(labels) * test_size)
    else:
        test_count = int(test_size)
    train_count = len(labels) - test_count
    if min(test_count, train_count) < len(label_counts):
        return None
    return labels


def _split_group_ids(group_ids: list[str], group_labels: list[str], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    if len(set(group_labels)) < 2:
        return group_ids, [], []
    groups_trainval, groups_test = train_test_split(
        group_ids,
        test_size=test_ratio,
        stratify=_stratify_labels_or_none(group_labels, test_ratio),
        random_state=seed,
    )
    label_by_group = dict(zip(group_ids, group_labels))
    trainval_labels = [label_by_group[group] for group in groups_trainval]
    val_fraction = val_ratio / (train_ratio + val_ratio)
    groups_train, groups_val = train_test_split(
        groups_trainval,
        test_size=val_fraction,
        stratify=_stratify_labels_or_none(trainval_labels, val_fraction),
        random_state=seed,
    )
    return groups_train, groups_val, groups_test


def split_indices(rows: list[dict], input_scheme: str, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    group_to_labels: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        group_key = _row_group_key(row, input_scheme)
        row["leakage_group_id"] = group_key
        group_to_indices[group_key].append(index)
        group_to_labels[group_key].add(row["target_label"])

    single_label_groups = []
    group_labels = []
    mixed_label_groups = {}
    for group, labels in sorted(group_to_labels.items()):
        if len(labels) == 1:
            single_label_groups.append(group)
            group_labels.append(next(iter(labels)))
        else:
            mixed_label_groups[group] = sorted(labels)

    groups_train, groups_val, groups_test = _split_group_ids(
        single_label_groups, group_labels, train_ratio, val_ratio, test_ratio, seed
    )
    train_groups = set(groups_train) | set(mixed_label_groups)
    val_groups = set(groups_val)
    test_groups = set(groups_test)

    def expand(groups: set[str]) -> list[int]:
        return sorted(index for group in groups for index in group_to_indices[group])

    audit = {
        "split_strategy": "stratified_group_split_by_input_hash",
        "input_scheme": input_scheme,
        "num_samples": len(rows),
        "num_leakage_groups": len(group_to_indices),
        "num_duplicate_groups": sum(1 for indices in group_to_indices.values() if len(indices) > 1),
        "max_group_size": max((len(indices) for indices in group_to_indices.values()), default=0),
        "mixed_label_groups": mixed_label_groups,
        "group_label_counts": dict(sorted(Counter(group_labels).items())),
    }
    return expand(train_groups), expand(val_groups), expand(test_groups), audit


def build_dataloaders(rows: list[dict], config: dict, run_dir: str | Path):
    data_cfg = config["data"]
    train_cfg = config["training"]
    runtime_cfg = config["runtime"]
    seed = config["project"]["seed"]

    rows = _subset_rows(rows, data_cfg.get("max_samples"), data_cfg.get("max_samples_percent"))
    dataset = ECGManifestDataset(
        rows=rows,
        class_names=data_cfg["class_names"],
        input_scheme=data_cfg["input_scheme"],
        image_size=data_cfg["image_size"],
    )
    idx_train, idx_val, idx_test, split_audit = split_indices(
        rows,
        data_cfg["input_scheme"],
        data_cfg["train_ratio"],
        data_cfg["val_ratio"],
        data_cfg["test_ratio"],
        seed,
    )

    splits_dir = ensure_dir(Path(run_dir) / "artifacts" / "splits")
    for name, indices in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        write_csv([rows[i] for i in indices], splits_dir / f"{name}_split.csv")
    split_audit["split_sizes"] = {"train": len(idx_train), "val": len(idx_val), "test": len(idx_test)}
    split_audit["split_group_counts"] = {
        name: len({rows[i]["leakage_group_id"] for i in indices})
        for name, indices in [("train", idx_train), ("val", idx_val), ("test", idx_test)]
    }
    write_json(split_audit, splits_dir / "split_audit.json")

    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": runtime_cfg["num_workers"],
        "pin_memory": False,
    }
    return (
        DataLoader(Subset(dataset, idx_train), shuffle=True, **loader_kwargs),
        DataLoader(Subset(dataset, idx_val), shuffle=False, **loader_kwargs),
        DataLoader(Subset(dataset, idx_test), shuffle=False, **loader_kwargs),
        dataset,
    )


def build_all_data_loader(rows: list[dict], config: dict):
    data_cfg = config["data"]
    train_cfg = config["training"]
    runtime_cfg = config["runtime"]

    rows = _subset_rows(rows, data_cfg.get("max_samples"), data_cfg.get("max_samples_percent"))
    dataset = ECGManifestDataset(
        rows=rows,
        class_names=data_cfg["class_names"],
        input_scheme=data_cfg["input_scheme"],
        image_size=data_cfg["image_size"],
    )
    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": runtime_cfg["num_workers"],
        "pin_memory": False,
    }
    return DataLoader(dataset, shuffle=False, **loader_kwargs), dataset
