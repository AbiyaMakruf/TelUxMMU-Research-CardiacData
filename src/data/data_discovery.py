from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from src.data.lead_parser import ALL_13_LEADS, lead_column_name
from src.utils.io import write_csv, write_json

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _index_by_stem(folder: Path) -> dict[str, Path]:
    return {p.stem: p for p in _image_files(folder)}


def build_manifest(data_dir: str | Path, label_mapping: dict[str, str]) -> list[dict]:
    data_dir = Path(data_dir)
    raw_root = data_dir / "raw"
    clean_root = data_dir / "preprocessed" / "clean_ecg_signal"
    leads_root = data_dir / "preprocessed" / "cropped_leads"

    rows = []
    for class_folder, target_label in sorted(label_mapping.items()):
        raw_index = _index_by_stem(raw_root / class_folder)
        clean_index = _index_by_stem(clean_root / class_folder)
        sample_names = set(raw_index) | set(clean_index)

        lead_class_dir = leads_root / class_folder
        if lead_class_dir.exists():
            sample_names |= {p.name for p in lead_class_dir.iterdir() if p.is_dir()}

        for sample_name in sorted(sample_names):
            sample_id = f"{class_folder}/{sample_name}"
            lead_dir = lead_class_dir / sample_name
            lead_paths = {}
            for prefix in ALL_13_LEADS:
                path = lead_dir / f"{prefix}.png"
                lead_paths[lead_column_name(prefix)] = str(path) if path.exists() else ""

            row = {
                "sample_id": sample_id,
                "sample_name": sample_name,
                "class_folder": class_folder,
                "target_label": target_label,
                "raw_image_path": str(raw_index.get(sample_name, "")),
                "clean_image_path": str(clean_index.get(sample_name, "")),
                **lead_paths,
            }
            row["has_all_leads"] = all(row[lead_column_name(prefix)] for prefix in ALL_13_LEADS)
            rows.append(row)
    return rows


def discover_data(data_dir: str | Path, label_mapping: dict[str, str]) -> dict:
    data_dir = Path(data_dir)
    manifest = build_manifest(data_dir, label_mapping)
    class_counts = Counter(row["class_folder"] for row in manifest)
    label_counts = Counter(row["target_label"] for row in manifest)
    complete_leads = Counter(row["class_folder"] for row in manifest if row["has_all_leads"])

    roots = {
        "raw": data_dir / "raw",
        "clean_ecg_signal": data_dir / "preprocessed" / "clean_ecg_signal",
        "cropped_leads": data_dir / "preprocessed" / "cropped_leads",
        "crop_ecg_area": data_dir / "preprocessed" / "crop_ecg_area",
        "extracted_ecg_signal": data_dir / "preprocessed" / "extracted_ecg_signal",
    }

    file_ext_counts = Counter()
    if data_dir.exists():
        for path in data_dir.rglob("*"):
            if path.is_file():
                file_ext_counts[path.suffix.lower() or "[no_ext]"] += 1

    missing_by_lead = defaultdict(int)
    for row in manifest:
        for prefix in ALL_13_LEADS:
            if not row[lead_column_name(prefix)]:
                missing_by_lead[prefix] += 1

    return {
        "data_dir": str(data_dir),
        "structure_type": "folder_per_class_with_preprocessed_variants",
        "has_train_val_test": False,
        "has_metadata": False,
        "roots": {name: {"path": str(path), "exists": path.exists()} for name, path in roots.items()},
        "number_of_samples": len(manifest),
        "file_extension_counts": dict(sorted(file_ext_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "target_label_counts": dict(sorted(label_counts.items())),
        "complete_13_lead_counts": dict(sorted(complete_leads.items())),
        "missing_lead_counts": dict(sorted(missing_by_lead.items())),
        "detected_classes": sorted(class_counts),
        "label_mapping": label_mapping,
        "recommended_loader_strategy": "manifest_from_existing_folders",
        "input_scheme_compatibility": {
            "single_raw_image": True,
            "single_clean_image": True,
            "single_long_lead_ii": True,
            "single_12_lead": True,
            "stacked_12lead_longlead": True,
            "stacked_6lead_6lead_longlead": True,
            "stacked_13lead_individual": True,
            "multibranch_12lead_longlead": True,
            "multibranch_6lead_6lead_longlead": True,
            "multibranch_13lead_individual": True,
        },
    }


def save_discovery_outputs(discovery: dict, manifest: list[dict], artifacts_dir: str | Path) -> None:
    artifacts_dir = Path(artifacts_dir)
    write_json(discovery, artifacts_dir / "data_discovery.json")
    write_json(discovery["label_mapping"], artifacts_dir / "label_mapping.json")
    write_csv(manifest, artifacts_dir / "manifest.csv")

    lines = [
        "# Data Discovery Report",
        "",
        "## Root Data Directory",
        f"- data_dir: {discovery['data_dir']}",
        "",
        "## Detected Structure",
        f"- structure_type: {discovery['structure_type']}",
        f"- has_train_val_test: {discovery['has_train_val_test']}",
        f"- has_metadata: {discovery['has_metadata']}",
        f"- number_of_samples: {discovery['number_of_samples']}",
        f"- detected_classes: {', '.join(discovery['detected_classes'])}",
        "",
        "## Class Counts",
    ]
    for class_name, count in discovery["class_counts"].items():
        lines.append(f"- {class_name}: {count}")
    lines.extend(["", "## Recommended Loader Strategy", "- loader_type: manifest_from_existing_folders"])
    (artifacts_dir / "data_discovery.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
