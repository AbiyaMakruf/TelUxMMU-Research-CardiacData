from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import matplotlib.pyplot as plt


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "test_loss",
]

OUTPUT_COLUMNS = [
    "global_rank_accuracy",
    "global_rank_macro_f1",
    "model_rank_accuracy",
    "model_rank_macro_f1",
    "model_name",
    "run_group",
    "input_scheme",
    "run_name",
    "run_dir",
    "completed_epoch",
    "total_epochs",
    "best_epoch",
    "best_val_macro_f1",
    "status",
    "batch_size",
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "scheduler",
    "early_stopping",
    "checkpoint_interval",
    *METRIC_COLUMNS,
    "metrics_path",
    "confusion_matrix_path",
    "confusion_matrix_image_path",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_metric_files(runs_root: Path) -> list[Path]:
    return sorted(runs_root.rglob("metrics/test_metrics.json"))


def load_run_row(metrics_path: Path, runs_root: Path) -> dict:
    run_dir = metrics_path.parents[1]
    metrics = read_json(metrics_path)
    config = read_yaml(run_dir / "config_resolved.yaml") or read_yaml(run_dir / "config.yaml")
    progress_path = run_dir / "artifacts" / "training_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    run_cfg = config.get("run", {})

    row = {
        "model_name": model_cfg.get("model_name") or infer_model_name_from_path(run_dir),
        "input_scheme": data_cfg.get("input_scheme") or infer_input_scheme_from_run_name(run_dir.name),
        "run_name": run_cfg.get("run_name") or run_dir.name,
        "run_dir": str(run_dir),
        "completed_epoch": progress.get("completed_epoch"),
        "total_epochs": progress.get("total_epochs") or train_cfg.get("epochs"),
        "best_epoch": progress.get("best_epoch"),
        "best_val_macro_f1": safe_float(progress.get("best_val_macro_f1")),
        "status": progress.get("status", "unknown"),
        "batch_size": train_cfg.get("batch_size"),
        "learning_rate": train_cfg.get("learning_rate"),
        "min_learning_rate": train_cfg.get("min_learning_rate"),
        "weight_decay": train_cfg.get("weight_decay"),
        "scheduler": train_cfg.get("scheduler"),
        "early_stopping": train_cfg.get("early_stopping"),
        "checkpoint_interval": train_cfg.get("checkpoint_interval"),
        "metrics_path": str(metrics_path),
        "confusion_matrix_path": str(run_dir / "metrics" / "confusion_matrix.csv"),
        "confusion_matrix_image_path": str(run_dir / "plots" / "confusion_matrix.png"),
    }
    for metric in METRIC_COLUMNS:
        row[metric] = safe_float(metrics.get(metric))
    row["relative_run_dir"] = str(run_dir.relative_to(runs_root)) if run_dir.is_relative_to(runs_root) else str(run_dir)
    row["run_group"] = row["relative_run_dir"].split("/", 1)[0] if row["relative_run_dir"] else row["model_name"]
    return row


def infer_model_name_from_path(run_dir: Path) -> str:
    for part in run_dir.parts:
        if part in {"mobilenet_v2", "efficientnet_v2s", "efficientnet_v2_s"}:
            return "efficientnet_v2_s" if part == "efficientnet_v2s" else part
    if "efficientnet_v2_s" in run_dir.name:
        return "efficientnet_v2_s"
    if "mobilenet_v2" in run_dir.name:
        return "mobilenet_v2"
    return "unknown"


def infer_input_scheme_from_run_name(run_name: str) -> str:
    for prefix in ["efficientnet_v2_s_", "mobilenet_v2_"]:
        if run_name.startswith(prefix):
            return run_name.removeprefix(prefix)
    parts = run_name.split("_", 2)
    return parts[-1] if parts else "unknown"


def sort_by_metric(rows: list[dict], metric: str) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row.get(metric) if row.get(metric) is not None else float("-inf"),
            row.get("accuracy") if row.get("accuracy") is not None else float("-inf"),
            row.get("macro_f1") if row.get("macro_f1") is not None else float("-inf"),
        ),
        reverse=True,
    )


def add_ranks(rows: list[dict]) -> list[dict]:
    for rank, row in enumerate(sort_by_metric(rows, "accuracy"), start=1):
        row["global_rank_accuracy"] = rank
    for rank, row in enumerate(sort_by_metric(rows, "macro_f1"), start=1):
        row["global_rank_macro_f1"] = rank

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(row)
    for model_rows in by_model.values():
        for rank, row in enumerate(sort_by_metric(model_rows, "accuracy"), start=1):
            row["model_rank_accuracy"] = rank
        for rank, row in enumerate(sort_by_metric(model_rows, "macro_f1"), start=1):
            row["model_rank_macro_f1"] = rank
    return rows


def build_model_summary(rows: list[dict]) -> list[dict]:
    summary = []
    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model_name"]].append(row)

    for model_name, model_rows in sorted(by_model.items()):
        best_acc = sort_by_metric(model_rows, "accuracy")[0]
        best_f1 = sort_by_metric(model_rows, "macro_f1")[0]
        summary.append(
            {
                "model_name": model_name,
                "num_runs": len(model_rows),
                "best_accuracy": best_acc.get("accuracy"),
                "best_accuracy_input_scheme": best_acc.get("input_scheme"),
                "best_accuracy_run_dir": best_acc.get("run_dir"),
                "best_macro_f1": best_f1.get("macro_f1"),
                "best_macro_f1_input_scheme": best_f1.get("input_scheme"),
                "best_macro_f1_run_dir": best_f1.get("run_dir"),
            }
        )
    return summary


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _plot_label(row: dict) -> str:
    scheme = str(row.get("input_scheme", "unknown")).replace("_", " ")
    group = row.get("run_group") or row.get("model_name") or "unknown"
    return f"{group}\n{scheme}"


def save_bar_chart(rows: list[dict], metric: str, title: str, path: Path, ascending: bool = False) -> None:
    usable_rows = [row for row in rows if row.get(metric) is not None]
    if not usable_rows:
        return
    plot_rows = sorted(usable_rows, key=lambda row: row[metric], reverse=not ascending)
    labels = [_plot_label(row) for row in plot_rows]
    values = [row[metric] for row in plot_rows]

    height = max(4.0, 0.55 * len(plot_rows) + 1.5)
    fig, ax = plt.subplots(figsize=(11, height))
    colors = ["#2f80ed" if not ascending else "#eb5757"] * len(plot_rows)
    ax.barh(labels, values, color=colors)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.4f}", va="center", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_ranking_plots(rows: list[dict], output_dir: Path, n: int = 5) -> list[Path]:
    plots_dir = output_dir / "plots"
    generated: list[Path] = []
    if not rows:
        return generated

    for metric in ["accuracy", "macro_f1"]:
        sorted_rows = sort_by_metric(rows, metric)
        chart_specs = [
            (sorted_rows[:n], f"global_top_{n}_{metric}.png", f"Global Top {n} by {metric.replace('_', ' ').title()}", False),
            (list(reversed(sorted_rows[-n:])), f"global_bottom_{n}_{metric}.png", f"Global Bottom {n} by {metric.replace('_', ' ').title()}", True),
        ]
        for chart_rows, filename, title, ascending in chart_specs:
            path = plots_dir / filename
            save_bar_chart(chart_rows, metric, title, path, ascending=ascending)
            if path.exists():
                generated.append(path)

    by_group = defaultdict(list)
    for row in rows:
        by_group[row.get("run_group") or row.get("model_name") or "unknown"].append(row)

    for group_name, group_rows in sorted(by_group.items()):
        safe_group = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in group_name)
        for metric in ["accuracy", "macro_f1"]:
            chart_rows = sort_by_metric(group_rows, metric)[:n]
            path = plots_dir / f"{safe_group}_top_{n}_{metric}.png"
            title = f"{group_name} Top {n} Schemes by {metric.replace('_', ' ').title()}"
            save_bar_chart(chart_rows, metric, title, path)
            if path.exists():
                generated.append(path)
    return generated


def write_markdown_report(
    rows: list[dict],
    model_summary: list[dict],
    output_path: Path,
    top_n: int,
    plot_paths: list[Path] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_paths or []
    lines = [
        "# Run Ranking Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Total completed metric files: {len(rows)}",
        "",
        "## Global Rank by Accuracy",
        "",
    ]
    top_accuracy = sort_by_metric(rows, "accuracy")[:top_n]
    lines.extend(
        markdown_table(
            top_accuracy,
            ["global_rank_accuracy", "model_name", "input_scheme", "accuracy", "macro_f1", "run_dir"],
        )
    )
    lines.extend(["", "## Global Rank by Macro F1", ""])
    top_f1 = sort_by_metric(rows, "macro_f1")[:top_n]
    lines.extend(
        markdown_table(
            top_f1,
            ["global_rank_macro_f1", "model_name", "input_scheme", "macro_f1", "accuracy", "run_dir"],
        )
    )
    lines.extend(["", "## Best per Model", ""])
    lines.extend(
        markdown_table(
            model_summary,
            [
                "model_name",
                "num_runs",
                "best_accuracy",
                "best_accuracy_input_scheme",
                "best_macro_f1",
                "best_macro_f1_input_scheme",
            ],
        )
    )
    if plot_paths:
        lines.extend(["", "## Ranking Plots", ""])
        for path in plot_paths:
            rel_path = path.relative_to(output_path.parent)
            title = path.stem.replace("_", " ").title()
            lines.extend([f"### {title}", "", f"![{title}]({rel_path.as_posix()})", ""])
    lines.extend(["", "## Output Files", ""])
    lines.extend(
        [
            "- `global_rank.csv`",
            "- `global_rank_by_macro_f1.csv`",
            "- `best_per_model.csv`",
            "- `rank_per_model_accuracy.csv`",
            "- `rank_per_model_macro_f1.csv`",
            "- `plots/global_top_5_accuracy.png`",
            "- `plots/global_bottom_5_accuracy.png`",
            "- `plots/global_top_5_macro_f1.png`",
            "- `plots/global_bottom_5_macro_f1.png`",
            "- `plots/[run_group]_top_5_accuracy.png`",
            "- `plots/[run_group]_top_5_macro_f1.png`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank all experiment runs under runs/ recursively.")
    parser.add_argument("--runs_root", default="runs", help="Root folder that contains run outputs.")
    parser.add_argument("--output_dir", default="summary/run_rankings", help="Folder for CSV and Markdown reports.")
    parser.add_argument("--top_n", type=int, default=20, help="Number of rows shown in Markdown top tables.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    metric_files = discover_metric_files(runs_root)
    rows = [load_run_row(path, runs_root) for path in metric_files]
    rows = add_ranks(rows)
    model_summary = build_model_summary(rows)

    global_accuracy = sort_by_metric(rows, "accuracy")
    global_f1 = sort_by_metric(rows, "macro_f1")
    per_model_accuracy = sorted(rows, key=lambda row: (row["model_name"], row["model_rank_accuracy"]))
    per_model_f1 = sorted(rows, key=lambda row: (row["model_name"], row["model_rank_macro_f1"]))

    write_csv(global_accuracy, output_dir / "global_rank.csv", OUTPUT_COLUMNS)
    write_csv(global_f1, output_dir / "global_rank_by_macro_f1.csv", OUTPUT_COLUMNS)
    write_csv(model_summary, output_dir / "best_per_model.csv", list(model_summary[0].keys()) if model_summary else [])
    write_csv(per_model_accuracy, output_dir / "rank_per_model_accuracy.csv", OUTPUT_COLUMNS)
    write_csv(per_model_f1, output_dir / "rank_per_model_macro_f1.csv", OUTPUT_COLUMNS)
    plot_paths = save_ranking_plots(rows, output_dir, n=5)
    write_markdown_report(rows, model_summary, output_dir / "RUN_RANKING_SUMMARY.md", args.top_n, plot_paths)

    best = global_accuracy[0] if global_accuracy else {}
    print(f"runs_found={len(rows)}")
    if best:
        print(
            "best_accuracy="
            f"{best.get('accuracy'):.4f} model={best.get('model_name')} "
            f"scheme={best.get('input_scheme')} run_dir={best.get('run_dir')}"
        )
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
