"""Classification report and run-summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dl_midterm.evaluation.metrics import per_class_frame
from dl_midterm.evaluation.plots import (
    save_confusion_matrix_plot,
    save_macro_f1_comparison_plot,
    save_training_curve_plot,
)


def write_run_report(
    run_dir: str | Path,
    *,
    resolved_config: dict[str, Any],
    history: pd.DataFrame,
    metrics: dict[str, Any],
    class_names: list[str],
    backbone: str,
) -> None:
    """Write all standard artifacts for one MLP run."""

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    serializable_metrics = dict(metrics)
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2),
        encoding="utf-8",
    )
    history.to_csv(output_dir / "history.csv", index=False)
    per_class_frame(metrics, backbone=backbone).to_csv(
        output_dir / "classification_report.csv",
        index=False,
    )
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{backbone} frozen MLP confusion matrix",
    )
    save_training_curve_plot(
        history,
        output_dir / "training_curve.png",
        title=f"{backbone} frozen MLP training curves",
    )


def collect_run_summaries(run_root: str | Path) -> pd.DataFrame:
    """Collect metrics.json files from run directories into a table."""

    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(run_root).glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config_path = metrics_path.parent / "config_resolved.yaml"
        config = (
            yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        rows.append(
            {
                "run_id": metrics_path.parent.name,
                "backbone": config.get("backbone", metrics.get("backbone")),
                "feature_source": config.get("feature_source", metrics.get("feature_source")),
                "fusion_method": config.get("fusion_method", metrics.get("fusion_method")),
                "seed": config.get("seed", metrics.get("seed")),
                "feature_dim": config.get("feature_dim", metrics.get("feature_dim")),
                "class_weighting": config.get("class_weighting", metrics.get("class_weighting")),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    return pd.DataFrame(rows)


def export_single_backbone_report_assets(
    run_root: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    *,
    feature_source: str = "frozen",
) -> dict[str, Path]:
    """Export report-ready aggregate tables/plots for single-backbone frozen runs."""

    summaries = collect_run_summaries(run_root)
    if summaries.empty:
        raise FileNotFoundError(f"No run metrics found under {run_root}")
    filtered = summaries[
        (summaries["feature_source"] == feature_source) & (summaries["fusion_method"] == "none")
    ].copy()
    if filtered.empty:
        raise FileNotFoundError(f"No {feature_source} single-backbone runs found under {run_root}")

    table_root = Path(tables_dir)
    figure_root = Path(figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    results_path = table_root / "single_backbone_frozen_results.csv"
    filtered.sort_values("backbone").to_csv(results_path, index=False)

    per_class_frames: list[pd.DataFrame] = []
    for run_id in filtered["run_id"]:
        report_path = Path(run_root) / run_id / "classification_report.csv"
        if report_path.exists():
            per_class_frames.append(pd.read_csv(report_path))
    per_class_path = table_root / "single_backbone_frozen_per_class_f1.csv"
    if per_class_frames:
        pd.concat(per_class_frames, ignore_index=True).to_csv(per_class_path, index=False)
    else:
        pd.DataFrame().to_csv(per_class_path, index=False)

    plot_path = save_macro_f1_comparison_plot(
        filtered,
        figure_root / "frozen_single_backbone_f1.png",
    )
    return {
        "results_table": results_path,
        "per_class_table": per_class_path,
        "macro_f1_plot": plot_path,
    }
