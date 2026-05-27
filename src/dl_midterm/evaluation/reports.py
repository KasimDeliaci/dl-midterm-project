"""Classification report and run-summary helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dl_midterm.evaluation.complementarity import (
    build_fusion_complementarity_summary,
    compute_representation_complementarity,
)
from dl_midterm.evaluation.metrics import per_class_frame
from dl_midterm.evaluation.plots import (
    save_accuracy_macro_f1_scatter,
    save_concat_vs_weighted_plot,
    save_confusion_matrix_plot,
    save_frozen_fusion_comparison_plot,
    save_fusion_gain_plot,
    save_fusion_gain_vs_complementarity_plot,
    save_learned_fusion_weights_plot,
    save_macro_f1_comparison_plot,
    save_mlp_search_macro_f1_plot,
    save_per_class_f1_heatmap,
    save_per_class_fusion_gain_heatmap,
    save_representation_similarity_heatmap,
    save_single_pairwise_three_plot,
    save_training_curve_plot,
)
from dl_midterm.models.backbones import backbone_alias


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
                "backbones": json.dumps(config.get("backbones", metrics.get("backbones"))),
                "backbone_combination": config.get(
                    "backbone_combination",
                    metrics.get("backbone_combination"),
                ),
                "backbone_count": config.get("backbone_count", metrics.get("backbone_count")),
                "feature_source": config.get("feature_source", metrics.get("feature_source")),
                "fusion_method": config.get("fusion_method", metrics.get("fusion_method")),
                "seed": config.get("seed", metrics.get("seed")),
                "feature_dim": config.get("feature_dim", metrics.get("feature_dim")),
                "fusion_input_dim": config.get(
                    "fusion_input_dim",
                    metrics.get("fusion_input_dim"),
                ),
                "fusion_output_dim": config.get(
                    "fusion_output_dim",
                    metrics.get("fusion_output_dim"),
                ),
                "projection_dim": config.get("projection_dim", metrics.get("projection_dim")),
                "class_weighting": config.get("class_weighting", metrics.get("class_weighting")),
                "optimizer": config.get("optimizer", metrics.get("optimizer")),
                "learning_rate": config.get("learning_rate", metrics.get("learning_rate")),
                "weight_decay": config.get("weight_decay", metrics.get("weight_decay")),
                "dropout": config.get("dropout", metrics.get("dropout")),
                "hidden_dims": json.dumps(config.get("hidden_dims", metrics.get("hidden_dims"))),
                "experiment_name": config.get(
                    "experiment_name",
                    metrics.get("experiment_name"),
                ),
                "run_tag": config.get("run_tag", metrics.get("run_tag")),
                "best_val_macro_f1": config.get(
                    "best_val_macro_f1",
                    metrics.get("best_val_macro_f1"),
                ),
                "best_val_epoch": config.get("best_val_epoch", metrics.get("best_val_epoch")),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    return pd.DataFrame(rows)


def export_frozen_matrix_report_assets(
    run_root: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    *,
    feature_source: str = "frozen",
    feature_root: str | Path | None = None,
    dataset_config: dict[str, Any] | None = None,
    complementarity_split: str = "test",
    complementarity_max_samples: int = 1500,
) -> dict[str, Path]:
    """Export report-ready tables and figures for all frozen single/fusion runs."""

    summaries = collect_run_summaries(run_root)
    if summaries.empty:
        raise FileNotFoundError(f"No run metrics found under {run_root}")

    filtered = summaries[summaries["feature_source"] == feature_source].copy()
    filtered = filtered[filtered["fusion_method"].isin(["none", "concat", "weighted"])].copy()
    if filtered.empty:
        raise FileNotFoundError(f"No frozen matrix runs found under {run_root}")

    filtered = _select_latest_matrix_rows(filtered)
    filtered = _add_display_columns(filtered)

    table_root = Path(tables_dir)
    figure_root = Path(figures_dir)
    run_figure_root = figure_root / "fusion_runs"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    run_figure_root.mkdir(parents=True, exist_ok=True)

    all_results_path = table_root / "frozen_all_results.csv"
    filtered.sort_values(["backbone_count", "backbone_combination", "fusion_method"]).to_csv(
        all_results_path,
        index=False,
    )

    per_class = _collect_per_class_reports(run_root, filtered)
    per_class_path = table_root / "per_class_f1_frozen.csv"
    per_class.to_csv(per_class_path, index=False)

    fusion_rows = filtered[filtered["fusion_method"].isin(["concat", "weighted"])].copy()
    best_single_macro_f1 = float(
        filtered[filtered["fusion_method"] == "none"]["macro_f1"].max()
    )
    gains = fusion_rows.copy()
    gains["best_single_macro_f1"] = best_single_macro_f1
    gains["macro_f1_gain"] = gains["macro_f1"] - best_single_macro_f1
    gain_path = table_root / "fusion_gain_summary.csv"
    gains.to_csv(gain_path, index=False)

    per_class_gains = _build_per_class_fusion_gains(per_class, filtered, gains)
    per_class_gain_path = table_root / "per_class_fusion_gain.csv"
    per_class_gains.to_csv(per_class_gain_path, index=False)

    weights = _collect_fusion_weights(run_root, filtered)
    weights_path = table_root / "fusion_weight_summary.csv"
    weights.to_csv(weights_path, index=False)

    exported: dict[str, Path] = {
        "all_results_table": all_results_path,
        "per_class_table": per_class_path,
        "fusion_gain_table": gain_path,
        "per_class_fusion_gain_table": per_class_gain_path,
        "fusion_weight_table": weights_path,
        "comparison_plot": save_frozen_fusion_comparison_plot(
            filtered,
            figure_root / "frozen_fusion_comparison.png",
        ),
        "single_pairwise_three_plot": save_single_pairwise_three_plot(
            filtered,
            figure_root / "single_pairwise_three_macro_f1.png",
        ),
        "concat_vs_weighted_plot": save_concat_vs_weighted_plot(
            filtered,
            figure_root / "concat_vs_weighted.png",
        ),
        "fusion_gain_plot": save_fusion_gain_plot(
            gains,
            figure_root / "fusion_gain_macro_f1.png",
        ),
        "per_class_heatmap": save_per_class_f1_heatmap(
            per_class,
            figure_root / "per_class_f1_frozen_heatmap.png",
        ),
        "per_class_fusion_gain_heatmap": save_per_class_fusion_gain_heatmap(
            per_class_gains,
            figure_root / "per_class_fusion_gain_heatmap.png",
        ),
        "accuracy_macro_f1_scatter": save_accuracy_macro_f1_scatter(
            filtered,
            figure_root / "accuracy_vs_macro_f1_frozen.png",
        ),
    }

    if feature_root is not None and dataset_config is not None:
        backbones = _matrix_backbones(filtered)
        complementarity = compute_representation_complementarity(
            feature_root=feature_root,
            dataset_name=str(dataset_config["name"]),
            feature_source=feature_source,
            backbones=backbones,
            splits_dir=dataset_config["splits_dir"],
            split=complementarity_split,
            max_samples=complementarity_max_samples,
            seed=int(dataset_config.get("seed", 42)),
        )
        complementarity_path = table_root / "representation_complementarity_summary.csv"
        complementarity.to_csv(complementarity_path, index=False)
        complementarity_results = filtered.merge(
            gains[["run_id", "macro_f1_gain"]],
            on="run_id",
            how="left",
        )
        fusion_complementarity = build_fusion_complementarity_summary(
            complementarity_results,
            complementarity,
        )
        fusion_complementarity_path = table_root / "fusion_complementarity_summary.csv"
        fusion_complementarity.to_csv(fusion_complementarity_path, index=False)
        exported.update(
            {
                "representation_complementarity_table": complementarity_path,
                "fusion_complementarity_table": fusion_complementarity_path,
                "representation_similarity_heatmap": save_representation_similarity_heatmap(
                    complementarity,
                    figure_root / "representation_similarity_heatmap.png",
                ),
                "fusion_gain_vs_complementarity_plot": (
                    save_fusion_gain_vs_complementarity_plot(
                        fusion_complementarity,
                        figure_root / "fusion_gain_vs_complementarity.png",
                    )
                ),
            }
        )
    if not weights.empty:
        exported["learned_weights_plot"] = save_learned_fusion_weights_plot(
            weights,
            figure_root / "learned_fusion_weights.png",
        )

    best_row = filtered.sort_values("macro_f1", ascending=False).iloc[0]
    best_source = Path(run_root) / str(best_row["run_id"]) / "confusion_matrix.png"
    best_dest = figure_root / "frozen_best_confusion_matrix.png"
    if best_source.exists():
        shutil.copy2(best_source, best_dest)
        exported["best_confusion_matrix"] = best_dest

    for row in fusion_rows.itertuples(index=False):
        run_dir = Path(run_root) / row.run_id
        for source_name, suffix in (
            ("confusion_matrix.png", "confusion_matrix"),
            ("training_curve.png", "training_curve"),
        ):
            source_path = run_dir / source_name
            if source_path.exists():
                dest_path = run_figure_root / f"{row.short_name}_{suffix}.png"
                shutil.copy2(source_path, dest_path)
                exported[f"{row.short_name}_{suffix}"] = dest_path

    return exported


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


def _select_latest_matrix_rows(summaries: pd.DataFrame) -> pd.DataFrame:
    frame = summaries.copy()
    frame["backbone_combination"] = frame.apply(_resolve_backbone_combination, axis=1)
    frame["backbone_count"] = frame["backbone_combination"].str.count(r"\+") + 1
    frame = frame.sort_values("run_id")
    return (
        frame.groupby(["backbone_combination", "fusion_method"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _add_display_columns(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["experiment_group"] = frame["backbone_count"].map(
        {1: "single", 2: "pairwise", 3: "three-backbone"}
    )
    frame["experiment_group"] = frame["experiment_group"].fillna("fusion")
    frame["short_name"] = frame.apply(_short_experiment_name, axis=1)
    frame["display_name"] = frame["short_name"]
    return frame


def _resolve_backbone_combination(row: pd.Series) -> str:
    combination = row.get("backbone_combination")
    if isinstance(combination, str) and combination and combination != "nan":
        return combination
    backbone = row.get("backbone")
    if isinstance(backbone, str) and backbone and backbone != "nan":
        return backbone
    backbones_raw = row.get("backbones")
    if isinstance(backbones_raw, str) and backbones_raw and backbones_raw != "null":
        try:
            backbones = json.loads(backbones_raw)
        except json.JSONDecodeError:
            backbones = None
        if isinstance(backbones, list) and backbones:
            return "+".join(str(value) for value in backbones)
    return str(row.get("run_id"))


def _short_experiment_name(row: pd.Series) -> str:
    names = str(row["backbone_combination"]).split("+")
    alias = "+".join(backbone_alias(name) for name in names)
    method = str(row["fusion_method"])
    return alias if method == "none" else f"{alias} {method}"


def _collect_per_class_reports(run_root: str | Path, results: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in results.itertuples(index=False):
        report_path = Path(run_root) / row.run_id / "classification_report.csv"
        if not report_path.exists():
            continue
        report = pd.read_csv(report_path)
        report.insert(0, "run_id", row.run_id)
        report.insert(1, "display_name", row.display_name)
        report.insert(2, "backbone_combination", row.backbone_combination)
        report.insert(3, "fusion_method", row.fusion_method)
        report.insert(4, "macro_f1", row.macro_f1)
        frames.append(report)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _collect_fusion_weights(run_root: str | Path, results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weighted = results[results["fusion_method"] == "weighted"]
    for row in weighted.itertuples(index=False):
        weights_path = Path(run_root) / row.run_id / "fusion_weights.csv"
        if not weights_path.exists():
            continue
        weights = pd.read_csv(weights_path)
        weight_sum = float(weights["weight"].sum())
        for weight_row in weights.itertuples(index=False):
            rows.append(
                {
                    "run_id": row.run_id,
                    "display_name": row.display_name,
                    "backbone_combination": row.backbone_combination,
                    "fusion_method": row.fusion_method,
                    "backbone": weight_row.backbone,
                    "weight": float(weight_row.weight),
                    "weight_sum": weight_sum,
                }
            )
    return pd.DataFrame(rows)


def _build_per_class_fusion_gains(
    per_class: pd.DataFrame,
    results: pd.DataFrame,
    fusion_gains: pd.DataFrame,
) -> pd.DataFrame:
    single_rows = results[results["fusion_method"] == "none"].sort_values(
        "macro_f1",
        ascending=False,
    )
    if single_rows.empty:
        return pd.DataFrame()
    reference = single_rows.iloc[0]
    reference_rows = per_class[per_class["run_id"] == reference["run_id"]][
        ["label", "f1", "precision", "recall", "support"]
    ].rename(
        columns={
            "f1": "reference_f1",
            "precision": "reference_precision",
            "recall": "reference_recall",
        }
    )
    fusion_rows = per_class[per_class["fusion_method"].isin(["concat", "weighted"])].copy()
    gain_rows = fusion_rows.merge(reference_rows, on=["label", "support"], how="left")
    gain_rows = gain_rows.merge(
        fusion_gains[["run_id", "macro_f1_gain"]],
        on="run_id",
        how="left",
    )
    gain_rows["reference_run_id"] = reference["run_id"]
    gain_rows["reference_display_name"] = reference["display_name"]
    gain_rows["f1_gain"] = gain_rows["f1"] - gain_rows["reference_f1"]
    gain_rows["precision_gain"] = gain_rows["precision"] - gain_rows["reference_precision"]
    gain_rows["recall_gain"] = gain_rows["recall"] - gain_rows["reference_recall"]
    return gain_rows.sort_values(["macro_f1_gain", "display_name", "label"], ascending=False)


def _matrix_backbones(results: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for combination in results["backbone_combination"].dropna():
        for backbone in str(combination).split("+"):
            if backbone not in names:
                names.append(backbone)
    return names


def export_mlp_search_report_assets(
    run_root: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    *,
    feature_source: str = "frozen",
    experiment_name: str | None = None,
) -> dict[str, Path]:
    """Export report-ready tables and figures for an MLP hyperparameter search."""

    summaries = collect_run_summaries(run_root)
    if summaries.empty:
        raise FileNotFoundError(f"No run metrics found under {run_root}")

    filtered = summaries[
        (summaries["feature_source"] == feature_source) & (summaries["fusion_method"] == "none")
    ].copy()
    if experiment_name is not None:
        filtered = filtered[filtered["experiment_name"] == experiment_name].copy()
    if filtered.empty:
        raise FileNotFoundError(f"No matching MLP search runs found under {run_root}")

    table_root = Path(tables_dir)
    figure_root = Path(figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    sorted_results = filtered.sort_values(["macro_f1", "weighted_f1"], ascending=False)
    results_path = table_root / "mlp_hyperparam_search_results.csv"
    sorted_results.to_csv(results_path, index=False)

    best_by_backbone = (
        sorted_results.sort_values(["backbone", "macro_f1"], ascending=[True, False])
        .groupby("backbone", as_index=False)
        .head(1)
        .sort_values("macro_f1", ascending=False)
    )
    best_path = table_root / "mlp_hyperparam_best_by_backbone.csv"
    best_by_backbone.to_csv(best_path, index=False)

    per_class_frames: list[pd.DataFrame] = []
    for run_id in sorted_results["run_id"]:
        report_path = Path(run_root) / run_id / "classification_report.csv"
        if report_path.exists():
            report = pd.read_csv(report_path)
            report.insert(0, "run_id", run_id)
            per_class_frames.append(report)
    per_class_path = table_root / "mlp_hyperparam_per_class_f1.csv"
    if per_class_frames:
        pd.concat(per_class_frames, ignore_index=True).to_csv(per_class_path, index=False)
    else:
        pd.DataFrame().to_csv(per_class_path, index=False)

    plot_path = save_mlp_search_macro_f1_plot(
        sorted_results,
        figure_root / "mlp_hyperparam_search_macro_f1.png",
    )

    copied_plots: list[Path] = []
    for row in best_by_backbone.itertuples(index=False):
        run_dir = Path(run_root) / row.run_id
        safe_tag = str(row.run_tag or row.run_id)
        for source_name, suffix in (
            ("confusion_matrix.png", "confusion_matrix"),
            ("training_curve.png", "training_curve"),
        ):
            source_path = run_dir / source_name
            if source_path.exists():
                dest_path = figure_root / f"best_{row.backbone}_{safe_tag}_{suffix}.png"
                shutil.copy2(source_path, dest_path)
                copied_plots.append(dest_path)

    exported = {
        "results_table": results_path,
        "best_by_backbone_table": best_path,
        "per_class_table": per_class_path,
        "macro_f1_plot": plot_path,
    }
    for index, path in enumerate(copied_plots, start=1):
        exported[f"best_run_plot_{index}"] = path
    return exported
