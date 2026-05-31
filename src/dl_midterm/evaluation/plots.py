"""Plot helpers for confusion matrices, comparisons, and training curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_confusion_matrix_plot(
    confusion_matrix: list[list[int]],
    class_names: list[str],
    path: str | Path,
    title: str,
) -> Path:
    """Save a confusion matrix heatmap."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_training_curve_plot(
    history: pd.DataFrame,
    path: str | Path,
    title: str,
) -> Path:
    """Save train/validation loss, macro-F1, and accuracy curves."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_lr = "learning_rate" in history.columns
    column_count = 4 if has_lr else 3
    fig, axes = plt.subplots(1, column_count, figsize=(5 * column_count, 4))
    best_epoch = None
    if "val_macro_f1" in history.columns and not history.empty:
        best_epoch = int(history.sort_values("val_macro_f1", ascending=False).iloc[0]["epoch"])
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    if best_epoch is not None:
        axes[0].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_macro_f1"], label="val macro-F1", color="tab:green")
    if best_epoch is not None:
        axes[1].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    axes[2].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[2].plot(history["epoch"], history["val_accuracy"], label="val")
    if best_epoch is not None:
        axes[2].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    if has_lr:
        axes[3].plot(history["epoch"], history["learning_rate"], label="learning rate")
        if best_epoch is not None:
            axes[3].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
        axes[3].set_title("Learning Rate")
        axes[3].set_xlabel("Epoch")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def save_macro_f1_comparison_plot(
    results: pd.DataFrame,
    path: str | Path,
    title: str = "Frozen Single-Backbone Macro-F1",
) -> Path:
    """Save a bar chart comparing macro-F1 across single backbones."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = results.sort_values("macro_f1", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=ordered, x="backbone", y="macro_f1", hue="backbone", legend=False)
    plt.ylim(0, 1)
    plt.xlabel("Backbone")
    plt.ylabel("Macro-F1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_mlp_search_macro_f1_plot(
    results: pd.DataFrame,
    path: str | Path,
    title: str = "MLP Hyperparameter Search Macro-F1",
) -> Path:
    """Save macro-F1 by candidate and backbone for MLP search runs."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = results.copy()
    plot_frame["candidate"] = plot_frame["run_tag"].fillna(plot_frame["run_id"])
    plt.figure(figsize=(max(10, 0.55 * len(plot_frame["candidate"].unique())), 5.5))
    sns.barplot(data=plot_frame, x="candidate", y="macro_f1", hue="backbone")
    plt.ylim(0, 1)
    plt.xlabel("MLP candidate")
    plt.ylabel("Test macro-F1")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_frozen_fusion_comparison_plot(results: pd.DataFrame, path: str | Path) -> Path:
    """Save a sorted macro-F1 bar chart for all frozen experiments."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = results.sort_values("macro_f1", ascending=False).copy()
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=frame, x="display_name", y="macro_f1", hue="experiment_group")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frozen experiment")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Frozen Feature Experiments Macro-F1")
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.legend(title="Group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_single_pairwise_three_plot(results: pd.DataFrame, path: str | Path) -> Path:
    """Save the best macro-F1 by backbone-count group."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = (
        results.sort_values("macro_f1", ascending=False)
        .groupby("experiment_group", as_index=False)
        .head(1)
        .sort_values("macro_f1", ascending=False)
    )
    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=frame, x="experiment_group", y="macro_f1", hue="experiment_group")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Experiment group")
    ax.set_ylabel("Best test macro-F1")
    ax.set_title("Best Single vs Pairwise vs Three-Backbone Fusion")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_concat_vs_weighted_plot(results: pd.DataFrame, path: str | Path) -> Path:
    """Save grouped macro-F1 bars for concat versus weighted fusion."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = results[results["fusion_method"].isin(["concat", "weighted"])].copy()
    frame = frame.sort_values(["backbone_combination", "fusion_method"])
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=frame, x="backbone_combination", y="macro_f1", hue="fusion_method")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Backbone combination")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Concatenation vs Weighted Fusion")
    ax.tick_params(axis="x", rotation=25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.legend(title="Fusion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_fusion_gain_plot(gains: pd.DataFrame, path: str | Path) -> Path:
    """Save macro-F1 gains over the best single-backbone frozen baseline."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = gains.sort_values("macro_f1_gain", ascending=False).copy()
    plt.figure(figsize=(11, 5.5))
    ax = sns.barplot(data=frame, x="display_name", y="macro_f1_gain", hue="fusion_method")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Fusion experiment")
    ax.set_ylabel("Macro-F1 gain over best single")
    ax.set_title("Fusion Gain Over Best Single-Backbone Baseline")
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%+.3f", padding=2, fontsize=8)
    plt.legend(title="Fusion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_frozen_vs_finetuned_macro_f1_plot(summary: pd.DataFrame, path: str | Path) -> Path:
    """Save paired frozen/fine-tuned macro-F1 bars by matching experiment."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary.sort_values("finetuned_macro_f1", ascending=False).copy()
    melted = frame.melt(
        id_vars=["display_name"],
        value_vars=["frozen_macro_f1", "finetuned_macro_f1"],
        var_name="feature_source",
        value_name="macro_f1",
    )
    melted["feature_source"] = melted["feature_source"].map(
        {"frozen_macro_f1": "frozen", "finetuned_macro_f1": "fine-tuned"}
    )
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=melted, x="display_name", y="macro_f1", hue="feature_source")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Frozen vs Fine-Tuned Macro-F1")
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.legend(title="Feature source")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_finetuning_gain_plot(summary: pd.DataFrame, path: str | Path) -> Path:
    """Save macro-F1 gain for fine-tuned representations over frozen counterparts."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary.sort_values("macro_f1_gain", ascending=False).copy()
    plt.figure(figsize=(12, 5.5))
    ax = sns.barplot(data=frame, x="display_name", y="macro_f1_gain", hue="fusion_method")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Fine-tuned minus frozen macro-F1")
    ax.set_title("Fine-Tuning Gain Over Frozen Counterpart")
    ax.tick_params(axis="x", rotation=35)
    for container in ax.containers:
        ax.bar_label(container, fmt="%+.3f", padding=2, fontsize=8)
    plt.legend(title="Fusion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_per_class_source_comparison_heatmap(per_class: pd.DataFrame, path: str | Path) -> Path:
    """Save per-class F1 heatmap across frozen and fine-tuned experiments."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = per_class.pivot_table(
        index="source_display_name",
        columns="label",
        values="f1",
        aggfunc="first",
    )
    order = (
        per_class[["source_display_name", "macro_f1"]]
        .drop_duplicates()
        .sort_values("macro_f1", ascending=False)["source_display_name"]
    )
    pivot = pivot.loc[order]
    plt.figure(figsize=(10, max(5, 0.34 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("Class")
    plt.ylabel("Experiment")
    plt.title("Per-Class F1: Frozen vs Fine-Tuned")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_per_class_f1_heatmap(per_class: pd.DataFrame, path: str | Path) -> Path:
    """Save a per-class F1 heatmap for frozen experiments."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = per_class.pivot_table(
        index="display_name",
        columns="label",
        values="f1",
        aggfunc="first",
    )
    ordered_index = (
        per_class[["display_name", "macro_f1"]]
        .drop_duplicates()
        .sort_values("macro_f1", ascending=False)["display_name"]
    )
    pivot = pivot.loc[ordered_index]
    plt.figure(figsize=(10, max(5, 0.38 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
    plt.xlabel("Class")
    plt.ylabel("Frozen experiment")
    plt.title("Per-Class F1 Across Frozen Experiments")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_per_class_fusion_gain_heatmap(gains: pd.DataFrame, path: str | Path) -> Path:
    """Save per-class F1 gains versus the best overall single-backbone baseline."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = gains.pivot_table(
        index="display_name",
        columns="label",
        values="f1_gain",
        aggfunc="first",
    )
    ordered_index = (
        gains[["display_name", "macro_f1_gain"]]
        .drop_duplicates()
        .sort_values("macro_f1_gain", ascending=False)["display_name"]
    )
    pivot = pivot.loc[ordered_index]
    limit = max(0.25, float(pivot.abs().max().max()))
    plt.figure(figsize=(10, max(5, 0.38 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="+.2f",
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
    )
    plt.xlabel("Class")
    plt.ylabel("Fusion experiment")
    plt.title("Per-Class F1 Gain vs Best Single Backbone")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_learned_fusion_weights_plot(weights: pd.DataFrame, path: str | Path) -> Path:
    """Save learned weighted-fusion softmax weights."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = weights.sort_values(["display_name", "backbone"]).copy()
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=frame, x="display_name", y="weight", hue="backbone")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Weighted fusion run")
    ax.set_ylabel("Learned softmax weight")
    ax.set_title("Learned Global Fusion Weights")
    ax.tick_params(axis="x", rotation=30)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    plt.legend(title="Backbone")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_accuracy_macro_f1_scatter(results: pd.DataFrame, path: str | Path) -> Path:
    """Save an accuracy versus macro-F1 scatter plot."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    ax = sns.scatterplot(
        data=results,
        x="accuracy",
        y="macro_f1",
        hue="experiment_group",
        style="fusion_method",
        s=90,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Test accuracy")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Accuracy vs Macro-F1")
    for row in results.itertuples(index=False):
        ax.text(row.accuracy + 0.004, row.macro_f1 + 0.004, row.short_name, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_sprint4b_val_macro_f1_screening_plot(results: pd.DataFrame, path: str | Path) -> Path:
    """Save validation macro-F1 bars for Sprint 4B single-backbone screening."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = results.dropna(subset=["best_val_macro_f1"]).copy()
    frame = frame.sort_values("best_val_macro_f1", ascending=False)
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=frame,
        x="backbone",
        y="best_val_macro_f1",
        hue="source_label",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Sprint 4B Single-Backbone Screening")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.legend(title="Feature source")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_sprint4b_test_macro_f1_vs_canonical_plot(
    comparison: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save canonical Sprint 4 versus Sprint 4B test macro-F1 bars."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    melted = comparison.melt(
        id_vars=["backbone", "feature_source", "source_label"],
        value_vars=["canonical_macro_f1", "macro_f1"],
        var_name="series",
        value_name="test_macro_f1",
    )
    melted["series"] = melted["series"].map(
        {"canonical_macro_f1": "canonical Sprint 4", "macro_f1": "Sprint 4B"}
    )
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=melted,
        x="backbone",
        y="test_macro_f1",
        hue="series",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Backbone")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Sprint 4B vs Canonical Sprint 4 Single-Backbone Macro-F1")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_sprint4b_per_class_gain_heatmap(gains: pd.DataFrame, path: str | Path) -> Path:
    """Save per-class F1 gains for Sprint 4B screening runs versus canonical Sprint 4."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = gains.pivot_table(
        index="candidate_display_name",
        columns="label",
        values="f1_gain",
        aggfunc="first",
    )
    order = (
        gains[["candidate_display_name", "macro_f1_gain"]]
        .drop_duplicates()
        .sort_values("macro_f1_gain", ascending=False)["candidate_display_name"]
    )
    pivot = pivot.loc[order]
    limit = max(0.25, float(pivot.abs().max().max()))
    plt.figure(figsize=(10, max(4, 0.45 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="+.2f",
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
    )
    plt.xlabel("Class")
    plt.ylabel("Sprint 4B run")
    plt.title("Sprint 4B Per-Class F1 Gain vs Canonical Sprint 4")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_representation_similarity_heatmap(pairwise: pd.DataFrame, path: str | Path) -> Path:
    """Save a heatmap of pairwise representation similarity."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backbones = sorted(
        set(pairwise["left_backbone"].astype(str)) | set(pairwise["right_backbone"].astype(str))
    )
    matrix = pd.DataFrame(1.0, index=backbones, columns=backbones)
    for row in pairwise.itertuples(index=False):
        matrix.loc[row.left_backbone, row.right_backbone] = row.representation_similarity
        matrix.loc[row.right_backbone, row.left_backbone] = row.representation_similarity

    plt.figure(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="mako", vmin=0, vmax=1, square=True)
    plt.xlabel("Backbone")
    plt.ylabel("Backbone")
    plt.title("Frozen Feature Representation Similarity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_fusion_gain_vs_complementarity_plot(summary: pd.DataFrame, path: str | Path) -> Path:
    """Save fusion macro-F1 gain against average representation complementarity."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        data=summary,
        x="avg_pairwise_representation_complementarity",
        y="macro_f1_gain",
        hue="fusion_method",
        style="backbone_count",
        s=100,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Average pairwise representation complementarity")
    ax.set_ylabel("Macro-F1 gain over best single")
    ax.set_title("Fusion Gain vs Representation Complementarity")
    for row in summary.itertuples(index=False):
        ax.text(
            row.avg_pairwise_representation_complementarity + 0.002,
            row.macro_f1_gain + 0.002,
            row.display_name,
            fontsize=7,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
