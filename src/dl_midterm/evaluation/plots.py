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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_macro_f1"], label="val macro-F1", color="tab:green")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)

    axes[2].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[2].plot(history["epoch"], history["val_accuracy"], label="val")
    axes[2].set_title("Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].legend()

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
