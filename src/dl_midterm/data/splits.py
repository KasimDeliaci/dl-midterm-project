"""Train/validation/test split creation and leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitResult:
    """Container for generated split frames and audit warnings."""

    splits: dict[str, pd.DataFrame]
    warnings: list[str]


def create_lesion_aware_splits(
    metadata: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    label_col: str = "label",
    group_col: str = "lesion_id",
) -> SplitResult:
    """Create 70/15/15 splits, grouping by lesion ID when available.

    Stratification is attempted first. If class/group constraints make that impossible,
    the function falls back to grouped random splitting and records a warning.
    """

    _validate_split_ratios(train_size, val_size, test_size)
    _require_columns(metadata, ["image_id", label_col])

    if group_col in metadata.columns and metadata[group_col].notna().all():
        return _create_grouped_splits(metadata, train_size, val_size, seed, label_col, group_col)

    warnings = [f"`{group_col}` unavailable or incomplete; using image-level stratified split."]
    return SplitResult(
        splits=_create_row_level_splits(metadata, train_size, val_size, seed, label_col),
        warnings=warnings,
    )


def write_split_csvs(splits: dict[str, pd.DataFrame], splits_dir: Path) -> None:
    """Write train/validation/test split CSV files."""

    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        splits[split_name].to_csv(splits_dir / f"{split_name}.csv", index=False)


def check_lesion_leakage(
    splits: dict[str, pd.DataFrame], group_col: str = "lesion_id"
) -> list[str]:
    """Return human-readable leakage messages for lesion IDs crossing splits."""

    if any(group_col not in frame.columns for frame in splits.values()):
        return []

    group_sets = {
        name: set(frame[group_col].dropna().astype(str))
        for name, frame in splits.items()
        if group_col in frame.columns
    }
    leaks: list[str] = []
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        overlap = sorted(group_sets[left] & group_sets[right])
        if overlap:
            preview = ", ".join(overlap[:5])
            suffix = "..." if len(overlap) > 5 else ""
            leaks.append(f"{left}/{right} share {len(overlap)} lesion_id values: {preview}{suffix}")
    return leaks


def _create_grouped_splits(
    metadata: pd.DataFrame,
    train_size: float,
    val_size: float,
    seed: int,
    label_col: str,
    group_col: str,
) -> SplitResult:
    group_labels = (
        metadata.groupby(group_col, dropna=False)[label_col]
        .agg(lambda values: values.mode().iat[0])
        .reset_index()
        .rename(columns={label_col: "group_label"})
    )

    train_groups, holdout_groups, warnings = _safe_train_test_split(
        group_labels,
        train_size=train_size,
        seed=seed,
        stratify_col="group_label",
        split_name="train/holdout",
    )

    relative_val = val_size / (1.0 - train_size)
    val_groups, test_groups, second_warnings = _safe_train_test_split(
        holdout_groups,
        train_size=relative_val,
        seed=seed,
        stratify_col="group_label",
        split_name="val/test",
    )
    warnings.extend(second_warnings)

    group_to_split = {
        **{group: "train" for group in train_groups[group_col]},
        **{group: "val" for group in val_groups[group_col]},
        **{group: "test" for group in test_groups[group_col]},
    }

    split_metadata = metadata.copy()
    split_metadata["split"] = split_metadata[group_col].map(group_to_split)
    splits = {
        split_name: split_metadata[split_metadata["split"] == split_name]
        .drop(columns=["split"])
        .sort_values("image_id")
        .reset_index(drop=True)
        for split_name in SPLIT_NAMES
    }

    leakage = check_lesion_leakage(splits, group_col=group_col)
    if leakage:
        raise ValueError("Lesion-level leakage detected: " + "; ".join(leakage))

    return SplitResult(splits=splits, warnings=warnings)


def _create_row_level_splits(
    metadata: pd.DataFrame,
    train_size: float,
    val_size: float,
    seed: int,
    label_col: str,
) -> dict[str, pd.DataFrame]:
    train_rows, holdout_rows, _ = _safe_train_test_split(
        metadata,
        train_size=train_size,
        seed=seed,
        stratify_col=label_col,
        split_name="train/holdout",
    )
    relative_val = val_size / (1.0 - train_size)
    val_rows, test_rows, _ = _safe_train_test_split(
        holdout_rows,
        train_size=relative_val,
        seed=seed,
        stratify_col=label_col,
        split_name="val/test",
    )
    return {
        "train": train_rows.sort_values("image_id").reset_index(drop=True),
        "val": val_rows.sort_values("image_id").reset_index(drop=True),
        "test": test_rows.sort_values("image_id").reset_index(drop=True),
    }


def _safe_train_test_split(
    frame: pd.DataFrame,
    train_size: float,
    seed: int,
    stratify_col: str,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    warnings: list[str] = []
    stratify = frame[stratify_col] if _can_stratify(frame[stratify_col]) else None
    if stratify is None:
        warnings.append(
            f"Could not stratify {split_name}; "
            "at least one class has fewer than two samples/groups."
        )

    try:
        left, right = train_test_split(
            frame,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as error:
        warnings.append(f"Fell back to unstratified {split_name} split: {error}")
        left, right = train_test_split(
            frame,
            train_size=train_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return left.copy(), right.copy(), warnings


def _can_stratify(series: pd.Series) -> bool:
    return bool((series.value_counts() >= 2).all())


def _validate_split_ratios(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}.")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Split ratios must all be positive.")


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")
