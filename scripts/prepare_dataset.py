"""Audit HAM10000 metadata and create leakage-aware dataset splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from dl_midterm.data.datasets import (
    attach_image_paths,
    audit_metadata,
    find_metadata_file,
    load_ham10000_metadata,
)
from dl_midterm.data.splits import create_lesion_aware_splits, write_split_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(Path(args.config))
    dataset_config = config["dataset"]

    metadata_dir = Path(dataset_config["metadata_dir"])
    raw_dir = Path(args.raw_dir or dataset_config["raw_dir"])
    metadata_path = find_metadata_file(metadata_dir, args.metadata_path)

    metadata = load_ham10000_metadata(metadata_path, dataset_config["class_names"])
    metadata = attach_image_paths(metadata, raw_dir)
    audit = audit_metadata(metadata, metadata_path, raw_dir)

    processed_dir = Path(dataset_config["processed_dir"])
    tables_dir = Path("artifacts/report_assets/tables")
    figures_dir = Path("artifacts/report_assets/figures")
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    audited_metadata_path = processed_dir / "ham10000_audited_metadata.csv"
    metadata.to_csv(audited_metadata_path, index=False)
    audit.class_distribution.to_csv(tables_dir / "class_distribution.csv", index=False)
    _plot_class_distribution(audit.class_distribution, figures_dir / "class_distribution.png")

    if audit.missing_images:
        missing_path = tables_dir / "missing_images.csv"
        pd.DataFrame({"image_id": audit.missing_images}).to_csv(missing_path, index=False)
        print(f"Missing image list written to {missing_path}")

    _print_audit_summary(audit)
    if audit.has_blocking_errors and not args.allow_incomplete:
        raise SystemExit(
            "Dataset audit found blocking issues. Fix missing/duplicate metadata or rerun with "
            "--allow-incomplete only for manual inspection; splits were not written."
        )

    split_config = dataset_config["split"]
    split_result = create_lesion_aware_splits(
        metadata,
        train_size=float(split_config["train"]),
        val_size=float(split_config["val"]),
        test_size=float(split_config["test"]),
        seed=int(dataset_config.get("seed", 42)),
    )
    write_split_csvs(split_result.splits, Path(dataset_config["splits_dir"]))

    split_distribution = _split_distribution(split_result.splits)
    split_distribution.to_csv(tables_dir / "split_class_distribution.csv", index=False)
    _plot_split_distribution(split_distribution, figures_dir / "split_class_distribution.png")
    for warning in split_result.warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote splits to {dataset_config['splits_dir']}")


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _plot_class_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(distribution["label"], distribution["count"], color="#3b82f6")
    plt.xlabel("HAM10000 class")
    plt.ylabel("Image count")
    plt.title("HAM10000 class distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_split_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    pivot = distribution.pivot(index="label", columns="split", values="count").fillna(0)
    pivot = pivot[["train", "val", "test"]]

    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#2563eb", "#f59e0b", "#10b981"])
    ax.set_xlabel("HAM10000 class")
    ax.set_ylabel("Image count")
    ax.set_title("HAM10000 split distribution by class")
    ax.legend(title="Split")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _split_distribution(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, frame in splits.items():
        counts = frame["label"].value_counts().sort_index()
        for label, count in counts.items():
            rows.append({"split": split_name, "label": label, "count": int(count)})
    return pd.DataFrame(rows)


def _print_audit_summary(audit) -> None:
    print(f"Metadata: {audit.metadata_path}")
    print(f"Raw images: {audit.raw_dir}")
    print("Audited metadata: data/processed/ham10000_audited_metadata.csv")
    print(f"Rows: {audit.image_rows}")
    print(f"Unique image IDs: {audit.unique_image_ids}")
    print(f"Duplicate image IDs: {len(audit.duplicate_image_ids)}")
    print(f"Missing images: {len(audit.missing_images)}")
    print(f"Missing labels: {audit.missing_labels}")
    print(f"Lesion ID available: {audit.lesion_id_available}")


if __name__ == "__main__":
    main()
