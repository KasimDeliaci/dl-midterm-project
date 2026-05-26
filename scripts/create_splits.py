"""Create leakage-aware HAM10000 train/validation/test splits from audited metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from dl_midterm.data.splits import (
    check_lesion_leakage,
    create_lesion_aware_splits,
    write_split_csvs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="CSV with image_id, label, image_path, and optional lesion_id columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(Path(args.config))["dataset"]
    metadata = pd.read_csv(args.metadata_csv)
    split_config = config["split"]

    result = create_lesion_aware_splits(
        metadata,
        train_size=float(split_config["train"]),
        val_size=float(split_config["val"]),
        test_size=float(split_config["test"]),
        seed=int(config.get("seed", 42)),
    )
    leaks = check_lesion_leakage(result.splits)
    if leaks:
        raise SystemExit("Lesion leakage detected: " + "; ".join(leaks))
    write_split_csvs(result.splits, Path(config["splits_dir"]))
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote train/val/test splits to {config['splits_dir']}")


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
