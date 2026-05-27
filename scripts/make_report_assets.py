"""Generate report-ready frozen experiment tables and plots."""

from __future__ import annotations

import argparse

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.reports import export_frozen_matrix_report_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for command consistency.")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--feature-source", default="frozen")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--complementarity-split", default="test")
    parser.add_argument("--complementarity-max-samples", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_yaml(args.dataset_config)["dataset"]
    exported = export_frozen_matrix_report_assets(
        args.run_root,
        args.tables_dir,
        args.figures_dir,
        feature_source=args.feature_source,
        feature_root=args.feature_root,
        dataset_config=dataset_config,
        complementarity_split=args.complementarity_split,
        complementarity_max_samples=args.complementarity_max_samples,
    )
    print("Exported report assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
