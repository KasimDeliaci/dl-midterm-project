"""Aggregate saved run metrics and produce report-ready assets."""

from __future__ import annotations

import argparse

from dl_midterm.evaluation.reports import export_single_backbone_report_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Accepted for command consistency.")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--feature-source", default="frozen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported = export_single_backbone_report_assets(
        args.run_root,
        args.tables_dir,
        args.figures_dir,
        feature_source=args.feature_source,
    )
    print("Exported report assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
