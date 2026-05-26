"""Run a focused MLP hyperparameter search on cached frozen features."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from train_mlp import run_single_backbone

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.reports import export_mlp_search_report_assets
from dl_midterm.utils.device import resolve_device
from dl_midterm.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/mlp_hyperparam_search.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--search-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_config = load_yaml(args.config)["mlp_hyperparam_search"]
    default_config = load_yaml(args.default_config)
    dataset_config = load_yaml(args.dataset_config)["dataset"]
    training_config = default_config.get("training", {})

    seed = int(search_config.get("seed", dataset_config.get("seed", 42)))
    seed_everything(seed)
    device = resolve_device(args.device or default_config.get("runtime", {}).get("device", "auto"))
    search_id = args.search_id or (
        f"{search_config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_root = Path(search_config["run_root"]) / search_id
    tables_dir = Path(search_config["tables_dir"]) / search_id
    figures_dir = Path(search_config["figures_dir"]) / search_id
    run_root.mkdir(parents=True, exist_ok=True)

    backbones = args.backbones or list(search_config["backbones"])
    candidates = _select_candidates(search_config["candidates"], args.candidates)
    batch_size = args.batch_size or int(training_config.get("batch_size", 32))
    epochs = args.epochs or int(training_config.get("epochs", 25))
    completed_runs: list[dict[str, Any]] = []

    for candidate in candidates:
        for backbone in backbones:
            if args.max_runs is not None and len(completed_runs) >= args.max_runs:
                break
            print(f"Running {search_id}: {backbone} / {candidate['name']}")
            run_args = SimpleNamespace(
                feature_root=args.feature_root,
                feature_source=search_config.get("feature_source", "frozen"),
                fusion=search_config.get("fusion_method", "none"),
                run_root=str(run_root),
            )
            run_dir = run_single_backbone(
                backbone=backbone,
                args=run_args,
                dataset_config=dataset_config,
                seed=seed,
                device=device,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=float(candidate["learning_rate"]),
                weight_decay=float(candidate["weight_decay"]),
                optimizer_name=str(candidate["optimizer"]),
                momentum=float(candidate.get("momentum", 0.9)),
                dropout=float(candidate["dropout"]),
                hidden_dims=list(candidate["hidden_dims"]),
                class_weighting=bool(candidate["class_weighting"]),
                training_config=training_config,
                experiment_name=search_config["name"],
                run_tag=str(candidate["name"]),
            )
            completed_runs.append(
                {
                    "backbone": backbone,
                    "candidate": candidate["name"],
                    "run_dir": str(run_dir),
                }
            )
        if args.max_runs is not None and len(completed_runs) >= args.max_runs:
            break

    manifest_path = run_root / "search_manifest.yaml"
    manifest = {
        "search_id": search_id,
        "search_config": search_config,
        "dataset_config": str(Path(args.dataset_config)),
        "default_config": str(Path(args.default_config)),
        "feature_root": args.feature_root,
        "device": str(device),
        "batch_size": batch_size,
        "epochs": epochs,
        "completed_runs": completed_runs,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    exported = export_mlp_search_report_assets(
        run_root,
        tables_dir,
        figures_dir,
        feature_source=search_config.get("feature_source", "frozen"),
        experiment_name=search_config["name"],
    )
    (run_root / "exported_assets.json").write_text(
        json.dumps({name: str(path) for name, path in exported.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote search manifest: {manifest_path}")
    print("Exported report assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


def _select_candidates(
    candidates: list[dict[str, Any]],
    requested_names: list[str] | None,
) -> list[dict[str, Any]]:
    if requested_names is None:
        return candidates
    by_name = {str(candidate["name"]): candidate for candidate in candidates}
    missing = sorted(set(requested_names) - set(by_name))
    if missing:
        raise ValueError(f"Unknown candidate names: {missing}")
    return [by_name[name] for name in requested_names]


if __name__ == "__main__":
    main()
