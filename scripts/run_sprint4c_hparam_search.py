"""Run Sprint 4C cached-feature MLP/fusion hyperparameter search."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from run_experiment_matrix import run_fusion_experiment
from train_mlp import run_single_backbone

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.reports import export_sprint4c_hparam_search_report_assets
from dl_midterm.utils.device import resolve_device
from dl_midterm.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/sprint4c_finetuned_mlpfusion_screen.yaml",
    )
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--canonical-run-root", default="artifacts/runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--search-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_config = load_yaml(args.config)["mlp_fusion_hparam_search"]
    default_config = load_yaml(args.default_config)
    dataset_config = load_yaml(args.dataset_config)["dataset"]
    training_config = default_config.get("training", {})

    seed = int(search_config.get("seed", dataset_config.get("seed", 42)))
    device = resolve_device(args.device or default_config.get("runtime", {}).get("device", "auto"))
    batch_size = args.batch_size or int(training_config.get("batch_size", 32))
    epochs = args.epochs or int(search_config.get("epochs", training_config.get("epochs", 25)))
    search_id = args.search_id or (
        f"{search_config['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    run_root = Path(search_config["run_root"]) / search_id
    tables_dir = Path(search_config["tables_dir"]) / search_id
    figures_dir = Path(search_config["figures_dir"]) / search_id
    run_root.mkdir(parents=True, exist_ok=True)

    planned_runs = expand_search_runs(search_config)
    if args.max_runs is not None:
        planned_runs = _limit_planned_runs(planned_runs, args.max_runs)
    if not planned_runs:
        raise ValueError("No Sprint 4C runs selected.")

    manifest_path = run_root / "search_manifest.yaml"
    completed_runs: list[dict[str, Any]] = []
    manifest = {
        "search_id": search_id,
        "search_config": search_config,
        "config": str(Path(args.config)),
        "default_config": str(Path(args.default_config)),
        "dataset_config": str(Path(args.dataset_config)),
        "feature_root": args.feature_root,
        "canonical_run_root": args.canonical_run_root,
        "device": str(device),
        "batch_size": batch_size,
        "epochs": epochs,
        "planned_run_count": len(planned_runs),
        "completed_runs": completed_runs,
    }
    _write_manifest(manifest_path, manifest)

    for index, run_spec in enumerate(planned_runs, start=1):
        print(
            f"[{index}/{len(planned_runs)}] "
            f"{'+'.join(run_spec['backbones'])} / {run_spec['fusion_method']} / "
            f"{run_spec['candidate_name']}"
        )
        seed_everything(seed)
        started = time.perf_counter()
        if run_spec["fusion_method"] == "none":
            run_dir = _run_single(
                run_spec,
                args,
                run_root,
                dataset_config,
                training_config,
                seed,
                device,
                batch_size,
                epochs,
            )
        else:
            run_dir = _run_fusion(
                run_spec,
                args,
                run_root,
                dataset_config,
                training_config,
                seed,
                device,
                batch_size,
                epochs,
            )
        runtime_seconds = time.perf_counter() - started
        _annotate_run(
            run_dir,
            search_stage=str(search_config.get("stage", "screen")),
            candidate_name=run_spec["candidate_name"],
            candidate_profile=run_spec["candidate_profile"],
            planned_index=index,
            runtime_seconds=runtime_seconds,
        )
        completed_runs.append(
            {
                "planned_index": index,
                "run_dir": str(run_dir),
                "runtime_seconds": runtime_seconds,
                **run_spec,
            }
        )
        _write_manifest(manifest_path, manifest)

    exported = export_sprint4c_hparam_search_report_assets(
        run_root,
        tables_dir,
        figures_dir,
        canonical_run_root=args.canonical_run_root,
        feature_source=str(search_config.get("feature_source", "finetuned")),
        experiment_name=str(search_config["name"]),
    )
    (run_root / "exported_assets.json").write_text(
        json.dumps({name: str(path) for name, path in exported.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote search manifest: {manifest_path}")
    print("Exported report assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


def expand_search_runs(search_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand Sprint 4C config into concrete run specs."""

    non_weighted = list(search_config["candidate_groups"]["non_weighted"])
    weighted = list(search_config["candidate_groups"]["weighted"])
    projection_dims = list(search_config.get("projection_dims", [512]))
    runs: list[dict[str, Any]] = []
    for combination in search_config["combinations"]:
        method = str(combination["fusion_method"])
        backbones = list(combination["backbones"])
        if method in {"none", "concat"}:
            for candidate in non_weighted:
                runs.append(_candidate_run_spec(backbones, method, candidate))
        elif method == "weighted":
            for candidate in weighted:
                candidate_projection_dims = candidate.get("projection_dims", projection_dims)
                for projection_dim in candidate_projection_dims:
                    runs.append(
                        _candidate_run_spec(
                            backbones,
                            method,
                            candidate,
                            projection_dim=int(projection_dim),
                        )
                    )
        else:
            raise ValueError(f"Unsupported Sprint 4C fusion method: {method}")
    return runs


def _limit_planned_runs(runs: list[dict[str, Any]], max_runs: int) -> list[dict[str, Any]]:
    if max_runs >= len(runs):
        return runs
    if max_runs == 3:
        selected: list[dict[str, Any]] = []
        for method in ("none", "concat", "weighted"):
            selected.extend(run for run in runs if run["fusion_method"] == method)
            if selected and selected[-1]["fusion_method"] == method:
                selected = selected[: len({run["fusion_method"] for run in selected})]
        if len(selected) == 3:
            return selected
    return runs[:max_runs]


def _candidate_run_spec(
    backbones: list[str],
    fusion_method: str,
    candidate: dict[str, Any],
    *,
    projection_dim: int | None = None,
) -> dict[str, Any]:
    base_name = str(candidate["name"])
    candidate_name = base_name if projection_dim is None else f"{base_name}_p{projection_dim}"
    return {
        "backbones": backbones,
        "fusion_method": fusion_method,
        "candidate_name": candidate_name,
        "candidate_profile": base_name,
        "projection_dim": projection_dim,
        "class_weighting": bool(candidate["class_weighting"]),
        "optimizer": str(candidate["optimizer"]),
        "learning_rate": float(candidate["learning_rate"]),
        "weight_decay": float(candidate["weight_decay"]),
        "momentum": float(candidate.get("momentum", 0.9)),
        "dropout": float(candidate["dropout"]),
        "hidden_dims": list(candidate["hidden_dims"]),
    }


def _run_single(
    run_spec: dict[str, Any],
    args: argparse.Namespace,
    run_root: Path,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    device: Any,
    batch_size: int,
    epochs: int,
) -> Path:
    run_args = SimpleNamespace(
        feature_root=args.feature_root,
        run_root=str(run_root),
        fusion="none",
    )
    search_config = load_yaml(args.config)["mlp_fusion_hparam_search"]
    return run_single_backbone(
        backbone=run_spec["backbones"][0],
        args=run_args,
        dataset_config=dataset_config,
        seed=seed,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=run_spec["learning_rate"],
        weight_decay=run_spec["weight_decay"],
        optimizer_name=run_spec["optimizer"],
        momentum=run_spec["momentum"],
        dropout=run_spec["dropout"],
        hidden_dims=run_spec["hidden_dims"],
        class_weighting=run_spec["class_weighting"],
        training_config=training_config,
        experiment_name=str(search_config["name"]),
        feature_source=str(search_config.get("feature_source", "finetuned")),
        run_tag=str(run_spec["candidate_name"]),
    )


def _run_fusion(
    run_spec: dict[str, Any],
    args: argparse.Namespace,
    run_root: Path,
    dataset_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    device: Any,
    batch_size: int,
    epochs: int,
) -> Path:
    search_config = load_yaml(args.config)["mlp_fusion_hparam_search"]
    run_args = SimpleNamespace(
        feature_root=args.feature_root,
        run_root=str(run_root),
    )
    projection_dim = int(
        run_spec["projection_dim"] or search_config.get("default_projection_dim", 512)
    )
    return run_fusion_experiment(
        run_spec={
            "feature_source": str(search_config.get("feature_source", "finetuned")),
            "backbones": run_spec["backbones"],
            "fusion_method": run_spec["fusion_method"],
        },
        args=run_args,
        dataset_config=dataset_config,
        seed=seed,
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=run_spec["learning_rate"],
        weight_decay=run_spec["weight_decay"],
        optimizer_name=run_spec["optimizer"],
        momentum=run_spec["momentum"],
        dropout=run_spec["dropout"],
        hidden_dims=run_spec["hidden_dims"],
        projection_dim=projection_dim,
        class_weighting=run_spec["class_weighting"],
        training_config=training_config,
        experiment_name=str(search_config["name"]),
        feature_source=str(search_config.get("feature_source", "finetuned")),
        run_tag=str(run_spec["candidate_name"]),
    )


def _annotate_run(
    run_dir: Path,
    *,
    search_stage: str,
    candidate_name: str,
    candidate_profile: str,
    planned_index: int,
    runtime_seconds: float,
) -> None:
    config_path = run_dir / "config_resolved.yaml"
    metrics_path = run_dir / "metrics.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    extra = {
        "search_stage": search_stage,
        "candidate_name": candidate_name,
        "candidate_profile": candidate_profile,
        "planned_index": planned_index,
        "runtime_seconds": runtime_seconds,
    }
    config.update(extra)
    metrics.update(extra)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
