"""Run frozen cached-feature MLP fusion experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.reports import (
    export_frozen_matrix_report_assets,
    export_frozen_vs_finetuned_report_assets,
    export_sprint4b_full_classaware_report_assets,
    write_run_report,
)
from dl_midterm.features.cache import (
    FeatureCache,
    backbone_cache_dir,
    cache_allows_prefix_split_verification,
    class_weights_from_cache,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)
from dl_midterm.models.backbones import backbone_alias, expected_feature_dim
from dl_midterm.models.fusion import WeightedFusionMLP, expected_concat_dim
from dl_midterm.models.mlp import FeatureMLP
from dl_midterm.training.loops import evaluate_model, train_mlp_model
from dl_midterm.training.optim import build_optimizer
from dl_midterm.utils.device import resolve_device
from dl_midterm.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/frozen_feature_matrix.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-source", default=None)
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--optimizer", default=None, choices=["adamw", "adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=None)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--fusion-methods", nargs="+", choices=["concat", "weighted"], default=None)
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def expand_fusion_run_matrix(
    experiment_config: dict[str, Any],
    *,
    feature_source: str = "frozen",
    fusion_methods: list[str] | None = None,
    backbones: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand the configured matrix to Sprint 3 fusion runs only."""

    requested_methods = set(fusion_methods) if fusion_methods else None
    requested_backbones = set(backbones) if backbones else None
    runs: list[dict[str, Any]] = []
    for combination in experiment_config["combinations"]:
        names = list(combination.get("backbones", []))
        if len(names) < 2:
            continue
        if requested_backbones and not set(names).issubset(requested_backbones):
            continue
        for method in combination.get("fusion_methods", []):
            if method == "none":
                continue
            if requested_methods and method not in requested_methods:
                continue
            runs.append(
                {
                    "feature_source": feature_source,
                    "backbones": names,
                    "fusion_method": method,
                    "fusion_input_dim": expected_concat_dim(names),
                }
            )
    return runs


def main() -> None:
    args = parse_args()
    experiment_config = load_yaml(args.config)["experiment_matrix"]
    default_config = load_yaml(args.default_config)
    dataset_config = load_yaml(args.dataset_config)["dataset"]

    seed = int(experiment_config.get("seed", dataset_config.get("seed", 42)))
    seed_everything(seed)

    runtime_config = default_config.get("runtime", {})
    training_config = default_config.get("training", {})
    mlp_config = default_config.get("mlp", {})
    feature_source = str(args.feature_source or experiment_config.get("feature_source", "frozen"))
    device = resolve_device(args.device or runtime_config.get("device", "auto"))
    batch_size = args.batch_size or int(training_config.get("batch_size", 32))
    epochs = args.epochs or int(training_config.get("epochs", 25))
    learning_rate = args.learning_rate or float(training_config.get("learning_rate", 1e-3))
    weight_decay = args.weight_decay or float(training_config.get("weight_decay", 1e-4))
    optimizer_name = args.optimizer or str(training_config.get("optimizer", "adamw"))
    dropout = args.dropout if args.dropout is not None else float(mlp_config.get("dropout", 0.3))
    hidden_dims = args.hidden_dims or list(mlp_config.get("hidden_dims", [512, 256]))
    projection_dim = args.projection_dim or int(mlp_config.get("projection_dim", 512))
    class_weighting = not args.no_class_weights
    experiment_name = args.experiment_name or f"{feature_source}_fusion_matrix"

    run_specs = expand_fusion_run_matrix(
        experiment_config,
        feature_source=feature_source,
        fusion_methods=args.fusion_methods,
        backbones=args.backbones,
    )
    if args.max_runs is not None:
        run_specs = run_specs[: args.max_runs]
    if not run_specs:
        raise ValueError("No fusion runs selected.")

    completed_runs: list[dict[str, Any]] = []
    for run_spec in run_specs:
        print(
            "Running fusion experiment: "
            f"{'+'.join(run_spec['backbones'])} / {run_spec['fusion_method']}"
        )
        run_dir = run_fusion_experiment(
            run_spec=run_spec,
            args=args,
            dataset_config=dataset_config,
            seed=seed,
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            momentum=float(args.momentum),
            dropout=dropout,
            hidden_dims=hidden_dims,
            projection_dim=projection_dim,
            class_weighting=class_weighting,
            training_config=training_config,
            experiment_name=experiment_name,
            feature_source=feature_source,
        )
        completed_runs.append({**run_spec, "run_dir": str(run_dir)})

    manifest_path = Path(args.run_root) / f"{experiment_name}_manifest.yaml"
    manifest = {
        "experiment_name": experiment_name,
        "feature_source": feature_source,
        "seed": seed,
        "config": str(Path(args.config)),
        "default_config": str(Path(args.default_config)),
        "dataset_config": str(Path(args.dataset_config)),
        "completed_runs": completed_runs,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    exported = export_frozen_matrix_report_assets(
        args.run_root,
        args.tables_dir,
        args.figures_dir,
        feature_source=feature_source,
        feature_root=args.feature_root,
        dataset_config=dataset_config,
    )
    if feature_source == "finetuned":
        try:
            exported.update(
                export_frozen_vs_finetuned_report_assets(
                    args.run_root,
                    args.tables_dir,
                    args.figures_dir,
                )
            )
        except FileNotFoundError as exc:
            print(f"Skipping frozen-vs-finetuned export: {exc}")
            print(
                "Fine-tuned matrix assets were still exported. Run the comparison where "
                "Sprint 3 frozen run folders are available."
            )
    if feature_source == "finetuned_classaware":
        try:
            exported.update(
                export_sprint4b_full_classaware_report_assets(
                    args.run_root,
                    args.tables_dir,
                    args.figures_dir,
                )
            )
        except FileNotFoundError as exc:
            print(f"Skipping Sprint 4B full-matrix comparison export: {exc}")
    (Path(args.run_root) / f"{experiment_name}_exported_assets.json").write_text(
        json.dumps({name: str(path) for name, path in exported.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote experiment manifest: {manifest_path}")
    print("Exported report assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


def run_fusion_experiment(
    *,
    run_spec: dict[str, Any],
    args: argparse.Namespace,
    dataset_config: dict[str, Any],
    seed: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    momentum: float,
    dropout: float,
    hidden_dims: list[int],
    projection_dim: int,
    class_weighting: bool,
    training_config: dict[str, Any],
    experiment_name: str,
    feature_source: str,
) -> Path:
    class_names = list(dataset_config["class_names"])
    backbones = list(run_spec["backbones"])
    fusion_method = str(run_spec["fusion_method"])
    caches_by_split = _load_aligned_caches(
        feature_root=args.feature_root,
        dataset_name=dataset_config["name"],
        feature_source=feature_source,
        backbones=backbones,
        splits_dir=Path(dataset_config["splits_dir"]),
    )
    input_dims = [expected_feature_dim(backbone) for backbone in backbones]
    fusion_input_dim = expected_concat_dim(backbones)
    fusion_output_dim = fusion_input_dim if fusion_method == "concat" else projection_dim

    datasets = {
        split: _build_concat_tensor_dataset(caches) for split, caches in caches_by_split.items()
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    if fusion_method == "concat":
        model: nn.Module = FeatureMLP(
            input_dim=fusion_input_dim,
            num_classes=len(class_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif fusion_method == "weighted":
        model = WeightedFusionMLP(
            input_dims=input_dims,
            num_classes=len(class_names),
            projection_dim=projection_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    weights = (
        class_weights_from_cache(caches_by_split["train"][0], len(class_names)).to(device)
        if class_weighting
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    model, history, val_metrics = train_mlp_model(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        device=device,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        early_stopping_patience=int(training_config.get("early_stopping_patience", 5)),
    )
    test_metrics = evaluate_model(model, test_loader, class_names=class_names, device=device)
    fusion_weights = _extract_fusion_weights(model, backbones)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alias = "-".join(backbone_alias(backbone) for backbone in backbones)
    run_id = f"{timestamp}_{feature_source}_{alias}_{fusion_method}_mlp_s{seed}"
    run_dir = Path(args.run_root) / run_id
    backbone_combination = "+".join(backbones)
    resolved_config = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "seed": seed,
        "dataset": dataset_config["name"],
        "feature_source": feature_source,
        "backbones": backbones,
        "backbone_combination": backbone_combination,
        "backbone_count": len(backbones),
        "fusion_method": fusion_method,
        "feature_dim": fusion_output_dim,
        "fusion_input_dim": fusion_input_dim,
        "fusion_output_dim": fusion_output_dim,
        "input_dims": dict(zip(backbones, input_dims, strict=True)),
        "projection_dim": projection_dim if fusion_method == "weighted" else None,
        "class_weighting": class_weighting,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum if optimizer_name.lower() == "sgd" else None,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "best_val_epoch": val_metrics.get("best_epoch"),
        "best_val_macro_f1": val_metrics.get("macro_f1"),
        "feature_cache_dirs": {
            backbone: str(
                backbone_cache_dir(
                    args.feature_root,
                    dataset_config["name"],
                    feature_source,
                    backbone,
                )
            )
            for backbone in backbones
        },
        "split_files": {
            split: str(Path(dataset_config["splits_dir"]) / f"{split}.csv")
            for split in ("train", "val", "test")
        },
        "learned_fusion_weights": fusion_weights,
    }
    test_metrics.update(
        {
            "run_id": run_id,
            "seed": seed,
            "feature_source": feature_source,
            "backbones": backbones,
            "backbone_combination": backbone_combination,
            "backbone_count": len(backbones),
            "fusion_method": fusion_method,
            "feature_dim": fusion_output_dim,
            "fusion_input_dim": fusion_input_dim,
            "fusion_output_dim": fusion_output_dim,
            "projection_dim": projection_dim if fusion_method == "weighted" else None,
            "class_weighting": class_weighting,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "hidden_dims": hidden_dims,
            "experiment_name": experiment_name,
            "best_val_macro_f1": val_metrics.get("macro_f1"),
            "best_val_epoch": val_metrics.get("best_epoch"),
            "learned_fusion_weights": fusion_weights,
        }
    )
    write_run_report(
        run_dir,
        resolved_config=resolved_config,
        history=history,
        metrics=test_metrics,
        class_names=class_names,
        backbone=backbone_combination,
    )
    if fusion_weights:
        _write_fusion_weights(run_dir, fusion_weights)
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Wrote fusion run: {run_dir}")
    return run_dir


def _load_aligned_caches(
    *,
    feature_root: str | Path,
    dataset_name: str,
    feature_source: str,
    backbones: list[str],
    splits_dir: Path,
) -> dict[str, list[FeatureCache]]:
    caches_by_split: dict[str, list[FeatureCache]] = {}
    for split in ("train", "val", "test"):
        split_caches: list[FeatureCache] = []
        for backbone in backbones:
            cache_dir = backbone_cache_dir(feature_root, dataset_name, feature_source, backbone)
            cache = load_feature_cache(feature_cache_path(cache_dir, split))
            verify_cache_matches_split(
                cache,
                splits_dir / f"{split}.csv",
                allow_prefix=cache_allows_prefix_split_verification(cache),
            )
            split_caches.append(cache)
        verify_cache_alignment(split_caches)
        caches_by_split[split] = split_caches
    return caches_by_split


def verify_cache_alignment(caches: list[FeatureCache]) -> None:
    """Ensure all backbone caches describe the same split rows in the same order."""

    if len(caches) < 2:
        raise ValueError("Fusion cache alignment requires at least two caches.")
    reference = caches[0]
    for cache in caches[1:]:
        if cache.split != reference.split:
            raise ValueError("Cannot fuse caches from different splits.")
        if cache.image_ids != reference.image_ids:
            raise ValueError("Fusion cache image_id order does not align.")
        if cache.label_names != reference.label_names:
            raise ValueError("Fusion cache labels do not align.")
        if not torch.equal(cache.labels.cpu(), reference.labels.cpu()):
            raise ValueError("Fusion cache label indices do not align.")


def _build_concat_tensor_dataset(caches: list[FeatureCache]) -> TensorDataset:
    verify_cache_alignment(caches)
    features = torch.cat([cache.features.float() for cache in caches], dim=1)
    labels = caches[0].labels.long()
    return TensorDataset(features, labels)


def _extract_fusion_weights(model: nn.Module, backbones: list[str]) -> dict[str, float] | None:
    if not hasattr(model, "normalized_weights"):
        return None
    with torch.no_grad():
        weights = model.normalized_weights().detach().cpu().tolist()
    return {backbone: float(weight) for backbone, weight in zip(backbones, weights, strict=True)}


def _write_fusion_weights(run_dir: Path, weights: dict[str, float] | None) -> None:
    if not weights:
        return
    frame = pd.DataFrame(
        [{"backbone": backbone, "weight": weight} for backbone, weight in weights.items()]
    )
    frame.to_csv(run_dir / "fusion_weights.csv", index=False)
    (run_dir / "fusion_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
