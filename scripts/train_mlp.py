"""Train single-backbone MLP classifiers on cached frozen features."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.reports import (
    export_single_backbone_report_assets,
    export_sprint4b_screening_report_assets,
    write_run_report,
)
from dl_midterm.features.cache import (
    FeatureDataset,
    backbone_cache_dir,
    cache_allows_prefix_split_verification,
    class_weights_from_cache,
    feature_cache_path,
    load_feature_cache,
    sample_weights_from_cache,
    verify_cache_matches_split,
)
from dl_midterm.features.extract import select_single_backbone_combinations
from dl_midterm.models.backbones import backbone_alias
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
    parser.add_argument("--fusion", default="none", choices=["none"])
    parser.add_argument("--backbones", nargs="+", default=None)
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
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--train-sampling",
        choices=["shuffle", "class_balanced"],
        default="shuffle",
        help="Train-set sampling strategy for cached-feature MLP training.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-tag", default=None)
    return parser.parse_args()


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
    class_weighting = not args.no_class_weights
    experiment_name = args.experiment_name or f"single_backbone_{feature_source}_baseline"

    backbones = args.backbones
    if backbones is None:
        backbones = select_single_backbone_combinations(experiment_config["combinations"])
    for backbone in backbones:
        run_single_backbone(
            backbone=backbone,
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
            class_weighting=class_weighting,
            train_sampling=args.train_sampling,
            training_config=training_config,
            experiment_name=experiment_name,
            feature_source=feature_source,
            run_tag=args.run_tag,
        )

    exported = export_single_backbone_report_assets(
        args.run_root,
        args.tables_dir,
        args.figures_dir,
        feature_source=feature_source,
    )
    if feature_source in {"finetuned_classaware", "finetuned_deeper"}:
        try:
            exported.update(
                export_sprint4b_screening_report_assets(
                    args.run_root,
                    args.tables_dir,
                    args.figures_dir,
                )
            )
        except FileNotFoundError as exc:
            print(f"Skipping Sprint 4B screening export: {exc}")
    print("Exported report assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


def run_single_backbone(
    *,
    backbone: str,
    args: argparse.Namespace,
    dataset_config: dict,
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
    class_weighting: bool,
    train_sampling: str,
    training_config: dict,
    experiment_name: str,
    feature_source: str,
    run_tag: str | None = None,
) -> Path:
    class_names = list(dataset_config["class_names"])
    cache_dir = backbone_cache_dir(
        args.feature_root,
        dataset_config["name"],
        feature_source,
        backbone,
    )
    caches = {
        split: load_feature_cache(feature_cache_path(cache_dir, split))
        for split in ("train", "val", "test")
    }
    for split, cache in caches.items():
        verify_cache_matches_split(
            cache,
            Path(dataset_config["splits_dir"]) / f"{split}.csv",
            allow_prefix=cache_allows_prefix_split_verification(cache),
        )

    train_loader = _build_train_loader(
        FeatureDataset(caches["train"]),
        cache=caches["train"],
        num_classes=len(class_names),
        batch_size=batch_size,
        seed=seed,
        train_sampling=train_sampling,
    )
    val_loader = DataLoader(FeatureDataset(caches["val"]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FeatureDataset(caches["test"]), batch_size=batch_size, shuffle=False)

    feature_dim = caches["train"].feature_dim
    model = FeatureMLP(
        input_dim=feature_dim,
        num_classes=len(class_names),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    weights = (
        class_weights_from_cache(caches["train"], len(class_names)).to(device)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{run_tag}" if run_tag else ""
    run_id = f"{timestamp}_{feature_source}_{backbone_alias(backbone)}_none_mlp{tag}_s{seed}"
    run_dir = Path(args.run_root) / run_id
    resolved_config = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "run_tag": run_tag,
        "seed": seed,
        "dataset": dataset_config["name"],
        "feature_source": feature_source,
        "backbone": backbone,
        "backbones": [backbone],
        "backbone_combination": backbone,
        "backbone_count": 1,
        "fusion_method": args.fusion,
        "feature_dim": feature_dim,
        "class_weighting": class_weighting,
        "train_sampling": train_sampling,
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
        "feature_cache_dir": str(cache_dir),
        "split_files": {
            split: str(Path(dataset_config["splits_dir"]) / f"{split}.csv")
            for split in ("train", "val", "test")
        },
    }
    test_metrics.update(
        {
            "run_id": run_id,
            "seed": seed,
            "feature_source": feature_source,
            "backbone": backbone,
            "backbones": [backbone],
            "backbone_combination": backbone,
            "backbone_count": 1,
            "fusion_method": args.fusion,
            "feature_dim": feature_dim,
            "class_weighting": class_weighting,
            "train_sampling": train_sampling,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "hidden_dims": hidden_dims,
            "experiment_name": experiment_name,
            "run_tag": run_tag,
            "best_val_macro_f1": val_metrics.get("macro_f1"),
            "best_val_epoch": val_metrics.get("best_epoch"),
        }
    )
    write_run_report(
        run_dir,
        resolved_config=resolved_config,
        history=history,
        metrics=test_metrics,
        class_names=class_names,
        backbone=backbone,
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Wrote MLP run: {run_dir}")
    return run_dir


def _build_train_loader(
    dataset: FeatureDataset,
    *,
    cache,
    num_classes: int,
    batch_size: int,
    seed: int,
    train_sampling: str,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    if train_sampling == "class_balanced":
        sampler = WeightedRandomSampler(
            weights=sample_weights_from_cache(cache, num_classes),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    if train_sampling == "shuffle":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    raise ValueError(f"Unsupported train sampling strategy: {train_sampling}")


if __name__ == "__main__":
    main()
