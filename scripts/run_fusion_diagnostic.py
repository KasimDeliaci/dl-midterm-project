"""Run Sprint 4E cached-feature fusion diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.metrics import per_class_frame
from dl_midterm.evaluation.plots import save_confusion_matrix_plot, save_training_curve_plot
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
from dl_midterm.models.fusion import (
    PerClassWeightedFusionMLP,
    WeightedFusionMLP,
    expected_concat_dim,
)
from dl_midterm.models.mlp import FeatureMLP
from dl_midterm.training.loops import evaluate_model, train_mlp_model
from dl_midterm.training.optim import build_optimizer
from dl_midterm.utils.device import resolve_device
from dl_midterm.utils.seed import seed_everything


@dataclass(frozen=True)
class NormalizationStats:
    """Train-fitted normalization statistics for one backbone."""

    mean: torch.Tensor
    std: torch.Tensor


class FocalLoss(nn.Module):
    """Small multi-class focal loss for cached-feature diagnostics."""

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/sprint4e_fusion_diagnostic.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument(
        "--test-only-from-selection",
        action="store_true",
        help=(
            "Reuse existing Sprint 4E validation/selection tables and run folders "
            "for test export."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)["sprint4e"]
    default_config = load_yaml(args.default_config)
    dataset_config = load_yaml(args.dataset_config)["dataset"]

    seed = int(config.get("seed", dataset_config.get("seed", 42)))
    seed_everything(seed)
    runtime_config = default_config.get("runtime", {})
    training_config = default_config.get("training", {})
    device = resolve_device(args.device or runtime_config.get("device", "auto"))
    batch_size = args.batch_size or int(training_config.get("batch_size", 32))
    epochs = args.epochs or int(training_config.get("epochs", 25))
    feature_root = args.feature_root or config.get("feature_root", "artifacts/features")
    run_root = Path(args.run_root or config["run_root"])
    tables_dir = Path(args.tables_dir or config["tables_dir"])
    figures_dir = Path(args.figures_dir or config["figures_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    candidates = _select_candidates(config, args.candidates, args.max_candidates)
    backbones = list(config["backbones"])
    caches_by_split = _load_caches(
        feature_root=feature_root,
        dataset_name=dataset_config["name"],
        feature_source=str(config["feature_source"]),
        backbones=backbones,
        splits_dir=Path(dataset_config["splits_dir"]),
    )
    if args.test_only_from_selection:
        _run_test_only_from_selection(
            candidates=candidates,
            config=config,
            dataset_config=dataset_config,
            caches_by_split=caches_by_split,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
            class_names=list(dataset_config["class_names"]),
            device=device,
            batch_size=batch_size,
            args=args,
            run_root=run_root,
        )
        return

    feature_scale = _build_feature_scale_summary(caches_by_split)
    feature_scale.to_csv(tables_dir / "sprint4e_feature_scale_summary.csv", index=False)
    _plot_feature_norms(feature_scale, figures_dir / "sprint4e_feature_norms_by_backbone.png")

    results: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    run_dirs: dict[str, Path] = {}
    for index, candidate in enumerate(candidates):
        print(f"Running Sprint 4E candidate {index + 1}/{len(candidates)}: {candidate['name']}")
        seed_everything(seed)
        result = _run_candidate(
            candidate=candidate,
            config=config,
            dataset_config=dataset_config,
            caches_by_split=caches_by_split,
            run_root=run_root,
            class_names=list(dataset_config["class_names"]),
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            training_config=training_config,
        )
        results.append(result["summary"])
        per_class_rows.extend(result["per_class_rows"])
        weight_rows.extend(result["weight_rows"])
        run_dirs[str(candidate["name"])] = result["run_dir"]

    validation = pd.DataFrame(results)
    validation = _annotate_validation_deltas(validation, config)
    validation.to_csv(tables_dir / "sprint4e_validation_results.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(
        tables_dir / "sprint4e_validation_per_class_f1.csv",
        index=False,
    )
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(tables_dir / "sprint4e_fusion_weight_summary.csv", index=False)
    _write_concat_weighted_audit(validation, weights, tables_dir, figures_dir)
    selected = _select_for_test(validation, config)
    selected.to_csv(tables_dir / "sprint4e_selection_log.csv", index=False)

    test_rows: list[dict[str, Any]] = []
    test_per_class_rows: list[dict[str, Any]] = []
    if not args.skip_test and not selected.empty:
        for row in selected.itertuples(index=False):
            candidate_name = str(row.candidate_name)
            run_dir = run_dirs[candidate_name]
            candidate = next(c for c in candidates if str(c["name"]) == candidate_name)
            test_result = _evaluate_candidate_test(
                candidate=candidate,
                run_dir=run_dir,
                config=config,
                dataset_config=dataset_config,
                caches_by_split=caches_by_split,
                class_names=list(dataset_config["class_names"]),
                device=device,
                batch_size=batch_size,
            )
            test_rows.append(test_result["summary"])
            test_per_class_rows.extend(test_result["per_class_rows"])
            print(
                f"Test {candidate_name}: macro-F1={test_result['summary']['macro_f1']:.4f}, "
                f"accuracy={test_result['summary']['accuracy']:.4f}"
            )
    test = pd.DataFrame(test_rows)
    test.to_csv(tables_dir / "sprint4e_test_results.csv", index=False)
    pd.DataFrame(test_per_class_rows).to_csv(
        tables_dir / "sprint4e_test_per_class_f1.csv",
        index=False,
    )
    _write_final_comparisons(
        validation,
        test,
        per_class_rows,
        test_per_class_rows,
        config,
        tables_dir,
    )
    _write_figures(validation, test, weights, per_class_rows, test_per_class_rows, figures_dir)
    _write_manifest(
        config=config,
        args=args,
        run_root=run_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        selected=selected,
    )
    print(f"Wrote Sprint 4E tables: {tables_dir}")
    print(f"Wrote Sprint 4E figures: {figures_dir}")


def _select_candidates(
    config: dict[str, Any],
    requested: list[str] | None,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    candidates = list(config["candidates"])
    if requested is not None:
        requested_set = set(requested)
        known = {str(candidate["name"]) for candidate in candidates}
        missing = sorted(requested_set - known)
        if missing:
            raise ValueError(f"Unknown Sprint 4E candidates: {missing}")
        candidates = [
            candidate for candidate in candidates if str(candidate["name"]) in requested_set
        ]
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    if not candidates:
        raise ValueError("No Sprint 4E candidates selected.")
    return candidates


def _run_test_only_from_selection(
    *,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    caches_by_split: dict[str, list[FeatureCache]],
    tables_dir: Path,
    figures_dir: Path,
    class_names: list[str],
    device: torch.device,
    batch_size: int,
    args: argparse.Namespace,
    run_root: Path,
) -> None:
    validation_path = tables_dir / "sprint4e_validation_results.csv"
    selection_path = tables_dir / "sprint4e_selection_log.csv"
    validation_per_class_path = tables_dir / "sprint4e_validation_per_class_f1.csv"
    weights_path = tables_dir / "sprint4e_fusion_weight_summary.csv"
    missing = [
        path
        for path in (validation_path, selection_path, validation_per_class_path, weights_path)
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Cannot run test-only Sprint 4E export; missing: {missing_text}")

    validation = pd.read_csv(validation_path)
    selected = pd.read_csv(selection_path)
    val_per_class = pd.read_csv(validation_per_class_path).to_dict("records")
    weights = pd.read_csv(weights_path)
    candidate_by_name = {str(candidate["name"]): candidate for candidate in candidates}

    test_rows: list[dict[str, Any]] = []
    test_per_class_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        candidate_name = str(row.candidate_name)
        if candidate_name not in candidate_by_name:
            raise ValueError(f"Selection references unknown Sprint 4E candidate: {candidate_name}")
        run_dir = Path(str(row.run_dir))
        test_result = _evaluate_candidate_test(
            candidate=candidate_by_name[candidate_name],
            run_dir=run_dir,
            config=config,
            dataset_config=dataset_config,
            caches_by_split=caches_by_split,
            class_names=class_names,
            device=device,
            batch_size=batch_size,
        )
        test_rows.append(test_result["summary"])
        test_per_class_rows.extend(test_result["per_class_rows"])
        print(
            f"Test {candidate_name}: macro-F1={test_result['summary']['macro_f1']:.4f}, "
            f"accuracy={test_result['summary']['accuracy']:.4f}"
        )

    test = pd.DataFrame(test_rows)
    test.to_csv(tables_dir / "sprint4e_test_results.csv", index=False)
    pd.DataFrame(test_per_class_rows).to_csv(
        tables_dir / "sprint4e_test_per_class_f1.csv",
        index=False,
    )
    _write_final_comparisons(
        validation,
        test,
        val_per_class,
        test_per_class_rows,
        config,
        tables_dir,
    )
    _write_figures(validation, test, weights, val_per_class, test_per_class_rows, figures_dir)
    _write_manifest(
        config=config,
        args=args,
        run_root=run_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        selected=selected,
    )
    print(f"Wrote Sprint 4E test-only tables: {tables_dir}")
    print(f"Wrote Sprint 4E test-only figures: {figures_dir}")


def _load_caches(
    *,
    feature_root: str | Path,
    dataset_name: str,
    feature_source: str,
    backbones: list[str],
    splits_dir: Path,
) -> dict[str, list[FeatureCache]]:
    caches_by_split: dict[str, list[FeatureCache]] = {}
    for split in ("train", "val", "test"):
        split_caches = []
        for backbone in backbones:
            cache_dir = backbone_cache_dir(feature_root, dataset_name, feature_source, backbone)
            cache = load_feature_cache(feature_cache_path(cache_dir, split))
            verify_cache_matches_split(
                cache,
                splits_dir / f"{split}.csv",
                allow_prefix=cache_allows_prefix_split_verification(cache),
            )
            split_caches.append(cache)
        _verify_cache_alignment(split_caches)
        caches_by_split[split] = split_caches
    return caches_by_split


def _verify_cache_alignment(caches: list[FeatureCache]) -> None:
    reference = caches[0]
    for cache in caches[1:]:
        if cache.split != reference.split:
            raise ValueError("Cannot fuse caches from different splits.")
        if cache.image_ids != reference.image_ids:
            raise ValueError("Fusion cache image_id order does not align.")
        if cache.label_names != reference.label_names:
            raise ValueError("Fusion cache labels do not align.")
        if not torch.equal(cache.labels.cpu(), reference.labels.cpu()):
            raise ValueError("Fusion cache labels do not align.")


def _build_feature_scale_summary(caches_by_split: dict[str, list[FeatureCache]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, caches in caches_by_split.items():
        for cache in caches:
            features = cache.features.float()
            norms = torch.linalg.vector_norm(features, dim=1)
            rows.append(
                {
                    "split": split,
                    "backbone": cache.backbone,
                    "feature_dim": cache.feature_dim,
                    "samples": int(features.shape[0]),
                    "mean_feature_norm": float(norms.mean()),
                    "std_feature_norm": float(norms.std(unbiased=False)),
                    "mean_abs_activation": float(features.abs().mean()),
                    "mean_dimension_std": float(features.std(dim=0, unbiased=False).mean()),
                    "max_abs_activation": float(features.abs().max()),
                }
            )
    return pd.DataFrame(rows)


def _fit_normalization(
    train_caches: list[FeatureCache],
    normalization: str,
) -> dict[str, NormalizationStats]:
    if normalization not in {"none", "standardize_per_backbone", "l2_per_backbone"}:
        raise ValueError(f"Unsupported feature normalization: {normalization}")
    stats: dict[str, NormalizationStats] = {}
    if normalization != "standardize_per_backbone":
        return stats
    for cache in train_caches:
        features = cache.features.float()
        stats[cache.backbone] = NormalizationStats(
            mean=features.mean(dim=0),
            std=features.std(dim=0, unbiased=False).clamp_min(1e-6),
        )
    return stats


def _apply_normalization(
    features: torch.Tensor,
    *,
    backbone: str,
    normalization: str,
    stats: dict[str, NormalizationStats],
) -> torch.Tensor:
    normalized = features.float()
    if normalization == "standardize_per_backbone":
        fitted = stats[backbone]
        normalized = (normalized - fitted.mean) / fitted.std
    elif normalization == "l2_per_backbone":
        normalized = torch.nn.functional.normalize(normalized, p=2, dim=1, eps=1e-12)
    elif normalization != "none":
        raise ValueError(f"Unsupported feature normalization: {normalization}")
    return normalized


def _build_datasets(
    caches_by_split: dict[str, list[FeatureCache]],
    normalization: str,
) -> dict[str, TensorDataset]:
    stats = _fit_normalization(caches_by_split["train"], normalization)
    datasets: dict[str, TensorDataset] = {}
    for split, caches in caches_by_split.items():
        _verify_cache_alignment(caches)
        parts = [
            _apply_normalization(
                cache.features,
                backbone=cache.backbone,
                normalization=normalization,
                stats=stats,
            )
            for cache in caches
        ]
        datasets[split] = TensorDataset(torch.cat(parts, dim=1), caches[0].labels.long())
    return datasets


def _run_candidate(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    caches_by_split: dict[str, list[FeatureCache]],
    run_root: Path,
    class_names: list[str],
    device: torch.device,
    batch_size: int,
    epochs: int,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    backbones = list(config["backbones"])
    normalization = str(candidate.get("normalization", "none"))
    datasets = _build_datasets(caches_by_split, normalization)
    seed = int(config.get("seed", dataset_config.get("seed", 42)))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    model = _build_model(candidate, backbones, class_names)
    criterion = _build_criterion(
        candidate,
        train_cache=caches_by_split["train"][0],
        class_names=class_names,
        device=device,
    )
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=str(candidate.get("optimizer", "adamw")),
        learning_rate=float(candidate.get("learning_rate", 1e-3)),
        weight_decay=float(candidate.get("weight_decay", 1e-4)),
        momentum=float(candidate.get("momentum", 0.9)),
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
    run_dir = _write_candidate_run(
        candidate=candidate,
        config=config,
        dataset_config=dataset_config,
        run_root=run_root,
        model=model,
        history=history,
        metrics=val_metrics,
        class_names=class_names,
        split="val",
        batch_size=batch_size,
        epochs=epochs,
        backbones=backbones,
    )
    weight_rows = _extract_weight_rows(model, backbones, class_names, str(candidate["name"]))
    summary = _summary_from_metrics(candidate, config, val_metrics, "val", run_dir, backbones)
    return {
        "summary": summary,
        "per_class_rows": _per_class_rows(candidate, val_metrics, "val"),
        "weight_rows": weight_rows,
        "run_dir": run_dir,
    }


def _build_model(
    candidate: dict[str, Any],
    backbones: list[str],
    class_names: list[str],
) -> nn.Module:
    input_dims = [expected_feature_dim(backbone) for backbone in backbones]
    fusion_method = str(candidate["fusion_method"])
    hidden_dims = list(candidate.get("hidden_dims", [512, 256]))
    dropout = float(candidate.get("dropout", 0.3))
    projection_dim = int(candidate.get("projection_dim", 512))
    if fusion_method == "concat":
        return FeatureMLP(
            input_dim=expected_concat_dim(backbones),
            num_classes=len(class_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if fusion_method == "weighted":
        return WeightedFusionMLP(
            input_dims=input_dims,
            num_classes=len(class_names),
            projection_dim=projection_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if fusion_method == "per_class_weighted":
        return PerClassWeightedFusionMLP(
            input_dims=input_dims,
            num_classes=len(class_names),
            projection_dim=projection_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported fusion method: {fusion_method}")


def _build_criterion(
    candidate: dict[str, Any],
    *,
    train_cache: FeatureCache,
    class_names: list[str],
    device: torch.device,
) -> nn.Module:
    class_weighting = bool(candidate.get("class_weighting", True))
    weights = (
        class_weights_from_cache(train_cache, len(class_names)).to(device)
        if class_weighting
        else None
    )
    loss_name = str(candidate.get("loss", "weighted_ce"))
    if loss_name in {"weighted_ce", "ce"}:
        return nn.CrossEntropyLoss(weight=weights)
    if loss_name == "label_smoothing":
        return nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=float(candidate.get("label_smoothing", 0.05)),
        )
    if loss_name == "focal":
        return FocalLoss(
            gamma=float(candidate.get("focal_gamma", 1.0)),
            class_weights=weights,
        )
    raise ValueError(f"Unsupported Sprint 4E loss: {loss_name}")


def _write_candidate_run(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    run_root: Path,
    model: nn.Module,
    history: pd.DataFrame,
    metrics: dict[str, Any],
    class_names: list[str],
    split: str,
    batch_size: int,
    epochs: int,
    backbones: list[str],
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    alias = "-".join(backbone_alias(backbone) for backbone in backbones)
    run_id = f"{timestamp}_{config['feature_source']}_{alias}_{candidate['name']}_s{config['seed']}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "run_id": run_id,
        "experiment_name": config["name"],
        "seed": int(config["seed"]),
        "dataset": dataset_config["name"],
        "feature_source": config["feature_source"],
        "candidate_name": candidate["name"],
        "family": candidate.get("family"),
        "backbones": backbones,
        "backbone_combination": "+".join(backbones),
        "fusion_method": candidate["fusion_method"],
        "normalization": candidate.get("normalization", "none"),
        "loss": candidate.get("loss", "weighted_ce"),
        "projection_dim": candidate.get("projection_dim"),
        "class_weighting": bool(candidate.get("class_weighting", True)),
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": candidate.get("optimizer", "adamw"),
        "learning_rate": float(candidate.get("learning_rate", 1e-3)),
        "weight_decay": float(candidate.get("weight_decay", 1e-4)),
        "hidden_dims": list(candidate.get("hidden_dims", [512, 256])),
        "dropout": float(candidate.get("dropout", 0.3)),
        "best_val_epoch": metrics.get("best_epoch"),
        "best_val_macro_f1": metrics.get("macro_f1"),
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / f"{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    history.to_csv(run_dir / "history.csv", index=False)
    per_class_frame(metrics, backbone=str(candidate["name"])).to_csv(
        run_dir / f"{split}_classification_report.csv",
        index=False,
    )
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        run_dir / f"{split}_confusion_matrix.png",
        title=f"Sprint 4E {candidate['name']} {split} confusion matrix",
    )
    save_training_curve_plot(
        history,
        run_dir / "training_curve.png",
        title=f"Sprint 4E {candidate['name']} training curves",
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    weight_rows = _extract_weight_rows(model, backbones, class_names, str(candidate["name"]))
    _write_fusion_weights(run_dir, weight_rows)
    return run_dir


def _summary_from_metrics(
    candidate: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    split: str,
    run_dir: Path,
    backbones: list[str],
) -> dict[str, Any]:
    return {
        "candidate_name": candidate["name"],
        "family": candidate.get("family"),
        "split": split,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "feature_source": config["feature_source"],
        "backbone_combination": "+".join(backbones),
        "fusion_method": candidate["fusion_method"],
        "normalization": candidate.get("normalization", "none"),
        "loss": candidate.get("loss", "weighted_ce"),
        "projection_dim": candidate.get("projection_dim"),
        "learning_rate": float(candidate.get("learning_rate", 1e-3)),
        "weight_decay": float(candidate.get("weight_decay", 1e-4)),
        "dropout": float(candidate.get("dropout", 0.3)),
        "hidden_dims": json.dumps(list(candidate.get("hidden_dims", [512, 256]))),
        "class_weighting": bool(candidate.get("class_weighting", True)),
        "best_epoch": metrics.get("best_epoch"),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }


def _per_class_rows(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in metrics["per_class"]:
        rows.append(
            {
                "candidate_name": candidate["name"],
                "family": candidate.get("family"),
                "fusion_method": candidate["fusion_method"],
                "normalization": candidate.get("normalization", "none"),
                "loss": candidate.get("loss", "weighted_ce"),
                "split": split,
                **row,
            }
        )
    return rows


def _extract_weight_rows(
    model: nn.Module,
    backbones: list[str],
    class_names: list[str],
    candidate_name: str,
) -> list[dict[str, Any]]:
    if not hasattr(model, "normalized_weights"):
        return []
    with torch.no_grad():
        weights = model.normalized_weights().detach().cpu()
    rows: list[dict[str, Any]] = []
    if weights.ndim == 1:
        for backbone, weight in zip(backbones, weights.tolist(), strict=True):
            rows.append(
                {
                    "candidate_name": candidate_name,
                    "weight_type": "global",
                    "class_name": "global",
                    "backbone": backbone,
                    "weight": float(weight),
                }
            )
    elif weights.ndim == 2:
        for backbone_index, backbone in enumerate(backbones):
            for class_index, class_name in enumerate(class_names):
                rows.append(
                    {
                        "candidate_name": candidate_name,
                        "weight_type": "per_class",
                        "class_name": class_name,
                        "backbone": backbone,
                        "weight": float(weights[backbone_index, class_index]),
                    }
                )
    return rows


def _write_fusion_weights(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "fusion_weights.csv", index=False)


def _annotate_validation_deltas(
    validation: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame = validation.copy()
    baseline_by_method = {}
    for method, name in {
        "concat": "concat_none_base",
        "weighted": "weighted_none_p512_low_lr",
        "per_class_weighted": "weighted_none_p512_low_lr",
    }.items():
        rows = frame[frame["candidate_name"] == name]
        baseline_by_method[method] = float(rows.iloc[0]["macro_f1"]) if not rows.empty else None
    frame["matched_baseline_macro_f1"] = frame["fusion_method"].map(baseline_by_method)
    frame["matched_baseline_macro_f1"] = frame["matched_baseline_macro_f1"].astype(float)
    frame["val_macro_f1_gain"] = frame["macro_f1"] - frame["matched_baseline_macro_f1"]
    frame["canonical_sprint4_test_macro_f1"] = float(
        config.get("baselines", {}).get("canonical_sprint4_concat_macro_f1", 0.7059)
    )
    return frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def _select_for_test(validation: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    gate = dict(config["gate"])
    frame = validation.copy()
    baseline_names = {
        "concat": "concat_none_base",
        "weighted": "weighted_none_p512_low_lr",
        "per_class_weighted": "weighted_none_p512_low_lr",
    }
    baseline_metrics: dict[str, dict[str, float]] = {}
    for method, name in baseline_names.items():
        rows = frame[frame["candidate_name"] == name]
        if rows.empty:
            baseline_metrics[method] = {"accuracy": 0.0, "weighted_f1": 0.0}
            continue
        baseline_metrics[method] = {
            "accuracy": float(rows.iloc[0]["accuracy"]),
            "weighted_f1": float(rows.iloc[0]["weighted_f1"]),
        }
    frame["accuracy_gain"] = frame.apply(
        lambda row: float(row["accuracy"])
        - baseline_metrics[str(row["fusion_method"])]["accuracy"],
        axis=1,
    )
    frame["weighted_f1_gain"] = frame.apply(
        lambda row: float(row["weighted_f1"])
        - baseline_metrics[str(row["fusion_method"])]["weighted_f1"],
        axis=1,
    )
    frame["gate_pass"] = (
        (frame["val_macro_f1_gain"] >= float(gate["min_val_macro_f1_gain"]))
        & (frame["accuracy_gain"] >= -float(gate["max_accuracy_drop"]))
        & (frame["weighted_f1_gain"] >= -float(gate["max_weighted_f1_drop"]))
    )
    selected = frame[frame["gate_pass"]].sort_values("macro_f1", ascending=False)
    return selected.head(int(config.get("test_top_k", 2))).reset_index(drop=True)


def _evaluate_candidate_test(
    *,
    candidate: dict[str, Any],
    run_dir: Path,
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    caches_by_split: dict[str, list[FeatureCache]],
    class_names: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    backbones = list(config["backbones"])
    datasets = _build_datasets(caches_by_split, str(candidate.get("normalization", "none")))
    loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)
    model = _build_model(candidate, backbones, class_names)
    state_dict = torch.load(run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    metrics = evaluate_model(model, loader, class_names=class_names, device=device)
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_class_frame(metrics, backbone=str(candidate["name"])).to_csv(
        run_dir / "test_classification_report.csv",
        index=False,
    )
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        run_dir / "test_confusion_matrix.png",
        title=f"Sprint 4E {candidate['name']} test confusion matrix",
    )
    summary = _summary_from_metrics(candidate, config, metrics, "test", run_dir, backbones)
    return {"summary": summary, "per_class_rows": _per_class_rows(candidate, metrics, "test")}


def _write_concat_weighted_audit(
    validation: pd.DataFrame,
    weights: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    audit = (
        validation.groupby(["fusion_method", "normalization"], as_index=False)
        .agg(
            best_val_macro_f1=("macro_f1", "max"),
            mean_val_macro_f1=("macro_f1", "mean"),
            candidate_count=("candidate_name", "count"),
        )
        .sort_values("best_val_macro_f1", ascending=False)
    )
    audit.to_csv(tables_dir / "sprint4e_concat_weighted_audit.csv", index=False)
    if not weights.empty:
        weights.to_csv(tables_dir / "sprint4e_learned_weights_audit.csv", index=False)
    _plot_concat_weighted_gap(audit, figures_dir / "sprint4e_concat_vs_weighted_gap.png")


def _write_final_comparisons(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    val_per_class: list[dict[str, Any]],
    test_per_class: list[dict[str, Any]],
    config: dict[str, Any],
    tables_dir: Path,
) -> None:
    baselines = dict(config.get("baselines", {}))
    rows = []
    for name, value in baselines.items():
        rows.append({"source": name, "split": "test", "macro_f1": float(value)})
    for row in test.to_dict("records"):
        rows.append(
            {
                "source": row["candidate_name"],
                "split": "test",
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
                "weighted_f1": row["weighted_f1"],
            }
        )
    pd.DataFrame(rows).to_csv(tables_dir / "sprint4e_final_comparison.csv", index=False)
    _write_per_class_gap_summary(val_per_class, test_per_class, tables_dir)


def _write_per_class_gap_summary(
    val_per_class: list[dict[str, Any]],
    test_per_class: list[dict[str, Any]],
    tables_dir: Path,
) -> None:
    frame = pd.DataFrame([*val_per_class, *test_per_class])
    if frame.empty:
        frame.to_csv(tables_dir / "sprint4e_per_class_gap_summary.csv", index=False)
        return
    baseline = frame[frame["candidate_name"] == "weighted_none_p512_low_lr"][
        ["split", "label", "f1"]
    ].rename(columns={"f1": "weighted_baseline_f1"})
    merged = frame.merge(baseline, on=["split", "label"], how="left")
    merged["f1_gain_vs_weighted_baseline"] = merged["f1"] - merged["weighted_baseline_f1"]
    merged.to_csv(tables_dir / "sprint4e_per_class_gap_summary.csv", index=False)


def _write_figures(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    weights: pd.DataFrame,
    val_per_class: list[dict[str, Any]],
    test_per_class: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    _plot_validation_candidates(
        validation,
        figures_dir / "sprint4e_validation_macro_f1_by_candidate.png",
    )
    _plot_weights(weights, figures_dir / "sprint4e_learned_weights_audit.png")
    _plot_accuracy_macro_scatter(validation, figures_dir / "sprint4e_accuracy_macro_f1_scatter.png")
    per_class = pd.DataFrame([*val_per_class, *test_per_class])
    _plot_per_class_gain(per_class, figures_dir / "sprint4e_per_class_f1_gain_heatmap.png")
    _copy_best_training_curves(validation, test, figures_dir)
    _copy_best_confusion_matrix(test, figures_dir)


def _plot_feature_norms(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=frame, x="backbone", y="mean_feature_norm", hue="split")
    plt.ylabel("Mean Feature L2 Norm")
    plt.xlabel("Backbone")
    plt.title("Sprint 4E Feature Norms by Backbone")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_validation_candidates(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values("macro_f1", ascending=False)
    plt.figure(figsize=(12, max(5, 0.4 * len(ordered))))
    sns.barplot(data=ordered, y="candidate_name", x="macro_f1", hue="fusion_method", dodge=False)
    plt.xlim(0, 1)
    plt.xlabel("Validation Macro-F1")
    plt.ylabel("Candidate")
    plt.title("Sprint 4E Validation Macro-F1 by Candidate")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_concat_weighted_gap(audit: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=audit.sort_values("best_val_macro_f1", ascending=False),
        x="normalization",
        y="best_val_macro_f1",
        hue="fusion_method",
    )
    plt.ylim(0, 1)
    plt.xlabel("Normalization")
    plt.ylabel("Best Validation Macro-F1")
    plt.title("Sprint 4E Concat vs Weighted Diagnostic")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_weights(weights: pd.DataFrame, path: Path) -> None:
    if weights.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    global_weights = weights[weights["weight_type"] == "global"]
    if not global_weights.empty:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=global_weights, x="candidate_name", y="weight", hue="backbone")
        plt.xticks(rotation=35, ha="right")
        plt.ylim(0, 1)
        plt.title("Sprint 4E Learned Global Fusion Weights")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return
    first = weights["candidate_name"].iloc[0]
    pivot = weights[weights["candidate_name"] == first].pivot(
        index="backbone",
        columns="class_name",
        values="weight",
    )
    plt.figure(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
    plt.title(f"Sprint 4E Per-Class Fusion Weights: {first}")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_accuracy_macro_scatter(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=frame,
        x="accuracy",
        y="macro_f1",
        hue="fusion_method",
        style="normalization",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Sprint 4E Accuracy vs Macro-F1")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_per_class_gain(per_class: pd.DataFrame, path: Path) -> None:
    if per_class.empty or "weighted_none_p512_low_lr" not in set(per_class["candidate_name"]):
        return
    baseline = per_class[per_class["candidate_name"] == "weighted_none_p512_low_lr"][
        ["split", "label", "f1"]
    ].rename(columns={"f1": "baseline_f1"})
    merged = per_class.merge(baseline, on=["split", "label"], how="left")
    merged["f1_gain"] = merged["f1"] - merged["baseline_f1"]
    top_candidates = (
        merged.groupby("candidate_name")["f1_gain"].mean().sort_values(ascending=False).head(8).index
    )
    pivot = merged[merged["candidate_name"].isin(top_candidates)].pivot_table(
        index="candidate_name",
        columns="label",
        values="f1_gain",
        aggfunc="mean",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(4, 0.45 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Sprint 4E Per-Class F1 Gain vs Weighted Baseline")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _copy_best_training_curves(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    figures_dir: Path,
) -> None:
    curves_dir = figures_dir / "training_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    source = test if not test.empty else validation.head(3)
    for row in source.head(3).itertuples(index=False):
        src = Path(str(row.run_dir)) / "training_curve.png"
        if src.exists():
            dst = curves_dir / f"{row.candidate_name}_training_curve.png"
            dst.write_bytes(src.read_bytes())


def _copy_best_confusion_matrix(test: pd.DataFrame, figures_dir: Path) -> None:
    if test.empty:
        return
    row = test.sort_values("macro_f1", ascending=False).iloc[0]
    src = Path(str(row["run_dir"])) / "test_confusion_matrix.png"
    if src.exists():
        dst = figures_dir / "sprint4e_best_confusion_matrix.png"
        dst.write_bytes(src.read_bytes())


def _write_manifest(
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    run_root: Path,
    tables_dir: Path,
    figures_dir: Path,
    selected: pd.DataFrame,
) -> None:
    manifest = {
        "name": config["name"],
        "seed": config["seed"],
        "run_root": str(run_root),
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "config": args.config,
        "selected_for_test": selected.to_dict("records"),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "sprint4e_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
