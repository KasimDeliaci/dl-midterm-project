"""Run Sprint 4G validation-gated cached-feature ensemble search."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.metrics import compute_classification_metrics, per_class_frame
from dl_midterm.features.cache import (
    FeatureCache,
    FeatureDataset,
    backbone_cache_dir,
    cache_allows_prefix_split_verification,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)
from dl_midterm.models.backbones import expected_feature_dim
from dl_midterm.models.fusion import (
    PerClassWeightedFusionMLP,
    WeightedFusionMLP,
    expected_concat_dim,
)
from dl_midterm.models.mlp import FeatureMLP
from dl_midterm.utils.device import resolve_device
from dl_midterm.utils.seed import seed_everything


@dataclass(frozen=True)
class Candidate:
    key: str
    run_dir: Path
    config: dict[str, Any]


@dataclass(frozen=True)
class PredictionBundle:
    candidate: Candidate
    labels: torch.Tensor
    logits: torch.Tensor
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/sprint4g_autoresearch_ensemble.yaml",
    )
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--feature-root", default=None)
    parser.add_argument("--tables-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)["sprint4g"]
    dataset_config = load_yaml(args.dataset_config)["dataset"]
    seed = int(config.get("seed", dataset_config.get("seed", 42)))
    seed_everything(seed)
    device = resolve_device(args.device)
    run_root = Path(args.run_root or config["run_root"])
    feature_root = Path(args.feature_root or config["feature_root"])
    tables_dir = Path(args.tables_dir or config["tables_dir"])
    figures_dir = Path(args.figures_dir or config["figures_dir"])
    full_results_dir = Path(
        config.get("full_results_dir", "artifacts/runs/sprint4g_autoresearch_ensemble")
    )
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    full_results_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_candidates(run_root, config)
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise FileNotFoundError("No eligible Sprint 4G ensemble candidates with model.pt found.")

    class_names = list(dataset_config["class_names"])
    val_predictions = [
        predict_candidate(
            candidate,
            split="val",
            dataset_config=dataset_config,
            feature_root=feature_root,
            class_names=class_names,
            device=device,
            batch_size=int(args.batch_size),
        )
        for candidate in candidates
    ]
    val_predictions = sorted(
        val_predictions,
        key=lambda pred: float(pred.metrics["macro_f1"]),
        reverse=True,
    )
    candidate_limit = int(config["candidate_pool"]["top_n_by_validation"])
    search_pool = val_predictions[:candidate_limit]

    individual_rows = [candidate_summary(pred, "val") for pred in val_predictions]
    pd.DataFrame(individual_rows).to_csv(
        tables_dir / "sprint4g_individual_validation_results.csv",
        index=False,
    )

    ensemble_rows, ensemble_specs = search_ensembles(
        search_pool,
        class_names=class_names,
        config=config,
        seed=seed,
    )
    ensemble_frame = pd.DataFrame(ensemble_rows).sort_values("macro_f1", ascending=False)
    ensemble_frame.to_csv(
        full_results_dir / "sprint4g_ensemble_validation_results_full.csv",
        index=False,
    )
    report_rows = int(config.get("report", {}).get("max_ensemble_rows", 100))
    ensemble_frame.head(report_rows).to_csv(
        tables_dir / "sprint4g_ensemble_validation_results.csv",
        index=False,
    )
    if ensemble_frame.empty:
        raise ValueError("Sprint 4G ensemble search produced no candidates.")

    selected = ensemble_frame.head(int(config["selection"].get("test_top_k", 1))).reset_index(
        drop=True
    )
    selected.to_csv(tables_dir / "sprint4g_selection_log.csv", index=False)

    test_rows: list[dict[str, Any]] = []
    test_per_class_rows: list[dict[str, Any]] = []
    selected_specs = {spec["ensemble_id"]: spec for spec in ensemble_specs}
    if not args.skip_test:
        for row in selected.itertuples(index=False):
            spec = selected_specs[str(row.ensemble_id)]
            test_result = evaluate_ensemble_on_split(
                spec,
                split="test",
                dataset_config=dataset_config,
                feature_root=feature_root,
                class_names=class_names,
                device=device,
                batch_size=int(args.batch_size),
            )
            test_rows.append(test_result["summary"])
            test_per_class_rows.extend(test_result["per_class_rows"])
    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(tables_dir / "sprint4g_test_results.csv", index=False)
    pd.DataFrame(test_per_class_rows).to_csv(
        tables_dir / "sprint4g_test_per_class_f1.csv",
        index=False,
    )

    write_final_comparison(config, selected, test_frame, tables_dir)
    write_figures(ensemble_frame, selected, test_frame, selected_specs, figures_dir, class_names)
    print(f"Wrote Sprint 4G tables: {tables_dir}")
    print(f"Wrote Sprint 4G figures: {figures_dir}")


def discover_candidates(run_root: Path, config: dict[str, Any]) -> list[Candidate]:
    eligible_sources = set(config["eligible_feature_sources"])
    eligible_methods = set(config["eligible_fusion_methods"])
    min_val_macro_f1 = float(config["candidate_pool"].get("min_val_macro_f1", 0.0))
    candidates: list[Candidate] = []
    for config_path in sorted(run_root.glob("**/config_resolved.yaml")):
        run_dir = config_path.parent
        if not (run_dir / "model.pt").exists():
            continue
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(resolved, dict):
            continue
        if str(resolved.get("feature_source")) not in eligible_sources:
            continue
        if str(resolved.get("fusion_method")) not in eligible_methods:
            continue
        if str(resolved.get("normalization", "none")) != "none":
            continue
        if float(resolved.get("best_val_macro_f1") or 0.0) < min_val_macro_f1:
            continue
        run_id = str(resolved.get("run_id") or run_dir.name)
        candidates.append(Candidate(key=run_id, run_dir=run_dir, config=resolved))
    return sorted(
        candidates,
        key=lambda candidate: float(candidate.config.get("best_val_macro_f1") or 0.0),
        reverse=True,
    )


def predict_candidate(
    candidate: Candidate,
    *,
    split: str,
    dataset_config: dict[str, Any],
    feature_root: Path,
    class_names: list[str],
    device: torch.device,
    batch_size: int,
) -> PredictionBundle:
    loader = build_candidate_loader(candidate, split, dataset_config, feature_root, batch_size)
    model = build_candidate_model(candidate, class_names)
    state_dict = torch.load(candidate.run_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    labels, logits = predict_logits(model, loader, device)
    metrics = metrics_from_logits(labels, logits, class_names)
    return PredictionBundle(candidate=candidate, labels=labels, logits=logits, metrics=metrics)


def build_candidate_model(candidate: Candidate, class_names: list[str]) -> nn.Module:
    resolved = candidate.config
    backbones = list(resolved.get("backbones") or [resolved["backbone"]])
    fusion_method = str(resolved["fusion_method"])
    hidden_dims = list(resolved.get("hidden_dims") or [512, 256])
    dropout = float(resolved.get("dropout", 0.3))
    num_classes = len(class_names)
    if fusion_method == "none":
        return FeatureMLP(
            input_dim=expected_feature_dim(backbones[0]),
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if fusion_method == "concat":
        return FeatureMLP(
            input_dim=expected_concat_dim(backbones),
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    projection_dim = int(resolved.get("projection_dim") or resolved.get("feature_dim") or 512)
    input_dims = [expected_feature_dim(backbone) for backbone in backbones]
    if fusion_method == "weighted":
        return WeightedFusionMLP(
            input_dims=input_dims,
            num_classes=num_classes,
            projection_dim=projection_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if fusion_method == "per_class_weighted":
        return PerClassWeightedFusionMLP(
            input_dims=input_dims,
            num_classes=num_classes,
            projection_dim=projection_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported Sprint 4G candidate fusion method: {fusion_method}")


def build_candidate_loader(
    candidate: Candidate,
    split: str,
    dataset_config: dict[str, Any],
    feature_root: Path,
    batch_size: int,
) -> DataLoader:
    backbones = list(candidate.config.get("backbones") or [candidate.config["backbone"]])
    feature_source = str(candidate.config["feature_source"])
    caches = []
    for backbone in backbones:
        cache_dir = backbone_cache_dir(
            feature_root,
            dataset_config["name"],
            feature_source,
            backbone,
        )
        cache = load_feature_cache(feature_cache_path(cache_dir, split))
        verify_cache_matches_split(
            cache,
            Path(dataset_config["splits_dir"]) / f"{split}.csv",
            allow_prefix=cache_allows_prefix_split_verification(cache),
        )
        caches.append(cache)
    if len(caches) == 1:
        dataset = FeatureDataset(caches[0])
    else:
        dataset = build_concat_dataset(caches)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def build_concat_dataset(caches: list[FeatureCache]) -> TensorDataset:
    first = caches[0]
    for cache in caches[1:]:
        if cache.image_ids != first.image_ids:
            raise ValueError("Feature cache image_id alignment mismatch.")
        if not torch.equal(cache.labels, first.labels):
            raise ValueError("Feature cache label alignment mismatch.")
    features = torch.cat([cache.features.float() for cache in caches], dim=1)
    return TensorDataset(features, first.labels)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    labels: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    for features, batch_labels in loader:
        logits.append(model(features.to(device)).detach().cpu())
        labels.append(batch_labels.detach().cpu())
    return torch.cat(labels), torch.cat(logits)


def metrics_from_logits(
    labels: torch.Tensor,
    logits: torch.Tensor,
    class_names: list[str],
) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    return compute_classification_metrics(labels.tolist(), predictions.tolist(), class_names)


def search_ensembles(
    predictions: list[PredictionBundle],
    *,
    class_names: list[str],
    config: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    search_config = config["ensemble_search"]
    min_size = int(search_config.get("min_size", 2))
    max_size = min(int(search_config.get("max_size", 5)), len(predictions))
    torch.manual_seed(seed)
    rows: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    for size in range(min_size, max_size + 1):
        for combo in itertools.combinations(predictions, size):
            if bool(search_config.get("include_uniform", True)):
                weights = torch.ones(size) / size
                add_ensemble_row(combo, weights, "uniform", class_names, rows, specs)
            if bool(search_config.get("include_rank_weighted", True)):
                scores = torch.tensor([float(item.metrics["macro_f1"]) for item in combo])
                weights = scores / scores.sum()
                add_ensemble_row(combo, weights, "rank_weighted", class_names, rows, specs)
    random_top_n = min(int(search_config.get("random_weight_top_n", 0)), len(predictions))
    samples = int(search_config.get("random_weight_samples", 0))
    if random_top_n >= min_size and samples > 0:
        random_pool = predictions[:random_top_n]
        for size in range(min_size, min(max_size, random_top_n) + 1):
            for combo in itertools.combinations(random_pool, size):
                for sample_index in range(samples):
                    weights = torch.distributions.Dirichlet(torch.ones(size)).sample()
                    add_ensemble_row(
                        combo,
                        weights,
                        f"dirichlet_{sample_index:03d}",
                        class_names,
                        rows,
                        specs,
                    )
    return rows, specs


def add_ensemble_row(
    combo: tuple[PredictionBundle, ...],
    weights: torch.Tensor,
    weight_policy: str,
    class_names: list[str],
    rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
) -> None:
    labels = combo[0].labels
    for item in combo[1:]:
        if not torch.equal(item.labels, labels):
            raise ValueError("Candidate validation label alignment mismatch.")
    probabilities = [torch.softmax(item.logits, dim=1) for item in combo]
    averaged = sum(weight * prob for weight, prob in zip(weights, probabilities, strict=True))
    metrics = compute_classification_metrics(
        labels.tolist(),
        averaged.argmax(dim=1).tolist(),
        class_names,
    )
    keys = [item.candidate.key for item in combo]
    ensemble_id = f"{weight_policy}__" + "__".join(keys)
    row = {
        "ensemble_id": ensemble_id,
        "weight_policy": weight_policy,
        "candidate_count": len(combo),
        "members": "|".join(keys),
        "weights": json.dumps([float(value) for value in weights.tolist()]),
        **metric_summary(metrics),
    }
    rows.append(row)
    specs.append(
        {
            "ensemble_id": ensemble_id,
            "members": [item.candidate for item in combo],
            "weights": weights,
        }
    )


def evaluate_ensemble_on_split(
    spec: dict[str, Any],
    *,
    split: str,
    dataset_config: dict[str, Any],
    feature_root: Path,
    class_names: list[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    predictions = [
        predict_candidate(
            candidate,
            split=split,
            dataset_config=dataset_config,
            feature_root=feature_root,
            class_names=class_names,
            device=device,
            batch_size=batch_size,
        )
        for candidate in spec["members"]
    ]
    labels = predictions[0].labels
    probs = [torch.softmax(item.logits, dim=1) for item in predictions]
    averaged = sum(
        float(weight) * prob for weight, prob in zip(spec["weights"], probs, strict=True)
    )
    metrics = compute_classification_metrics(
        labels.tolist(),
        averaged.argmax(dim=1).tolist(),
        class_names,
    )
    summary = {
        "ensemble_id": spec["ensemble_id"],
        "split": split,
        "members": "|".join(candidate.key for candidate in spec["members"]),
        "weights": json.dumps([float(value) for value in spec["weights"].tolist()]),
        **metric_summary(metrics),
    }
    per_class = per_class_frame(metrics, backbone=spec["ensemble_id"])
    per_class["split"] = split
    return {"summary": summary, "per_class_rows": per_class.to_dict("records")}


def candidate_summary(prediction: PredictionBundle, split: str) -> dict[str, Any]:
    resolved = prediction.candidate.config
    return {
        "candidate_key": prediction.candidate.key,
        "split": split,
        "run_dir": str(prediction.candidate.run_dir),
        "feature_source": resolved.get("feature_source"),
        "backbone_combination": resolved.get("backbone_combination"),
        "fusion_method": resolved.get("fusion_method"),
        "configured_best_val_macro_f1": resolved.get("best_val_macro_f1"),
        **metric_summary(prediction.metrics),
    }


def metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
    }


def write_final_comparison(
    config: dict[str, Any],
    selected: pd.DataFrame,
    test: pd.DataFrame,
    tables_dir: Path,
) -> None:
    rows = [
        {"source": name, "split": "test", "macro_f1": value}
        for name, value in config.get("baselines", {}).items()
    ]
    rows.extend(test.to_dict("records"))
    rows.extend({**row, "split": "val_selected"} for row in selected.to_dict("records"))
    pd.DataFrame(rows).to_csv(tables_dir / "sprint4g_final_comparison.csv", index=False)


def write_figures(
    validation: pd.DataFrame,
    selected: pd.DataFrame,
    test: pd.DataFrame,
    specs: dict[str, dict[str, Any]],
    figures_dir: Path,
    class_names: list[str],
) -> None:
    top = validation.head(20).copy()
    top["plot_label"] = [
        f"E{index + 1:02d} {row.weight_policy} k={int(row.candidate_count)}"
        for index, row in enumerate(top.itertuples(index=False))
    ]
    plt.figure(figsize=(10, max(5, 0.35 * len(top))))
    plt.barh(top["plot_label"], top["macro_f1"])
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.xlabel("Validation Macro-F1")
    plt.title("Sprint 4G Top Validation Ensembles")
    plt.tight_layout()
    plt.savefig(figures_dir / "sprint4g_validation_ensemble_macro_f1.png", dpi=200)
    plt.close()

    if not test.empty:
        compare = test[["ensemble_id", "macro_f1"]].copy()
        compare["plot_label"] = [f"Selected E{index + 1:02d}" for index in range(len(compare))]
        plt.figure(figsize=(8, 4))
        plt.bar(compare["plot_label"], compare["macro_f1"])
        plt.ylim(0, 1)
        plt.ylabel("Test Macro-F1")
        plt.title("Sprint 4G Selected Ensemble Test Macro-F1")
        plt.tight_layout()
        plt.savefig(figures_dir / "sprint4g_test_macro_f1.png", dpi=200)
        plt.close()

    if not selected.empty:
        ensemble_id = str(selected.iloc[0]["ensemble_id"])
        spec = specs[ensemble_id]
        weights = pd.DataFrame(
            {
                "member": [short_candidate_label(candidate.key) for candidate in spec["members"]],
                "weight": [float(value) for value in spec["weights"].tolist()],
            }
        )
        weights.to_csv(figures_dir / "sprint4g_selected_weights.csv", index=False)
        plt.figure(figsize=(10, max(4, 0.35 * len(weights))))
        plt.barh(weights["member"], weights["weight"])
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.xlabel("Soft-vote weight")
        plt.title("Sprint 4G Selected Ensemble Weights")
        plt.tight_layout()
        plt.savefig(figures_dir / "sprint4g_selected_weights.png", dpi=200)
        plt.close()


def short_candidate_label(value: str, max_length: int = 56) -> str:
    label = value
    replacements = {
        "finetuned_classaware": "classaware",
        "finetuned_augmented": "augmented",
        "finetuned": "ft",
        "resnet50": "r50",
        "mobilenet_v2": "mnv2",
        "efficientnet_b0": "effb0",
        "_mlp": "",
        "_s42": "",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    parts = label.split("_", maxsplit=2)
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        label = parts[2]
    if len(label) <= max_length:
        return label
    return f"{label[: max_length - 3]}..."


if __name__ == "__main__":
    main()
