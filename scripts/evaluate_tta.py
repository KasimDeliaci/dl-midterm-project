"""Evaluate validation-gated test-time augmentation experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dl_midterm.config.load_config import load_yaml
from dl_midterm.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from dl_midterm.evaluation.metrics import compute_classification_metrics
from dl_midterm.evaluation.plots import save_confusion_matrix_plot
from dl_midterm.evaluation.tta import (
    average_probabilities,
    choose_tta_policy,
    expand_tta_policy,
)
from dl_midterm.models.backbones import (
    build_finetuned_feature_extractor,
    expected_feature_dim,
)
from dl_midterm.models.fusion import WeightedFusionMLP, expected_concat_dim
from dl_midterm.models.mlp import FeatureMLP
from dl_midterm.utils.device import resolve_device
from dl_midterm.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/sprint4d_tta.yaml")
    parser.add_argument(
        "--experiment-key",
        default=None,
        help=(
            "Top-level YAML key to read. Defaults to 'sprint4d' when present, "
            "otherwise the only top-level key."
        ),
    )
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--stage", choices=["validation", "test", "all"], default="validation")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--policies", nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_experiment_config(args.config, args.experiment_key)
    config = _with_smoke_output_dirs(config, args.max_samples)
    dataset_config = load_yaml(args.dataset_config)["dataset"]
    seed = int(config.get("seed", dataset_config.get("seed", 42)))
    seed_everything(seed)
    device = resolve_device(args.device)
    _validate_requested_artifacts(config, args.models)

    output_dirs = _prepare_output_dirs(config)
    if args.stage in {"validation", "all"}:
        validation_rows, per_class_rows, runtime_rows = run_validation_stage(
            config=config,
            dataset_config=dataset_config,
            args=args,
            device=device,
        )
        _write_stage_outputs(
            split="val",
            rows=validation_rows,
            per_class_rows=per_class_rows,
            runtime_rows=runtime_rows,
            config=config,
            output_dirs=output_dirs,
        )
        decisions = _write_decision_log(config, validation_rows, output_dirs["tables"])
        _write_validation_figures(config, validation_rows, decisions, output_dirs["figures"])
        _write_combined_outputs(config, output_dirs)
    else:
        decisions = _read_decision_log(config, output_dirs["tables"])

    if args.stage in {"test", "all"}:
        selected = _selected_test_pairs(decisions)
        if not selected:
            print("No model-policy pair passed the validation gate; skipping test evaluation.")
            return
        test_rows, per_class_rows, runtime_rows = run_test_stage(
            config=config,
            dataset_config=dataset_config,
            args=args,
            device=device,
            selected_pairs=selected,
        )
        _write_stage_outputs(
            split="test",
            rows=test_rows,
            per_class_rows=per_class_rows,
            runtime_rows=runtime_rows,
            config=config,
            output_dirs=output_dirs,
        )
        _write_combined_outputs(config, output_dirs)


def _load_experiment_config(path: str | Path, experiment_key: str | None) -> dict[str, Any]:
    raw = load_yaml(path)
    if experiment_key is None:
        experiment_key = "sprint4d" if "sprint4d" in raw else None
    if experiment_key is None:
        keys = list(raw)
        if len(keys) != 1:
            raise ValueError(
                "Config contains multiple top-level keys; pass --experiment-key explicitly."
            )
        experiment_key = str(keys[0])
    try:
        config = dict(raw[experiment_key])
    except KeyError as exc:
        raise KeyError(f"Config {path!s} does not contain key {experiment_key!r}.") from exc
    config.setdefault("artifact_prefix", str(experiment_key))
    config.setdefault("display_prefix", str(experiment_key).upper())
    return config


def run_validation_stage(
    *,
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    model_items = _selected_model_items(config, args.models)
    requested_policies = set(args.policies) if args.policies else None
    for model_key, model_config in model_items:
        policies = list(model_config["validation_policies"])
        if requested_policies is not None:
            policies = [policy for policy in policies if policy in requested_policies]
        for policy in policies:
            result = evaluate_model_policy(
                split="val",
                policy=policy,
                model_key=model_key,
                model_config=model_config,
                config=config,
                dataset_config=dataset_config,
                device=device,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                num_workers=args.num_workers,
            )
            rows.append(result.summary)
            per_class_rows.extend(result.per_class_rows)
            runtime_rows.append(result.runtime_row)
    return rows, per_class_rows, runtime_rows


def run_test_stage(
    *,
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    selected_pairs: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    models = dict(_selected_model_items(config, args.models))
    for pair in selected_pairs:
        model_key = pair["model_key"]
        if model_key not in models:
            continue
        policies = ["identity", pair["selected_policy"]]
        for policy in dict.fromkeys(policies):
            result = evaluate_model_policy(
                split="test",
                policy=policy,
                model_key=model_key,
                model_config=models[model_key],
                config=config,
                dataset_config=dataset_config,
                device=device,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                num_workers=args.num_workers,
            )
            rows.append(result.summary)
            per_class_rows.extend(result.per_class_rows)
            runtime_rows.append(result.runtime_row)
    return rows, per_class_rows, runtime_rows


class TTASplitDataset(Dataset[dict[str, Any]]):
    """Split-backed image dataset with one deterministic TTA view."""

    def __init__(
        self,
        split_csv: str | Path,
        *,
        class_names: list[str],
        image_size: int,
        view: str,
        split: str,
        max_samples: int | None = None,
    ) -> None:
        self.frame = pd.read_csv(split_csv)
        if max_samples is not None:
            self.frame = self.frame.head(max_samples).copy()
        self.class_names = class_names
        self.label_to_index = {label: index for index, label in enumerate(class_names)}
        self.transform = TTAViewTransform(image_size=image_size, view=view)
        self.view = view
        self.split = split
        self._validate()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label_name = str(row["label"])
        return {
            "image": tensor,
            "label": torch.tensor(self.label_to_index[label_name], dtype=torch.long),
            "label_name": label_name,
            "image_id": str(row["image_id"]),
            "lesion_id": str(row.get("lesion_id", "")),
            "split": self.split,
            "row_index": int(index),
        }

    def _validate(self) -> None:
        required = {"image_id", "label", "image_path"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")
        unknown = sorted(set(self.frame["label"].astype(str)) - set(self.class_names))
        if unknown:
            raise ValueError(f"Split CSV contains unknown labels: {unknown}")


class TTAViewTransform:
    """Apply one deterministic TTA view and canonical ImageNet preprocessing."""

    def __init__(self, *, image_size: int, view: str) -> None:
        self.image_size = int(image_size)
        self.view = view
        self.post = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.view == "identity":
            transformed = image
        elif self.view == "hflip":
            transformed = ImageOps.mirror(image)
        elif self.view == "vflip":
            transformed = ImageOps.flip(image)
        elif self.view == "hvflip":
            transformed = ImageOps.flip(ImageOps.mirror(image))
        elif self.view == "rot90":
            transformed = image.transpose(Image.Transpose.ROTATE_90)
        elif self.view == "rot180":
            transformed = image.transpose(Image.Transpose.ROTATE_180)
        elif self.view == "rot270":
            transformed = image.transpose(Image.Transpose.ROTATE_270)
        elif self.view == "hflip_rot90":
            transformed = ImageOps.mirror(image).transpose(Image.Transpose.ROTATE_90)
        elif self.view == "hflip_rot180":
            transformed = ImageOps.mirror(image).transpose(Image.Transpose.ROTATE_180)
        elif self.view == "hflip_rot270":
            transformed = ImageOps.mirror(image).transpose(Image.Transpose.ROTATE_270)
        else:
            raise ValueError(f"Unsupported TTA view: {self.view}")
        return self.post(transformed)


class EvaluationResult:
    """Container for one model/policy/split evaluation."""

    def __init__(
        self,
        *,
        summary: dict[str, Any],
        per_class_rows: list[dict[str, Any]],
        runtime_row: dict[str, Any],
    ) -> None:
        self.summary = summary
        self.per_class_rows = per_class_rows
        self.runtime_row = runtime_row


def evaluate_model_policy(
    *,
    split: str,
    policy: str,
    model_key: str,
    model_config: dict[str, Any],
    config: dict[str, Any],
    dataset_config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    num_workers: int,
) -> EvaluationResult:
    views = expand_tta_policy(policy)
    class_names = list(dataset_config["class_names"])
    split_csv = Path(dataset_config["splits_dir"]) / f"{split}.csv"
    run_dir = Path(model_config["run_dir"])
    resolved_config = load_yaml(run_dir / "config_resolved.yaml")
    backbones = list(resolved_config["backbones"])
    feature_extractors = _load_feature_extractors(
        backbones=backbones,
        checkpoint_paths=model_config["checkpoint_paths"],
        class_names=class_names,
        device=device,
    )
    classifier = _load_classifier(resolved_config, run_dir / "model.pt", class_names, device)

    started = time.perf_counter()
    view_probabilities: list[torch.Tensor] = []
    reference: pd.DataFrame | None = None
    for view in views:
        probabilities, metadata = _predict_one_view(
            split_csv=split_csv,
            split=split,
            view=view,
            class_names=class_names,
            image_size=int(dataset_config["image_size"]),
            feature_extractors=feature_extractors,
            classifier=classifier,
            backbones=backbones,
            device=device,
            batch_size=batch_size,
            max_samples=max_samples,
            num_workers=num_workers,
        )
        if reference is None:
            reference = metadata
        else:
            _verify_same_split_order(reference, metadata, split)
        view_probabilities.append(probabilities)
    averaged = average_probabilities(view_probabilities)
    runtime_seconds = time.perf_counter() - started
    assert reference is not None
    y_true = reference["label_index"].to_numpy()
    y_pred = averaged.argmax(dim=1).cpu().numpy()
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    summary = {
        "model_key": model_key,
        "display_name": model_config["display_name"],
        "run_id": resolved_config["run_id"],
        "split": split,
        "policy": policy,
        "views": "+".join(views),
        "num_views": len(views),
        "test_eligible": bool(model_config.get("test_eligible", False)),
        "feature_source": resolved_config.get("feature_source"),
        "backbone_combination": resolved_config.get("backbone_combination"),
        "fusion_method": resolved_config.get("fusion_method"),
        "candidate_name": resolved_config.get("candidate_name", resolved_config.get("run_tag")),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "runtime_seconds": runtime_seconds,
        "samples_per_second": len(reference) / max(runtime_seconds, 1e-12),
    }
    per_class_rows = [
        {
            "model_key": model_key,
            "display_name": model_config["display_name"],
            "run_id": resolved_config["run_id"],
            "split": split,
            "policy": policy,
            **row,
        }
        for row in metrics["per_class"]
    ]
    runtime_row = {
        "model_key": model_key,
        "display_name": model_config["display_name"],
        "split": split,
        "policy": policy,
        "num_views": len(views),
        "runtime_seconds": runtime_seconds,
        "samples": len(reference),
        "samples_per_second": len(reference) / max(runtime_seconds, 1e-12),
    }
    _write_predictions(
        config=config,
        model_key=model_key,
        split=split,
        policy=policy,
        reference=reference,
        probabilities=averaged,
        y_pred=y_pred,
        class_names=class_names,
    )
    _write_confusion_matrix(config, split, model_key, policy, metrics, class_names)
    print(
        f"{split} {model_key} {policy}: macro-F1={metrics['macro_f1']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}, views={len(views)}"
    )
    return EvaluationResult(
        summary=summary,
        per_class_rows=per_class_rows,
        runtime_row=runtime_row,
    )


def _predict_one_view(
    *,
    split_csv: Path,
    split: str,
    view: str,
    class_names: list[str],
    image_size: int,
    feature_extractors: dict[str, nn.Module],
    classifier: nn.Module,
    backbones: list[str],
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    num_workers: int,
) -> tuple[torch.Tensor, pd.DataFrame]:
    dataset = TTASplitDataset(
        split_csv,
        class_names=class_names,
        image_size=image_size,
        view=view,
        split=split,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    all_probabilities: list[torch.Tensor] = []
    metadata_rows: list[pd.DataFrame] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            features = [
                feature_extractors[backbone](images).float()
                for backbone in backbones
            ]
            fused = features[0] if len(features) == 1 else torch.cat(features, dim=1)
            probabilities = torch.softmax(classifier(fused), dim=1)
            all_probabilities.append(probabilities.detach().cpu())
            metadata_rows.append(
                pd.DataFrame(
                    {
                        "split": list(batch["split"]),
                        "row_index": batch["row_index"].cpu().tolist(),
                        "image_id": list(batch["image_id"]),
                        "lesion_id": list(batch["lesion_id"]),
                        "true_label": list(batch["label_name"]),
                        "label_index": batch["label"].cpu().tolist(),
                    }
                )
            )
    return torch.cat(all_probabilities, dim=0), pd.concat(metadata_rows, ignore_index=True)


def _load_feature_extractors(
    *,
    backbones: list[str],
    checkpoint_paths: dict[str, str],
    class_names: list[str],
    device: torch.device,
) -> dict[str, nn.Module]:
    extractors: dict[str, nn.Module] = {}
    for backbone in backbones:
        checkpoint_path = Path(checkpoint_paths[backbone])
        extractor = build_finetuned_feature_extractor(
            backbone,
            checkpoint_path=checkpoint_path,
            num_classes=len(class_names),
        )
        extractor.to(device)
        extractor.eval()
        extractors[backbone] = extractor
    return extractors


def _load_classifier(
    resolved_config: dict[str, Any],
    model_path: Path,
    class_names: list[str],
    device: torch.device,
) -> nn.Module:
    backbones = list(resolved_config["backbones"])
    fusion_method = str(resolved_config["fusion_method"])
    hidden_dims = list(resolved_config["hidden_dims"])
    dropout = float(resolved_config["dropout"])
    if fusion_method in {"none", "concat"}:
        input_dim = (
            expected_feature_dim(backbones[0])
            if fusion_method == "none"
            else expected_concat_dim(backbones)
        )
        model: nn.Module = FeatureMLP(
            input_dim=input_dim,
            num_classes=len(class_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif fusion_method == "weighted":
        model = WeightedFusionMLP(
            input_dims=[expected_feature_dim(backbone) for backbone in backbones],
            num_classes=len(class_names),
            projection_dim=int(resolved_config["projection_dim"]),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported fusion method for TTA: {fusion_method}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"MLP model checkpoint is required for TTA evaluation but was not found: "
            f"{model_path}. Restore this gitignored run artifact before evaluating this model."
        )
    state_dict = _torch_load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _torch_load(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _verify_same_split_order(reference: pd.DataFrame, current: pd.DataFrame, split: str) -> None:
    for column in ("split", "row_index", "image_id", "true_label", "label_index"):
        if reference[column].tolist() != current[column].tolist():
            raise ValueError(f"TTA view order mismatch for split {split!r} column {column!r}.")


def _prepare_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    dirs = {
        "tables": Path(config["tables_dir"]),
        "figures": Path(config["figures_dir"]),
        "predictions": Path(config["predictions_dir"]),
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _artifact_prefix(config: dict[str, Any]) -> str:
    return str(config.get("artifact_prefix", "sprint4d"))


def _display_prefix(config: dict[str, Any]) -> str:
    return str(config.get("display_prefix", "Sprint 4D"))


def _with_smoke_output_dirs(config: dict[str, Any], max_samples: int | None) -> dict[str, Any]:
    """Route max-sample smoke outputs away from canonical report assets."""

    if max_samples is None:
        return dict(config)
    routed = dict(config)
    suffix = f"{_artifact_prefix(config)}_smoke_n{max_samples}"
    routed["tables_dir"] = str(Path(config["tables_dir"]) / suffix)
    routed["figures_dir"] = str(Path(config["figures_dir"]) / suffix)
    routed["predictions_dir"] = str(Path(config["predictions_dir"]) / suffix)
    routed["output_run_root"] = str(Path(config["output_run_root"]) / suffix)
    return routed


def _selected_model_items(
    config: dict[str, Any],
    requested: list[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    models = dict(config["models"])
    if requested is None:
        return [
            (key, model)
            for key, model in models.items()
            if bool(model.get("run_by_default", True))
        ]
    missing = sorted(set(requested) - set(models))
    if missing:
        raise ValueError(f"Unknown TTA model keys: {missing}")
    return [(key, models[key]) for key in requested]


def _validate_requested_artifacts(config: dict[str, Any], requested: list[str] | None) -> None:
    missing: list[str] = []
    for model_key, model_config in _selected_model_items(config, requested):
        run_dir = Path(model_config["run_dir"])
        for path in (run_dir / "config_resolved.yaml", run_dir / "model.pt"):
            if not path.exists():
                missing.append(f"{model_key}: {path}")
        for backbone, checkpoint_path in dict(model_config["checkpoint_paths"]).items():
            path = Path(str(checkpoint_path))
            if not path.exists():
                missing.append(f"{model_key}/{backbone}: {path}")
    if missing:
        raise FileNotFoundError(
            "TTA evaluation requires gitignored model/checkpoint artifacts that are missing:\n"
            + "\n".join(f"- {item}" for item in missing)
        )


def _write_stage_outputs(
    *,
    split: str,
    rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    config: dict[str, Any],
    output_dirs: dict[str, Path],
) -> None:
    artifact_prefix = _artifact_prefix(config)
    prefix = "validation" if split == "val" else split
    pd.DataFrame(rows).to_csv(
        output_dirs["tables"] / f"{artifact_prefix}_{prefix}_results.csv",
        index=False,
    )
    pd.DataFrame(per_class_rows).to_csv(
        output_dirs["tables"] / f"{artifact_prefix}_{prefix}_per_class_f1.csv",
        index=False,
    )
    runtime_path = output_dirs["tables"] / f"{artifact_prefix}_runtime_summary.csv"
    runtime = pd.DataFrame(runtime_rows)
    if runtime_path.exists():
        existing = pd.read_csv(runtime_path)
        runtime = pd.concat(
            [existing[existing["split"] != split], runtime],
            ignore_index=True,
        )
    runtime.to_csv(runtime_path, index=False)
    (Path(config["output_run_root"]) / "latest_config.json").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (Path(config["output_run_root"]) / "latest_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def _write_decision_log(
    config: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    tables_dir: Path,
) -> pd.DataFrame:
    gate = dict(config["gate"])
    decisions = []
    for model_key, model_config in config["models"].items():
        if not any(row["model_key"] == model_key for row in validation_rows):
            continue
        decision = choose_tta_policy(
            validation_rows,
            model_key=model_key,
            min_macro_f1_gain=float(gate["min_macro_f1_gain"]),
            max_accuracy_drop=float(gate["max_accuracy_drop"]),
            max_weighted_f1_drop=float(gate["max_weighted_f1_drop"]),
        )
        row = asdict(decision)
        row["test_eligible"] = bool(model_config.get("test_eligible", False))
        row["test_allowed"] = bool(row["test_allowed"] and row["test_eligible"])
        expected = model_config.get("expected_val_macro_f1")
        identity = row["identity_val_macro_f1"]
        tolerance = float(gate.get("identity_sanity_tolerance", 0.03))
        row["expected_val_macro_f1"] = expected
        row["identity_sanity_abs_error"] = (
            abs(identity - float(expected)) if expected is not None else None
        )
        row["identity_sanity_pass"] = (
            True
            if expected is None
            else row["identity_sanity_abs_error"] <= tolerance
        )
        if not row["identity_sanity_pass"]:
            row["test_allowed"] = False
            row["reason"] = (
                f"Identity sanity check failed with abs error "
                f"{row['identity_sanity_abs_error']:.4f}; {row['reason']}"
            )
        if not row["test_eligible"]:
            row["reason"] = f"Validation-only diagnostic; {row['reason']}"
        decisions.append(row)
    frame = pd.DataFrame(decisions)
    frame.to_csv(tables_dir / f"{_artifact_prefix(config)}_decision_log.csv", index=False)
    return frame


def _read_decision_log(config: dict[str, Any], tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / f"{_artifact_prefix(config)}_decision_log.csv"
    if not path.exists():
        raise FileNotFoundError(
            "TTA test stage requires a validation decision log. "
            f"Missing: {path}"
        )
    return pd.read_csv(path)


def _selected_test_pairs(decisions: pd.DataFrame) -> list[dict[str, str]]:
    selected = decisions[decisions["test_allowed"] == True]  # noqa: E712
    return [
        {"model_key": str(row.model_key), "selected_policy": str(row.selected_policy)}
        for row in selected.itertuples(index=False)
        if str(row.selected_policy) != "identity"
    ]


def _write_combined_outputs(config: dict[str, Any], output_dirs: dict[str, Path]) -> None:
    tables_dir = output_dirs["tables"]
    artifact_prefix = _artifact_prefix(config)
    validation = pd.read_csv(tables_dir / f"{artifact_prefix}_validation_results.csv")
    test_path = tables_dir / f"{artifact_prefix}_test_results.csv"
    test = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()
    combined = pd.concat([validation, test], ignore_index=True)
    delta = _build_delta_vs_identity(combined)
    delta.to_csv(tables_dir / f"{artifact_prefix}_delta_vs_identity.csv", index=False)
    _write_per_class_gain(config, tables_dir, output_dirs["figures"])
    _write_test_figures(config, delta, output_dirs["figures"])
    _write_delta_figure(config, delta, output_dirs["figures"])
    _write_runtime_figure(config, tables_dir, output_dirs["figures"])


def _build_delta_vs_identity(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, model_key), group in results.groupby(["split", "model_key"]):
        identity = group[group["policy"] == "identity"]
        if identity.empty:
            continue
        baseline = identity.iloc[0]
        for row in group[group["policy"] != "identity"].itertuples(index=False):
            rows.append(
                {
                    "split": split,
                    "model_key": model_key,
                    "display_name": row.display_name,
                    "policy": row.policy,
                    "identity_macro_f1": baseline["macro_f1"],
                    "macro_f1": row.macro_f1,
                    "macro_f1_gain": row.macro_f1 - baseline["macro_f1"],
                    "identity_accuracy": baseline["accuracy"],
                    "accuracy": row.accuracy,
                    "accuracy_gain": row.accuracy - baseline["accuracy"],
                    "identity_weighted_f1": baseline["weighted_f1"],
                    "weighted_f1": row.weighted_f1,
                    "weighted_f1_gain": row.weighted_f1 - baseline["weighted_f1"],
                    "runtime_multiplier": row.runtime_seconds
                    / max(float(baseline["runtime_seconds"]), 1e-12),
                }
            )
    return pd.DataFrame(rows)


def _write_per_class_gain(
    config: dict[str, Any],
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    artifact_prefix = _artifact_prefix(config)
    display_prefix = _display_prefix(config)
    frames = []
    for prefix in ("validation", "test"):
        path = tables_dir / f"{artifact_prefix}_{prefix}_per_class_f1.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["stage"] = prefix
            frames.append(frame)
    if not frames:
        return
    per_class = pd.concat(frames, ignore_index=True)
    rows = []
    for (stage, model_key), group in per_class.groupby(["stage", "model_key"]):
        identity = group[group["policy"] == "identity"]
        if identity.empty:
            continue
        identity_ref = identity[["label", "precision", "recall", "f1", "support"]].rename(
            columns={
                "precision": "identity_precision",
                "recall": "identity_recall",
                "f1": "identity_f1",
                "support": "identity_support",
            }
        )
        for _, candidate in group[group["policy"] != "identity"].groupby("policy"):
            merged = candidate.merge(identity_ref, on="label", how="left")
            merged["stage"] = stage
            merged["model_key"] = model_key
            merged["f1_gain"] = merged["f1"] - merged["identity_f1"]
            merged["precision_gain"] = merged["precision"] - merged["identity_precision"]
            merged["recall_gain"] = merged["recall"] - merged["identity_recall"]
            rows.append(merged)
    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output.to_csv(tables_dir / f"{artifact_prefix}_per_class_f1_gain.csv", index=False)
    if output.empty:
        return
    selected_stage = "test" if "test" in set(output["stage"]) else "validation"
    plot_frame = output[output["stage"] == selected_stage].copy()
    plot_frame["column"] = plot_frame["display_name"] + " / " + plot_frame["policy"]
    pivot = plot_frame.pivot_table(index="label", columns="column", values="f1_gain")
    plt.figure(figsize=(max(8, 1.4 * len(pivot.columns)), 5))
    sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="coolwarm", center=0)
    plt.xlabel("Model / policy")
    plt.ylabel("Class")
    plt.title(f"{display_prefix} {selected_stage.title()} Per-Class F1 Gain vs Identity")
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / f"{artifact_prefix}_per_class_f1_gain_heatmap.png", dpi=200)
    plt.close()


def _write_predictions(
    *,
    config: dict[str, Any],
    model_key: str,
    split: str,
    policy: str,
    reference: pd.DataFrame,
    probabilities: torch.Tensor,
    y_pred: Any,
    class_names: list[str],
) -> None:
    frame = reference.copy()
    frame["pred_label"] = [class_names[int(index)] for index in y_pred]
    frame["correct"] = frame["label_index"].to_numpy() == y_pred
    for index, class_name in enumerate(class_names):
        frame[f"prob_{class_name}"] = probabilities[:, index].cpu().numpy()
    path = Path(config["predictions_dir"]) / f"{split}_{model_key}_{policy}_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_confusion_matrix(
    config: dict[str, Any],
    split: str,
    model_key: str,
    policy: str,
    metrics: dict[str, Any],
    class_names: list[str],
) -> None:
    path = (
        Path(config["figures_dir"])
        / "confusion_matrices"
        / f"{split}_{model_key}_{policy}_confusion_matrix.png"
    )
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        path,
        title=f"{_display_prefix(config)} {split} {model_key} {policy}",
    )


def _write_validation_figures(
    config: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    decisions: pd.DataFrame,
    figures_dir: Path,
) -> None:
    frame = pd.DataFrame(validation_rows)
    if frame.empty:
        return
    artifact_prefix = _artifact_prefix(config)
    display_prefix = _display_prefix(config)
    path = figures_dir / f"{artifact_prefix}_val_policy_comparison.png"
    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(data=frame, x="display_name", y="macro_f1", hue="policy")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title(f"{display_prefix} Validation TTA Policy Comparison")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    decisions.to_json(
        figures_dir / f"{artifact_prefix}_decision_log.json",
        orient="records",
        indent=2,
    )


def _write_test_figures(
    config: dict[str, Any],
    delta: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if delta.empty:
        return
    test = delta[delta["split"] == "test"].copy()
    if test.empty:
        return
    artifact_prefix = _artifact_prefix(config)
    display_prefix = _display_prefix(config)
    path = figures_dir / f"{artifact_prefix}_test_macro_f1_delta.png"
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=test, x="display_name", y="macro_f1_gain", hue="policy")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Test macro-F1 gain vs identity")
    ax.set_title(f"{display_prefix} Test Macro-F1 Delta")
    for container in ax.containers:
        ax.bar_label(container, fmt="%+.3f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_delta_figure(
    config: dict[str, Any],
    delta: pd.DataFrame,
    figures_dir: Path,
) -> None:
    if delta.empty:
        return
    validation = delta[delta["split"] == "val"].copy()
    if validation.empty:
        return
    artifact_prefix = _artifact_prefix(config)
    display_prefix = _display_prefix(config)
    path = figures_dir / f"{artifact_prefix}_validation_macro_f1_delta.png"
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=validation, x="display_name", y="macro_f1_gain", hue="policy")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Validation macro-F1 gain vs identity")
    ax.set_title(f"{display_prefix} Validation Macro-F1 Delta")
    for container in ax.containers:
        ax.bar_label(container, fmt="%+.3f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _write_runtime_figure(
    config: dict[str, Any],
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    artifact_prefix = _artifact_prefix(config)
    display_prefix = _display_prefix(config)
    path = tables_dir / f"{artifact_prefix}_runtime_summary.csv"
    if not path.exists():
        return
    runtime = pd.read_csv(path)
    if runtime.empty:
        return
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=runtime, x="display_name", y="runtime_seconds", hue="policy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Runtime seconds")
    ax.set_title(f"{display_prefix} Runtime by TTA Policy")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(figures_dir / f"{artifact_prefix}_runtime_multiplier.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
