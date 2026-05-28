"""Fine-tune final CNN blocks and extract fine-tuned feature caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dl_midterm.config.load_config import load_yaml
from dl_midterm.data.dataloaders import (
    create_feature_extraction_loaders,
    create_finetuning_loaders,
)
from dl_midterm.features.cache import backbone_cache_dir, save_backbone_manifest
from dl_midterm.models.backbones import supported_backbones
from dl_midterm.training.finetune import extract_finetuned_feature_cache, finetune_backbone
from dl_midterm.utils.device import resolve_device, supports_mixed_precision
from dl_midterm.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/finetune_backbones.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--backbone", choices=supported_backbones(), default=None)
    parser.add_argument("--backbones", nargs="+", choices=supported_backbones(), default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--feature-source", default="finetuned")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-feature-extraction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finetune_config = load_yaml(args.config)["finetuning"]
    dataset_config = load_yaml(args.dataset_config)["dataset"]
    default_config = load_yaml(args.default_config)
    runtime_config = default_config.get("runtime", {})
    training_config = default_config.get("training", {})

    seed = int(finetune_config.get("seed", dataset_config.get("seed", 42)))
    seed_everything(seed)

    device = resolve_device(args.device or runtime_config.get("device", "auto"))
    mixed_precision = (
        bool(runtime_config.get("mixed_precision", True))
        and supports_mixed_precision(device)
        and not args.no_mixed_precision
    )
    batch_size = args.batch_size or int(
        finetune_config.get("batch_size", training_config.get("batch_size", 32))
    )
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(runtime_config.get("num_workers", 2))
    epochs = args.epochs or int(finetune_config.get("epochs", 10))
    learning_rate = args.learning_rate or float(finetune_config.get("learning_rate", 1e-4))
    weight_decay = args.weight_decay or float(finetune_config.get("weight_decay", 1e-4))
    patience = args.early_stopping_patience or int(
        finetune_config.get("early_stopping_patience", 3)
    )
    checkpoint_dir = Path(args.checkpoint_dir or finetune_config["checkpoint_dir"])
    backbones = _selected_backbones(args, finetune_config)

    class_names = list(dataset_config["class_names"])
    image_size = int(dataset_config.get("image_size", 224))
    train_loaders = create_finetuning_loaders(
        splits_dir=dataset_config["splits_dir"],
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples_per_split=args.limit_per_split,
    )
    extraction_loaders = create_feature_extraction_loaders(
        splits_dir=dataset_config["splits_dir"],
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples_per_split=args.limit_per_split,
    )

    summary: list[dict[str, Any]] = []
    for backbone in backbones:
        print(f"Fine-tuning {backbone} on {device}.")
        policy = _policy_for_backbone(finetune_config, backbone)
        checkpoint_path = checkpoint_dir / f"{backbone}_best.pt"
        run_dir: Path | None = None
        if not args.skip_training:
            checkpoint_path, run_dir = finetune_backbone(
                backbone=backbone,
                loaders=train_loaders,
                class_names=class_names,
                device=device,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                early_stopping_patience=patience,
                policy=policy,
                mixed_precision=mixed_precision,
                pretrained=not args.no_pretrained,
                class_weighting=not args.no_class_weights,
                output_run_root=args.run_root,
                limit_per_split=args.limit_per_split,
            )
            print(f"Wrote best checkpoint: {checkpoint_path}")
        if not args.skip_feature_extraction:
            cache_dir = backbone_cache_dir(
                args.feature_root,
                dataset_config["name"],
                args.feature_source,
                backbone,
            )
            caches = extract_finetuned_feature_cache(
                backbone=backbone,
                checkpoint_path=checkpoint_path,
                loaders=extraction_loaders,
                output_dir=cache_dir,
                class_names=class_names,
                seed=seed,
                device=device,
                mixed_precision=mixed_precision,
                config={
                    "checkpoint_path": str(checkpoint_path),
                    "dataset_config": str(Path(args.dataset_config)),
                    "default_config": str(Path(args.default_config)),
                    "finetune_config": str(Path(args.config)),
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "feature_source": args.feature_source,
                    "limit_per_split": args.limit_per_split,
                },
            )
            manifest_path = save_backbone_manifest(cache_dir, caches)
            print(f"Wrote fine-tuned feature manifest: {manifest_path}")
        summary.append(
            {
                "backbone": backbone,
                "checkpoint_path": str(checkpoint_path),
                "run_dir": str(run_dir) if run_dir is not None else None,
                "feature_cache_dir": str(
                    backbone_cache_dir(
                        args.feature_root,
                        dataset_config["name"],
                        args.feature_source,
                        backbone,
                    )
                ),
            }
        )

    summary_path = Path(args.run_root) / "sprint4_finetune_backbones_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote fine-tuning summary: {summary_path}")


def _selected_backbones(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if args.backbone is not None:
        return [args.backbone]
    if args.backbones is not None:
        return list(args.backbones)
    return list(config.get("backbones", supported_backbones()))


def _policy_for_backbone(config: dict[str, Any], backbone: str) -> str | None:
    policies = config.get("unfreeze_policies", {})
    if isinstance(policies, dict) and backbone in policies:
        return str(policies[backbone])
    if backbone == "resnet50":
        return "layer4"
    return "last_feature_blocks"


if __name__ == "__main__":
    main()
