"""Extract frozen CNN feature vectors and save feature caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from dl_midterm.config.load_config import load_yaml
from dl_midterm.data.dataloaders import create_feature_extraction_loaders
from dl_midterm.features.cache import backbone_cache_dir, save_backbone_manifest
from dl_midterm.features.extract import extract_and_cache_backbone
from dl_midterm.models.backbones import build_frozen_feature_extractor, supported_backbones
from dl_midterm.utils.device import resolve_device, supports_mixed_precision
from dl_midterm.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--source", default="frozen", choices=["frozen"])
    parser.add_argument("--backbones", nargs="+", default=supported_backbones())
    parser.add_argument("--output-root", default="artifacts/features")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_yaml(args.config)["dataset"]
    default_config = load_yaml(args.default_config)
    runtime_config = default_config.get("runtime", {})
    seed = int(dataset_config.get("seed", runtime_config.get("seed", 42)))
    seed_everything(seed)

    device = resolve_device(args.device or runtime_config.get("device", "auto"))
    mixed_precision = (
        bool(runtime_config.get("mixed_precision", True))
        and supports_mixed_precision(device)
        and not args.no_mixed_precision
    )
    batch_size = args.batch_size or int(default_config.get("training", {}).get("batch_size", 32))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = int(runtime_config.get("num_workers", 2))

    class_names = list(dataset_config["class_names"])
    loaders = create_feature_extraction_loaders(
        splits_dir=dataset_config["splits_dir"],
        class_names=class_names,
        image_size=int(dataset_config.get("image_size", 224)),
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples_per_split=args.limit_per_split,
    )

    for backbone in args.backbones:
        print(f"Extracting {args.source} features for {backbone} on {device}.")
        model = build_frozen_feature_extractor(backbone, pretrained=not args.no_pretrained)
        cache_dir = backbone_cache_dir(
            args.output_root,
            dataset_config["name"],
            args.source,
            backbone,
        )
        caches = extract_and_cache_backbone(
            model=model,
            backbone=backbone,
            loaders=loaders,
            output_dir=cache_dir,
            class_names=class_names,
            feature_source=args.source,
            seed=seed,
            device=device,
            mixed_precision=mixed_precision,
            config={
                "dataset_config": str(Path(args.config)),
                "default_config": str(Path(args.default_config)),
                "batch_size": batch_size,
                "image_size": int(dataset_config.get("image_size", 224)),
                "pretrained": not args.no_pretrained,
                "limit_per_split": args.limit_per_split,
            },
        )
        manifest_path = save_backbone_manifest(cache_dir, caches)
        print(f"Wrote {backbone} cache manifest: {manifest_path}")


if __name__ == "__main__":
    main()
