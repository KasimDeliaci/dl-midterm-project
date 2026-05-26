"""Feature extraction, normalization, cache IO, and feature metadata helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dl_midterm.features.cache import FeatureCache, save_feature_cache
from dl_midterm.models.backbones import expected_feature_dim


@torch.no_grad()
def extract_features_from_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    mixed_precision: bool = False,
    progress_label: str = "extract",
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], float]:
    """Run a frozen feature extractor over one split dataloader."""

    model.to(device)
    model.eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    image_ids: list[str] = []
    label_names: list[str] = []
    started = perf_counter()
    device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"
    autocast_enabled = mixed_precision and device.type in {"cuda", "mps"}

    for batch in tqdm(loader, desc=progress_label):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].cpu().long()
        with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled):
            features = model(images)
        feature_batches.append(features.detach().cpu().float())
        label_batches.append(labels)
        image_ids.extend(str(value) for value in batch["image_id"])
        label_names.extend(str(value) for value in batch["label_name"])

    elapsed = perf_counter() - started
    features = torch.cat(feature_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    return features, labels, image_ids, label_names, elapsed


def extract_and_cache_backbone(
    *,
    model: torch.nn.Module,
    backbone: str,
    loaders: dict[str, DataLoader],
    output_dir: str | Path,
    class_names: list[str],
    feature_source: str,
    seed: int,
    device: torch.device,
    mixed_precision: bool,
    config: dict[str, Any] | None = None,
) -> list[FeatureCache]:
    """Extract and save train/validation/test caches for one backbone."""

    caches: list[FeatureCache] = []
    expected_dim = expected_feature_dim(backbone)
    for split, loader in loaders.items():
        features, labels, image_ids, label_names, elapsed = extract_features_from_loader(
            model,
            loader,
            device,
            mixed_precision=mixed_precision,
            progress_label=f"{backbone}:{split}",
        )
        actual_dim = int(features.shape[1])
        if actual_dim != expected_dim:
            raise ValueError(
                f"{backbone} produced feature_dim={actual_dim}, expected {expected_dim}."
            )
        split_config = dict(config or {})
        split_config["runtime_seconds"] = round(elapsed, 4)
        cache = save_feature_cache(
            Path(output_dir) / f"{split}.pt",
            features=features,
            labels=labels,
            image_ids=image_ids,
            label_names=label_names,
            split=split,
            backbone=backbone,
            class_names=class_names,
            feature_source=feature_source,
            seed=seed,
            config=split_config,
        )
        caches.append(cache)
    return caches


def select_single_backbone_combinations(
    combinations: Iterable[dict[str, Any]],
) -> list[str]:
    """Extract single-backbone `fusion: none` entries from an experiment matrix."""

    backbones: list[str] = []
    for combination in combinations:
        names = list(combination.get("backbones", []))
        fusion_methods = set(combination.get("fusion_methods", []))
        if len(names) == 1 and "none" in fusion_methods:
            backbones.append(names[0])
    return backbones
