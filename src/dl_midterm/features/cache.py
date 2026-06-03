"""Feature-cache read/write helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FeatureCache:
    """Loaded feature-cache payload."""

    path: Path
    features: torch.Tensor
    labels: torch.Tensor
    image_ids: list[str]
    label_names: list[str]
    split: str
    backbone: str
    feature_dim: int
    metadata: dict[str, Any]


class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tensor dataset for cached CNN features."""

    def __init__(self, cache: FeatureCache) -> None:
        self.cache = cache
        if len(cache.features) != len(cache.labels):
            raise ValueError("Feature and label counts do not match.")

    def __len__(self) -> int:
        return len(self.cache.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cache.features[index].float(), self.cache.labels[index].long()


def backbone_cache_dir(
    output_root: str | Path,
    dataset_name: str,
    feature_source: str,
    backbone: str,
) -> Path:
    """Return the directory for one dataset/source/backbone feature cache."""

    return Path(output_root) / dataset_name / feature_source / backbone


def feature_cache_path(cache_dir: str | Path, split: str) -> Path:
    """Return the `.pt` cache path for a split."""

    return Path(cache_dir) / f"{split}.pt"


def save_feature_cache(
    path: str | Path,
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    image_ids: list[str],
    label_names: list[str],
    split: str,
    backbone: str,
    class_names: list[str],
    feature_source: str,
    seed: int,
    config: dict[str, Any] | None = None,
) -> FeatureCache:
    """Save feature tensors and alignment metadata."""

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = features.detach().cpu().float()
    labels = labels.detach().cpu().long()
    feature_dim = int(features.shape[1])
    metadata: dict[str, Any] = {
        "split": split,
        "backbone": backbone,
        "feature_source": feature_source,
        "feature_dim": feature_dim,
        "num_samples": int(features.shape[0]),
        "class_names": class_names,
        "seed": seed,
        "config": config or {},
    }
    payload = {
        "features": features,
        "labels": labels,
        "image_ids": image_ids,
        "label_names": label_names,
        "metadata": metadata,
    }
    _validate_payload(payload)
    torch.save(payload, cache_path)
    _write_split_manifest(cache_path.with_name(f"{split}_manifest.csv"), payload)
    return load_feature_cache(cache_path)


def load_feature_cache(path: str | Path, map_location: str | torch.device = "cpu") -> FeatureCache:
    """Load a feature cache from disk."""

    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache does not exist: {cache_path}")
    try:
        payload = torch.load(cache_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location=map_location)
    _validate_payload(payload)
    metadata = payload["metadata"]
    return FeatureCache(
        path=cache_path,
        features=payload["features"],
        labels=payload["labels"],
        image_ids=list(payload["image_ids"]),
        label_names=list(payload["label_names"]),
        split=str(metadata["split"]),
        backbone=str(metadata["backbone"]),
        feature_dim=int(metadata["feature_dim"]),
        metadata=dict(metadata),
    )


def save_backbone_manifest(cache_dir: str | Path, caches: list[FeatureCache]) -> Path:
    """Write a small JSON manifest for all split caches of one backbone."""

    cache_root = Path(cache_dir)
    manifest = {
        "backbone": caches[0].backbone if caches else None,
        "feature_source": caches[0].metadata.get("feature_source") if caches else None,
        "feature_dim": caches[0].feature_dim if caches else None,
        "splits": {
            cache.split: {
                "path": str(cache.path),
                "num_samples": len(cache.labels),
                "feature_dim": cache.feature_dim,
            }
            for cache in caches
        },
    }
    path = cache_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def verify_cache_matches_split(
    cache: FeatureCache,
    split_csv: str | Path,
    *,
    allow_prefix: bool = False,
) -> None:
    """Ensure cache image IDs and labels align exactly with a split CSV."""

    split = pd.read_csv(split_csv)
    expected_ids = split["image_id"].astype(str).tolist()
    expected_labels = split["label"].astype(str).tolist()
    if allow_prefix:
        expected_ids = expected_ids[: len(cache.image_ids)]
        expected_labels = expected_labels[: len(cache.label_names)]
    if cache.image_ids != expected_ids:
        raise ValueError(f"Cache image_id order does not match split CSV: {split_csv}")
    if cache.label_names != expected_labels:
        raise ValueError(f"Cache label order does not match split CSV: {split_csv}")


def cache_allows_prefix_split_verification(cache: FeatureCache) -> bool:
    """Return true for explicitly limited smoke-test feature caches."""

    config = cache.metadata.get("config", {})
    return bool(isinstance(config, dict) and config.get("limit_per_split") is not None)


def class_weights_from_cache(cache: FeatureCache, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training cache only."""

    counts = torch.bincount(cache.labels.long(), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    weights[counts == 0] = 0.0
    return weights


def sample_weights_from_cache(cache: FeatureCache, num_classes: int) -> torch.Tensor:
    """Return per-sample inverse-frequency weights for balanced train sampling."""

    counts = torch.bincount(cache.labels.long(), minlength=num_classes).float()
    class_weights = torch.zeros(num_classes, dtype=torch.float)
    observed = counts > 0
    class_weights[observed] = 1.0 / counts[observed]
    return class_weights[cache.labels.long()]


def _write_split_manifest(path: Path, payload: dict[str, Any]) -> None:
    frame = pd.DataFrame(
        {
            "row_index": range(len(payload["labels"])),
            "image_id": payload["image_ids"],
            "label": payload["label_names"],
            "label_index": payload["labels"].tolist(),
            "split": payload["metadata"]["split"],
            "backbone": payload["metadata"]["backbone"],
        }
    )
    frame.to_csv(path, index=False)


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {"features", "labels", "image_ids", "label_names", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Feature cache is missing keys: {sorted(missing)}")
    features = payload["features"]
    labels = payload["labels"]
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("Feature cache `features` must be a 2D torch.Tensor.")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("Feature cache `labels` must be a 1D torch.Tensor.")
    sample_count = int(features.shape[0])
    if sample_count != int(labels.shape[0]):
        raise ValueError("Feature and label counts do not match.")
    if sample_count != len(payload["image_ids"]) or sample_count != len(payload["label_names"]):
        raise ValueError("Feature cache metadata length does not match feature rows.")
    metadata = payload["metadata"]
    feature_dim = int(features.shape[1])
    if int(metadata["feature_dim"]) != feature_dim:
        raise ValueError("Feature dimension metadata does not match feature tensor.")
