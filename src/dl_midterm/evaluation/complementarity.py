"""Representation complementarity analysis for cached backbone features."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dl_midterm.features.cache import (
    backbone_cache_dir,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)


def compute_representation_complementarity(
    *,
    feature_root: str | Path,
    dataset_name: str,
    feature_source: str,
    backbones: list[str],
    splits_dir: str | Path,
    split: str = "test",
    max_samples: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare backbone representations with sample-similarity correlation.

    Feature dimensions can differ across backbones, so this uses a simple representational
    similarity analysis: compute the sample-by-sample cosine matrix for each backbone, then
    correlate the upper triangles of those matrices.
    """

    caches = []
    for backbone in backbones:
        cache_dir = backbone_cache_dir(feature_root, dataset_name, feature_source, backbone)
        cache = load_feature_cache(feature_cache_path(cache_dir, split))
        verify_cache_matches_split(cache, Path(splits_dir) / f"{split}.csv")
        caches.append(cache)

    if not caches:
        raise ValueError("At least one backbone cache is required.")

    sample_count = len(caches[0].labels)
    for cache in caches[1:]:
        if cache.image_ids != caches[0].image_ids:
            raise ValueError("Backbone caches must be aligned before complementarity analysis.")

    indices = _sample_indices(sample_count, max_samples=max_samples, seed=seed)
    similarity_vectors = {
        cache.backbone: _upper_triangle_cosine_vector(cache.features[indices])
        for cache in caches
    }

    columns = [
        "split",
        "num_samples",
        "left_backbone",
        "right_backbone",
        "representation_similarity",
        "representation_complementarity",
        "method",
    ]
    rows: list[dict[str, float | int | str]] = []
    for left, right in combinations(backbones, 2):
        similarity = _safe_pearson(similarity_vectors[left], similarity_vectors[right])
        rows.append(
            {
                "split": split,
                "num_samples": int(len(indices)),
                "left_backbone": left,
                "right_backbone": right,
                "representation_similarity": similarity,
                "representation_complementarity": 1.0 - similarity,
                "method": "sample_cosine_rsa_pearson",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_fusion_complementarity_summary(
    results: pd.DataFrame,
    pairwise_complementarity: pd.DataFrame,
) -> pd.DataFrame:
    """Attach average pairwise complementarity to each fusion run."""

    fusion_rows = results[results["fusion_method"].isin(["concat", "weighted"])].copy()
    pair_values = {
        frozenset((row.left_backbone, row.right_backbone)): float(
            row.representation_complementarity
        )
        for row in pairwise_complementarity.itertuples(index=False)
    }
    similarity_values = {
        frozenset((row.left_backbone, row.right_backbone)): float(
            row.representation_similarity
        )
        for row in pairwise_complementarity.itertuples(index=False)
    }

    rows: list[dict[str, float | int | str]] = []
    for row in fusion_rows.itertuples(index=False):
        backbones = str(row.backbone_combination).split("+")
        pair_keys = [frozenset(pair) for pair in combinations(backbones, 2)]
        complementarity = [pair_values[key] for key in pair_keys if key in pair_values]
        similarity = [similarity_values[key] for key in pair_keys if key in similarity_values]
        if not complementarity:
            continue
        rows.append(
            {
                "run_id": row.run_id,
                "display_name": row.display_name,
                "backbone_combination": row.backbone_combination,
                "fusion_method": row.fusion_method,
                "backbone_count": int(row.backbone_count),
                "avg_pairwise_representation_similarity": float(np.mean(similarity)),
                "avg_pairwise_representation_complementarity": float(np.mean(complementarity)),
                "macro_f1": float(row.macro_f1),
                "macro_f1_gain": float(row.macro_f1_gain)
                if hasattr(row, "macro_f1_gain")
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _sample_indices(sample_count: int, *, max_samples: int, seed: int) -> torch.Tensor:
    if max_samples <= 0 or sample_count <= max_samples:
        return torch.arange(sample_count)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(sample_count, generator=generator)[:max_samples].sort().values


def _upper_triangle_cosine_vector(features: torch.Tensor) -> np.ndarray:
    normalized = F.normalize(features.float(), p=2, dim=1)
    similarity = normalized @ normalized.T
    row_index, col_index = torch.triu_indices(
        similarity.shape[0],
        similarity.shape[1],
        offset=1,
    )
    return similarity[row_index, col_index].cpu().numpy()


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Similarity vectors must have the same shape.")
    if np.isclose(left.std(), 0.0) or np.isclose(right.std(), 0.0):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])
