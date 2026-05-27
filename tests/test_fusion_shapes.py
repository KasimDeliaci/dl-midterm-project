"""Fusion-shape, matrix-expansion, and cache-alignment tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from dl_midterm.features.cache import save_feature_cache
from dl_midterm.models.fusion import (
    ConcatenationFusion,
    ProjectedWeightedFusion,
    WeightedFusionMLP,
    expected_concat_dim,
)


def _load_run_matrix_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concat_output_shape_and_expected_dimensions() -> None:
    fusion = ConcatenationFusion()
    output = fusion([torch.randn(4, 2048), torch.randn(4, 1280)])

    assert output.shape == (4, 3328)
    assert expected_concat_dim(["resnet50", "mobilenet_v2"]) == 3328
    assert expected_concat_dim(["resnet50", "efficientnet_b0"]) == 3328
    assert expected_concat_dim(["mobilenet_v2", "efficientnet_b0"]) == 2560
    assert expected_concat_dim(["resnet50", "mobilenet_v2", "efficientnet_b0"]) == 4608


def test_weighted_fusion_projection_shape_and_softmax_sum() -> None:
    fusion = ProjectedWeightedFusion([2048, 1280], projection_dim=512)
    output = fusion([torch.randn(3, 2048), torch.randn(3, 1280)])
    weights = fusion.normalized_weights()

    assert output.shape == (3, 512)
    assert weights.shape == (2,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_weighted_fusion_mlp_accepts_concatenated_cached_features() -> None:
    model = WeightedFusionMLP([2048, 1280, 1280], num_classes=7, projection_dim=512)
    logits = model(torch.randn(5, 4608))

    assert logits.shape == (5, 7)
    assert torch.isclose(model.normalized_weights().sum(), torch.tensor(1.0))


def test_fusion_run_matrix_expands_to_eight_runs() -> None:
    run_matrix = _load_run_matrix_module()
    config = {
        "combinations": [
            {"backbones": ["resnet50"], "fusion_methods": ["none"]},
            {"backbones": ["mobilenet_v2"], "fusion_methods": ["none"]},
            {"backbones": ["efficientnet_b0"], "fusion_methods": ["none"]},
            {"backbones": ["resnet50", "mobilenet_v2"], "fusion_methods": ["concat", "weighted"]},
            {
                "backbones": ["resnet50", "efficientnet_b0"],
                "fusion_methods": ["concat", "weighted"],
            },
            {
                "backbones": ["mobilenet_v2", "efficientnet_b0"],
                "fusion_methods": ["concat", "weighted"],
            },
            {
                "backbones": ["resnet50", "mobilenet_v2", "efficientnet_b0"],
                "fusion_methods": ["concat", "weighted"],
            },
        ]
    }

    runs = run_matrix.expand_fusion_run_matrix(config)

    assert len(runs) == 8
    assert {run["fusion_method"] for run in runs} == {"concat", "weighted"}
    assert all(len(run["backbones"]) >= 2 for run in runs)


def test_cache_alignment_detects_mismatched_order(tmp_path: Path) -> None:
    run_matrix = _load_run_matrix_module()
    first = save_feature_cache(
        tmp_path / "a.pt",
        features=torch.ones(2, 4),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="train",
        backbone="resnet50",
        class_names=["nv", "mel"],
        feature_source="frozen",
        seed=42,
    )
    second = save_feature_cache(
        tmp_path / "b.pt",
        features=torch.ones(2, 4),
        labels=torch.tensor([1, 0]),
        image_ids=["img2", "img1"],
        label_names=["mel", "nv"],
        split="train",
        backbone="mobilenet_v2",
        class_names=["nv", "mel"],
        feature_source="frozen",
        seed=42,
    )
    aligned = save_feature_cache(
        tmp_path / "c.pt",
        features=torch.ones(2, 4),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="train",
        backbone="efficientnet_b0",
        class_names=["nv", "mel"],
        feature_source="frozen",
        seed=42,
    )

    run_matrix.verify_cache_alignment([first, aligned])
    with pytest.raises(ValueError, match="image_id order"):
        run_matrix.verify_cache_alignment([first, second])
