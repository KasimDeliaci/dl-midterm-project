"""Sprint 4E fusion diagnostic tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from dl_midterm.config.load_config import load_yaml
from dl_midterm.features.cache import FeatureCache
from dl_midterm.models.fusion import PerClassWeightedFusionMLP

_SCRIPT_PATH = Path("scripts/run_fusion_diagnostic.py")
_SPEC = importlib.util.spec_from_file_location("run_fusion_diagnostic", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_apply_normalization = _MODULE._apply_normalization
_fit_normalization = _MODULE._fit_normalization


def _fake_cache(backbone: str, features: torch.Tensor) -> FeatureCache:
    labels = torch.tensor([0, 1], dtype=torch.long)
    return FeatureCache(
        path=Path(f"{backbone}.pt"),
        features=features.float(),
        labels=labels,
        image_ids=["a", "b"],
        label_names=["akiec", "bcc"],
        split="train",
        backbone=backbone,
        feature_dim=int(features.shape[1]),
        metadata={},
    )


def test_per_class_weighted_fusion_weights_sum_to_one_per_class() -> None:
    model = PerClassWeightedFusionMLP(
        input_dims=[4, 3, 2],
        num_classes=7,
        projection_dim=5,
        hidden_dims=[6],
        dropout=0.0,
    )

    weights = model.normalized_weights()

    assert tuple(weights.shape) == (3, 7)
    assert torch.allclose(weights.sum(dim=0), torch.ones(7))


def test_per_class_weighted_fusion_forward_shape() -> None:
    model = PerClassWeightedFusionMLP(
        input_dims=[4, 3, 2],
        num_classes=7,
        projection_dim=5,
        hidden_dims=[6],
        dropout=0.0,
    )
    features = torch.randn(8, 9)

    logits = model(features)

    assert tuple(logits.shape) == (8, 7)


def test_standardization_stats_are_fit_from_train_cache_only() -> None:
    train = _fake_cache("resnet50", torch.tensor([[1.0, 3.0], [3.0, 7.0]]))
    stats = _fit_normalization([train], "standardize_per_backbone")
    val_features = torch.tensor([[3.0, 7.0]])

    normalized = _apply_normalization(
        val_features,
        backbone="resnet50",
        normalization="standardize_per_backbone",
        stats=stats,
    )

    assert torch.allclose(stats["resnet50"].mean, torch.tensor([2.0, 5.0]))
    assert torch.allclose(normalized, torch.tensor([[1.0, 1.0]]))


def test_l2_normalization_preserves_feature_shape_and_unit_norm() -> None:
    features = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

    normalized = _apply_normalization(
        features,
        backbone="resnet50",
        normalization="l2_per_backbone",
        stats={},
    )

    assert tuple(normalized.shape) == tuple(features.shape)
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=1), torch.ones(2))


def test_sprint4e_config_is_small_local_cached_feature_diagnostic() -> None:
    config = load_yaml("configs/experiments/sprint4e_fusion_diagnostic.yaml")["sprint4e"]

    assert config["feature_source"] == "finetuned"
    assert config["test_top_k"] == 2
    assert len(config["candidates"]) <= 16
    assert {candidate["fusion_method"] for candidate in config["candidates"]} == {
        "concat",
        "weighted",
        "per_class_weighted",
    }


def test_sprint4e_script_does_not_use_raw_image_stage() -> None:
    source = Path("scripts/run_fusion_diagnostic.py").read_text(encoding="utf-8")

    assert "PIL" not in source
    assert "image_path" not in source
