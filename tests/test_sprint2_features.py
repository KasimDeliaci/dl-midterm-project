"""Sprint 2 feature extraction and cache tests."""

from __future__ import annotations

from pathlib import Path

import torch

from dl_midterm.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_image_transform
from dl_midterm.features.cache import (
    class_weights_from_cache,
    load_feature_cache,
    sample_weights_from_cache,
    save_feature_cache,
    verify_cache_matches_split,
)
from dl_midterm.models.backbones import build_frozen_feature_extractor, expected_feature_dim
from dl_midterm.models.feature_extractors import count_trainable_parameters


def test_transform_uses_imagenet_normalization() -> None:
    transform = build_image_transform(image_size=224)

    assert tuple(transform.transforms[-1].mean) == IMAGENET_MEAN
    assert tuple(transform.transforms[-1].std) == IMAGENET_STD


def test_feature_cache_roundtrip_and_split_alignment(tmp_path: Path) -> None:
    split_csv = tmp_path / "train.csv"
    split_csv.write_text(
        "image_id,label,image_path\nimg1,nv,/tmp/img1.jpg\nimg2,mel,/tmp/img2.jpg\n",
        encoding="utf-8",
    )
    cache_path = tmp_path / "train.pt"
    save_feature_cache(
        cache_path,
        features=torch.ones(2, 4),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="train",
        backbone="resnet50",
        class_names=["nv", "mel"],
        feature_source="frozen",
        seed=42,
        config={"test": True},
    )

    cache = load_feature_cache(cache_path)

    assert cache.feature_dim == 4
    assert cache.image_ids == ["img1", "img2"]
    verify_cache_matches_split(cache, split_csv)
    assert class_weights_from_cache(cache, num_classes=2).tolist() == [1.0, 1.0]


def test_sample_weights_from_cache_equalizes_class_sampling_mass(tmp_path: Path) -> None:
    cache_path = tmp_path / "train.pt"
    cache = save_feature_cache(
        cache_path,
        features=torch.ones(6, 4),
        labels=torch.tensor([0, 0, 0, 0, 1, 1]),
        image_ids=[f"img{i}" for i in range(6)],
        label_names=["nv"] * 4 + ["mel"] * 2,
        split="train",
        backbone="resnet50",
        class_names=["nv", "mel"],
        feature_source="frozen",
        seed=42,
    )

    weights = sample_weights_from_cache(cache, num_classes=2)

    assert torch.isclose(weights[cache.labels == 0].sum(), weights[cache.labels == 1].sum())
    assert weights[4] > weights[0]


def test_frozen_backbone_output_dimensions_without_pretrained_weights() -> None:
    for backbone in ("resnet50", "mobilenet_v2", "efficientnet_b0"):
        model = build_frozen_feature_extractor(backbone, pretrained=False)
        output = model(torch.randn(2, 3, 64, 64))

        assert output.shape == (2, expected_feature_dim(backbone))
        assert count_trainable_parameters(model) == 0
