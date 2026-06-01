"""Representation complementarity tests."""

from __future__ import annotations

from pathlib import Path

import torch

from dl_midterm.evaluation.complementarity import compute_representation_complementarity
from dl_midterm.features.cache import save_feature_cache


def test_representation_complementarity_handles_different_feature_dims(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "test.csv").write_text(
        "image_id,label,image_path\n"
        "img1,nv,/tmp/img1.jpg\n"
        "img2,mel,/tmp/img2.jpg\n"
        "img3,bcc,/tmp/img3.jpg\n",
        encoding="utf-8",
    )
    feature_root = tmp_path / "features"
    class_names = ["nv", "mel", "bcc"]
    for backbone, dim in (("resnet50", 4), ("mobilenet_v2", 3), ("efficientnet_b0", 5)):
        save_feature_cache(
            feature_root / "ham10000" / "frozen" / backbone / "test.pt",
            features=torch.randn(3, dim),
            labels=torch.tensor([0, 1, 2]),
            image_ids=["img1", "img2", "img3"],
            label_names=["nv", "mel", "bcc"],
            split="test",
            backbone=backbone,
            class_names=class_names,
            feature_source="frozen",
            seed=42,
        )

    summary = compute_representation_complementarity(
        feature_root=feature_root,
        dataset_name="ham10000",
        feature_source="frozen",
        backbones=["resnet50", "mobilenet_v2", "efficientnet_b0"],
        splits_dir=splits_dir,
        split="test",
    )

    assert len(summary) == 3
    assert set(summary["method"]) == {"sample_cosine_rsa_pearson"}
    assert summary["representation_similarity"].between(-1, 1).all()


def test_representation_complementarity_preserves_columns_for_single_backbone(
    tmp_path: Path,
) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "test.csv").write_text(
        "image_id,label,image_path\n"
        "img1,nv,/tmp/img1.jpg\n"
        "img2,mel,/tmp/img2.jpg\n",
        encoding="utf-8",
    )
    feature_root = tmp_path / "features"
    save_feature_cache(
        feature_root / "ham10000" / "finetuned_deeper" / "resnet50" / "test.pt",
        features=torch.randn(2, 4),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="test",
        backbone="resnet50",
        class_names=["nv", "mel"],
        feature_source="finetuned_deeper",
        seed=42,
    )

    summary = compute_representation_complementarity(
        feature_root=feature_root,
        dataset_name="ham10000",
        feature_source="finetuned_deeper",
        backbones=["resnet50"],
        splits_dir=splits_dir,
        split="test",
    )

    assert summary.empty
    assert {
        "left_backbone",
        "right_backbone",
        "representation_similarity",
        "representation_complementarity",
    }.issubset(summary.columns)
