"""Sprint 4K image-level balanced sampler tests."""

from __future__ import annotations

from pathlib import Path

from dl_midterm.config.load_config import load_yaml
from dl_midterm.data.dataloaders import create_image_dataloader


def test_image_level_balanced_sampler_is_configured_without_shuffle(tmp_path: Path) -> None:
    split_csv = tmp_path / "train.csv"
    split_csv.write_text(
        "image_id,label,image_path\n"
        "img0,nv,/tmp/img0.jpg\n"
        "img1,nv,/tmp/img1.jpg\n"
        "img2,nv,/tmp/img2.jpg\n"
        "img3,mel,/tmp/img3.jpg\n",
        encoding="utf-8",
    )

    loader = create_image_dataloader(
        split_csv,
        class_names=["nv", "mel"],
        batch_size=2,
        shuffle=True,
        sampler_config={"name": "class_balanced", "replacement": True},
        seed=42,
    )

    assert loader.sampler is not None
    assert loader.batch_sampler.sampler is loader.sampler


def test_sprint4k_config_uses_sampler_without_class_weights() -> None:
    config = load_yaml("configs/experiments/sprint4k_image_balanced_sampler_backbones.yaml")[
        "finetuning"
    ]

    assert config["feature_source"] == "finetuned_balanced_sampler"
    assert config["class_weighting"] is False
    assert config["sampler"]["name"] == "class_balanced"
    assert config["unfreeze_policies"]["resnet50"] == "layer3_layer4"
    assert config["backbone_learning_rate"] < config["head_learning_rate"]
