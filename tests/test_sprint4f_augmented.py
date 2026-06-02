"""Sprint 4F augmented fine-tuning extension tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from torchvision import transforms

from dl_midterm.config.load_config import load_yaml
from dl_midterm.data.transforms import build_image_transform
from dl_midterm.features.cache import backbone_cache_dir
from dl_midterm.features.extract import select_single_backbone_combinations


def _load_run_matrix_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sprint4f_augmented_config_uses_separate_feature_source() -> None:
    config = load_yaml("configs/experiments/sprint4f_augmented_backbones.yaml")["finetuning"]

    assert config["feature_source"] == "finetuned_augmented"
    assert config["checkpoint_dir"].endswith("finetuned_augmented_backbones")
    assert config["backbones"] == ["resnet50", "mobilenet_v2", "efficientnet_b0"]
    assert config["augmentation"]["random_resized_crop"]["scale"] == [0.85, 1.0]


def test_sprint4f_train_transform_uses_configured_augmentation() -> None:
    config = load_yaml("configs/experiments/sprint4f_augmented_backbones.yaml")["finetuning"]
    transform = build_image_transform(
        image_size=224,
        train=True,
        augmentation=config["augmentation"],
    )
    step_types = {type(step) for step in transform.transforms}

    assert transforms.RandomResizedCrop in step_types
    assert transforms.RandomHorizontalFlip in step_types
    assert transforms.RandomVerticalFlip in step_types
    assert transforms.RandomRotation in step_types
    assert transforms.RandomAffine in step_types
    assert transforms.ColorJitter in step_types


def test_feature_extraction_transform_remains_deterministic() -> None:
    config = load_yaml("configs/experiments/sprint4f_augmented_backbones.yaml")["finetuning"]
    transform = build_image_transform(
        image_size=224,
        train=False,
        augmentation=config["augmentation"],
    )
    step_types = [type(step) for step in transform.transforms]

    assert step_types[0] is transforms.Resize
    assert transforms.RandomResizedCrop not in step_types
    assert transforms.RandomRotation not in step_types
    assert transforms.RandomAffine not in step_types


def test_sprint4f_feature_source_path_expansion() -> None:
    path = backbone_cache_dir(
        "artifacts/features",
        "ham10000",
        "finetuned_augmented",
        "efficientnet_b0",
    )

    assert str(path) == "artifacts/features/ham10000/finetuned_augmented/efficientnet_b0"


def test_sprint4f_small_matrix_has_three_singles_and_two_three_backbone_fusions() -> None:
    run_matrix = _load_run_matrix_module()
    config = load_yaml("configs/experiments/sprint4f_augmented_feature_matrix.yaml")[
        "experiment_matrix"
    ]

    singles = select_single_backbone_combinations(config["combinations"])
    fusions = run_matrix.expand_fusion_run_matrix(config, feature_source="finetuned_augmented")

    assert singles == ["resnet50", "mobilenet_v2", "efficientnet_b0"]
    assert len(fusions) == 2
    assert {run["fusion_method"] for run in fusions} == {"concat", "weighted"}
    assert {run["feature_source"] for run in fusions} == {"finetuned_augmented"}
    assert all(len(run["backbones"]) == 3 for run in fusions)


def test_sprint4f_colab_mirror_does_not_exclude_model_artifacts() -> None:
    notebook = json.loads(
        Path("notebooks/04_sprint4f_augmented_finetuning.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
        if isinstance(line, str)
    )

    assert "--exclude" not in source
    assert "model.pt" in source
    assert "finetuned_augmented_backbones" in source
    assert "finetuned_augmented" in source
