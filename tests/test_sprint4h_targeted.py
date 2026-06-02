"""Sprint 4H targeted fine-tuning extension tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from torchvision import transforms

from dl_midterm.config.load_config import load_yaml
from dl_midterm.data.transforms import build_image_transform
from dl_midterm.features.cache import backbone_cache_dir
from dl_midterm.features.extract import select_single_backbone_combinations
from dl_midterm.models.backbones import apply_finetuning_policy, build_classification_backbone


def _load_run_matrix_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sprint4h_config_uses_targeted_feature_source_and_deeper_resnet() -> None:
    config = load_yaml("configs/experiments/sprint4h_targeted_backbones.yaml")["finetuning"]

    assert config["feature_source"] == "finetuned_targeted"
    assert config["checkpoint_dir"].endswith("finetuned_targeted_backbones")
    assert config["unfreeze_policies"]["resnet50"] == "layer3_layer4"
    assert config["unfreeze_policies"]["mobilenet_v2"] == "last_feature_blocks"
    assert config["unfreeze_policies"]["efficientnet_b0"] == "last_feature_blocks"
    assert config["loss"]["name"] == "class_balanced_focal"


def test_sprint4h_train_transform_is_crop_free_and_tta_compatible() -> None:
    config = load_yaml("configs/experiments/sprint4h_targeted_backbones.yaml")["finetuning"]
    transform = build_image_transform(
        image_size=224,
        train=True,
        augmentation=config["augmentation"],
    )
    step_types = {type(step) for step in transform.transforms}

    assert transforms.RandomResizedCrop not in step_types
    assert transforms.RandomAffine not in step_types
    assert transforms.RandomHorizontalFlip in step_types
    assert transforms.RandomVerticalFlip in step_types
    assert transforms.RandomRotation in step_types
    assert transforms.ColorJitter in step_types


def test_sprint4h_resnet_policy_unfreezes_layer3_and_layer4() -> None:
    model = build_classification_backbone("resnet50", num_classes=7, pretrained=False)
    summary = apply_finetuning_policy(model, "resnet50", policy="layer3_layer4")
    params = dict(model.named_parameters())

    assert summary["policy"] == "layer3_layer4"
    assert params["layer3.0.conv1.weight"].requires_grad
    assert params["layer4.0.conv1.weight"].requires_grad
    assert params["fc.weight"].requires_grad
    assert not params["layer2.0.conv1.weight"].requires_grad


def test_sprint4h_feature_source_path_expansion() -> None:
    path = backbone_cache_dir(
        "artifacts/features",
        "ham10000",
        "finetuned_targeted",
        "resnet50",
    )

    assert str(path) == "artifacts/features/ham10000/finetuned_targeted/resnet50"


def test_sprint4h_full_matrix_has_eleven_runs() -> None:
    run_matrix = _load_run_matrix_module()
    config = load_yaml("configs/experiments/sprint4h_targeted_feature_matrix.yaml")[
        "experiment_matrix"
    ]

    singles = select_single_backbone_combinations(config["combinations"])
    fusions = run_matrix.expand_fusion_run_matrix(config, feature_source="finetuned_targeted")

    assert singles == ["resnet50", "mobilenet_v2", "efficientnet_b0"]
    assert len(fusions) == 8
    assert len(singles) + len(fusions) == 11
    assert {run["feature_source"] for run in fusions} == {"finetuned_targeted"}


def test_sprint4h_colab_mirror_preserves_model_artifacts() -> None:
    notebook = json.loads(
        Path("notebooks/04_sprint4h_targeted_finetuning.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
        if isinstance(line, str)
    )

    assert "--exclude" not in source
    assert "model.pt" in source
    assert "finetuned_targeted_backbones" in source
    assert "finetuned_targeted" in source
