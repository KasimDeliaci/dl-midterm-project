"""Sprint 4B class-aware fine-tuning extension tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from dl_midterm.config.load_config import load_yaml
from dl_midterm.features.cache import (
    backbone_cache_dir,
    cache_allows_prefix_split_verification,
    save_feature_cache,
    verify_cache_matches_split,
)
from dl_midterm.features.extract import select_single_backbone_combinations
from dl_midterm.models.backbones import apply_finetuning_policy, build_classification_backbone
from dl_midterm.training.losses import ClassBalancedFocalLoss, class_weights_from_labels


def _load_run_matrix_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_class_balanced_focal_loss_is_finite_and_backpropagates() -> None:
    logits = torch.tensor(
        [[2.0, 0.1, -0.4], [0.2, 1.5, -0.5], [-0.2, 0.3, 1.7]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 2])
    weights = torch.tensor([1.0, 1.5, 2.0])
    loss = ClassBalancedFocalLoss(gamma=1.0, class_weights=weights)(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_class_weight_smoothing_clipping_and_normalization() -> None:
    labels = [0] * 8 + [1] * 2 + [2]
    raw = class_weights_from_labels(labels, 3)
    smoothed = class_weights_from_labels(
        labels,
        3,
        smoothing="sqrt",
        max_weight=2.0,
        normalize=True,
    )

    assert raw[2] > raw[1] > raw[0]
    assert smoothed[2] > smoothed[1] > smoothed[0]
    assert float(smoothed.max()) < float(raw.max())
    assert torch.isclose(smoothed.mean(), torch.tensor(1.0), atol=1e-6)


def test_canonical_sprint4_defaults_remain_cross_entropy() -> None:
    config = load_yaml("configs/experiments/finetune_backbones.yaml")["finetuning"]

    assert config["feature_source"] == "finetuned"
    assert config["unfreeze_policies"]["resnet50"] == "layer4"
    assert config.get("loss", {"name": "cross_entropy"})["name"] == "cross_entropy"


def test_finetuned_classaware_feature_source_path_expansion() -> None:
    path = backbone_cache_dir(
        "artifacts/features",
        "ham10000",
        "finetuned_classaware",
        "resnet50",
    )

    assert str(path) == "artifacts/features/ham10000/finetuned_classaware/resnet50"


def test_finetuned_deeper_resnet50_unfreeze_policy() -> None:
    model = build_classification_backbone("resnet50", num_classes=7, pretrained=False)
    summary = apply_finetuning_policy(model, "resnet50", policy="layer3_layer4")
    params = dict(model.named_parameters())

    assert summary["policy"] == "layer3_layer4"
    assert params["layer3.0.conv1.weight"].requires_grad
    assert params["layer4.0.conv1.weight"].requires_grad
    assert params["fc.weight"].requires_grad
    assert not params["layer2.0.conv1.weight"].requires_grad


def test_sprint4b_screening_matrix_selects_only_single_backbones() -> None:
    config = load_yaml("configs/experiments/sprint4b_classaware_feature_matrix.yaml")[
        "experiment_matrix"
    ]

    singles = select_single_backbone_combinations(config["combinations"])

    assert singles == ["resnet50", "mobilenet_v2", "efficientnet_b0"]


def test_sprint4b_optional_full_classaware_matrix_has_eleven_runs() -> None:
    run_matrix = _load_run_matrix_module()
    config = load_yaml("configs/experiments/sprint4b_classaware_feature_matrix.yaml")[
        "experiment_matrix"
    ]

    singles = select_single_backbone_combinations(config["combinations"])
    fusions = run_matrix.expand_fusion_run_matrix(config, feature_source="finetuned_classaware")

    assert len(singles) == 3
    assert len(fusions) == 8
    assert len(singles) + len(fusions) == 11
    assert {run["feature_source"] for run in fusions} == {"finetuned_classaware"}


def test_sprint4j_balanced_sampler_matrix_is_targeted_to_three_backbone_fusion() -> None:
    run_matrix = _load_run_matrix_module()
    config = load_yaml("configs/experiments/sprint4j_balanced_sampler_mlp.yaml")[
        "experiment_matrix"
    ]

    fusions = run_matrix.expand_fusion_run_matrix(config, feature_source="finetuned")

    assert len(fusions) == 2
    assert {run["fusion_method"] for run in fusions} == {"concat", "weighted"}
    assert all(
        run["backbones"] == ["resnet50", "mobilenet_v2", "efficientnet_b0"]
        for run in fusions
    )


def test_limited_smoke_cache_can_verify_split_prefix(tmp_path: Path) -> None:
    split_csv = tmp_path / "train.csv"
    split_csv.write_text(
        "image_id,label,image_path\n"
        "img1,nv,/tmp/img1.jpg\n"
        "img2,mel,/tmp/img2.jpg\n"
        "img3,bcc,/tmp/img3.jpg\n",
        encoding="utf-8",
    )
    cache = save_feature_cache(
        tmp_path / "train.pt",
        features=torch.ones(2, 4),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="train",
        backbone="resnet50",
        class_names=["nv", "mel", "bcc"],
        feature_source="finetuned_classaware",
        seed=42,
        config={"limit_per_split": 2},
    )

    assert cache_allows_prefix_split_verification(cache)
    verify_cache_matches_split(cache, split_csv, allow_prefix=True)
