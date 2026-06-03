"""Sprint 4D TTA protocol tests."""

from __future__ import annotations

import torch

from dl_midterm.config.load_config import load_yaml
from dl_midterm.evaluation.tta import (
    average_probabilities,
    choose_tta_policy,
    expand_tta_policy,
)


def test_tta_policy_expansion_is_deterministic_and_conservative() -> None:
    assert expand_tta_policy("identity") == ["identity"]
    assert expand_tta_policy("tta_hflip2") == ["identity", "hflip"]
    assert expand_tta_policy("tta_vflip2") == ["identity", "vflip"]
    assert expand_tta_policy("tta_flip4") == ["identity", "hflip", "vflip", "hvflip"]
    assert expand_tta_policy("tta_rot4") == ["identity", "rot90", "rot180", "rot270"]
    assert expand_tta_policy("tta_d4_8") == [
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "hflip",
        "hflip_rot90",
        "hflip_rot180",
        "hflip_rot270",
    ]


def test_average_probabilities_preserves_shape_and_normalization() -> None:
    first = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    second = torch.tensor([[0.6, 0.4], [0.2, 0.8]])

    averaged = average_probabilities([first, second])

    assert tuple(averaged.shape) == (2, 2)
    assert torch.allclose(averaged.sum(dim=1), torch.ones(2))
    assert torch.allclose(averaged, torch.tensor([[0.7, 0.3], [0.3, 0.7]]))


def test_tta_gate_uses_validation_metrics_without_test_fields() -> None:
    rows = [
        {
            "model_key": "canonical_concat",
            "policy": "identity",
            "macro_f1": 0.700,
            "accuracy": 0.810,
            "weighted_f1": 0.820,
            "test_macro_f1": 0.999,
        },
        {
            "model_key": "canonical_concat",
            "policy": "tta_flip4",
            "macro_f1": 0.706,
            "accuracy": 0.809,
            "weighted_f1": 0.819,
            "test_macro_f1": 0.100,
        },
    ]

    decision = choose_tta_policy(
        rows,
        model_key="canonical_concat",
        min_macro_f1_gain=0.005,
        max_accuracy_drop=0.020,
        max_weighted_f1_drop=0.020,
    )

    assert decision.selected_policy == "tta_flip4"
    assert decision.test_allowed
    assert decision.val_macro_f1_gain == 0.006000000000000005


def test_tta_gate_blocks_tiny_validation_gain() -> None:
    rows = [
        {
            "model_key": "canonical_concat",
            "policy": "identity",
            "macro_f1": 0.700,
            "accuracy": 0.810,
            "weighted_f1": 0.820,
        },
        {
            "model_key": "canonical_concat",
            "policy": "tta_flip4",
            "macro_f1": 0.703,
            "accuracy": 0.820,
            "weighted_f1": 0.830,
        },
    ]

    decision = choose_tta_policy(
        rows,
        model_key="canonical_concat",
        min_macro_f1_gain=0.005,
        max_accuracy_drop=0.020,
        max_weighted_f1_drop=0.020,
    )

    assert not decision.test_allowed
    assert "below" in decision.reason


def test_sprint4d_config_pre_registers_only_two_test_eligible_models() -> None:
    config = load_yaml("configs/experiments/sprint4d_tta.yaml")["sprint4d"]

    eligible = [
        key for key, model in config["models"].items() if bool(model.get("test_eligible", False))
    ]

    assert eligible == ["canonical_concat", "sprint4c_weighted"]
    assert not config["models"]["resnet50_diagnostic"]["test_eligible"]
    assert config["models"]["canonical_concat"]["validation_policies"] == [
        "identity",
        "tta_flip4",
        "tta_rot4",
    ]


def test_sprint4i_config_pre_registers_geometry_safe_tta_refinement() -> None:
    config = load_yaml("configs/experiments/sprint4i_geometry_safe_tta.yaml")["sprint4i"]

    eligible = [
        key for key, model in config["models"].items() if bool(model.get("test_eligible", False))
    ]

    assert eligible == ["canonical_concat", "sprint4c_weighted"]
    assert config["artifact_prefix"] == "sprint4i"
    assert config["gate"]["min_macro_f1_gain"] == 0.005
    for model in config["models"].values():
        assert model["validation_policies"] == [
            "identity",
            "tta_hflip2",
            "tta_vflip2",
            "tta_flip4",
            "tta_rot4",
            "tta_d4_8",
        ]
