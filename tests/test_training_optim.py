"""Optimizer construction tests."""

from __future__ import annotations

import pytest
import torch

from dl_midterm.training.optim import build_optimizer


def test_build_supported_optimizers() -> None:
    model = torch.nn.Linear(2, 2)

    assert isinstance(
        build_optimizer(
            model.parameters(),
            optimizer_name="adamw",
            learning_rate=0.001,
            weight_decay=0.0001,
        ),
        torch.optim.AdamW,
    )
    assert isinstance(
        build_optimizer(
            model.parameters(),
            optimizer_name="adam",
            learning_rate=0.001,
            weight_decay=0.0001,
        ),
        torch.optim.Adam,
    )
    assert isinstance(
        build_optimizer(
            model.parameters(),
            optimizer_name="sgd",
            learning_rate=0.01,
            weight_decay=0.0001,
        ),
        torch.optim.SGD,
    )


def test_build_optimizer_rejects_unknown_name() -> None:
    model = torch.nn.Linear(2, 2)

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        build_optimizer(
            model.parameters(),
            optimizer_name="rmsprop",
            learning_rate=0.001,
            weight_decay=0.0001,
        )
