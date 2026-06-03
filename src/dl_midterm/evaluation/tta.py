"""Test-time augmentation helpers for Sprint 4D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

TTA_VIEW_REGISTRY: dict[str, list[str]] = {
    "identity": ["identity"],
    "tta_hflip2": ["identity", "hflip"],
    "tta_vflip2": ["identity", "vflip"],
    "tta_hvflip2": ["identity", "hvflip"],
    "tta_flip4": ["identity", "hflip", "vflip", "hvflip"],
    "tta_rot4": ["identity", "rot90", "rot180", "rot270"],
    "tta_d4_8": [
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "hflip",
        "hflip_rot90",
        "hflip_rot180",
        "hflip_rot270",
    ],
}


@dataclass(frozen=True)
class TTAGateDecision:
    """Validation-gated decision for one model's TTA policy."""

    model_key: str
    identity_policy: str
    selected_policy: str
    identity_val_macro_f1: float
    selected_val_macro_f1: float
    val_macro_f1_gain: float
    accuracy_gain: float
    weighted_f1_gain: float
    test_allowed: bool
    reason: str


def expand_tta_policy(policy_name: str) -> list[str]:
    """Return deterministic TTA views for a registered policy."""

    try:
        return list(TTA_VIEW_REGISTRY[policy_name])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported TTA policy {policy_name!r}. "
            f"Supported policies: {', '.join(sorted(TTA_VIEW_REGISTRY))}"
        ) from exc


def average_probabilities(view_probabilities: list[torch.Tensor]) -> torch.Tensor:
    """Average per-view probability tensors and preserve probability normalization."""

    if not view_probabilities:
        raise ValueError("At least one view probability tensor is required.")
    reference_shape = tuple(view_probabilities[0].shape)
    if len(reference_shape) != 2:
        raise ValueError("View probabilities must be 2D tensors.")
    for probabilities in view_probabilities:
        if tuple(probabilities.shape) != reference_shape:
            raise ValueError("All view probability tensors must have the same shape.")
    averaged = torch.stack(view_probabilities, dim=0).mean(dim=0)
    row_sums = averaged.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return averaged / row_sums


def choose_tta_policy(
    validation_rows: list[dict[str, Any]],
    *,
    model_key: str,
    min_macro_f1_gain: float,
    max_accuracy_drop: float,
    max_weighted_f1_drop: float,
) -> TTAGateDecision:
    """Choose a TTA policy using validation metrics only."""

    model_rows = [row for row in validation_rows if row["model_key"] == model_key]
    if not model_rows:
        raise ValueError(f"No validation rows found for model {model_key!r}.")
    identity_rows = [row for row in model_rows if row["policy"] == "identity"]
    if len(identity_rows) != 1:
        raise ValueError(f"Expected exactly one identity row for model {model_key!r}.")
    identity = identity_rows[0]
    candidates = [row for row in model_rows if row["policy"] != "identity"]
    if not candidates:
        return TTAGateDecision(
            model_key=model_key,
            identity_policy="identity",
            selected_policy="identity",
            identity_val_macro_f1=float(identity["macro_f1"]),
            selected_val_macro_f1=float(identity["macro_f1"]),
            val_macro_f1_gain=0.0,
            accuracy_gain=0.0,
            weighted_f1_gain=0.0,
            test_allowed=False,
            reason="No non-identity TTA policies were evaluated on validation.",
        )

    best = sorted(
        candidates,
        key=lambda row: (
            float(row["macro_f1"]),
            1 if row["policy"] == "tta_flip4" else 0,
            str(row["policy"]),
        ),
        reverse=True,
    )[0]
    macro_gain = float(best["macro_f1"]) - float(identity["macro_f1"])
    accuracy_gain = float(best["accuracy"]) - float(identity["accuracy"])
    weighted_gain = float(best["weighted_f1"]) - float(identity["weighted_f1"])
    test_allowed = True
    reasons: list[str] = []
    if macro_gain < min_macro_f1_gain:
        test_allowed = False
        reasons.append(
            f"validation macro-F1 gain {macro_gain:+.4f} below {min_macro_f1_gain:.4f}"
        )
    if accuracy_gain < -max_accuracy_drop:
        test_allowed = False
        reasons.append(
            f"validation accuracy drop {accuracy_gain:+.4f} exceeds {max_accuracy_drop:.4f}"
        )
    if weighted_gain < -max_weighted_f1_drop:
        test_allowed = False
        reasons.append(
            f"validation weighted-F1 drop {weighted_gain:+.4f} exceeds "
            f"{max_weighted_f1_drop:.4f}"
        )
    if not reasons:
        reasons.append("Validation gate passed.")
    return TTAGateDecision(
        model_key=model_key,
        identity_policy="identity",
        selected_policy=str(best["policy"]),
        identity_val_macro_f1=float(identity["macro_f1"]),
        selected_val_macro_f1=float(best["macro_f1"]),
        val_macro_f1_gain=macro_gain,
        accuracy_gain=accuracy_gain,
        weighted_f1_gain=weighted_gain,
        test_allowed=test_allowed,
        reason="; ".join(reasons),
    )
