"""Loss helpers for image-level fine-tuning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def class_weights_from_labels(
    labels: list[int] | torch.Tensor,
    num_classes: int,
    *,
    smoothing: str | float | bool | None = None,
    max_weight: float | None = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from train labels only."""

    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    weights[counts == 0] = 0.0
    weights = smooth_class_weights(weights, smoothing=smoothing)
    if max_weight is not None:
        weights = weights.clamp(max=float(max_weight))
    if normalize:
        positive = weights > 0
        if bool(positive.any()):
            weights = weights.clone()
            weights[positive] = weights[positive] / weights[positive].mean()
    return weights


def smooth_class_weights(
    weights: torch.Tensor,
    *,
    smoothing: str | float | bool | None = None,
) -> torch.Tensor:
    """Apply optional class-weight smoothing for imbalance-aware losses."""

    if smoothing in {None, False, "none"}:
        return weights
    if smoothing is True:
        return weights.sqrt()
    if isinstance(smoothing, str):
        normalized = smoothing.lower()
        if normalized == "sqrt":
            return weights.sqrt()
        if normalized == "log":
            return torch.log1p(weights)
        raise ValueError("Unsupported class-weight smoothing. Expected none, sqrt, or log.")
    exponent = float(smoothing)
    if not 0.0 < exponent <= 1.0:
        raise ValueError("Numeric class-weight smoothing must be in the range (0, 1].")
    return weights.pow(exponent)


class ClassBalancedFocalLoss(nn.Module):
    """Focal loss with optional train-split class weights."""

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        class_weights: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("Focal gamma must be non-negative.")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("Reduction must be one of: mean, sum, none.")
        self.gamma = float(gamma)
        self.reduction = reduction
        if class_weights is None:
            self.register_buffer("class_weights", None)
        else:
            self.register_buffer("class_weights", class_weights.detach().float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        ce = F.nll_loss(
            log_probs,
            targets,
            weight=self.class_weights,
            reduction="none",
        )
        loss = ((1.0 - pt).pow(self.gamma)) * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_classification_loss(
    *,
    labels: list[int],
    num_classes: int,
    class_weighting: bool,
    loss_config: dict[str, Any] | None,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the configured image-classification loss and metadata."""

    config = loss_config or {}
    loss_name = str(config.get("name", "cross_entropy")).lower()
    smoothing = config.get("weight_smoothing")
    max_weight = config.get("max_class_weight")
    normalize = bool(config.get("normalize_class_weights", False))
    weights = (
        class_weights_from_labels(
            labels,
            num_classes,
            smoothing=smoothing,
            max_weight=float(max_weight) if max_weight is not None else None,
            normalize=normalize,
        ).to(device)
        if class_weighting
        else None
    )

    metadata = {
        "name": loss_name,
        "class_weighting": class_weighting,
        "weight_smoothing": smoothing,
        "max_class_weight": max_weight,
        "normalize_class_weights": normalize,
        "class_weights": weights.detach().cpu().tolist() if weights is not None else None,
    }
    if loss_name in {"cross_entropy", "weighted_cross_entropy"}:
        return nn.CrossEntropyLoss(weight=weights), metadata
    if loss_name in {"focal", "class_balanced_focal"}:
        gamma = float(config.get("gamma", 1.0))
        metadata["gamma"] = gamma
        return ClassBalancedFocalLoss(gamma=gamma, class_weights=weights), metadata
    raise ValueError(
        "Unsupported fine-tuning loss. Expected cross_entropy, weighted_cross_entropy, "
        "focal, or class_balanced_focal."
    )
