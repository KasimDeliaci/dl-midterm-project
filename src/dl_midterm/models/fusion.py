"""Concatenation and weighted feature fusion modules."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dl_midterm.models.backbones import expected_feature_dim
from dl_midterm.models.mlp import FeatureMLP


def expected_concat_dim(backbones: Sequence[str]) -> int:
    """Return the concatenated feature width for a backbone combination."""

    if len(backbones) < 2:
        raise ValueError("Fusion requires at least two backbones.")
    return sum(expected_feature_dim(backbone) for backbone in backbones)


class ConcatenationFusion(nn.Module):
    """Concatenate feature tensors along their feature dimension."""

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) < 2:
            raise ValueError("Concatenation fusion requires at least two feature tensors.")
        sample_counts = {int(feature.shape[0]) for feature in features}
        if len(sample_counts) != 1:
            raise ValueError("All feature tensors must have the same batch size.")
        return torch.cat(list(features), dim=1)


class ProjectedWeightedFusion(nn.Module):
    """Project each feature vector and combine it with global softmax weights."""

    def __init__(self, input_dims: Sequence[int], projection_dim: int = 512) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("Weighted fusion requires at least two input dimensions.")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.projection_dim = int(projection_dim)
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, self.projection_dim) for input_dim in self.input_dims
        )
        self.logits = nn.Parameter(torch.zeros(len(self.input_dims)))

    def normalized_weights(self) -> torch.Tensor:
        """Return softmax-normalized global fusion weights."""

        return torch.softmax(self.logits, dim=0)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.input_dims):
            raise ValueError(
                f"Expected {len(self.input_dims)} feature tensors, got {len(features)}."
            )
        projected = []
        for index, (feature, projection, expected_dim) in enumerate(
            zip(features, self.projections, self.input_dims, strict=True)
        ):
            if int(feature.shape[1]) != expected_dim:
                raise ValueError(
                    f"Feature tensor {index} has dim {int(feature.shape[1])}, "
                    f"expected {expected_dim}."
                )
            projected.append(projection(feature))
        stacked = torch.stack(projected, dim=1)
        weights = self.normalized_weights().view(1, -1, 1)
        return (stacked * weights).sum(dim=1)


class WeightedFusionMLP(nn.Module):
    """Classifier that splits concatenated cached features, fuses them, then runs an MLP."""

    def __init__(
        self,
        input_dims: Sequence[int],
        num_classes: int,
        *,
        projection_dim: int = 512,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.fusion = ProjectedWeightedFusion(self.input_dims, projection_dim=projection_dim)
        self.classifier = FeatureMLP(
            input_dim=projection_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parts = torch.split(features, self.input_dims, dim=1)
        fused = self.fusion(parts)
        return self.classifier(fused)

    def normalized_weights(self) -> torch.Tensor:
        """Return the learned softmax weights from the fusion layer."""

        return self.fusion.normalized_weights()
