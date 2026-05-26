"""Classifier-head removal and feature-extractor wrappers."""

from __future__ import annotations

import torch
from torch import nn


class FrozenFeatureExtractor(nn.Module):
    """Classifier-free CNN feature extractor with frozen weights."""

    def __init__(self, name: str, features: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.name = name
        self.features = features
        self.feature_dim = feature_dim
        self.freeze()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that still require gradients."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
