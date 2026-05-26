"""Backbone loading, classifier removal, and freeze/unfreeze policies."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torchvision import models

from dl_midterm.models.feature_extractors import FrozenFeatureExtractor


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    feature_dim: int
    alias: str


BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "resnet50": BackboneSpec(name="resnet50", feature_dim=2048, alias="r50"),
    "mobilenet_v2": BackboneSpec(name="mobilenet_v2", feature_dim=1280, alias="mnv2"),
    "efficientnet_b0": BackboneSpec(name="efficientnet_b0", feature_dim=1280, alias="effb0"),
}


def supported_backbones() -> list[str]:
    """Return supported Sprint 2 backbone names."""

    return list(BACKBONE_SPECS)


def backbone_alias(name: str) -> str:
    """Return the compact run-ID alias for a backbone."""

    return _spec(name).alias


def expected_feature_dim(name: str) -> int:
    """Return the expected classifier-free feature vector width."""

    return _spec(name).feature_dim


def build_frozen_feature_extractor(
    name: str,
    pretrained: bool = True,
) -> FrozenFeatureExtractor:
    """Build a frozen ImageNet-pretrained classifier-free backbone."""

    normalized = name.lower()
    if normalized == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        features = nn.Sequential(*list(model.children())[:-1], nn.Flatten(start_dim=1))
    elif normalized == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        features = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
    elif normalized == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        features = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )
    else:
        raise ValueError(
            f"Unsupported backbone {name!r}. Supported: {', '.join(supported_backbones())}"
        )

    return FrozenFeatureExtractor(
        name=normalized,
        features=features,
        feature_dim=expected_feature_dim(normalized),
    )


def _spec(name: str) -> BackboneSpec:
    try:
        return BACKBONE_SPECS[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone {name!r}. Supported: {', '.join(supported_backbones())}"
        ) from exc
