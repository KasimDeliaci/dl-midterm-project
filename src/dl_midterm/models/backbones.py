"""Backbone loading, classifier removal, and freeze/unfreeze policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
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


def build_classification_backbone(
    name: str,
    *,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """Build an ImageNet-pretrained backbone with a temporary classification head."""

    normalized = name.lower()
    if normalized == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif normalized == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif normalized == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(
            f"Unsupported backbone {name!r}. Supported: {', '.join(supported_backbones())}"
        )
    return model


def apply_finetuning_policy(
    model: nn.Module,
    name: str,
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    """Freeze all parameters, then unfreeze the final meaningful CNN blocks and head."""

    normalized = name.lower()
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_modules: list[tuple[str, nn.Module]]
    if normalized == "resnet50":
        if policy not in {None, "layer4", "layer3_layer4"}:
            raise ValueError("ResNet50 fine-tuning policy must be `layer4` or `layer3_layer4`.")
        if policy == "layer3_layer4":
            trainable_modules = [
                ("layer3", model.layer3),
                ("layer4", model.layer4),
                ("fc", model.fc),
            ]
            resolved_policy = "layer3_layer4"
        else:
            trainable_modules = [("layer4", model.layer4), ("fc", model.fc)]
            resolved_policy = "layer4"
    elif normalized == "mobilenet_v2":
        if policy not in {None, "last_feature_blocks"}:
            raise ValueError("MobileNetV2 fine-tuning policy must be `last_feature_blocks`.")
        trainable_modules = [
            ("features.16", model.features[16]),
            ("features.17", model.features[17]),
            ("features.18", model.features[18]),
            ("classifier", model.classifier),
        ]
        resolved_policy = "last_feature_blocks"
    elif normalized == "efficientnet_b0":
        if policy not in {None, "last_feature_blocks"}:
            raise ValueError("EfficientNetB0 fine-tuning policy must be `last_feature_blocks`.")
        trainable_modules = [
            ("features.7", model.features[7]),
            ("features.8", model.features[8]),
            ("classifier", model.classifier),
        ]
        resolved_policy = "last_feature_blocks"
    else:
        raise ValueError(
            f"Unsupported backbone {name!r}. Supported: {', '.join(supported_backbones())}"
        )

    for _, module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    return {
        "policy": resolved_policy,
        "trainable_modules": [module_name for module_name, _ in trainable_modules],
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_fraction": trainable_params / max(total_params, 1),
    }


def build_finetuned_feature_extractor(
    name: str,
    *,
    checkpoint_path: str | Path,
    num_classes: int,
) -> FrozenFeatureExtractor:
    """Build a classifier-free feature extractor from a fine-tuned checkpoint."""

    normalized = name.lower()
    model = build_classification_backbone(normalized, num_classes=num_classes, pretrained=False)
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    if normalized == "resnet50":
        features = nn.Sequential(*list(model.children())[:-1], nn.Flatten(start_dim=1))
    elif normalized in {"mobilenet_v2", "efficientnet_b0"}:
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


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary payload: {checkpoint_path}")
    return checkpoint
