"""Image preprocessing and augmentation transforms."""

from __future__ import annotations

from typing import Any

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(
    image_size: int = 224,
    train: bool = False,
    augmentation: dict[str, Any] | None = None,
) -> transforms.Compose:
    """Build TorchVision preprocessing for ImageNet-pretrained backbones.

    Feature extraction intentionally uses deterministic resizing instead of
    stochastic augmentation so cached vectors are reproducible across runs.
    """

    if train:
        policy = augmentation or _default_train_augmentation()
        steps = _spatial_steps(image_size, policy)
        if bool(policy.get("horizontal_flip", False)):
            steps.append(
                transforms.RandomHorizontalFlip(p=float(policy.get("horizontal_flip_p", 0.5)))
            )
        if bool(policy.get("vertical_flip", False)):
            steps.append(
                transforms.RandomVerticalFlip(p=float(policy.get("vertical_flip_p", 0.5)))
            )
        rotation_degrees = float(policy.get("rotation_degrees", 0) or 0)
        if rotation_degrees > 0:
            steps.append(transforms.RandomRotation(degrees=rotation_degrees))
        affine = policy.get("random_affine")
        if affine:
            steps.append(_build_random_affine(affine))
        color_jitter = _build_color_jitter(policy.get("color_jitter"))
        if color_jitter is not None:
            steps.append(color_jitter)
    else:
        steps = [transforms.Resize((image_size, image_size), antialias=True)]

    return transforms.Compose(
        steps
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _default_train_augmentation() -> dict[str, Any]:
    return {
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotation_degrees": 20,
        "color_jitter": "mild",
    }


def _spatial_steps(image_size: int, policy: dict[str, Any]) -> list[transforms.Transform]:
    crop = policy.get("random_resized_crop")
    if crop:
        if isinstance(crop, dict):
            scale = tuple(float(value) for value in crop.get("scale", [0.85, 1.0]))
            ratio = tuple(float(value) for value in crop.get("ratio", [0.9, 1.1]))
        else:
            scale = (0.85, 1.0)
            ratio = (0.9, 1.1)
        return [
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=ratio,
                antialias=True,
            )
        ]
    return [transforms.Resize((image_size, image_size), antialias=True)]


def _build_random_affine(config: Any) -> transforms.RandomAffine:
    if isinstance(config, dict):
        degrees = float(config.get("degrees", 0))
        translate = config.get("translate")
        scale = config.get("scale")
        if translate is not None:
            translate = tuple(float(value) for value in translate)
        if scale is not None:
            scale = tuple(float(value) for value in scale)
    else:
        degrees = 0.0
        translate = (0.03, 0.03)
        scale = (0.95, 1.05)
    return transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale)


def _build_color_jitter(config: Any) -> transforms.ColorJitter | None:
    if not config:
        return None
    if config == "mild":
        return transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08,
            saturation=0.05,
            hue=0.02,
        )
    if config == "moderate":
        return transforms.ColorJitter(
            brightness=0.12,
            contrast=0.12,
            saturation=0.08,
            hue=0.02,
        )
    if isinstance(config, dict):
        return transforms.ColorJitter(
            brightness=float(config.get("brightness", 0.0)),
            contrast=float(config.get("contrast", 0.0)),
            saturation=float(config.get("saturation", 0.0)),
            hue=float(config.get("hue", 0.0)),
        )
    raise ValueError(f"Unsupported color_jitter augmentation config: {config!r}")
