"""Image preprocessing and augmentation transforms."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    """Build TorchVision preprocessing for ImageNet-pretrained backbones.

    Feature extraction intentionally uses deterministic resizing instead of
    stochastic augmentation so cached vectors are reproducible across runs.
    """

    if train:
        steps = [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
        ]
    else:
        steps = [transforms.Resize((image_size, image_size), antialias=True)]

    return transforms.Compose(
        steps
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
