"""PyTorch dataloader construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dl_midterm.data.datasets import HAM10000ImageDataset
from dl_midterm.data.transforms import build_image_transform


def create_image_dataloader(
    split_csv: str | Path,
    class_names: list[str],
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = False,
    split_name: str | None = None,
    max_samples: int | None = None,
    train: bool = False,
    augmentation: dict[str, Any] | None = None,
    sampler_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> DataLoader:
    """Create an image dataloader for feature extraction or image-level training."""

    dataset = HAM10000ImageDataset(
        split_csv=split_csv,
        class_names=class_names,
        transform=build_image_transform(
            image_size=image_size,
            train=train,
            augmentation=augmentation,
        ),
        split_name=split_name,
        max_samples=max_samples,
    )
    sampler = _build_sampler(dataset, sampler_config=sampler_config, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_finetuning_loaders(
    splits_dir: str | Path,
    class_names: list[str],
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    max_samples_per_split: int | None = None,
    augmentation: dict[str, Any] | None = None,
    train_sampler: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """Create train/validation/test dataloaders for image-level fine-tuning."""

    split_root = Path(splits_dir)
    return {
        "train": create_image_dataloader(
            split_root / "train.csv",
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            split_name="train",
            max_samples=max_samples_per_split,
            train=True,
            augmentation=augmentation,
            sampler_config=train_sampler,
            seed=seed,
        ),
        "val": create_image_dataloader(
            split_root / "val.csv",
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            split_name="val",
            max_samples=max_samples_per_split,
            train=False,
        ),
        "test": create_image_dataloader(
            split_root / "test.csv",
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            split_name="test",
            max_samples=max_samples_per_split,
            train=False,
        ),
    }


def _build_sampler(
    dataset: HAM10000ImageDataset,
    *,
    sampler_config: dict[str, Any] | None,
    seed: int,
) -> WeightedRandomSampler | None:
    if not sampler_config or str(sampler_config.get("name", "none")) == "none":
        return None
    name = str(sampler_config.get("name"))
    if name != "class_balanced":
        raise ValueError(f"Unsupported image sampler: {name}")
    labels = [
        int(dataset.label_to_index[str(label)])
        for label in dataset.frame["label"].astype(str).tolist()
    ]
    label_tensor = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=len(dataset.class_names)).float()
    class_weights = torch.zeros(len(dataset.class_names), dtype=torch.float)
    observed = counts > 0
    class_weights[observed] = 1.0 / counts[observed]
    sample_weights = class_weights[label_tensor]
    num_samples = int(sampler_config.get("num_samples", len(dataset)))
    replacement = bool(sampler_config.get("replacement", True))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=torch.Generator().manual_seed(seed),
    )


def create_feature_extraction_loaders(
    splits_dir: str | Path,
    class_names: list[str],
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 2,
    max_samples_per_split: int | None = None,
) -> dict[str, DataLoader]:
    """Create train/validation/test dataloaders from Sprint 1 split CSVs."""

    split_root = Path(splits_dir)
    return {
        split: create_image_dataloader(
            split_root / f"{split}.csv",
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            split_name=split,
            max_samples=max_samples_per_split,
            train=False,
        )
        for split in ("train", "val", "test")
    }
