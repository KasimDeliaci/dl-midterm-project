"""PyTorch dataloader construction helpers."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

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
) -> DataLoader:
    """Create an image dataloader for feature extraction."""

    dataset = HAM10000ImageDataset(
        split_csv=split_csv,
        class_names=class_names,
        transform=build_image_transform(image_size=image_size, train=False),
        split_name=split_name,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
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
        )
        for split in ("train", "val", "test")
    }
