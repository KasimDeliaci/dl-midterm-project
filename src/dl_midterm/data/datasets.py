"""HAM10000 metadata loading and audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class DatasetAudit:
    """Summary of metadata/image integrity checks."""

    metadata_path: Path
    raw_dir: Path
    image_rows: int
    unique_image_ids: int
    duplicate_image_ids: list[str]
    missing_images: list[str]
    missing_labels: int
    lesion_id_available: bool
    class_distribution: pd.DataFrame

    @property
    def has_blocking_errors(self) -> bool:
        return bool(self.duplicate_image_ids or self.missing_images or self.missing_labels)


def find_metadata_file(metadata_dir: Path, configured_path: str | None = None) -> Path:
    """Resolve the HAM10000 metadata CSV path."""

    if configured_path:
        path = Path(configured_path)
        return path if path.is_absolute() else Path(configured_path)

    candidates = [
        metadata_dir / "HAM10000_metadata.csv",
        metadata_dir / "ham10000_metadata.csv",
        metadata_dir / "metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "HAM10000 metadata CSV was not found. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_ham10000_metadata(metadata_path: Path, class_names: list[str]) -> pd.DataFrame:
    """Load HAM10000 metadata and normalize core columns."""

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    if "image_id" not in metadata.columns:
        raise ValueError("Metadata must include an `image_id` column.")

    if "dx" in metadata.columns:
        label_col = "dx"
    elif "label" in metadata.columns:
        label_col = "label"
    else:
        label_col = None
    if label_col is None:
        raise ValueError("Metadata must include a `dx` or `label` class column.")

    normalized = metadata.copy()
    normalized["image_id"] = normalized["image_id"].astype(str)
    normalized["label"] = normalized[label_col].astype(str)

    unknown_labels = sorted(set(normalized["label"].dropna()) - set(class_names))
    if unknown_labels:
        raise ValueError(
            f"Metadata contains labels outside configured class_names: {unknown_labels}"
        )

    if "lesion_id" in normalized.columns:
        normalized["lesion_id"] = normalized["lesion_id"].astype(str)

    return normalized


def attach_image_paths(metadata: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """Attach resolved image paths by searching under the raw dataset directory."""

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"HAM10000 raw image directory does not exist: {raw_dir}. "
            "Download/extract the dataset before running Sprint 1 preparation."
        )

    image_index = _index_images(raw_dir)
    with_paths = metadata.copy()
    with_paths["image_path"] = with_paths["image_id"].map(image_index)
    return with_paths


def audit_metadata(metadata: pd.DataFrame, metadata_path: Path, raw_dir: Path) -> DatasetAudit:
    """Run metadata integrity checks and class distribution counts."""

    duplicate_ids = metadata.loc[
        metadata["image_id"].duplicated(keep=False), "image_id"
    ].dropna().unique().tolist()
    missing_images = metadata.loc[metadata["image_path"].isna(), "image_id"].tolist()
    missing_labels = int(metadata["label"].isna().sum())
    class_distribution = (
        metadata["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    class_distribution["percent"] = (
        class_distribution["count"] / class_distribution["count"].sum() * 100
    ).round(4)

    return DatasetAudit(
        metadata_path=metadata_path,
        raw_dir=raw_dir,
        image_rows=len(metadata),
        unique_image_ids=int(metadata["image_id"].nunique()),
        duplicate_image_ids=sorted(duplicate_ids),
        missing_images=missing_images,
        missing_labels=missing_labels,
        lesion_id_available="lesion_id" in metadata.columns and metadata["lesion_id"].notna().all(),
        class_distribution=class_distribution,
    )


def read_split(split_path: Path) -> pd.DataFrame:
    """Read a generated split CSV."""

    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV does not exist: {split_path}")
    return pd.read_csv(split_path)


class HAM10000ImageDataset(Dataset[dict[str, object]]):
    """PyTorch dataset backed by a Sprint 1 HAM10000 split CSV."""

    def __init__(
        self,
        split_csv: str | Path,
        class_names: list[str],
        transform: object | None = None,
        split_name: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.frame = read_split(self.split_csv)
        if max_samples is not None:
            self.frame = self.frame.head(max_samples).copy()
        self.class_names = class_names
        self.label_to_index = {label: index for index, label in enumerate(class_names)}
        self.transform = transform
        self.split_name = split_name or self.split_csv.stem
        self._validate_columns()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                image_tensor = image

        label_name = str(row["label"])
        return {
            "image": image_tensor,
            "label": torch.tensor(self.label_to_index[label_name], dtype=torch.long),
            "label_name": label_name,
            "image_id": str(row["image_id"]),
            "split": self.split_name,
            "image_path": str(image_path),
        }

    def _validate_columns(self) -> None:
        required = {"image_id", "label", "image_path"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")

        unknown_labels = sorted(set(self.frame["label"].astype(str)) - set(self.class_names))
        if unknown_labels:
            raise ValueError(f"Split CSV contains unknown labels: {unknown_labels}")


def _index_images(raw_dir: Path) -> dict[str, str]:
    image_index: dict[str, str] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in raw_dir.rglob(f"*{extension}"):
            image_index[path.stem] = str(path)
        for path in raw_dir.rglob(f"*{extension.upper()}"):
            image_index[path.stem] = str(path)
    return image_index
