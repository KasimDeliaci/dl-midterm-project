"""Sprint 1 dataset audit and split-generation tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dl_midterm.data.datasets import (
    attach_image_paths,
    audit_metadata,
    load_ham10000_metadata,
)
from dl_midterm.data.splits import check_lesion_leakage, create_lesion_aware_splits

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]


def test_load_metadata_normalizes_dx_to_label(tmp_path: Path) -> None:
    metadata_path = tmp_path / "HAM10000_metadata.csv"
    pd.DataFrame(
        {
            "image_id": ["ISIC_0000001"],
            "lesion_id": ["HAM_0001"],
            "dx": ["nv"],
        }
    ).to_csv(metadata_path, index=False)

    metadata = load_ham10000_metadata(metadata_path, CLASS_NAMES)

    assert metadata.loc[0, "label"] == "nv"
    assert metadata.loc[0, "image_id"] == "ISIC_0000001"


def test_attach_image_paths_and_audit_reports_missing_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "ISIC_0000001.jpg").write_bytes(b"not a real image but path exists")
    metadata = pd.DataFrame(
        {
            "image_id": ["ISIC_0000001", "ISIC_0000002"],
            "lesion_id": ["HAM_0001", "HAM_0002"],
            "label": ["nv", "mel"],
        }
    )

    with_paths = attach_image_paths(metadata, raw_dir)
    audit = audit_metadata(with_paths, tmp_path / "metadata.csv", raw_dir)

    assert with_paths.loc[0, "image_path"].endswith("ISIC_0000001.jpg")
    assert audit.missing_images == ["ISIC_0000002"]
    assert audit.has_blocking_errors


def test_lesion_aware_split_prevents_group_leakage() -> None:
    labels = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]
    rows = []
    for label in labels:
        for group_idx in range(6):
            lesion_id = f"{label}_{group_idx}"
            for image_idx in range(2):
                rows.append(
                    {
                        "image_id": f"{lesion_id}_{image_idx}",
                        "lesion_id": lesion_id,
                        "label": label,
                        "image_path": f"/tmp/{lesion_id}_{image_idx}.jpg",
                    }
                )
    metadata = pd.DataFrame(rows)

    result = create_lesion_aware_splits(metadata, seed=7)
    leaks = check_lesion_leakage(result.splits)

    assert leaks == []
    assert sum(len(frame) for frame in result.splits.values()) == len(metadata)
    assert {name for name, frame in result.splits.items() if not frame.empty} == {
        "train",
        "val",
        "test",
    }


def test_unknown_label_fails_fast(tmp_path: Path) -> None:
    metadata_path = tmp_path / "HAM10000_metadata.csv"
    pd.DataFrame({"image_id": ["x"], "dx": ["unknown"]}).to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="outside configured class_names"):
        load_ham10000_metadata(metadata_path, CLASS_NAMES)
