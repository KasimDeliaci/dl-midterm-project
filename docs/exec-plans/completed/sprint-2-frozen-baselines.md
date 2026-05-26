# Sprint 2 Execution Plan: Frozen Feature Baselines

## Scope

Implement the frozen ImageNet-pretrained feature extraction and single-backbone MLP baseline stage for HAM10000.

Included:

- Real HAM10000 image dataset and dataloaders backed by Sprint 1 split CSVs.
- Deterministic RGB 224x224 ImageNet preprocessing for feature extraction.
- Frozen ResNet50, MobileNetV2, and EfficientNetB0 feature extractors.
- `.pt` feature caches and CSV/JSON manifests for train/validation/test.
- MLP training from cached single-backbone features.
- Metrics, per-class reports, confusion matrices, training curves, and frozen single-backbone summary assets.
- Thin Colab notebook runner and command documentation.

Excluded until later sprints:

- Pairwise fusion.
- Three-CNN fusion.
- Fine-tuning.

## Verification

- `uv run pytest`
- `uv run ruff check .`
- Smoke extraction with 4 samples per split and no pretrained download under `/private/tmp`.
- Smoke MLP training/evaluation under `/private/tmp`.
- Full pretrained feature extraction for all three backbones and all three splits.
- Full frozen single-backbone MLP training for ResNet50, MobileNetV2, and EfficientNetB0.

## Artifact Policy

Feature caches, checkpoints, and run folders remain gitignored. The generated Sprint 2 report-ready summary CSV/PNG files are small enough to track after review:

- `artifacts/report_assets/tables/single_backbone_frozen_results.csv`: 660 B
- `artifacts/report_assets/tables/single_backbone_frozen_per_class_f1.csv`: 1.5 KB
- `artifacts/report_assets/figures/frozen_single_backbone_f1.png`: 38 KB

Full feature cache sizes are intentionally not tracked:

- ResNet50 frozen cache: 79 MB
- MobileNetV2 frozen cache: 50 MB
- EfficientNetB0 frozen cache: 50 MB
