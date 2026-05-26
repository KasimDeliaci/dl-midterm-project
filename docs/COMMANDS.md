# Commands

## Setup

```bash
uv sync
```

## Validate Configs

```bash
uv run python -c "import yaml; yaml.safe_load(open('configs/dataset/selected_dataset.yaml')); print('dataset yaml ok')"
```

## Sprint 1: Dataset Preparation

Place `HAM10000_metadata.csv` under `data/metadata/` and raw image files under `data/raw/`
before running the preparation command.

```bash
uv run python scripts/prepare_dataset.py --config configs/dataset/selected_dataset.yaml
```

Expected artifacts:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
artifacts/report_assets/tables/class_distribution.csv
artifacts/report_assets/figures/class_distribution.png
```

Optional split-only command for an already audited metadata CSV:

```bash
uv run python scripts/create_splits.py \
  --config configs/dataset/selected_dataset.yaml \
  --metadata-csv data/processed/ham10000_audited_metadata.csv
```

Sprint 1 verification:

```bash
uv run pytest tests/test_dataset_sprint1.py
uv run python -c "import yaml; yaml.safe_load(open('configs/dataset/selected_dataset.yaml')); print('dataset yaml ok')"
```

## Sprint 2: Frozen Feature Extraction

```bash
uv run python scripts/extract_features.py \
  --config configs/dataset/selected_dataset.yaml \
  --default-config configs/default.yaml \
  --source frozen \
  --batch-size 64
```

Expected cache layout:

```text
artifacts/features/ham10000/frozen/resnet50/{train,val,test}.pt
artifacts/features/ham10000/frozen/mobilenet_v2/{train,val,test}.pt
artifacts/features/ham10000/frozen/efficientnet_b0/{train,val,test}.pt
```

Each backbone directory also receives per-split CSV manifests and `manifest.json`.

For a quick local smoke test without writing full caches:

```bash
uv run python scripts/extract_features.py \
  --config configs/dataset/selected_dataset.yaml \
  --default-config configs/default.yaml \
  --source frozen \
  --backbones resnet50 \
  --limit-per-split 4 \
  --batch-size 2 \
  --no-pretrained
```

## Sprint 2: Single-Backbone MLP Baselines

```bash
uv run python scripts/train_mlp.py \
  --config configs/experiments/frozen_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source frozen \
  --fusion none
```

This trains the three frozen single-backbone baselines from cached features:

- ResNet50 features + MLP
- MobileNetV2 features + MLP
- EfficientNetB0 features + MLP

Each run writes:

```text
artifacts/runs/<run_id>/config_resolved.yaml
artifacts/runs/<run_id>/metrics.json
artifacts/runs/<run_id>/history.csv
artifacts/runs/<run_id>/classification_report.csv
artifacts/runs/<run_id>/confusion_matrix.png
artifacts/runs/<run_id>/training_curve.png
artifacts/runs/<run_id>/model.pt
```

Class weighting is enabled by default and is computed from the train split cache only. Disable it
for an ablation with:

```bash
uv run python scripts/train_mlp.py \
  --config configs/experiments/frozen_feature_matrix.yaml \
  --feature-source frozen \
  --fusion none \
  --no-class-weights
```

## Sprint 3: Fusion Matrix

```bash
uv run python scripts/run_experiment_matrix.py --config configs/experiments/frozen_feature_matrix.yaml --feature-source frozen
```

## Sprint 4: Fine-Tuning

```bash
uv run python scripts/finetune_backbone.py --config configs/backbones/resnet50.yaml --backbone resnet50
uv run python scripts/finetune_backbone.py --config configs/backbones/mobilenet_v2.yaml --backbone mobilenet_v2
uv run python scripts/finetune_backbone.py --config configs/backbones/efficientnet_b0.yaml --backbone efficientnet_b0
```

## Evaluation And Report Assets

```bash
uv run python scripts/evaluate_runs.py --config configs/report_assets.yaml
uv run python scripts/aggregate_results.py --config configs/report_assets.yaml
uv run python scripts/make_report_assets.py --config configs/report_assets.yaml
```

Sprint 2 single-backbone aggregation can be refreshed directly:

```bash
uv run python scripts/evaluate_runs.py --feature-source frozen
```

Expected Sprint 2 report-ready outputs:

```text
artifacts/report_assets/tables/single_backbone_frozen_results.csv
artifacts/report_assets/tables/single_backbone_frozen_per_class_f1.csv
artifacts/report_assets/figures/frozen_single_backbone_f1.png
```

The full feature caches, checkpoints, and run folders are generated artifacts and should not be
committed.

## Later Command Pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
