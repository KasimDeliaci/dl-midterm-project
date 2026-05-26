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
uv run python scripts/extract_features.py --config configs/dataset/selected_dataset.yaml --source frozen
```

## Sprint 2: Single-Backbone MLP Baselines

```bash
uv run python scripts/train_mlp.py --config configs/experiments/frozen_feature_matrix.yaml --feature-source frozen --fusion none
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

## Later Command Pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
