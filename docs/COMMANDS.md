# Commands

## Setup

```bash
uv sync
```

## Validate Configs

```bash
uv run python -c "import yaml; yaml.safe_load(open('configs/dataset.yaml')); print('dataset yaml ok')"
```

## Sprint 1: Dataset Preparation

```bash
uv run python scripts/prepare_dataset.py --config configs/dataset.yaml
```

Expected outputs:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
outputs/report_assets/tables/class_distribution.csv
outputs/report_assets/figures/class_distribution.png
```

## Sprint 2: Frozen Feature Extraction

```bash
uv run python scripts/extract_features.py --config configs/dataset.yaml --source frozen
```

## Sprint 2: Single-Backbone MLP Baselines

```bash
uv run python scripts/train_mlp.py --config configs/experiments.yaml --feature-source frozen --fusion none
```

## Sprint 3: Fusion Matrix

```bash
uv run python scripts/run_experiments.py --config configs/experiments.yaml --feature-source frozen
```

## Sprint 4: Fine-Tuning

```bash
uv run python scripts/finetune_backbone.py --config configs/backbones.yaml --backbone resnet50
uv run python scripts/finetune_backbone.py --config configs/backbones.yaml --backbone mobilenet_v2
uv run python scripts/finetune_backbone.py --config configs/backbones.yaml --backbone efficientnet_b0
```

## Evaluation And Report Assets

```bash
uv run python scripts/evaluate_results.py --config configs/report.yaml
uv run python scripts/make_report_assets.py --config configs/report.yaml
```

## Later Command Pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
