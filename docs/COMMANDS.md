# Commands

## Setup

```bash
uv sync
```

## Colab Data Restore

After uploading `ham10000_colab_bundle.tar` to Google Drive, run the restore flow in
`notebooks/00_colab_setup.ipynb`. The bundle is expected at:

```text
/content/drive/MyDrive/ham10000_colab_bundle.tar
```

The runner notebooks force-remount Drive and also check these fallback locations:

```text
/content/drive/MyDrive/Colab Notebooks/ham10000_colab_bundle.tar
/content/drive/MyDrive/dl-assignment/ham10000_colab_bundle.tar
```

It restores these repo-relative paths:

```text
data/raw/ham10000/
data/metadata/HAM10000_metadata.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

The split CSVs are the preserved Sprint 1 lesion-aware split. Do not rerun split generation in
Colab unless a new split is intentional.

The notebooks request a Colab GPU runtime with T4 metadata. Colab can still change the assigned
GPU based on availability; if the runtime is not T4, use Runtime > Change runtime type and choose
T4 GPU manually.

Runner notebooks bootstrap repo/data in fresh Colab runtimes. Sprint 2 notebook outputs frozen
feature caches and report-ready assets back to:

```text
/content/drive/MyDrive/dl-midterm-artifacts/
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

Planned Sprint 3 entrypoint after fusion orchestration is implemented:

```bash
uv run python scripts/run_experiment_matrix.py --config configs/experiments/frozen_feature_matrix.yaml --feature-source frozen
```

Current status before Sprint 3 implementation: `scripts/run_experiment_matrix.py` is still a placeholder. Implement fusion modules, shape tests, and matrix orchestration before treating this command as runnable.

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

## Sprint 2b: MLP Hyperparameter Search

After frozen features exist, run the focused MLP-only search:

```bash
uv run python scripts/run_mlp_hyperparam_search.py \
  --config configs/experiments/mlp_hyperparam_search.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --search-id mlp_hparam_v1_full
```

The current search varies:

- class weighting: enabled vs disabled,
- dropout: `0.1`, `0.3`, `0.5`,
- optimizer: AdamW, Adam, SGD with momentum,
- backbone: ResNet50, MobileNetV2, EfficientNetB0.

Each MLP run gets a separate folder under:

```text
artifacts/runs/mlp_hparam_search/<search_id>/<run_id>/
```

Each run folder contains:

```text
config_resolved.yaml
metrics.json
history.csv
classification_report.csv
confusion_matrix.png
training_curve.png
model.pt
```

Search-level report assets are exported to:

```text
artifacts/report_assets/tables/mlp_hparam_search/<search_id>/
artifacts/report_assets/figures/mlp_hparam_search/<search_id>/
```

Quick smoke test:

```bash
uv run python scripts/run_mlp_hyperparam_search.py \
  --config configs/experiments/mlp_hyperparam_search.yaml \
  --backbones resnet50 \
  --candidates cw_adamw_d03 nocw_adamw_d03 \
  --epochs 3 \
  --search-id smoke_mlp_search
```

## Later Command Pattern

```bash
uv run python scripts/<script>.py --config configs/<config>.yaml
```
