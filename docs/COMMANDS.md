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

```bash
uv run python scripts/run_experiment_matrix.py \
  --config configs/experiments/frozen_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source frozen
```

This trains the 8 Sprint 3 frozen fusion runs from the existing cached features:

- ResNet50 + MobileNetV2: concat, weighted
- ResNet50 + EfficientNetB0: concat, weighted
- MobileNetV2 + EfficientNetB0: concat, weighted
- ResNet50 + MobileNetV2 + EfficientNetB0: concat, weighted

Weighted fusion projects each backbone feature to 512 dimensions and learns global softmax
weights. The command does not run raw-image feature extraction or CNN forward passes.

Quick smoke test:

```bash
uv run python scripts/run_experiment_matrix.py \
  --config configs/experiments/frozen_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source frozen \
  --max-runs 1 \
  --epochs 1 \
  --batch-size 128
```

Expected Sprint 3 report-ready outputs:

```text
artifacts/report_assets/tables/frozen_all_results.csv
artifacts/report_assets/tables/fusion_weight_summary.csv
artifacts/report_assets/tables/per_class_f1_frozen.csv
artifacts/report_assets/tables/fusion_gain_summary.csv
artifacts/report_assets/tables/per_class_fusion_gain.csv
artifacts/report_assets/tables/representation_complementarity_summary.csv
artifacts/report_assets/tables/fusion_complementarity_summary.csv
artifacts/report_assets/figures/frozen_fusion_comparison.png
artifacts/report_assets/figures/single_pairwise_three_macro_f1.png
artifacts/report_assets/figures/concat_vs_weighted.png
artifacts/report_assets/figures/fusion_gain_macro_f1.png
artifacts/report_assets/figures/per_class_f1_frozen_heatmap.png
artifacts/report_assets/figures/per_class_fusion_gain_heatmap.png
artifacts/report_assets/figures/frozen_best_confusion_matrix.png
artifacts/report_assets/figures/learned_fusion_weights.png
artifacts/report_assets/figures/accuracy_vs_macro_f1_frozen.png
artifacts/report_assets/figures/representation_similarity_heatmap.png
artifacts/report_assets/figures/fusion_gain_vs_complementarity.png
artifacts/report_assets/figures/fusion_runs/
```

The per-run folders under `artifacts/runs/` contain `metrics.json`, `history.csv`,
`classification_report.csv`, `confusion_matrix.png`, `training_curve.png`, `model.pt`, and,
for weighted runs, `fusion_weights.csv`/`.json`.

## Sprint 4: Fine-Tuning

Sprint 4 full runs should execute on Colab GPU, not locally. The notebook
`notebooks/03_finetune_backbones.ipynb` is a thin runner around the commands below. It clones/pulls
the repo under `/content/dl-assignment`, restores the Drive HAM10000 bundle, runs scripts, and
mirrors selected generated artifacts to:

```bash
/content/drive/MyDrive/dl-midterm-artifacts/
```

Fine-tune all three backbones and extract fine-tuned feature caches:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/finetune_backbones.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned \
  --batch-size 32
```

Expected checkpoint layout:

```text
artifacts/checkpoints/finetuned_backbones/resnet50_best.pt
artifacts/checkpoints/finetuned_backbones/mobilenet_v2_best.pt
artifacts/checkpoints/finetuned_backbones/efficientnet_b0_best.pt
```

Expected fine-tuned feature cache layout:

```text
artifacts/features/ham10000/finetuned/resnet50/{train,val,test}.pt
artifacts/features/ham10000/finetuned/mobilenet_v2/{train,val,test}.pt
artifacts/features/ham10000/finetuned/efficientnet_b0/{train,val,test}.pt
```

Run the three single-backbone fine-tuned MLP baselines:

```bash
uv run python scripts/train_mlp.py \
  --config configs/experiments/finetuned_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned
```

Run the eight fine-tuned fusion MLP experiments:

```bash
uv run python scripts/run_experiment_matrix.py \
  --config configs/experiments/finetuned_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned
```

Refresh fine-tuned and frozen-vs-fine-tuned report assets:

```bash
uv run python scripts/make_report_assets.py \
  --feature-source finetuned \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-root artifacts/features
```

Expected Sprint 4 report-ready outputs:

```text
artifacts/report_assets/tables/finetuned_all_results.csv
artifacts/report_assets/tables/frozen_vs_finetuned_summary.csv
artifacts/report_assets/tables/finetuned_per_class_f1.csv
artifacts/report_assets/tables/finetuning_gain_summary.csv
artifacts/report_assets/tables/finetuned_fusion_weight_summary.csv
artifacts/report_assets/figures/frozen_vs_finetuned_macro_f1.png
artifacts/report_assets/figures/finetuned_fusion_comparison.png
artifacts/report_assets/figures/finetuned_concat_vs_weighted.png
artifacts/report_assets/figures/finetuning_gain_macro_f1.png
artifacts/report_assets/figures/finetuned_per_class_f1_heatmap.png
artifacts/report_assets/figures/finetuned_best_confusion_matrix.png
artifacts/report_assets/figures/finetuned_learned_fusion_weights.png
```

Non-canonical local smoke test, writing only to `/tmp`:

```bash
uv run python scripts/finetune_backbone.py \
  --backbone mobilenet_v2 \
  --epochs 1 \
  --batch-size 2 \
  --limit-per-split 2 \
  --no-pretrained \
  --no-mixed-precision \
  --checkpoint-dir /tmp/dlmidterm_sprint4_ckpt \
  --feature-root /tmp/dlmidterm_sprint4_features \
  --run-root /tmp/dlmidterm_sprint4_runs
```

## Sprint 4B: Class-Aware Extension

Sprint 4B is an exploratory extension and does not replace canonical Sprint 4. Full image-level
fine-tuning should run on Colab GPU. The thin runner is:

```text
notebooks/04_sprint4b_classaware.ipynb
```

Run the class-aware fine-tuning screen and extract feature caches:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/sprint4b_classaware_backbones.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware \
  --batch-size 32
```

Expected cache layout:

```text
artifacts/features/ham10000/finetuned_classaware/resnet50/{train,val,test}.pt
artifacts/features/ham10000/finetuned_classaware/mobilenet_v2/{train,val,test}.pt
artifacts/features/ham10000/finetuned_classaware/efficientnet_b0/{train,val,test}.pt
```

Run the class-aware single-backbone MLP screening:

```bash
uv run python scripts/train_mlp.py \
  --config configs/experiments/sprint4b_classaware_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware
```

Run the deeper ResNet50 probe and its single-backbone MLP screen:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/sprint4b_deeper_screen.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_deeper \
  --backbone resnet50 \
  --batch-size 32

uv run python scripts/train_mlp.py \
  --config configs/experiments/sprint4b_deeper_screen.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_deeper
```

Expected deeper cache layout:

```text
artifacts/features/ham10000/finetuned_deeper/resnet50/{train,val,test}.pt
```

Refresh Sprint 4B screening report assets after canonical Sprint 4 and Sprint 4B single-backbone
MLP runs are available locally:

```bash
uv run python scripts/make_report_assets.py \
  --feature-source finetuned_classaware \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-root artifacts/features
```

Expected screening outputs:

```text
artifacts/report_assets/tables/sprint4b_screening_results.csv
artifacts/report_assets/tables/sprint4b_vs_canonical_single_backbone.csv
artifacts/report_assets/tables/sprint4b_per_class_f1_gain.csv
artifacts/report_assets/figures/sprint4b_val_macro_f1_screening.png
artifacts/report_assets/figures/sprint4b_test_macro_f1_vs_canonical.png
artifacts/report_assets/figures/sprint4b_per_class_f1_gain_heatmap.png
```

Run the optional full class-aware matrix only if the validation-based stop/go criteria pass:

```bash
uv run python scripts/run_experiment_matrix.py \
  --config configs/experiments/sprint4b_classaware_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware
```

Expected full-matrix outputs:

```text
artifacts/report_assets/tables/sprint4b_classaware_all_results.csv
artifacts/report_assets/tables/sprint4b_classaware_vs_canonical_fusion_summary.csv
artifacts/report_assets/tables/sprint4b_classaware_fusion_weight_summary.csv
artifacts/report_assets/figures/sprint4b_classaware_fusion_comparison.png
artifacts/report_assets/figures/sprint4b_classaware_concat_vs_weighted.png
artifacts/report_assets/figures/sprint4b_classaware_learned_fusion_weights.png
artifacts/report_assets/figures/sprint4b_best_confusion_matrix.png
```

Non-canonical local smoke test, writing only to `/tmp`:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/sprint4b_classaware_backbones.yaml \
  --backbone mobilenet_v2 \
  --epochs 1 \
  --batch-size 2 \
  --limit-per-split 2 \
  --no-pretrained \
  --no-mixed-precision \
  --checkpoint-dir /tmp/dlmidterm_sprint4b_ckpt \
  --feature-root /tmp/dlmidterm_sprint4b_features \
  --run-root /tmp/dlmidterm_sprint4b_runs
```

## Evaluation And Report Assets

```bash
uv run python scripts/evaluate_runs.py --config configs/report_assets.yaml
uv run python scripts/aggregate_results.py --config configs/report_assets.yaml
uv run python scripts/make_report_assets.py --config configs/report_assets.yaml
```

Sprint 3 frozen matrix assets can be refreshed without rerunning training:

```bash
uv run python scripts/make_report_assets.py \
  --feature-source frozen \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-root artifacts/features
```

This also refreshes the Sprint 3 analysis add-ons:

- per-class F1 gain versus the best default single-backbone baseline,
- representation complementarity using cached test-split feature similarity matrices.

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
