# Sprint 3 Execution Plan: Frozen Feature Fusion

## Scope

Implement and run frozen feature fusion experiments from the existing Sprint 2 cached features.

Included:

- Pairwise and three-backbone fusion over cached frozen ResNet50, MobileNetV2, and EfficientNetB0 features.
- Concatenation fusion with expected summed input dimensions.
- Projected learnable weighted fusion with 512-dimensional projections and global softmax weights.
- MLP classifier training/evaluation using the existing Sprint 2 loop pattern, seed 42, train-only class weights, validation macro-F1 early stopping, and held-out test reporting.
- Per-run metrics, classification reports, confusion matrices, training curves, and learned weighted-fusion weights.
- Frozen matrix report assets comparing the Sprint 2 single-backbone runs with the Sprint 3 fusion runs.
- Documentation and Turkish Sprint 3 result notes.

Excluded:

- Raw-image feature extraction.
- CNN forward passes unless only validating existing cache metadata.
- Fine-tuning.
- Test-metric-based hyperparameter or model selection.

## Run Matrix

- ResNet50 + MobileNetV2: concat, weighted.
- ResNet50 + EfficientNetB0: concat, weighted.
- MobileNetV2 + EfficientNetB0: concat, weighted.
- ResNet50 + MobileNetV2 + EfficientNetB0: concat, weighted.

Total Sprint 3 runs: 8.

## Verification Gates

- Unit tests cover concat shape, weighted fusion output shape, softmax weight sum, expected concat dimensions, matrix expansion, and cache alignment.
- `uv run pytest` passes.
- `uv run ruff check src scripts tests` passes.
- `uv run python scripts/run_experiment_matrix.py --config configs/experiments/frozen_feature_matrix.yaml --default-config configs/default.yaml --dataset-config configs/dataset/selected_dataset.yaml --feature-source frozen` completes.
- `artifacts/report_assets/tables/frozen_all_results.csv` contains 11 frozen rows: 3 single-backbone baselines and 8 Sprint 3 fusion runs.
- Weighted fusion weight tables/plots exist and each learned softmax row sums to 1.

## Final Outcome

Implemented:

- `src/dl_midterm/models/fusion.py`
- `scripts/run_experiment_matrix.py`
- `scripts/make_report_assets.py`
- `tests/test_fusion_shapes.py`
- frozen matrix export helpers and plots under `src/dl_midterm/evaluation/`

Generated local Sprint 3 fusion run folders:

```text
artifacts/runs/20260527_161410_frozen_r50-mnv2_concat_mlp_s42/
artifacts/runs/20260527_161438_frozen_r50-mnv2_weighted_mlp_s42/
artifacts/runs/20260527_161503_frozen_r50-effb0_concat_mlp_s42/
artifacts/runs/20260527_161529_frozen_r50-effb0_weighted_mlp_s42/
artifacts/runs/20260527_161601_frozen_mnv2-effb0_concat_mlp_s42/
artifacts/runs/20260527_161627_frozen_mnv2-effb0_weighted_mlp_s42/
artifacts/runs/20260527_161706_frozen_r50-mnv2-effb0_concat_mlp_s42/
artifacts/runs/20260527_161732_frozen_r50-mnv2-effb0_weighted_mlp_s42/
```

Best frozen model by test macro-F1:

- ResNet50 + EfficientNetB0 concat: accuracy `0.746`, macro-F1 `0.595`, weighted-F1 `0.760`.

Best default Sprint 2 single-backbone baseline:

- ResNet50: accuracy `0.712`, macro-F1 `0.531`, weighted-F1 `0.730`.

Best Sprint 3 fusion gain over the default single-backbone baseline:

- ResNet50 + EfficientNetB0 concat: `+0.064` macro-F1.

Report-ready outputs:

```text
artifacts/report_assets/tables/frozen_all_results.csv
artifacts/report_assets/tables/fusion_weight_summary.csv
artifacts/report_assets/tables/per_class_f1_frozen.csv
artifacts/report_assets/tables/fusion_gain_summary.csv
artifacts/report_assets/figures/frozen_fusion_comparison.png
artifacts/report_assets/figures/concat_vs_weighted.png
artifacts/report_assets/figures/fusion_gain_macro_f1.png
artifacts/report_assets/figures/per_class_f1_frozen_heatmap.png
artifacts/report_assets/figures/frozen_best_confusion_matrix.png
artifacts/report_assets/figures/learned_fusion_weights.png
artifacts/report_assets/figures/fusion_runs/
```

## Artifact Policy

Feature caches, run directories, checkpoints, and `model.pt` files remain gitignored. The generated report-ready CSV/PNG assets are small: `artifacts/report_assets/tables` is about 80 KB and `artifacts/report_assets/figures` is about 3.2 MB after Sprint 3.
