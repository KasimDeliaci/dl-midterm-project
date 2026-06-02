# Sprint 4G Autoresearch Ensemble

## Purpose

Sprint 4G is a local cached-feature autoresearch extension after Sprint 4F. The goal is to test
whether existing checkpointed MLP/fusion models from Sprint 4B, 4C, 4E, and 4F can be combined by a
validation-gated soft-vote ensemble to improve macro-F1 without running new CNN fine-tuning.

This experiment does not read raw images, does not create new feature caches, and does not replace
canonical Sprint 4 or Sprint 4D. It is a post-hoc model-combination audit over already generated
checkpoints.

## Protocol

- Fixed Sprint 1 lesion-aware train/validation/test split.
- Candidate discovery requires existing `model.pt` and `config_resolved.yaml`.
- Eligible feature sources: `finetuned`, `finetuned_classaware`, `finetuned_deeper`,
  `finetuned_augmented`.
- Eligible fusion methods: `none`, `concat`, `weighted`, `per_class_weighted`.
- Candidate filtering uses validation macro-F1 only.
- Ensemble selection uses validation macro-F1 only.
- Test evaluation is run once for the selected ensemble.
- Random Dirichlet weight search is disabled because it showed validation overfit risk.

## Implementation

Added:

- `configs/experiments/sprint4g_autoresearch_ensemble.yaml`
- `scripts/run_autoresearch_ensemble.py`
- `tests/test_sprint4g_autoresearch.py`

The report CSV is capped to the top validation ensembles; the full search table is written under
`artifacts/runs/sprint4g_autoresearch_ensemble/` and remains gitignored.

## Run Command

```bash
uv run python scripts/run_autoresearch_ensemble.py \
  --config configs/experiments/sprint4g_autoresearch_ensemble.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --device auto \
  --batch-size 128
```

## Result

Selected validation ensemble:

- policy: uniform soft-vote;
- candidate count: 3;
- validation macro-F1: `0.725`;
- test accuracy: `0.806`;
- test macro-F1: `0.707`;
- test weighted-F1: `0.812`.

Compared with reference results:

- canonical Sprint 4 three-backbone concat: test macro-F1 `0.706`;
- Sprint 4D weighted + `tta_rot4`: test macro-F1 `0.733`;
- Sprint 4G selected ensemble: test macro-F1 `0.707`.

The gain over canonical Sprint 4 is very small (`+0.0015` absolute macro-F1), so Sprint 4G should
be interpreted as near-tie evidence that simple post-hoc ensemble averaging is not enough to close
the gap to the Sprint 4D TTA result.

## Report Assets

Tables:

- `artifacts/report_assets/tables/sprint4g/sprint4g_individual_validation_results.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_ensemble_validation_results.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_selection_log.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_test_results.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_test_per_class_f1.csv`
- `artifacts/report_assets/tables/sprint4g/sprint4g_final_comparison.csv`

Figures:

- `artifacts/report_assets/figures/sprint4g/sprint4g_validation_ensemble_macro_f1.png`
- `artifacts/report_assets/figures/sprint4g/sprint4g_test_macro_f1.png`
- `artifacts/report_assets/figures/sprint4g/sprint4g_selected_weights.png`

## Verification

```bash
uv run ruff check scripts/run_autoresearch_ensemble.py tests/test_sprint4g_autoresearch.py
uv run pytest tests/test_sprint4g_autoresearch.py
```

Both checks passed locally on 2026-06-03.
