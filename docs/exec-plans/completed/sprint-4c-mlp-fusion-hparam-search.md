# Sprint 4C Execution Plan: Fine-Tuned MLP/Fusion Hyperparameter Search

Status: completed on 2026-05-31.

Outcome:

- Stage A completed 72/72 cached-feature MLP/fusion runs.
- Stage B was justified by validation macro-F1 gain and completed 22/22 full-matrix runs.
- Best validation-selected Sprint 4C candidate: `r50+mnv2+effb0 weighted /
  w_cw_adamw_low_lr_p512`, validation macro-F1 `0.680`, test macro-F1 `0.699`.
- Sprint 4C did not replace canonical Sprint 4 overall best `r50+mnv2+effb0 concat`, test
  macro-F1 `0.706`, because the validation-selected tuned best remained lower on test macro-F1.
- Final interpretation is recorded in `docs/report_notes/sprint-4c-results-tr.md`.

## Purpose

Sprint 4C is a local, cached-feature extension after the completed canonical Sprint 4 and
exploratory Sprint 4B runs. It does not change the CNN feature extractors. It asks whether the final
MLP/fusion classifier stage is limiting the canonical fine-tuned result.

Scientific question:

> Given fixed canonical `finetuned` feature caches, can a broader but controlled MLP/fusion
> hyperparameter search improve validation-selected macro-F1 or show that the Sprint 4 fusion result
> is robust to classifier tuning?

Sprint 4C should strengthen the report discussion whether or not it improves the best test result.
It must not become test-set shopping.

## Current Baseline

Canonical Sprint 4 remains the assignment-complete baseline:

- Feature source: `finetuned`
- Best overall model: `r50+mnv2+effb0 concat`
- Accuracy: `0.811`
- Macro-F1: `0.706`
- Weighted-F1: `0.813`

Sprint 4B class-aware fine-tuning did not beat the canonical best fusion result:

- Best class-aware matrix result: `r50+mnv2 concat`, macro-F1 `0.695`
- Deeper ResNet50 single probe improved the ResNet50 single model, macro-F1 `0.688`
- Main Sprint 4B conclusion: better standalone macro-F1 did not necessarily translate into better
  fusion, so representation complementarity and final-stage calibration matter.

## Scope

Included:

- Use existing local canonical `finetuned` feature caches only.
- Search cached-feature MLP/fusion hyperparameters for selected single, concat, and weighted runs.
- Select candidates by validation macro-F1.
- Compare final selected candidates against canonical Sprint 4.
- Produce report-ready tables and figures for search behavior, tuned-vs-canonical comparison, and
  per-class effects.

Excluded:

- New CNN fine-tuning.
- Test-time augmentation.
- New model families such as ViT, Swin, EfficientNet-B4/V2, CLIP, or metadata/multimodal models.
- Image-level oversampling/GAN/SMOTE experiments.
- Changing the dataset split.
- Claiming clinical significance.

## Key Revision to GPT Pro Proposal

The staged search recommendation is good, but implementation must respect the current repo:

- `scripts/run_mlp_hyperparam_search.py` currently supports only single-backbone `fusion_method:
  none`.
- `scripts/run_experiment_matrix.py` supports fusion runs, but only one hyperparameter setting per
  invocation.
- `train_mlp.py` and `run_experiment_matrix.py` currently evaluate test metrics for every run.

Therefore Sprint 4C should add a dedicated search orchestrator rather than forcing the old Sprint 2B
script to handle fusion. Methodologically, all ranking and expansion decisions must still use
validation macro-F1. If test metrics are automatically written for all runs, the report must treat
them as audit fields and only make final claims for validation-selected candidates.

## Stage A: Screening Search

Run a curated screening search on the most informative subset.

### Combinations

Single-backbone runs:

- `resnet50`
- `mobilenet_v2`
- `efficientnet_b0`

Concat fusion runs:

- `resnet50 + mobilenet_v2`
- `resnet50 + efficientnet_b0`
- `resnet50 + mobilenet_v2 + efficientnet_b0`

Weighted fusion runs:

- `resnet50 + efficientnet_b0`
- `resnet50 + mobilenet_v2 + efficientnet_b0`

Rationale:

- Includes all single backbones to check whether classifier tuning changes the ranking.
- Includes canonical best `r50+mnv2+effb0 concat`.
- Includes `r50+mnv2 concat`, which was the best Sprint 4B class-aware matrix result.
- Includes the strongest weighted-fusion candidates without immediately running all weighted pairs.

### Non-Weighted Candidate Profiles

Use these profiles for single-backbone and concat runs:

| Candidate | Class weighting | Optimizer | LR | Weight decay | Dropout | Hidden dims |
|---|---:|---|---:|---:|---:|---|
| `cw_adamw_base` | true | adamw | 0.001 | 0.0001 | 0.3 | `[512, 256]` |
| `nocw_adamw_base` | false | adamw | 0.001 | 0.0001 | 0.3 | `[512, 256]` |
| `cw_adamw_low_lr` | true | adamw | 0.0003 | 0.0001 | 0.3 | `[512, 256]` |
| `cw_adamw_high_lr` | true | adamw | 0.003 | 0.0001 | 0.3 | `[512, 256]` |
| `cw_adamw_small` | true | adamw | 0.001 | 0.0001 | 0.3 | `[256]` |
| `cw_adamw_wide` | true | adamw | 0.001 | 0.0001 | 0.4 | `[1024, 512]` |
| `cw_adamw_regularized` | true | adamw | 0.001 | 0.001 | 0.5 | `[512, 256]` |
| `cw_sgd_base` | true | sgd | 0.01 | 0.0001 | 0.3 | `[512, 256]` |

Stage A non-weighted run count:

- 3 single combinations x 8 candidates = 24 runs
- 3 concat combinations x 8 candidates = 24 runs

### Weighted Candidate Profiles

Use a smaller candidate family for weighted fusion because projection dimension is the main added
degree of freedom.

Base profiles:

| Candidate | Class weighting | Optimizer | LR | Weight decay | Dropout | Hidden dims |
|---|---:|---|---:|---:|---:|---|
| `w_cw_adamw_base` | true | adamw | 0.001 | 0.0001 | 0.3 | `[512, 256]` |
| `w_nocw_adamw_base` | false | adamw | 0.001 | 0.0001 | 0.3 | `[512, 256]` |
| `w_cw_adamw_low_lr` | true | adamw | 0.0003 | 0.0001 | 0.3 | `[512, 256]` |
| `w_cw_adamw_regularized` | true | adamw | 0.001 | 0.001 | 0.5 | `[512, 256]` |

Projection dimensions:

- `256`
- `512`
- `1024`

Stage A weighted run count:

- 2 weighted combinations x 4 profiles x 3 projection dims = 24 runs

Total Stage A:

- 72 local cached-feature MLP/fusion runs

## Stage B: Conditional Full-Matrix Expansion

Run Stage B only if Stage A shows validation evidence that tuning matters.

Select:

- Top 2 non-weighted profiles by validation macro-F1, considering both absolute performance and
  matched canonical gain.
- Top 2 weighted profiles by validation macro-F1, including projection dimension.

Then run them across the full canonical 11-combination matrix:

- 3 single-backbone runs
- 3 pairwise concat runs
- 3 pairwise weighted runs
- 1 three-backbone concat run
- 1 three-backbone weighted run

Expected Stage B run count:

- 3 singles x 2 non-weighted profiles = 6
- 4 concat runs x 2 non-weighted profiles = 8
- 4 weighted runs x 2 weighted profiles = 8
- Total = 22 runs

If Stage A gives no meaningful validation gain, stop after Stage A and write Sprint 4C as a
negative/robustness result.

## Stop/Go Criteria

All decisions use validation macro-F1.

Go from Stage A to Stage B if at least one is true:

- A candidate improves matched canonical validation macro-F1 by at least `0.015`.
- A candidate improves the canonical best three-backbone concat validation macro-F1 by at least
  `0.010`.
- Weighted fusion closes the gap to concat with a validation macro-F1 gain of at least `0.015`
  over its matched canonical weighted baseline.

Do not expand if:

- Gains appear only in test metrics.
- Validation macro-F1 gain is below `0.005`.
- Accuracy or weighted-F1 drops by more than roughly `0.030`.
- The apparent macro-F1 gain is driven mainly by one very low-support class such as `df` or `vasc`.

Interpret final test macro-F1 gain over canonical best:

| Test macro-F1 gain | Interpretation |
|---:|---|
| `< +0.005` | Noise-level, do not claim improvement |
| `+0.005` to `+0.010` | Suggestive only |
| `+0.010` to `+0.020` | Useful if validation also supports it |
| `> +0.020` | Meaningful Sprint 4C improvement |

Acceptable tradeoff for a new tuned best:

- Macro-F1 gain at least `+0.010` to `+0.015`.
- Accuracy drop no worse than about `0.020`.
- Weighted-F1 drop no worse than about `0.020`.

## Suggested Configs

Add:

- `configs/experiments/sprint4c_finetuned_mlpfusion_screen.yaml`
- `configs/experiments/sprint4c_finetuned_mlpfusion_full.yaml` only after Stage A justifies it.

Suggested screen config shape:

```yaml
mlp_fusion_hparam_search:
  name: sprint4c_finetuned_mlpfusion_screen
  feature_source: finetuned
  seed: 42
  selection_metric: best_val_macro_f1
  run_root: artifacts/runs/sprint4c_hparam_search
  tables_dir: artifacts/report_assets/tables/sprint4c
  figures_dir: artifacts/report_assets/figures/sprint4c
  stage: screen
  combinations:
    - backbones: [resnet50]
      fusion_method: none
    - backbones: [mobilenet_v2]
      fusion_method: none
    - backbones: [efficientnet_b0]
      fusion_method: none
    - backbones: [resnet50, mobilenet_v2]
      fusion_method: concat
    - backbones: [resnet50, efficientnet_b0]
      fusion_method: concat
    - backbones: [resnet50, mobilenet_v2, efficientnet_b0]
      fusion_method: concat
    - backbones: [resnet50, efficientnet_b0]
      fusion_method: weighted
    - backbones: [resnet50, mobilenet_v2, efficientnet_b0]
      fusion_method: weighted
```

Candidate profiles should live in the same config so run manifests fully describe the search.

## Implementation Tasks

1. Add a Sprint 4C search orchestrator.
   - Preferred path: `scripts/run_sprint4c_hparam_search.py`.
   - Reuse `run_single_backbone` from `scripts/train_mlp.py`.
   - Reuse `run_fusion_experiment` from `scripts/run_experiment_matrix.py`.
   - Add run tags that include candidate name and projection dimension where relevant.
   - Write a search manifest with all planned and completed runs.

2. Generalize report export for MLP/fusion hyperparameter search.
   - Current `export_mlp_search_report_assets` filters `fusion_method == "none"`.
   - Sprint 4C needs all `none`, `concat`, and `weighted` runs.
   - Add a dedicated exporter if that is cleaner than overloading the Sprint 2B exporter.

3. Add Sprint 4C tables.
   - `artifacts/report_assets/tables/sprint4c/sprint4c_hparam_search_all.csv`
   - `artifacts/report_assets/tables/sprint4c/sprint4c_screening_summary.csv`
   - `artifacts/report_assets/tables/sprint4c/sprint4c_best_by_combination.csv`
   - `artifacts/report_assets/tables/sprint4c/sprint4c_vs_canonical_summary.csv`
   - `artifacts/report_assets/tables/sprint4c/sprint4c_concat_vs_weighted_tuned.csv`
   - `artifacts/report_assets/tables/sprint4c/sprint4c_per_class_f1_gain.csv`
   - `artifacts/report_assets/tables/sprint4c/sprint4c_weighted_fusion_weights.csv`

4. Add Sprint 4C figures.
   - `artifacts/report_assets/figures/sprint4c/sprint4c_val_macro_f1_by_candidate.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_best_by_combination_macro_f1.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_canonical_vs_tuned_macro_f1.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_concat_vs_weighted_after_tuning.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_runtime_vs_val_macro_f1.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_val_vs_test_macro_f1_selected.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_per_class_f1_gain_heatmap.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_best_confusion_matrix.png`
   - `artifacts/report_assets/figures/sprint4c/sprint4c_best_weighted_fusion_weights.png`

5. Add report note skeleton.
   - `docs/report_notes/sprint-4c-results-tr.md`
   - Include purpose, search space, selection rule, best validation-selected candidate, tuned-vs-
     canonical comparison, per-class observations, and final interpretation.

## Expected Commands

Stage A smoke run:

```bash
uv run python scripts/run_sprint4c_hparam_search.py \
  --config configs/experiments/sprint4c_finetuned_mlpfusion_screen.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --max-runs 3
```

Stage A full run:

```bash
uv run python scripts/run_sprint4c_hparam_search.py \
  --config configs/experiments/sprint4c_finetuned_mlpfusion_screen.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml
```

Stage B, only if justified:

```bash
uv run python scripts/run_sprint4c_hparam_search.py \
  --config configs/experiments/sprint4c_finetuned_mlpfusion_full.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml
```

## Verification Gates

Implementation is ready to run only if:

- `uv run ruff check src scripts tests` passes.
- `uv run pytest` passes.
- A `--max-runs 3` smoke run writes:
  - a single-backbone run,
  - a concat run,
  - a weighted run,
  - a manifest,
  - a search summary table.
- Weighted search rows preserve `projection_dim` and learned fusion weights.
- Report exporters compare against canonical `finetuned` runs, not Sprint 4B class-aware runs.

Stage A is complete only if:

- 72 planned runs either complete or failures are explicitly documented.
- `sprint4c_hparam_search_all.csv` contains all completed runs.
- `sprint4c_best_by_combination.csv` has one best validation row per searched combination.
- `sprint4c_vs_canonical_summary.csv` includes validation and test deltas against matched canonical
  Sprint 4 rows.
- Stop/go decision is recorded in `docs/report_notes/sprint-4c-results-tr.md`.

## Report Interpretation

If Sprint 4C improves the best canonical macro-F1:

> Sprint 4C shows that final-stage classifier/fusion calibration matters even after CNN fine-tuning.
> The CNN feature extractors were unchanged; the improvement came from validation-selected MLP/fusion
> hyperparameters on cached features.

If Sprint 4C does not improve the best canonical macro-F1:

> Sprint 4C shows that the canonical Sprint 4 result was not merely limited by weak MLP
> hyperparameters. Combined with Sprint 4B, this supports the conclusion that the remaining macro-F1
> limitation is tied to representation difficulty, class imbalance, calibration, and visually similar
> HAM10000 classes.

In both cases:

- Do not replace canonical Sprint 4 unless validation-selected Sprint 4C improves test macro-F1 with
  acceptable accuracy/weighted-F1 tradeoff.
- Keep all claims benchmark-focused and non-clinical.

## Artifact Hygiene

Do not commit:

- `artifacts/runs/sprint4c_hparam_search/**`
- `artifacts/features/**/*.pt`
- `artifacts/checkpoints/**/*.pt`
- `model.pt`
- large logs

Small CSV/PNG report assets may be versioned only if consistent with the rest of the project report
asset policy.
