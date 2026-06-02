# Sprint 4D Execution Plan: Validation-Gated Test-Time Augmentation Audit

Status: completed. Implementation and validation-gated test audit are complete. Full evaluation used
the local regenerated
Sprint 4 concat MLP checkpoint documented below. The exact May 29 canonical `model.pt` artifact was
not present in the Drive mirror because MLP checkpoints were not persisted from the Colab runtime.

## Purpose

Sprint 4D is a small post-hoc inference extension after the completed canonical Sprint 4,
Sprint 4B, and Sprint 4C experiments. It does not train new CNNs, change dataset splits, or tune
new MLP classifiers. It asks whether deterministic test-time augmentation (TTA) improves inference
robustness for already selected fine-tuned fusion models.

Scientific question:

> Given the fixed lesion-aware HAM10000 split and already trained fine-tuned CNN/MLP fusion models,
> does conservative inference-time augmentation improve validation-selected macro-F1 without
> introducing test-set shopping or an uncontrolled preprocessing search?

Canonical Sprint 4 remains the main project result unless Sprint 4D passes the validation gate and
improves the test audit under the pre-registered policy.

## Current Baselines

Canonical Sprint 4 best:

- Run ID: `20260529_173806_finetuned_r50-mnv2-effb0_concat_mlp_s42`
- Feature source: `finetuned`
- Backbones: `resnet50 + mobilenet_v2 + efficientnet_b0`
- Fusion: `concat`
- Validation macro-F1: `0.655`
- Test accuracy: `0.811`
- Test macro-F1: `0.706`
- Test weighted-F1: `0.813`

Local Sprint 4D checkpoint restoration:

- Exact May 29 MLP `model.pt` files were not found in the local repo, Drive download, or May 29
  artifact bundle.
- The May 29 run folders contain `metrics.json`, `history.csv`, `config_resolved.yaml`, and figures,
  but not the MLP checkpoint.
- For Sprint 4D only, the three-backbone concat MLP was regenerated locally from the existing
  fine-tuned feature caches using the same configured matrix order and seed.
- Restored run ID:
  `20260602_095314_finetuned_r50-mnv2-effb0_concat_mlp_restore_matrix_replay_s42`
- Restored run dir:
  `artifacts/runs/sprint4d_restore_matrix/20260602_095314_finetuned_r50-mnv2-effb0_concat_mlp_restore_matrix_replay_s42`
- Restored validation macro-F1: `0.664`
- Restored test macro-F1: `0.685`

This restored checkpoint is not a replacement for the canonical Sprint 4 reported result. It is a
local inference artifact that makes the Sprint 4D TTA audit runnable. Any Sprint 4D result using it
must be described as using a regenerated Sprint 4 MLP checkpoint.

Sprint 4C validation-selected weighted candidate:

- Run ID: `20260531_213529_finetuned_r50-mnv2-effb0_weighted_mlp_w_cw_adamw_low_lr_p512_s42`
- Feature source: `finetuned`
- Backbones: `resnet50 + mobilenet_v2 + efficientnet_b0`
- Fusion: `weighted`
- Candidate: `w_cw_adamw_low_lr_p512`
- Validation macro-F1: `0.680`
- Test accuracy: `0.802`
- Test macro-F1: `0.699`
- Test weighted-F1: `0.808`

Sprint 4C showed that final-stage MLP/fusion tuning improved some matched baselines but did not
reliably beat the canonical Sprint 4 concat result. Sprint 4D should test the same question for
inference-time augmentation: useful robustness check, not a new broad search.

## Definition of TTA in This Project

For each original image, Sprint 4D creates a small set of deterministic label-preserving views.
Each view is passed through the already fine-tuned CNN feature extractors and the already trained
MLP/fusion classifier. Final probabilities are averaged across views:

```text
p_final = mean(softmax(logits_view_1), ..., softmax(logits_view_n))
```

No training occurs:

- Do not update CNN weights.
- Do not update MLP/fusion weights.
- Do not update BatchNorm statistics.
- Do not tune thresholds.
- Do not change class weights.
- Do not change the train/validation/test split.

Use `model.eval()` and `torch.no_grad()` for every inference path.

## Scope

Included:

- Local MPS/CPU inference using existing image files, checkpoints, and MLP `model.pt` files.
- Validation-first TTA policy screening.
- Test evaluation only for pre-registered model-policy pairs that pass the validation gate.
- Probability averaging across deterministic TTA views.
- Report-ready tables/figures and a Turkish result note.

Excluded:

- New fine-tuning or retraining.
- New feature-cache training runs.
- Full 11-run matrix TTA.
- Random augmentations.
- Color jitter, brightness/contrast changes, blur/sharpen filters, hair-removal preprocessing,
  crops, large affine transforms, or color normalization experiments.
- Test-set selection or changing TTA policy after test metrics are observed.
- Clinical diagnosis claims.

## Models

### Test-Eligible Models

Only these two models are eligible for test evaluation, and only if they pass validation criteria:

1. Canonical Sprint 4 best concat model:
   - run ID: `20260529_173806_finetuned_r50-mnv2-effb0_concat_mlp_s42`
   - fusion: `concat`
   - backbones: `[resnet50, mobilenet_v2, efficientnet_b0]`

2. Sprint 4C validation-selected weighted model:
   - run ID: `20260531_213529_finetuned_r50-mnv2-effb0_weighted_mlp_w_cw_adamw_low_lr_p512_s42`
   - fusion: `weighted`
   - projection dim: `512`
   - backbones: `[resnet50, mobilenet_v2, efficientnet_b0]`

### Optional Validation-Only Diagnostic

One optional diagnostic may be run on validation only:

- Fine-tuned ResNet50 single-backbone canonical MLP:
  - run ID: `20260529_173629_finetuned_r50_none_mlp_s42`
  - fusion: `none`
  - backbone: `[resnet50]`

Do not evaluate this diagnostic on test unless the plan is explicitly revised before any Sprint 4D
test results are produced.

## TTA Policies

### Required Policies

`identity`:

- `identity`

`tta_flip4`:

- `identity`
- horizontal flip
- vertical flip
- horizontal + vertical flip

Rationale: flips are mild and usually label-preserving for dermoscopic benchmark images.

### Optional Validation-Only Policy

`tta_rot4`:

- `identity`
- rotate 90 degrees
- rotate 180 degrees
- rotate 270 degrees

Rationale: dermoscopic orientation is usually not class-defining, but rotations can introduce
interpolation and distribution-shift risk. Use this only on validation unless it clearly passes the
gate.

### Deferred Policy

`tta_fliprot8` is deferred. Do not implement it in the first pass unless `tta_flip4` or `tta_rot4`
show clear validation gains and a larger policy is explicitly justified before test evaluation.

## Split and Leakage Rules

Use only the existing Sprint 1 split files:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

Sprint 4D must not create new splits. It must not move images between splits. It must not use test
metrics for policy choice, model choice, threshold choice, or aggregation-rule choice.

Validation is the only split used for TTA policy selection. Test is evaluated once after decisions
are frozen.

Implementation should verify:

- Every evaluated image ID comes from the requested split CSV.
- Image order and labels are read from the split CSV.
- The test split is not loaded during validation-only screening except when an explicit final test
  stage is requested.
- Prediction dumps include `split`, `image_id`, `lesion_id`, `true_label`, and `pred_label` so
  split provenance is auditable.

## Stage A: Validation Screening

Run validation only:

| Model | Policies |
|---|---|
| canonical three-backbone concat | `identity`, `tta_flip4`, optional `tta_rot4` |
| Sprint 4C three-backbone weighted | `identity`, `tta_flip4`, optional `tta_rot4` |
| optional ResNet50 single diagnostic | `identity`, `tta_flip4` |

Before comparing TTA policies, run an identity sanity check. Identity inference through the new
Sprint 4D script should reproduce the existing run metrics closely enough to trust the new
inference path. If identity differs materially, stop and debug the script before running TTA.

Primary selection metric:

- validation macro-F1

Secondary guard metrics:

- validation accuracy
- validation weighted-F1
- per-class precision/recall/F1
- runtime multiplier

## Stage B: Test Once

Freeze the decision before loading test:

- selected model or models,
- selected TTA policy,
- probability aggregation rule,
- metrics and artifact outputs.

Then evaluate test once for only the model-policy pairs that passed Stage A. If no policy passes
Stage A, do not run test TTA just to inspect it; write Sprint 4D as a negative validation-gated
extension.

## Stop/Go Criteria

Go to test if all are true:

- TTA improves validation macro-F1 by at least `+0.005` over that model's identity result.
- Validation accuracy does not drop by more than `0.020`.
- Validation weighted-F1 does not drop by more than `0.020`.
- The gain is not explained only by one low-support class such as `df` or `vasc`.
- Per-class precision/recall behavior is not obviously pathological.

Prefer `tta_flip4` over `tta_rot4` if the validation gains are similar. Simpler and milder policies
are easier to justify.

Stop after validation if:

- validation macro-F1 does not improve,
- the improvement is below `+0.005`,
- accuracy/weighted-F1 collapse,
- rare-class recall improves only by severe precision collapse,
- identity sanity check fails.

Interpret test macro-F1 gains conservatively:

| Test macro-F1 gain over identity | Interpretation |
|---:|---|
| `< +0.005` | noise-level, do not claim improvement |
| `+0.005` to `+0.010` | suggestive only |
| `+0.010` to `+0.020` | useful if validation also supports it |
| `> +0.020` | meaningful Sprint 4D improvement |

## Implementation Plan

Add:

- `configs/experiments/sprint4d_tta.yaml`
- `scripts/evaluate_tta.py`
- any small support helpers under `src/dl_midterm/` if the logic is reusable

The script should:

1. Load selected split CSVs.
2. Build deterministic view transforms using existing ImageNet normalization.
3. Load fine-tuned CNN checkpoints for required backbones.
4. Load the selected MLP/fusion `model.pt`.
5. For each image batch and each TTA view:
   - extract backbone features,
   - concatenate/project/fuse according to the selected run,
   - compute softmax probabilities.
6. Average probabilities across views.
7. Compute metrics with the existing evaluation helpers.
8. Write tables, figures, confusion matrices, and a decision log.

Stream features during evaluation rather than writing large TTA feature caches.

## Expected Artifacts

Config:

- `configs/experiments/sprint4d_tta.yaml`

Script:

- `scripts/evaluate_tta.py`

Tables:

- `artifacts/report_assets/tables/sprint4d/sprint4d_validation_results.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_test_results.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_delta_vs_identity.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_per_class_f1_gain.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_runtime_summary.csv`
- `artifacts/report_assets/tables/sprint4d/sprint4d_decision_log.csv`

Figures:

- `artifacts/report_assets/figures/sprint4d/sprint4d_val_policy_comparison.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_test_macro_f1_delta.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_per_class_f1_gain_heatmap.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_best_confusion_matrix.png`
- `artifacts/report_assets/figures/sprint4d/sprint4d_runtime_multiplier.png`

Documentation:

- `docs/report_notes/sprint-4d-results-tr.md`
- update `docs/COMMANDS.md` with exact Sprint 4D commands after implementation
- update `docs/DECISIONS.md` with TTA policy and validation-gating decision
- move this execution plan to `docs/exec-plans/completed/` when finished

Prediction dumps may be useful for debugging but should not be committed if large. If kept, place
them under a gitignored run directory rather than under report assets.

## Tests

Add focused tests for:

- TTA policy expansion (`identity`, `tta_flip4`, optional `tta_rot4`).
- Probability averaging shape and normalization.
- Validation/test gating logic does not select by test metrics.
- Config model expansion resolves the expected run IDs.
- Split provenance: prediction rows must carry the split and image IDs from the split CSV.
- Identity policy contains exactly one view and does not alter labels.

If practical, include a small fake-model test that proves averaged probabilities are used rather
than averaged class labels.

## Local Runtime Notes

Local environment check before implementation:

- Apple MPS is available.
- Fine-tuned backbone checkpoints are present under `artifacts/checkpoints/finetuned_backbones/`.
- Required MLP `model.pt` files are present under `artifacts/runs/`.
- Validation/test split CSVs and image paths are present locally.

Use conservative batch sizes on MPS because TTA multiplies inference cost by number of views.

## Implementation Status - 2026-06-02

Implemented locally:

- Added `configs/experiments/sprint4d_tta.yaml` with pre-registered test-eligible models,
  validation policies, gate thresholds, and smoke-output routing.
- Added `scripts/evaluate_tta.py` for image-level TTA inference, probability averaging, split
  provenance, decision logging, report tables, report figures, and prediction dumps under
  gitignored run artifacts.
- Added `src/dl_midterm/evaluation/tta.py` with deterministic policy expansion, probability
  averaging, and validation-only gate selection helpers.
- Added tests for policy expansion, probability averaging, validation-only gate behavior, tiny-gain
  blocking, and config-level test eligibility.
- Updated `docs/COMMANDS.md`, `docs/DECISIONS.md`, and Sprint report notes.
- Added `.gitignore` rules so Sprint 4D smoke report folders cannot be mistaken for canonical
  report assets.

Validation performed:

- `uv run ruff check src scripts tests` passed.
- `uv run pytest` passed with 41 tests.
- A 16-sample smoke run for `sprint4c_weighted` completed and wrote outputs under
  `artifacts/report_assets/{tables,figures}/sprint4d/sprint4d_smoke_n16/`.
- Full validation screening completed locally on MPS:
  - `canonical_concat`: identity macro-F1 `0.660`, `tta_flip4` `0.695`, `tta_rot4` `0.695`.
  - `sprint4c_weighted`: identity macro-F1 `0.678`, `tta_flip4` `0.703`, `tta_rot4` `0.707`.
- Both test-eligible models passed the validation gate with `tta_rot4`.
- Test was run once for identity and selected `tta_rot4` pairs:
  - regenerated Sprint 4 concat: test macro-F1 `0.685 -> 0.690` (`+0.005`), accuracy
    `0.804 -> 0.814`.
  - Sprint 4C weighted: test macro-F1 `0.699 -> 0.733` (`+0.034`), accuracy `0.803 -> 0.815`.

Artifact policy:

- `model.pt`, backbone checkpoints, prediction dumps, and run folders remain gitignored.
- Small report-ready Sprint 4D tables and figures were written under
  `artifacts/report_assets/{tables,figures}/sprint4d/`.
- Smoke outputs remain under `sprint4d_smoke_n16/` and are not canonical results.

## Report Framing

Suggested summary:

> Sprint 4D evaluated deterministic test-time augmentation as a post-hoc inference strategy for
> already selected fine-tuned fusion models. The TTA policy was selected only on validation
> macro-F1 and then evaluated once on test. Four-view right-angle rotation TTA improved the Sprint
> 4C weighted model's test macro-F1 from `0.699` to `0.733`, with accuracy increasing from `0.803`
> to `0.815`. The regenerated Sprint 4 concat checkpoint showed only a small test macro-F1 gain
> from `0.685` to `0.690`, so the strongest Sprint 4D result is the validation-gated Sprint 4C
> weighted + `tta_rot4` audit.

In all cases:

- Use "benchmark dermoscopic image classification," not clinical diagnosis.
- Report macro-F1 and per-class F1 prominently because HAM10000 is imbalanced.
- Treat `df` and `vasc` per-class movement cautiously because support is low.
- Compare TTA gains against the much larger Sprint 4 fine-tuning gain over Sprint 3 frozen fusion.
