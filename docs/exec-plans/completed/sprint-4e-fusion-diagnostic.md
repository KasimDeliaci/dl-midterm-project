# Sprint 4E Execution Plan: Fusion Training Diagnostic

Status: completed on 2026-06-02.

## Purpose

Sprint 4E is a local cached-feature diagnostic extension after Sprint 4C and Sprint 4D. It does not
fine-tune CNN backbones and does not change the Sprint 1 lesion-aware train/validation/test split.

Scientific question:

> Why did learned weighted fusion underperform or only inconsistently compete with concat fusion,
> and can targeted fusion-stage training changes improve validation-selected macro-F1?

This responds directly to the observation that a more flexible or better-trained fusion stage may be
needed. Sprint 4E should be framed as fusion-stage diagnosis, not as an uncontrolled score chase.

## Current Context

Canonical Sprint 4:

- Best model: `resnet50 + mobilenet_v2 + efficientnet_b0 concat`
- Test macro-F1: approximately `0.706`
- Uses canonical fine-tuned feature caches.

Sprint 4C:

- Best validation-selected tuned model: three-backbone weighted fusion with low LR and projection
  dim `512`.
- Test macro-F1: approximately `0.699`.
- Useful tuning signal, but it still did not beat canonical Sprint 4 concat on test.

Sprint 4D:

- Validation-gated TTA selected `tta_rot4`.
- Sprint 4C weighted + `tta_rot4` reached test macro-F1 approximately `0.733`.
- This suggests inference-time averaging helped the weighted model, but does not explain why the
  fusion training itself underperformed concat before TTA.

## Scope

Included:

- Use existing `finetuned` cached features only.
- Diagnose concat vs weighted fusion using train/validation metrics and feature statistics.
- Add or evaluate lightweight fusion-stage changes:
  - per-backbone feature normalization,
  - weighted-fusion projection capacity,
  - learned weight behavior,
  - per-class weighted fusion,
  - loss/class-imbalance variants if needed.
- Select candidates by validation macro-F1.
- Test only the final 1-2 validation-selected candidates.
- Produce concise report-ready tables and figures.

Excluded:

- New CNN fine-tuning.
- New image-level augmentation training.
- New TTA search beyond using Sprint 4D as an optional final audit for the selected candidate.
- New dataset splits or test-set model selection.
- Large architecture families such as ViT/Swin/CLIP.
- Clinical diagnosis claims.

## Hypotheses

1. **Feature scale mismatch:** ResNet50, MobileNetV2, and EfficientNetB0 cached features may have
   different norm/variance distributions. Concat MLP can partly adapt to this, while weighted fusion
   may be more sensitive to projection scale.
2. **Weighted fusion capacity limit:** The current weighted fusion uses global backbone weights. A
   single global weight per backbone cannot express class-specific complementarity.
3. **Projection bottleneck:** Projection dim `512` may be too narrow for three high-dimensional
   backbone features, especially for minority classes.
4. **Optimization sensitivity:** Weighted fusion may need lower LR, different dropout, or less
   weight decay than concat.
5. **Class imbalance interaction:** Macro-F1 may improve with focal/class-balanced variants, but
   gains must not come from rare-class precision collapse.

## Stage A: Artifact and Feature Diagnostics

No training changes in this stage.

Produce:

- Matched concat vs weighted summary for Sprint 4, Sprint 4C, and Sprint 4D where applicable.
- Learned weighted-fusion weights summary:
  - final normalized weights,
  - whether weights are near-uniform,
  - whether one backbone dominates.
- Per-class F1 gap table:
  - concat minus weighted by class,
  - weighted minus concat by class,
  - focus on `akiec`, `bcc`, `bkl`, `df`, `mel`, `vasc`.
- Feature scale summary for each backbone and split:
  - mean feature norm,
  - std of feature norm,
  - mean absolute activation,
  - per-dimension std aggregate.

Report-ready outputs:

- `artifacts/report_assets/tables/sprint4e/sprint4e_concat_weighted_audit.csv`
- `artifacts/report_assets/tables/sprint4e/sprint4e_feature_scale_summary.csv`
- `artifacts/report_assets/tables/sprint4e/sprint4e_per_class_gap_summary.csv`
- `artifacts/report_assets/figures/sprint4e/sprint4e_feature_norms_by_backbone.png`
- `artifacts/report_assets/figures/sprint4e/sprint4e_concat_vs_weighted_gap.png`
- `artifacts/report_assets/figures/sprint4e/sprint4e_learned_weights_audit.png`

## Stage B: Controlled Fusion Training Experiments

Use local cached features under:

- `artifacts/features/ham10000/finetuned/resnet50/`
- `artifacts/features/ham10000/finetuned/mobilenet_v2/`
- `artifacts/features/ham10000/finetuned/efficientnet_b0/`

Run only three-backbone fusion first:

- `resnet50 + mobilenet_v2 + efficientnet_b0 concat`
- `resnet50 + mobilenet_v2 + efficientnet_b0 weighted`

### Candidate Families

#### 1. Feature Normalization

Compare:

- `none`: current behavior.
- `standardize_per_backbone`: fit mean/std on train features per backbone, apply to val/test.
- `l2_per_backbone`: L2 normalize each backbone feature vector.

Do not fit normalization on validation or test.

#### 2. Weighted Fusion Capacity

Keep the current global weighted fusion but test a small projection/capacity grid:

| Candidate | Projection dim | LR | Dropout | Hidden dims | Class weights |
|---|---:|---:|---:|---|---:|
| `weighted_p512_low_lr` | 512 | 0.0003 | 0.3 | `[512, 256]` | true |
| `weighted_p1024_low_lr` | 1024 | 0.0003 | 0.3 | `[512, 256]` | true |
| `weighted_p1024_reg` | 1024 | 0.0003 | 0.5 | `[512, 256]` | true |

#### 3. More Expressive Fusion

Implement only if Stage B base diagnostics are stable:

- `per_class_weighted`: learn a backbone weight per class, not one global scalar per backbone.

Optional, validation-only diagnostic if time permits:

- `gated_weighted`: example-wise gating network over backbone projections.

Keep gated fusion optional because it has higher overfit risk and is harder to explain cleanly.

#### 4. Loss Variants

Use only for the best 2-3 fusion architecture candidates:

- class-weighted CE baseline,
- focal loss with train-split class weights and modest gamma, e.g. `gamma=1.0`,
- label smoothing with class weights, e.g. `label_smoothing=0.05`.

Do not combine every architecture with every loss. Keep this diagnostic small.

## Suggested MVP Run Matrix

Start with this local MVP:

| Family | Fusion | Normalization | Candidate count |
|---|---|---|---:|
| concat baseline check | concat | none / standardize / l2 | 3 |
| weighted normalization | weighted | none / standardize / l2 | 3 |
| weighted projection | weighted | best normalization from above | 3 |
| per-class weighted | per-class weighted | best normalization from above | 1 |
| loss variants | best 2 candidates | best normalization | 4-6 |

Expected MVP run count: roughly `14-16` cached-feature runs.

This is intentionally smaller than Sprint 4C. If the MVP does not show validation improvement, stop
and write the result as a diagnostic explanation rather than expanding.

## Stop/Go Criteria

All decisions use validation macro-F1.

Go to final test if a candidate satisfies all:

- validation macro-F1 improves over its matched baseline by at least `+0.010`;
- accuracy drop is no worse than `-0.020`;
- weighted-F1 drop is no worse than `-0.020`;
- per-class gains are not explained only by `df` or `vasc`;
- training curve does not show obvious late overfit before selected epoch.

Test policy:

- Test only the best 1-2 validation-selected candidates.
- Compare against:
  - canonical Sprint 4 concat,
  - Sprint 4C weighted,
  - Sprint 4D weighted + `tta_rot4` as an inference-time upper audit.
- Do not choose additional candidates after seeing test results.

Interpret test macro-F1 gain:

| Test macro-F1 gain | Interpretation |
|---:|---|
| `< +0.005` | noise-level |
| `+0.005` to `+0.010` | suggestive only |
| `+0.010` to `+0.020` | useful if validation supports it |
| `> +0.020` | meaningful Sprint 4E improvement |

## Implementation Notes

Prefer adding a dedicated script rather than stretching old matrix scripts too far:

- `scripts/run_fusion_diagnostic.py`
- helpers under `src/dl_midterm/models/` or `src/dl_midterm/evaluation/` only if reusable.

Expected config:

- `configs/experiments/sprint4e_fusion_diagnostic.yaml`

Implementation should support:

- feature normalization fitted on train only,
- concat / weighted / per-class weighted fusion,
- class-weighted CE,
- optional focal loss or label smoothing,
- validation-selected checkpointing,
- report asset export.

Keep run folders under:

- `artifacts/runs/sprint4e_fusion_diagnostic/`

Keep report assets under:

- `artifacts/report_assets/tables/sprint4e/`
- `artifacts/report_assets/figures/sprint4e/`

Do not commit:

- `model.pt`,
- full run folders,
- feature caches,
- prediction dumps.

## Visualization Plan

Keep figures meaningful and not crowded.

Required:

- Feature norm distribution by backbone and split.
- Validation macro-F1 by candidate, sorted descending.
- Matched concat vs weighted macro-F1 gap.
- Learned fusion weights plot for weighted/per-class weighted models.
- Per-class F1 gain heatmap for the final selected candidate.
- Best candidate confusion matrix.
- Training curve for the best 2-3 candidates:
  - train loss,
  - val loss,
  - val macro-F1,
  - train/val accuracy.

Optional if useful:

- Accuracy vs macro-F1 scatter for candidates.
- Runtime vs validation macro-F1 scatter.

Avoid:

- One giant plot containing every candidate and every metric.
- Reporting test-only exploratory plots for non-selected candidates.

## Tests

Add or update tests for:

- train-only feature normalization fit and val/test transform behavior;
- normalized feature shape/alignment;
- per-class fusion weights shape and probability path;
- fusion weights sum to one where applicable;
- validation selection ignores test metrics;
- no raw image access in cached-feature fusion diagnostic training.

## Completed Implementation

Implemented:

- `configs/experiments/sprint4e_fusion_diagnostic.yaml`
- `scripts/run_fusion_diagnostic.py`
- `src/dl_midterm/models/fusion.py` per-class weighted fusion support
- `tests/test_sprint4e_fusion_diagnostic.py`
- report assets under `artifacts/report_assets/tables/sprint4e/`
- report assets under `artifacts/report_assets/figures/sprint4e/`

The local full MVP run trained 14 cached-feature fusion candidates using the existing
`finetuned` feature caches. The run used the fixed lesion-aware split and did not read raw image
paths during the fusion diagnostic stage.

Validation results:

- Best validation macro-F1: `perclass_l2_p512`, macro-F1 `0.683`, accuracy `0.816`.
- Strongest global weighted baseline: `weighted_none_p512_low_lr`, macro-F1 `0.680`.
- Best concat diagnostic: `concat_standardize_base`, macro-F1 `0.662`, improving over
  `concat_none_base` validation macro-F1 `0.639`.

Selection and test policy:

- The pre-registered validation gate selected `concat_standardize_base` and `concat_l2_base` for
  test evaluation.
- `concat_standardize_base` test macro-F1: `0.691`, accuracy `0.790`, weighted-F1 `0.798`.
- `concat_l2_base` test macro-F1: `0.683`, accuracy `0.787`, weighted-F1 `0.795`.

Interpretation:

- Feature normalization helped the concat diagnostic on validation, so feature scale mismatch is a
  plausible contributor to the concat-vs-weighted training behavior.
- Sprint 4E did not produce a validation-gated test result above canonical Sprint 4 concat
  (`0.706`) or Sprint 4D weighted + `tta_rot4` (`0.733`).
- The result is still useful for the report because it narrows the explanation: local fusion-stage
  normalization/capacity changes alone were not enough to explain or surpass the stronger
  fine-tuned/TTA results.

## Expected Report Framing

If Sprint 4E improves:

> Sprint 4E diagnosed the weighted-vs-concat fusion gap using fixed fine-tuned feature caches. After
> fitting normalization only on train features and selecting fusion-stage variants by validation
> macro-F1, the best diagnostic candidate improved test macro-F1 from X to Y. This suggests that the
> earlier weighted-fusion gap was partly due to final-stage training/capacity rather than only
> backbone representation quality.

If Sprint 4E does not improve:

> Sprint 4E tested whether feature normalization, projection capacity, and more expressive weighted
> fusion could close the concat-vs-weighted gap. Validation-selected variants did not reliably beat
> canonical concat, suggesting that concat's simpler full-feature MLP remained the most robust
> cached-feature fusion strategy in this setup.

Always mention:

- fixed lesion-aware split,
- validation-only selection,
- benchmark dermoscopic image classification,
- low-support classes require cautious per-class interpretation.
