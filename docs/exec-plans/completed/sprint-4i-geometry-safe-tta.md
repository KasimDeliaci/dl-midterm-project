# Sprint 4I Geometry-Safe TTA Refinement

## Goal

Evaluate whether deterministic, geometry-preserving test-time augmentation can improve the best Sprint 4 / Sprint 4C fine-tuned fusion models without retraining or changing the lesion-aware data splits.

## Scope

- Use existing fine-tuned backbone checkpoints and restored MLP `model.pt` artifacts.
- Keep selection validation-gated: choose TTA policy using validation macro-F1 only, then evaluate the selected policy on test.
- Report accuracy, macro precision, macro recall, macro-F1, weighted-F1, per-class metrics, confusion matrices, and runtime.
- Produce compact report-ready visualizations for policy comparison, macro-F1 gains, per-class F1 gains, confusion matrices, and runtime.

## Candidate Policies

- `identity`: baseline inference.
- `tta_hflip2`: identity + horizontal flip.
- `tta_vflip2`: identity + vertical flip.
- `tta_flip4`: identity + horizontal, vertical, and horizontal+vertical flips.
- `tta_rot4`: identity + 90/180/270 rotations.
- `tta_d4_8`: full deterministic dihedral group with rotations and mirrored rotations.

These transforms preserve the full image frame. No crop, affine warp, color jitter, or stochastic augmentation is used.

## Models

- Canonical Sprint 4 three-backbone concat MLP.
- Best Sprint 4C three-backbone weighted MLP.

Both use the canonical Sprint 4 fine-tuned backbone checkpoints.

## Validation Gate

- Selection metric: validation macro-F1.
- Minimum validation macro-F1 gain: 0.005 over identity.
- Maximum validation accuracy drop: 0.020.
- Maximum validation weighted-F1 drop: 0.020.
- Identity sanity check against recorded validation macro-F1 must pass before test evaluation.

## Outputs

- `artifacts/report_assets/tables/sprint4i/`
- `artifacts/report_assets/figures/sprint4i/`
- Ignored prediction CSVs under `artifacts/runs/sprint4i_tta_refinement/predictions/`

## Risks

- TTA can improve rare-class recall but also hurt calibration or majority-class precision.
- Because policy selection is validation-gated, the result remains single-seed and should be reported as an inference-time refinement, not a new training breakthrough.
- Test results must not be used to choose among policies after the fact.
