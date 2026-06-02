# Sprint 4F Execution Plan: Augmented Three-Backbone Fine-Tuned Fusion

Status: active. Implementation scaffolding complete; Colab execution not started.

## Purpose

Sprint 4F moves back to image-level training because Sprint 4E showed that fusion-stage MLP
diagnostics alone did not beat the stronger Sprint 4 and Sprint 4D results. The main goal is to
test whether training-time augmentation improves the three-backbone fine-tuned fusion result.

Scientific question:

> Can augmentation-aware fine-tuning of ResNet50, MobileNetV2, and EfficientNetB0 produce stronger
> feature caches for the best three-backbone fusion setup?

This is a higher-impact experiment than more cached-feature fusion micro-tuning because Sprint 4D
showed that transformation averaging at inference time helped, especially for the weighted fusion
model. Sprint 4F tests whether similar robustness can be learned during fine-tuning.

## Current Context

Canonical Sprint 4:

- Best canonical fine-tuned fusion: `resnet50 + mobilenet_v2 + efficientnet_b0 concat`
- Test macro-F1: approximately `0.706`
- Uses `finetuned` feature source.

Sprint 4D:

- Validation-gated TTA improved Sprint 4C weighted fusion from test macro-F1 `0.699` to `0.733`.
- This suggests that geometric robustness is useful.

Sprint 4E:

- Feature normalization helped concat validation behavior but did not beat canonical Sprint 4 or
  Sprint 4D on test.
- This suggests that higher-impact work should target image-level training rather than more MLP
  fusion tuning.

## Scope

Included:

- Colab GPU execution for image-level fine-tuning.
- Keep notebooks thin; core logic remains in `src/dl_midterm/` and `scripts/`.
- Use the fixed Sprint 1 lesion-aware splits.
- Fine-tune all three backbones separately with stronger but still plausible augmentation:
  - ResNet50,
  - MobileNetV2,
  - EfficientNetB0.
- Produce a new feature source, not overwriting canonical Sprint 4:
  - `finetuned_augmented`
- Evaluate the best three-backbone fusion path first:
  - 3 single-backbone MLP runs,
  - 1 three-backbone concat run,
  - 1 three-backbone weighted run.
- Select by validation macro-F1 only.
- Test only pre-registered or validation-selected candidates.
- Persist full operational artifacts to Drive so checkpoint loss does not repeat.

Excluded:

- Changing train/validation/test splits.
- Test-set model selection.
- Mixing Sprint 4F artifacts into canonical Sprint 4 tables without explicit labeling.
- Heavy uncontrolled augmentation such as hair removal, strong color transforms, blur/sharpen
  pipelines, or arbitrary preprocessing searches.
- End-to-end joint training of all three CNNs as one giant model.
- Clinical diagnosis claims.

## Colab Runner Policy

Use the existing `notebooks/03_finetune_backbones.ipynb` pattern as the reference. Add a dedicated
Sprint 4F runner if needed, for example:

- `notebooks/04_sprint4f_augmented_finetuning.ipynb`

The notebook should:

1. Mount Drive.
2. Clone or pull the repo under `/content/dl-assignment`.
3. Restore `ham10000_colab_bundle.tar`.
4. Install with `uv`.
5. Run script commands only; do not embed project logic in notebook cells.
6. Mirror outputs to Drive.

Important artifact rule:

- Do not use `rsync --exclude 'model.pt'` for Sprint 4F Drive mirror.
- Do not exclude backbone checkpoints, feature caches, or MLP checkpoints from Drive.
- Still keep these large artifacts out of Git locally.

Suggested Drive root:

- `/content/drive/MyDrive/dl-midterm-artifacts/sprint4f/`

## Artifact Layout

Colab and local paths should remain compatible with existing scripts:

- Checkpoints:
  - `artifacts/checkpoints/finetuned_augmented_backbones/resnet50_best.pt`
  - `artifacts/checkpoints/finetuned_augmented_backbones/mobilenet_v2_best.pt`
  - `artifacts/checkpoints/finetuned_augmented_backbones/efficientnet_b0_best.pt`
- Feature caches:
  - `artifacts/features/ham10000/finetuned_augmented/resnet50/`
  - `artifacts/features/ham10000/finetuned_augmented/mobilenet_v2/`
  - `artifacts/features/ham10000/finetuned_augmented/efficientnet_b0/`
- Run folders:
  - `artifacts/runs/sprint4f_augmented_finetuning/`
  - `artifacts/runs/sprint4f_augmented_feature_matrix/`
- Report assets:
  - `artifacts/report_assets/tables/sprint4f/`
  - `artifacts/report_assets/figures/sprint4f/`

Do not commit:

- `.pt`, `.pth`, `.ckpt`
- feature cache files
- full run folders
- raw images or Drive bundles

Commit candidates:

- configs,
- scripts,
- notebook thin runner,
- tests,
- report-ready small CSV/PNG assets after size check,
- report notes.

## Fine-Tuning Strategy

### Stage A: Augmented Last-Block Fine-Tuning

Start with the same conservative unfreeze policy as canonical Sprint 4:

- ResNet50: `layer4 + fc`
- MobileNetV2: final feature blocks + classifier
- EfficientNetB0: final feature blocks + classifier

Use ImageNet pretrained weights, train-split class weights, early stopping, and validation macro-F1
selection.

Recommended initial training settings:

- seed: `42`
- epochs: `12-15`
- batch size: `32` if Colab memory allows, otherwise `16`
- backbone LR: `3e-5` if discriminative LR is implemented
- head LR: `1e-4`
- weight decay: `1e-4`
- early stopping patience: `4`
- selection metric: validation macro-F1
- mixed precision: enabled on CUDA

Recommended augmentation policy:

- random horizontal flip,
- random vertical flip,
- random rotation up to `30` degrees,
- mild random resized crop, e.g. scale `[0.85, 1.0]`,
- mild color jitter, lower than aggressive natural-image augmentation,
- optional random affine with small translate/scale.

Avoid aggressive augmentation in the first pass. The aim is robustness, not changing lesion
appearance.

### Stage B: Optional Deeper Probe

Only run deeper unfreezing if Stage A does not improve validation macro-F1 enough or if training
curves show underfitting.

Low-risk first deeper probe:

- ResNet50: `layer3 + layer4 + fc`

Do not immediately run deeper unfreezing for all three backbones. MobileNetV2 and EfficientNetB0
deeper policies require explicit implementation and freeze/unfreeze tests before Colab execution.

Stop/go rule:

- Expand deeper probes only if validation macro-F1 improves by at least `+0.010` over the matched
  augmented last-block backbone or training curves clearly show underfitting.

## Feature Extraction

After each accepted backbone checkpoint:

- extract deterministic feature caches with no stochastic augmentation,
- verify train/validation/test cache shape and split alignment,
- write manifests for each split and backbone,
- mirror caches to Drive.

Feature extraction must use deterministic resize/ImageNet normalization, not train-time random
augmentation.

## Fusion Matrix

Run the small high-impact matrix first:

| Run type | Backbones | Fusion |
|---|---|---|
| single | ResNet50 | none |
| single | MobileNetV2 | none |
| single | EfficientNetB0 | none |
| three-backbone | ResNet50 + MobileNetV2 + EfficientNetB0 | concat |
| three-backbone | ResNet50 + MobileNetV2 + EfficientNetB0 | weighted |

Do not start with the full 11-run matrix. If the three-backbone concat or weighted run improves
over canonical Sprint 4 by validation macro-F1, then expand to pairwise concat/weighted runs.

Optional final audit:

- Apply Sprint 4D-style validation-gated TTA only to the best Sprint 4F model after the fine-tuned
  feature/fusion result is selected.

## Stop/Go Criteria

Use validation macro-F1 for decisions.

Go to test if:

- a Sprint 4F candidate improves validation macro-F1 over the matched canonical Sprint 4 candidate
  by at least `+0.010`, and
- accuracy drop is not worse than `-0.020`, and
- weighted-F1 drop is not worse than `-0.020`, and
- the gain is not explained only by `df` or `vasc`.

Test policy:

- Test only pre-registered baselines and the best 1-2 validation-selected Sprint 4F candidates.
- Do not add new candidates after seeing test metrics.

Interpretation:

| Test macro-F1 gain vs canonical Sprint 4 concat | Interpretation |
|---:|---|
| `< +0.005` | noise-level |
| `+0.005` to `+0.010` | suggestive |
| `+0.010` to `+0.020` | useful |
| `> +0.020` | meaningful Sprint 4F improvement |

## Expected Outputs

Tables:

- `artifacts/report_assets/tables/sprint4f/sprint4f_finetuning_summary.csv`
- `artifacts/report_assets/tables/sprint4f/sprint4f_feature_cache_audit.csv`
- `artifacts/report_assets/tables/sprint4f/sprint4f_single_backbone_results.csv`
- `artifacts/report_assets/tables/sprint4f/sprint4f_fusion_results.csv`
- `artifacts/report_assets/tables/sprint4f/sprint4f_vs_canonical_summary.csv`
- `artifacts/report_assets/tables/sprint4f/sprint4f_per_class_f1.csv`

Figures:

- per-backbone fine-tuning training curves,
- per-backbone fine-tuning confusion matrices,
- single-backbone vs three-backbone fusion macro-F1 comparison,
- canonical Sprint 4 vs Sprint 4F macro-F1 comparison,
- concat vs weighted Sprint 4F comparison,
- per-class F1 gain heatmap,
- best Sprint 4F confusion matrix.

Visual standards:

- y-axis metric scale should be `0-1` where applicable,
- labels must be readable,
- bar plots should be sorted where useful,
- avoid crowded all-in-one plots,
- use consistent color semantics for canonical, augmented, concat, and weighted.

## Tests

Add or update tests for:

- augmentation config parsing and transform construction,
- deterministic feature extraction uses eval transform, not train augmentation,
- new `finetuned_augmented` feature source expansion,
- three-backbone Sprint 4F matrix expansion,
- no raw image leakage into cached-feature MLP stage,
- Colab mirror command documentation does not exclude model/checkpoint/cache artifacts.

If deeper policies are implemented:

- add freeze/unfreeze tests for each new policy before Colab execution.

## Documentation Updates

During or after implementation:

- update `docs/COMMANDS.md` with exact Colab and local commands,
- update `docs/DECISIONS.md` with augmentation and artifact persistence decisions,
- update `docs/planning/5-sprint-project-plan.md` with Sprint 4F status,
- create `docs/report_notes/sprint-4f-results-tr.md`,
- move this plan to `docs/exec-plans/completed/` only after the run and verification finish.

## Implementation Scaffolding - 2026-06-02

Implemented before Colab execution:

- configurable train-time augmentation in `src/dl_midterm/data/transforms.py`;
- augmentation config plumbing through fine-tuning dataloaders and run metadata;
- `configs/experiments/sprint4f_augmented_backbones.yaml`;
- `configs/experiments/sprint4f_augmented_feature_matrix.yaml`;
- `notebooks/04_sprint4f_augmented_finetuning.ipynb`;
- `tests/test_sprint4f_augmented.py`;
- command and decision documentation for Sprint 4F.

Remaining work:

- run the notebook on Colab GPU;
- mirror full operational artifacts to Drive without excluding `.pt` files;
- download or sync report-ready assets locally;
- verify cache alignment and generated metrics;
- create `docs/report_notes/sprint-4f-results-tr.md`;
- move this plan to `completed/` after results are verified.

## Report Framing

If Sprint 4F improves:

> Sprint 4F shows that the main performance bottleneck was not only final fusion tuning. Training
> the CNN backbones with controlled augmentation produced more robust fine-tuned representations,
> and the three-backbone fusion result improved under validation-selected evaluation.

If Sprint 4F does not improve:

> Sprint 4F tested whether controlled image-level augmentation could improve the strongest
> three-backbone fine-tuned fusion setup. The validation/test results did not reliably exceed
> canonical Sprint 4 or Sprint 4D, suggesting that the remaining ceiling may require either more
> data, stronger architecture changes, or multi-seed validation rather than further small
> augmentation/fusion changes.

Always mention:

- fixed lesion-aware split,
- validation-only selection,
- benchmark dermoscopic image classification,
- low-support class caution for per-class claims,
- Sprint 4F artifacts are separate from canonical Sprint 4 unless explicitly promoted.

## Completion - 2026-06-02

Sprint 4F was run on Colab T4 and mirrored to Drive under
`/content/drive/MyDrive/dl-midterm-artifacts/sprint4f/`. The local project restored the report-ready
tables/figures plus gitignored operational artifacts needed for auditability:

- 3 backbone checkpoints in `artifacts/checkpoints/finetuned_augmented_backbones/`;
- 9 deterministic fine-tuned augmented feature cache `.pt` files under
  `artifacts/features/ham10000/finetuned_augmented/`;
- 5 MLP `model.pt` files for the Sprint 4F single-backbone and three-backbone fusion runs.

Main test macro-F1 results:

- augmented ResNet50 single-backbone: `0.589`;
- augmented MobileNetV2 single-backbone: `0.575`;
- augmented EfficientNetB0 single-backbone: `0.579`;
- augmented three-backbone concat: `0.645`;
- augmented three-backbone weighted: `0.615`.

Sprint 4F did not improve over canonical Sprint 4 concat (`0.706`) or Sprint 4D weighted +
`tta_rot4` (`0.733`). It is retained as a negative but informative augmentation experiment rather
than promoted to the project best result.
