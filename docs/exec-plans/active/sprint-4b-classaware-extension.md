# Sprint 4B Execution Plan: Class-Aware Fine-Tuning Extension

## Purpose

Sprint 4B is an exploratory extension after the completed canonical Sprint 4 fine-tuning matrix.
Sprint 4 remains the assignment-complete canonical result. Sprint 4B tests whether a controlled
class-aware fine-tuning variant can improve macro-F1 and minority-class behavior without changing
the core cached-feature MLP/fusion protocol.

The extension is intentionally narrow: one main class-aware fine-tuning family plus one deeper
ResNet50 probe. It should strengthen the final report discussion, not become an uncontrolled search.

## Current Baseline

Canonical Sprint 4 result:

- Feature source: `finetuned`
- Fine-tuning policy:
  - ResNet50: `layer4` + `fc`
  - MobileNetV2: `features[16]`, `features[17]`, `features[18]` + `classifier`
  - EfficientNetB0: `features[7]`, `features[8]` + `classifier`
- Best fine-tuned single backbone: ResNet50, macro-F1 `0.658`
- Best fine-tuned fusion: `r50+mnv2+effb0 concat`, macro-F1 `0.706`, accuracy `0.811`,
  weighted-F1 `0.813`
- Best frozen Sprint 3 result: `r50+effb0 concat`, macro-F1 `0.595`

Sprint 4B should compare against these results and should not overwrite or relabel them.

## Scientific Questions

1. Does a more explicitly class-aware fine-tuning objective improve validation macro-F1 compared
   with canonical Sprint 4?
2. Does class-aware fine-tuning improve per-class precision/recall/F1 for minority or visually
   similar classes such as `akiec`, `bcc`, `df`, `mel`, and `vasc`?
3. Is deeper partial unfreezing worth expanding beyond the conservative Sprint 4 policy?

## Main Experiment: `finetuned_classaware`

Keep the same canonical Sprint 4 unfreeze policy, but change the fine-tuning objective.

Included:

- Fine-tune all three ImageNet-pretrained backbones on Colab GPU.
- Use the canonical Sprint 4 unfreeze policy.
- Use a mild class-aware loss:
  - preferred starting point: class-balanced focal loss,
  - gamma `1.0`,
  - train-split class weights only,
  - no weighted sampler in the first pass,
  - optional class-weight smoothing/clipping if raw weights are too aggressive.
- Select checkpoints by validation macro-F1.
- Extract classifier-free cached features under
  `artifacts/features/ham10000/finetuned_classaware/<backbone>/`.
- Run single-backbone MLP screening first.
- Expand to the full 11-run fusion matrix only if stop/go criteria pass.

Excluded from the first pass:

- Weighted sampler or balanced batch sampling.
- Test-time augmentation.
- Grad-CAM.
- Deeper + class-aware combined training.
- Any test-metric-based model selection.

## Secondary Probe: `finetuned_deeper`

Run one deeper partial fine-tuning screen for ResNet50 only.

Policy:

- Feature source: `finetuned_deeper`
- Backbone: ResNet50 only for the first probe.
- Trainable blocks: `layer3`, `layer4`, and `fc`.
- Use lower backbone learning rate than canonical Sprint 4.
- Keep validation macro-F1 checkpoint selection.
- Extract features under `artifacts/features/ham10000/finetuned_deeper/resnet50/`.
- Train only a single-backbone MLP for screening.

Do not run deeper MobileNetV2/EfficientNetB0 or a deeper fusion matrix unless the ResNet50 probe
shows a clear validation macro-F1 gain and stable validation curves.

## Run Count

Must-run screening phase:

| Experiment | CNN fine-tuning runs | Feature extraction jobs | MLP runs | Purpose |
|---|---:|---:|---:|---|
| `finetuned_classaware` | 3 | 3 | 3 single-backbone | Main class-aware screening |
| `finetuned_deeper` ResNet50 | 1 | 1 | 1 single-backbone | Deeper unfreezing probe |
| Total | 4 | 4 | 4 | Controlled Sprint 4B screen |

Conditional expansion:

- If `finetuned_classaware` passes stop/go: run the remaining 8 fusion MLP runs for a full 11-run
  class-aware matrix.
- If `finetuned_deeper` ResNet50 passes stop/go: decide whether to fine-tune the other two
  backbones before any deeper fusion work.

## Stop/Go Criteria

All decisions must be based on validation metrics, not test metrics.

Use the fine-tuned CNN validation macro-F1 for checkpoint selection. Use the downstream
single-backbone MLP validation macro-F1 to decide whether a feature source should expand to fusion.

Expand `finetuned_classaware` to the full 11-run matrix if at least one is true:

- Best class-aware single-backbone MLP validation macro-F1 beats the matched canonical Sprint 4
  single-backbone validation macro-F1 by at least `0.015`.
- At least two of three class-aware backbones improve over their matched canonical Sprint 4
  single-backbone validation macro-F1 by at least `0.010`.

And all of these must also hold:

- Accuracy or weighted-F1 does not collapse by more than roughly `0.03`.
- Per-class validation behavior does not show obvious rare-class precision collapse.
- Training curves do not show severe overfitting after the selected epoch.

Stop after screening if:

- Gains appear only on test but not validation.
- Minority-class recall improves while precision collapses badly.
- Validation macro-F1 is flat or worse than canonical Sprint 4.
- Run count would threaten final report quality.

Expand `finetuned_deeper` beyond ResNet50 only if:

- Deeper ResNet50 single-backbone MLP validation macro-F1 beats canonical ResNet50 by at least
  `0.020`.
- Validation loss/accuracy curves remain stable.
- The gain is large enough to justify two more CNN fine-tuning runs.

## Suggested Configs

Add these configs only when implementation begins:

- `configs/experiments/sprint4b_classaware_backbones.yaml`
- `configs/experiments/sprint4b_classaware_feature_matrix.yaml`
- `configs/experiments/sprint4b_deeper_screen.yaml`

Recommended public feature-source names:

- `finetuned_classaware`
- `finetuned_deeper`

Keep more detailed settings in resolved configs, not in long public names. For example:

```yaml
feature_source: finetuned_classaware
selection_metric: val_macro_f1
loss:
  name: class_balanced_focal
  gamma: 1.0
  class_weights: train_only
  weight_smoothing: true
sampler:
  name: none
unfreeze_policy:
  name: canonical_sprint4
augmentation:
  name: sprint4_conservative
```

## Colab Runner

Sprint 4B should use the same Colab pattern as Sprint 4:

- Full runs execute on Colab GPU.
- Repo is cloned or pulled under `/content/dl-assignment`.
- Dataset bundle is restored from Drive.
- Notebook remains a thin runner around scripts.
- Outputs can be mirrored to `/content/drive/MyDrive/dl-midterm-artifacts/`.
- Smoke-test outputs must not be mixed into canonical local results.

Suggested notebook:

- `notebooks/04_sprint4b_classaware.ipynb`

The notebook should only orchestrate commands, not contain core training logic.

## Expected Commands

Exact commands may change during implementation, but the run shape should remain:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/sprint4b_classaware_backbones.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware \
  --batch-size 32

uv run python scripts/train_mlp.py \
  --config configs/experiments/sprint4b_classaware_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware
```

If stop/go passes, run the class-aware fusion matrix:

```bash
uv run python scripts/run_experiment_matrix.py \
  --config configs/experiments/sprint4b_classaware_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_classaware
```

For the deeper ResNet50 probe:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/sprint4b_deeper_screen.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_deeper \
  --backbone resnet50 \
  --batch-size 32
```

## Report-Ready Outputs

Must-produce screening tables:

- `artifacts/report_assets/tables/sprint4b_screening_results.csv`
- `artifacts/report_assets/tables/sprint4b_vs_canonical_single_backbone.csv`
- `artifacts/report_assets/tables/sprint4b_per_class_f1_gain.csv`

Must-produce screening figures:

- `artifacts/report_assets/figures/sprint4b_val_macro_f1_screening.png`
- `artifacts/report_assets/figures/sprint4b_test_macro_f1_vs_canonical.png`
- `artifacts/report_assets/figures/sprint4b_per_class_f1_gain_heatmap.png`

If full class-aware matrix is run:

- `artifacts/report_assets/tables/sprint4b_classaware_all_results.csv`
- `artifacts/report_assets/tables/sprint4b_classaware_vs_canonical_fusion_summary.csv`
- `artifacts/report_assets/tables/sprint4b_classaware_fusion_weight_summary.csv`
- `artifacts/report_assets/figures/sprint4b_classaware_fusion_comparison.png`
- `artifacts/report_assets/figures/sprint4b_classaware_concat_vs_weighted.png`
- `artifacts/report_assets/figures/sprint4b_classaware_learned_fusion_weights.png`
- `artifacts/report_assets/figures/sprint4b_best_confusion_matrix.png`

Optional only after the quantitative work is stable:

- `artifacts/report_assets/figures/gradcam_examples_resnet50_canonical_vs_classaware.png`

Grad-CAM, if used, must be framed as qualitative benchmark visualization for a single fine-tuned
backbone, not clinical interpretability.

## Implementation Tasks

- Add class-aware loss support in `src/dl_midterm/training/` without changing canonical Sprint 4
  defaults.
- Add or extend config parsing for loss name, focal gamma, and class-weight smoothing/clipping.
- Add `finetuned_classaware` and `finetuned_deeper` feature-source expansion.
- Add deeper ResNet50 unfreeze policy.
- Add report exporters for Sprint 4B screening and optional full class-aware matrix.
- Add thin Colab runner notebook.
- Update `docs/COMMANDS.md`, `docs/DECISIONS.md`, and final report notes after results exist.

## Tests

Add or update tests for:

- Class-balanced focal loss shape, finite output, and gradient path.
- Class-weight smoothing/clipping behavior.
- Canonical Sprint 4 defaults remaining unchanged.
- `finetuned_classaware` feature-source path expansion.
- `finetuned_deeper` ResNet50 unfreeze policy.
- Screening matrix expansion producing only single-backbone runs before stop/go.
- Optional full class-aware matrix expansion producing 11 runs.
- No raw image paths entering the cached-feature MLP stage.

Run before committing implementation:

```bash
uv run ruff check src scripts tests
uv run pytest
```

## Artifact Policy

Do not commit:

- raw images,
- feature cache `.pt` files,
- fine-tuned checkpoints,
- MLP `model.pt`,
- full run folders,
- Colab runtime artifacts.

Small report-ready CSV/PNG assets may be committed after checking their size and relevance.

## Reporting Interpretation

If Sprint 4B improves results:

> After completing the canonical frozen and fine-tuned matrices, Sprint 4B tested whether a
> class-aware fine-tuning objective could improve balanced HAM10000 benchmark classification.
> The extension preserved the same split, backbones, feature extraction protocol, cached-feature
> MLP/fusion evaluation, and validation macro-F1 selection rule. Compared with canonical Sprint 4,
> the class-aware variant improved macro-F1 by X, especially on classes Y and Z, with runtime and
> precision/recall tradeoffs discussed separately.

If Sprint 4B does not improve results:

> Sprint 4B did not outperform the conservative Sprint 4 fine-tuning strategy under validation
> macro-F1. This suggests that canonical last-block fine-tuning already provided a strong
> bias-variance tradeoff for this dataset. The negative result is still useful because stronger
> minority emphasis may improve recall while reducing precision or stability.

In both cases:

- Keep macro-F1 as the primary ranking metric.
- Report accuracy and weighted-F1, but do not use them to hide minority-class weaknesses.
- Treat `df` and `vasc` gains cautiously because their test support is low.
- Avoid clinical diagnosis claims.

## Current Status

Implementation scaffold is complete locally; full Colab GPU runs and result interpretation are
still pending.

Completed locally:

- Added class-aware loss support with class-balanced focal loss, train-only class weights,
  smoothing, and clipping.
- Added `finetuned_classaware` and `finetuned_deeper` configs and feature-source routing.
- Added ResNet50 `layer3_layer4` unfreeze policy with optional backbone/head learning-rate groups.
- Added Sprint 4B screening and optional full-matrix report exporters.
- Added thin Colab runner `notebooks/04_sprint4b_classaware.ipynb`.
- Added tests for focal-loss gradients, class-weight smoothing/clipping, canonical Sprint 4
  defaults, feature-source path expansion, deeper ResNet50 unfreezing, screening expansion, full
  11-run matrix expansion, and cached-feature MLP boundaries.

Pending:

- Run class-aware fine-tuning on Colab GPU.
- Run class-aware single-backbone MLP screening and deeper ResNet50 MLP screening.
- Apply validation-based stop/go before any full class-aware fusion matrix.
- Generate and interpret real Sprint 4B tables/figures.
- Move this execution plan to `docs/exec-plans/completed/` only after the Sprint 4B runs and
  result notes are complete.
