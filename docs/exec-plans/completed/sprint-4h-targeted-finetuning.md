# Sprint 4H Targeted Fine-Tuning Plan

## Purpose

Sprint 4H is the next high-impact macro-F1 experiment after Sprint 4F and 4G. Sprint 4F showed
that stronger train-time augmentation with random crop/affine hurt performance, while Sprint 4D
showed that deterministic TTA can help. Sprint 4H therefore tests a more targeted training change:
slightly deeper ResNet50 fine-tuning, class-balanced focal loss, and conservative crop-free
augmentation.

Research question:

> Can class-aware targeted fine-tuning with a deeper ResNet50 policy improve the fine-tuned
> three-backbone fusion result without the damaging crop/affine policy used in Sprint 4F?

## Scope

Run on Colab T4 using the thin notebook:

- `notebooks/04_sprint4h_targeted_finetuning.ipynb`

Configs:

- `configs/experiments/sprint4h_targeted_backbones.yaml`
- `configs/experiments/sprint4h_targeted_feature_matrix.yaml`

Feature source and checkpoints:

- feature source: `finetuned_targeted`;
- checkpoint directory: `artifacts/checkpoints/finetuned_targeted_backbones/`;
- feature cache directory: `artifacts/features/ham10000/finetuned_targeted/<backbone>/`.

## Training Policy

- ResNet50: unfreeze `layer3`, `layer4`, and `fc`.
- MobileNetV2: keep Sprint 4 last feature blocks.
- EfficientNetB0: keep Sprint 4 last feature blocks.
- Loss: class-balanced focal, gamma `1.0`, sqrt-smoothed train-split class weights, max class
  weight `4.0`, normalized class weights.
- Augmentation: horizontal/vertical flip, rotation up to `20` degrees, mild color jitter.
- Explicitly exclude random resized crop and affine in the first Sprint 4H pass.
- Selection remains validation macro-F1 only.
- Test metrics are not used for tuning.

## Matrix

Run the full 11-run cached-feature MLP matrix:

- 3 single-backbone MLP runs;
- 3 pairwise concat;
- 3 pairwise weighted;
- 1 three-backbone concat;
- 1 three-backbone weighted.

## Stop/Go Interpretation

Compare against:

- canonical Sprint 4 concat test macro-F1 `0.706`;
- Sprint 4D weighted + `tta_rot4` test macro-F1 `0.733`;
- Sprint 4F augmented concat test macro-F1 `0.645`.

Interpretation:

| Outcome | Interpretation |
|---|---|
| Beats `0.733` | Strong new best; evaluate TTA on selected Sprint 4H model. |
| Beats `0.706` but not `0.733` | Useful training improvement, but Sprint 4D remains best. |
| Below `0.706` | Deeper/class-aware training did not improve the canonical result; report as negative. |

## Artifacts

Commit only report-ready small CSV/PNG outputs. Keep these out of Git:

- raw HAM10000 images;
- feature `.pt` cache files;
- backbone checkpoints;
- MLP `model.pt`;
- full run folders.

Drive mirror must preserve operational `.pt` artifacts under
`/content/drive/MyDrive/dl-midterm-artifacts/sprint4h/` and must not use `--exclude model.pt`.

## Verification

Before Colab:

```bash
uv run ruff check src scripts tests
uv run pytest tests/test_sprint4h_targeted.py
```

After Colab:

- confirm 3 backbone checkpoints on Drive;
- confirm 9 feature cache `.pt` files on Drive;
- confirm 11 MLP `model.pt` files on Drive for the full matrix;
- download/sync report-ready CSV/PNG outputs locally;
- update report notes and move this plan to `completed/`.

## Outcome

Sprint 4H completed on Colab T4 and was restored locally from Drive. The artifact audit passed:

- 3 backbone checkpoints;
- 9 fine-tuned feature cache `.pt` files;
- 11 MLP `model.pt` files;
- 14 `metrics.json` files;
- 4H-specific report-ready CSV/PNG outputs.

The best cached-feature matrix result was ResNet50 + MobileNetV2 + EfficientNetB0 concat with test
macro-F1 `0.643`, accuracy `0.768`, and weighted-F1 `0.779`. The best image-level fine-tuned head
was targeted ResNet50 with test macro-F1 `0.647`.

Sprint 4H did not beat canonical Sprint 4 concat (`0.706`) or Sprint 4D weighted + `tta_rot4`
(`0.733`). It should be reported as an informative negative result: deeper/class-aware targeted
training improved minority-class recall in places, but did not improve the downstream
cached-feature fusion objective.
