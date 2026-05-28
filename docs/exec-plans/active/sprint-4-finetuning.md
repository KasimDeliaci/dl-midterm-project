# Sprint 4 Execution Plan: Fine-Tuning and Frozen-vs-Fine-Tuned Comparison

## Scope

Implement the fine-tuning stage for the HAM10000 benchmark dermoscopic image classification
project.

Included:

- Fine-tune the final meaningful blocks of ResNet50, MobileNetV2, and EfficientNetB0.
- Use ImageNet pretrained weights for full runs.
- Keep only the final CNN stage/block and temporary classifier head trainable.
- Use train-split class weights, seed 42, conservative learning rate, and validation macro-F1
  early stopping.
- Save best checkpoints by validation macro-F1.
- Extract classifier-free feature caches from the best checkpoints under
  `artifacts/features/ham10000/finetuned/<backbone>/`.
- Run the fine-tuned cached-feature MLP matrix:
  - 3 single-backbone runs,
  - 3 pairwise concat,
  - 3 pairwise weighted,
  - 1 three-backbone concat,
  - 1 three-backbone weighted.
- Export fine-tuned report tables/figures and frozen-vs-fine-tuned comparison assets.

Excluded:

- Test-metric-based model selection.
- Mixing Colab smoke-test outputs into canonical local Sprint results.
- Committing checkpoints, feature cache `.pt` files, run folders, raw images, or Colab artifacts.

## Unfreeze Policy

| Backbone | Trainable CNN block(s) | Trainable head |
|---|---|---|
| ResNet50 | `layer4` | `fc` |
| MobileNetV2 | `features[16]`, `features[17]`, `features[18]` | `classifier` |
| EfficientNetB0 | `features[7]`, `features[8]` | `classifier` |

All earlier backbone parameters stay frozen.

## Verification Gates

- Unit tests cover fine-tuning freeze/unfreeze policy.
- Unit tests cover fine-tuned cache shape/alignment.
- Unit tests cover feature-source matrix expansion for `finetuned`.
- Unit tests cover weighted fusion softmax weight sum.
- Unit tests cover that cached-feature MLP stages consume tensors, not raw image paths.
- `uv run pytest` passes.
- `uv run ruff check src scripts tests` passes.
- A non-canonical `/tmp` smoke run can fine-tune one backbone for one epoch, write a checkpoint,
  extract a tiny fine-tuned cache, and train a tiny MLP from that cache.

## Full Colab Commands

Run on Colab GPU after restoring the Drive HAM10000 bundle:

```bash
uv run python scripts/finetune_backbone.py \
  --config configs/experiments/finetune_backbones.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned \
  --batch-size 32

uv run python scripts/train_mlp.py \
  --config configs/experiments/finetuned_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned

uv run python scripts/run_experiment_matrix.py \
  --config configs/experiments/finetuned_feature_matrix.yaml \
  --default-config configs/default.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned

uv run python scripts/make_report_assets.py \
  --feature-source finetuned \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-root artifacts/features
```

## Current Status

Implementation is complete locally and verified with unit tests plus small `/tmp` smoke runs.
The canonical full Sprint 4 result matrix still needs to be executed on Colab GPU.
