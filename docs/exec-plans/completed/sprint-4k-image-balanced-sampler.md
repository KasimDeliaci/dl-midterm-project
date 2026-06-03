# Sprint 4K Image-Level Balanced Sampler

## Goal

Test whether class-balanced sampling helps when applied during image-level backbone fine-tuning,
rather than only at the cached-feature MLP stage.

## Rationale

Sprint 4J showed that balanced sampling at the cached-feature MLP stage did not improve test
macro-F1. That does not rule out image-level balanced sampling, because image-level sampling can
change the fine-tuned CNN representation itself. This Sprint 4K diagnostic tests that narrower
hypothesis.

## Initial Policy

- Feature source: `finetuned_balanced_sampler`.
- First diagnostic backbone: ResNet50 only.
- ResNet50 unfreeze policy: `layer3_layer4`.
- Train sampler: class-balanced `WeightedRandomSampler`.
- Loss: plain cross-entropy.
- Class weights: disabled, to avoid double-compensating minority classes.
- Augmentation: crop-free mild flips, rotation, and color jitter.
- Selection: validation macro-F1 only.

## Decision Gate

Escalate to all three backbones and an MLP/fusion matrix only if the ResNet50 image-level run shows
a credible validation macro-F1 improvement without a major validation accuracy or weighted-F1 drop.

## Result

The ResNet50 diagnostic completed on Colab and was restored through the Drive mirror.

- Best validation epoch: `14`.
- Best validation macro-F1: `0.672`.
- Test accuracy: `0.756`.
- Test macro-F1: `0.657`.
- Test weighted-F1: `0.772`.

The result did not beat the canonical Sprint 4 matrix or the Sprint 4D TTA result, so the decision
gate was not met. Sprint 4K was not escalated to all three backbones or the 11-run MLP/fusion
matrix.

## Outputs

Large operational artifacts stay gitignored:

- `artifacts/checkpoints/finetuned_balanced_sampler_backbones/`
- `artifacts/features/ham10000/finetuned_balanced_sampler/`
- `artifacts/runs/`

Small report assets can be added later only if Sprint 4K is discussed beyond the concise negative
diagnostic note.
