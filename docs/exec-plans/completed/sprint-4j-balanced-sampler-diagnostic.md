# Sprint 4J Balanced Sampler Diagnostic

## Goal

Test whether class-balanced mini-batch sampling improves cached-feature MLP training before spending
Colab time on image-level balanced-sampler fine-tuning.

## Scope

- Use existing canonical Sprint 4 `finetuned` feature caches.
- Do not create synthetic images or synthetic features.
- Keep validation/test splits unchanged and lesion-aware.
- Add `WeightedRandomSampler` support to cached-feature MLP training only.
- Run a small diagnostic on the three-backbone concat and weighted MLPs.

## Policy

- Train sampler: inverse-frequency sample weights from the train cache.
- Replacement: enabled, with `num_samples = len(train_split)`.
- Validation/test: no sampling, no shuffling.
- Selection remains validation macro-F1 oriented.

## Decision Rule

Treat the diagnostic as promising only if validation macro-F1 improves over the comparable cached
feature MLP setup without a large weighted-F1 or accuracy collapse. If it looks promising, a later
Colab experiment can apply balanced sampling during image-level backbone fine-tuning.
