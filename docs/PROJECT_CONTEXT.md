# Project Context

This project investigates multi-CNN feature extraction and feature fusion for seven-class HAM10000 dermoscopic image classification.

The assignment requires:

- ResNet50, MobileNetV2, and EfficientNetB0.
- Final classifier removal and feature-vector extraction.
- Concatenation fusion and weighted fusion.
- MLP classifier over extracted or fused features.
- Single-CNN, two-CNN, and three-CNN comparisons.
- Frozen feature extraction versus fine-tuning the last meaningful CNN blocks.
- Accuracy, precision, recall, and F1-score reporting.

## Current Direction

- Primary dataset: HAM10000.
- Backup dataset: EuroSAT.
- Primary metric for interpretation: macro-F1.
- Required reported metrics: accuracy, precision, recall, and F1-score.
- Creative angle: projected learnable weighted fusion plus learned-weight and per-class F1 analysis.

## Core Rules

- Use `uv` with `pyproject.toml` and `uv.lock`.
- Use PyTorch and TorchVision.
- Use Colab for GPU-heavy execution.
- Keep notebooks as launchers; scripts and `src/dl_fusion/` contain real logic.
- Cache features so MLP ablations are cheap and reproducible.
- Avoid medical diagnosis claims; describe the task as benchmark dermoscopic image classification.

## Near-Term Path

1. Complete Sprint 1 dataset acquisition, audit, and leakage-aware split generation.
2. Build frozen feature extraction and single-backbone MLP baselines.
3. Add concatenation and projected learnable weighted fusion.
4. Fine-tune final meaningful backbone blocks and rerun the comparison matrix.
5. Aggregate results into report-ready tables, plots, and discussion notes.
