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
- Current status: Sprint 1 dataset audit/splits and Sprint 2 frozen single-backbone baselines are complete locally.

## Core Rules

- Use `uv` with `pyproject.toml` and `uv.lock`.
- Use PyTorch and TorchVision.
- Use Colab for GPU-heavy execution, especially fine-tuning; frozen cached-feature MLP work can continue locally.
- Keep notebooks as launchers; scripts and `src/dl_midterm/` contain real logic.
- Cache features so MLP ablations are cheap and reproducible.
- Avoid medical diagnosis claims; describe the task as benchmark dermoscopic image classification.

## Near-Term Path

1. Use the completed Sprint 1 lesion-aware split and Sprint 2 frozen feature caches as the fixed baseline.
2. Add Sprint 3 concatenation and projected learnable weighted fusion on cached frozen features.
3. Compare single-backbone, pairwise-fusion, and three-backbone frozen MLP results with macro-F1 as the main interpretation metric.
4. Fine-tune final meaningful backbone blocks in Sprint 4 and rerun the comparison matrix.
5. Aggregate results into report-ready tables, plots, and discussion notes.
