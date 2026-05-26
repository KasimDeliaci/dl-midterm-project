# Sprint 2b Execution Plan: MLP Hyperparameter Search

## Scope

Push the Sprint 2 frozen-feature baseline by varying only the MLP classifier setup. CNN feature caches remain fixed and frozen.

Included:

- Class weighting ablation.
- Dropout variants.
- Optimizer variants.
- Separate run folders for each backbone/candidate pair.
- Search-level aggregate tables and plots.
- Best-run confusion matrices and training curves copied to report-ready figure paths.

Excluded:

- Pairwise fusion.
- Three-CNN fusion.
- CNN fine-tuning.

## Search ID

`mlp_hparam_v1_full`

## Candidates

- `cw_adamw_d03`
- `nocw_adamw_d03`
- `cw_adamw_d01`
- `cw_adamw_d05`
- `cw_adam_d03`
- `cw_sgd_d03`

Each candidate was run for:

- ResNet50 frozen features.
- MobileNetV2 frozen features.
- EfficientNetB0 frozen features.

Total: 18 MLP runs.

## Verification

- `uv run pytest`
- `uv run ruff check .`
- `uv run python scripts/run_mlp_hyperparam_search.py --config configs/experiments/mlp_hyperparam_search.yaml --default-config configs/default.yaml --dataset-config configs/dataset/selected_dataset.yaml --search-id mlp_hparam_v1_full`

## Results Summary

Best test macro-F1 by backbone:

| Backbone | Candidate | Test macro-F1 | Accuracy |
|---|---|---:|---:|
| EfficientNetB0 | `cw_sgd_d03` | 0.5605 | 0.6924 |
| ResNet50 | `nocw_adamw_d03` | 0.5426 | 0.7483 |
| MobileNetV2 | `cw_adamw_d03` | 0.4675 | 0.6678 |

Compared with the original weighted AdamW/dropout-0.3 baseline:

| Backbone | Best Candidate | Macro-F1 Change | Accuracy Change |
|---|---|---:|---:|
| EfficientNetB0 | `cw_sgd_d03` | +0.0540 | -0.0047 |
| ResNet50 | `nocw_adamw_d03` | +0.0115 | +0.0360 |
| MobileNetV2 | `cw_adamw_d03` | +0.0000 | +0.0000 |

## Artifact Policy

Full run folders and model weights remain generated artifacts. Search-level CSV/PNG assets are small and report-ready.
