# Deep Learning Midterm Project

Multi-CNN feature extraction and feature fusion for image classification.

The project compares ResNet50, MobileNetV2, and EfficientNetB0 features under frozen feature extraction and fine-tuning settings. Fused representations are evaluated with a shared MLP classifier so the report can isolate the effect of backbone choice, fusion strategy, and transfer learning policy.

## Workflow

Local development uses `uv` for dependency management, code quality, report assets, and Git history. Google Colab is used for GPU-heavy dataset preparation, feature extraction, fine-tuning, and experiment runs.

```bash
uv sync
uv run python scripts/make_report_assets.py --config configs/report.yaml
```

Colab notebooks should remain thin launchers around the scripts in `scripts/`. Core logic belongs in `src/dl_fusion/`.

## Repository Layout

- `configs/`: YAML configuration for dataset, backbones, training, experiments, and report outputs.
- `notebooks/`: Colab launcher notebooks.
- `scripts/`: Command-line entrypoints used locally and from Colab.
- `src/dl_fusion/`: Reusable Python package.
- `data/`: Dataset and split files. Raw and processed data are gitignored.
- `outputs/`: Generated features, checkpoints, runs, and report assets. Heavy outputs are gitignored.
- `report/`: LaTeX report source.
- `submission/`: Final package/archive staging and YouTube link.

