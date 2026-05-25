# Deep Learning Midterm Project

Multi-CNN feature extraction and feature fusion for image classification.

The project compares ResNet50, MobileNetV2, and EfficientNetB0 features under frozen feature extraction and fine-tuning settings. Fused representations are evaluated with a shared MLP classifier so the report can isolate the effect of backbone choice, fusion strategy, and transfer learning policy.

## Dataset Direction

Primary dataset: `HAM10000`, the seven-class dermoscopic skin lesion classification benchmark released through the ISIC Archive / Harvard Dataverse.

Backup dataset: `EuroSAT`, a 10-class remote-sensing land-use classification dataset.

The project uses HAM10000 because it is public, Colab-feasible, class-imbalanced, visually fine-grained, and well-suited for meaningful analysis of frozen features, fine-tuning, feature fusion, macro-F1, per-class F1, and runtime tradeoffs. The report should avoid clinical claims and describe the task as benchmark dermoscopic image classification.

Primary creative angle: project each backbone feature vector into a shared latent dimension, apply learnable softmax weighted fusion, and analyze learned fusion weights plus per-class F1 gains over the best single backbone.

## Workflow

Local development uses `uv` for dependency management, code quality, report assets, and Git history. Google Colab is used for GPU-heavy dataset preparation, feature extraction, fine-tuning, and experiment runs.

```bash
uv sync
uv run python scripts/make_report_assets.py --config configs/report_assets.yaml
```

Colab notebooks should remain thin launchers around the scripts in `scripts/`. Core logic belongs in `src/dl_midterm/`.

## Repository Layout

- `AGENTS.md`: Short guide for coding agents working in this repo.
- `PROJECT_FOLDER_STRUCTURE.md`: Repository organization and workflow rationale.
- `configs/`: YAML configuration for dataset, backbones, training, experiments, and report artifacts.
- `docs/`: Planning notes, research decisions, and sprint reports.
- `notebooks/`: Colab launcher notebooks.
- `scripts/`: Command-line entrypoints used locally and from Colab.
- `src/dl_midterm/`: Reusable Python package.
- `data/`: Dataset and split files. Raw and processed data are gitignored.
- `artifacts/`: Generated features, checkpoints, runs, and report assets. Heavy artifacts are gitignored.
- `reports/final_report/`: LaTeX report source.
- `submission/`: Final package/archive staging and YouTube link.

## Planning

The working implementation plan lives in [docs/planning/5-sprint-project-plan.md](docs/planning/5-sprint-project-plan.md).

Key project-memory files:

- [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
- [docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md)
- [docs/COMMANDS.md](docs/COMMANDS.md)
