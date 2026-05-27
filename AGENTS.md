# Agent Guide

This repository is a uv-managed PyTorch project for the Deep Learning midterm assignment: multi-CNN feature extraction and feature fusion for HAM10000 image classification.

## Start Here

- Project context: `docs/PROJECT_CONTEXT.md`
- Current plan: `docs/planning/5-sprint-project-plan.md`
- Key decisions: `docs/DECISIONS.md`
- Evaluation rules: `docs/EVALUATION_PROTOCOL.md`
- Commands: `docs/COMMANDS.md`
- Dataset audit: `docs/DATASET_AUDIT.md`
- Tech debt: `docs/exec-plans/tech-debt.md`

## Working Rules

- Use `uv` with `pyproject.toml` and `uv.lock`; do not maintain ad-hoc requirements files unless an external environment explicitly needs one.
- Keep core implementation in `src/dl_midterm/`.
- Keep Colab notebooks as thin runners around scripts.
- When work requires Google Colab execution or notebook interaction, use the Codex Chrome extension (`@chrome`) with Chrome skills for Google Chrome.
- Keep datasets, checkpoints, feature caches, and run artifacts out of Git.
- Do not make clinical claims. This project is benchmark dermoscopic image classification, not diagnosis.
- Record important scientific or engineering decisions in `docs/DECISIONS.md`.
- For substantial work, create an execution plan under `docs/exec-plans/active/` and move it to `completed/` when done.

## Colab Notes

- Do not add noisy `nvidia-smi` cells to notebooks just to show GPU state. Colab's runtime selector/status is enough unless debugging a GPU issue.
- GitHub-opened Colab notebooks are not the saved project notebooks; use `Drive'a kopyala` when outputs need to persist.
- Colab runs in `/content/dl-assignment`; local repo files are not automatically present there. Runner notebooks must clone/pull the repo and restore the Drive bundle/cache explicitly.
- The HAM10000 bundle lives on Drive as `ham10000_colab_bundle.tar`; frozen feature caches and small report assets can be mirrored under `/content/drive/MyDrive/dl-midterm-artifacts/`.
- Colab smoke outputs are for reproducibility checks. Do not mix them into the local Sprint result set unless explicitly requested.
