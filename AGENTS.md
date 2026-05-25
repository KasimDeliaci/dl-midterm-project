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
- Keep datasets, checkpoints, feature caches, and run artifacts out of Git.
- Do not make clinical claims. This project is benchmark dermoscopic image classification, not diagnosis.
- Record important scientific or engineering decisions in `docs/DECISIONS.md`.
- For substantial work, create an execution plan under `docs/exec-plans/active/` and move it to `completed/` when done.
