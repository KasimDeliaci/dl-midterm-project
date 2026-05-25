# Project Folder Structure

This repository follows a uv-first, script-driven research workflow:

```text
GitHub repository
  code + configs + docs + report source + lightweight placeholders

Google Drive / Colab storage
  HAM10000 images + checkpoints + feature caches + large run artifacts

Colab
  GPU runner that clones the repo and executes scripts

Markdown docs
  project memory for decisions, evaluation policy, sprint notes, and report evidence
```

## Main Rules

- Colab is a runner, not the development environment.
- Core project logic belongs in `src/dl_midterm/`.
- Entry points belong in `scripts/`.
- Reproducible settings belong in `configs/`.
- Large generated artifacts belong outside Git under `data/`, `artifacts/`, or `submission/`.
- Important decisions belong in `docs/DECISIONS.md`.

## Layout

```text
dl-assignment/
├── AGENTS.md
├── README.md
├── PROJECT_FOLDER_STRUCTURE.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── configs/
│   ├── default.yaml
│   ├── report_assets.yaml
│   ├── dataset/
│   ├── backbones/
│   └── experiments/
│
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── DATASET_AUDIT.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── DECISIONS.md
│   ├── COMMANDS.md
│   ├── planning/
│   ├── exec-plans/
│   └── literature/
│
├── notebooks/
├── scripts/
├── src/
│   └── dl_midterm/
│       ├── config/
│       ├── data/
│       ├── models/
│       ├── features/
│       ├── training/
│       ├── evaluation/
│       └── utils/
├── data/
├── artifacts/
├── reports/
│   ├── final_report/
│   └── presentation/
├── tests/
└── submission/
```

## Documentation Roles

- `docs/PROJECT_CONTEXT.md`: current scientific/engineering state.
- `docs/DECISIONS.md`: dated decisions and rationale.
- `docs/EVALUATION_PROTOCOL.md`: metrics, split rules, leakage rules, and reporting rules.
- `docs/DATASET_AUDIT.md`: canonical dataset audit summary.
- `docs/COMMANDS.md`: reproducible commands.
- `docs/exec-plans/`: task-level implementation plans and tech debt.
- `docs/literature/`: paper registry and literature notes.
