# Sprint 1 Dataset Preparation Execution Plan

## Objective

Make HAM10000 dataset preparation reproducible before any model training, feature extraction, or fusion work begins.

## Constraints

- Keep raw images, processed data, feature caches, checkpoints, and generated run artifacts out of Git.
- Use `uv` for local commands.
- Keep notebooks as thin Colab runners around scripts.
- Do not create dummy dataset results when HAM10000 is absent locally.
- Use benchmark dermoscopic image classification language, not clinical diagnosis claims.

## Affected Files

- `configs/dataset/selected_dataset.yaml`
- `src/dl_midterm/data/datasets.py`
- `src/dl_midterm/data/splits.py`
- `scripts/prepare_dataset.py`
- `scripts/create_splits.py`
- `tests/test_dataset_sprint1.py`
- `notebooks/01_dataset_prepare.ipynb`
- `docs/DATASET_AUDIT.md`
- `docs/DECISIONS.md`
- `docs/COMMANDS.md`
- `docs/planning/5-sprint-project-plan.md`

## Verification Gates

- Config YAML parses.
- Unit tests cover metadata normalization, missing image audit, unknown-label failure, and lesion-aware split leakage checks.
- `scripts/prepare_dataset.py` fails clearly when local HAM10000 metadata/images are absent.
- `scripts/create_splits.py` can create grouped 70/15/15 splits from a valid audited metadata CSV.

## Rollback/Fallback Notes

If grouped stratification is impossible because a class has too few lesion groups, split generation falls back to grouped random splitting and emits a warning. If metadata or images are missing, preparation stops before split files are written unless explicitly run with `--allow-incomplete` for manual inspection.

## Final Outcome

Implemented the Sprint 1 dataset audit and split-generation infrastructure, including
`data/processed/ham10000_audited_metadata.csv` export for repeatable split-only runs.
The real local HAM10000 audit later completed successfully: 10,015 images were verified,
7,470 unique lesion IDs were found, and lesion-aware 70/15/15 train/validation/test splits
were generated with zero cross-split lesion leakage.

Verified:

```bash
uv run pytest tests/test_dataset_sprint1.py
uv run python -c "import yaml; yaml.safe_load(open('configs/dataset/selected_dataset.yaml')); print('dataset yaml ok')"
uv run python scripts/prepare_dataset.py --config configs/dataset/selected_dataset.yaml
```

All commands passed once the local HAM10000 metadata and raw images were available.

Generated local outputs:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
artifacts/report_assets/tables/class_distribution.csv
artifacts/report_assets/tables/split_class_distribution.csv
artifacts/report_assets/figures/class_distribution.png
artifacts/report_assets/figures/split_class_distribution.png
```
