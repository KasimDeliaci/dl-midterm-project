# Decisions

## 2026-05-22 - Initialize uv-first repository

Decision: use `uv`, `pyproject.toml`, and `uv.lock` as the dependency source of truth.

Reason: the project needs a reproducible workflow that works locally and in Colab without manually maintained dependency files.

## 2026-05-22 - Keep Colab as a runner

Decision: keep Colab notebooks as thin launchers around scripts.

Reason: model logic, data handling, and evaluation should live in `src/dl_midterm/` and `scripts/` so experiments are reproducible and maintainable.

## 2026-05-22 - Use a flat package for the student-project scope

Decision: keep the reusable package flat under `src/dl_midterm/` rather than creating many nested subpackages.

Reason: this project needs clear boundaries without turning a midterm assignment into an over-engineered ML platform.

Outcome: superseded on 2026-05-25 after the project scope clarified around dataset preparation, feature caching, fine-tuning, evaluation, and report-asset generation.

## 2026-05-24 - Select HAM10000 as the primary dataset

Decision: use HAM10000 as the primary dataset and EuroSAT as the backup.

Reason: HAM10000 is public, multi-class, Colab-feasible, class-imbalanced, visually fine-grained, and strong for feature-representation and fusion analysis. EuroSAT is simpler operationally and remains the fallback if HAM10000 acquisition or splitting becomes too risky.

## 2026-05-24 - Use macro-F1 as the primary interpretation metric

Decision: use macro-F1 as the primary model-selection and discussion metric, while still reporting assignment-required accuracy, precision, recall, and F1-score.

Reason: HAM10000 is class-imbalanced, so accuracy and weighted averages can hide poor minority-class behavior.

## 2026-05-24 - Use projected learnable weighted fusion

Decision: implement weighted fusion by projecting each backbone feature vector to a shared latent dimension and learning global softmax-normalized weights.

Reason: ResNet50, MobileNetV2, and EfficientNetB0 produce different feature dimensions. Projection makes weighted fusion well-defined and gives interpretable learned weights for the discussion section.

## 2026-05-24 - Avoid clinical claims

Decision: describe the work as benchmark dermoscopic image classification, not diagnosis or clinical decision support.

Reason: the project is an educational benchmark study and is not validated for clinical use.

## 2026-05-25 - Adopt structured `dl_midterm` package

Decision: organize source code under `src/dl_midterm/` with subpackages for `config`, `data`, `models`, `features`, `training`, `evaluation`, and `utils`.

Reason: the project now has enough distinct responsibilities that a flat module layout would blur boundaries. The nested package keeps implementation maintainable while still avoiding a heavy framework.

Outcome: scripts remain the command-line entrypoints; reusable logic lives in the package submodules.

## 2026-05-25 - Use `artifacts/` for generated experiment outputs

Decision: replace `outputs/` with `artifacts/` for feature caches, checkpoints, run folders, and report-ready tables/figures.

Reason: `artifacts/` better communicates generated, mostly-gitignored experiment products and separates them from source docs/reports.

## 2026-05-26 - Use lesion-aware 70/15/15 HAM10000 splits

Decision: generate the primary HAM10000 train/validation/test split with `70/15/15` ratios and group by `lesion_id` whenever metadata provides complete lesion IDs.

Reason: HAM10000 can contain multiple images per lesion. Grouping by lesion ID reduces leakage risk between train, validation, and test while preserving the assignment's multi-class benchmark framing.

## 2026-05-26 - Fail fast when local dataset files are absent

Decision: dataset preparation must stop with clear errors when metadata or raw images are absent, rather than generating dummy splits or placeholder statistics.

Reason: Sprint 1 is about trustworthy, reproducible dataset preparation. Fabricated class counts or split files would contaminate the report and later model comparisons.

## 2026-05-26 - Cache frozen features as tensors plus manifests

Decision: Sprint 2 stores frozen CNN features as one `.pt` tensor payload per backbone and split, with companion CSV manifests and a per-backbone JSON manifest.

Reason: tensor caches keep MLP experiments fast, while CSV/JSON manifests make image ID, label, split, feature source, feature dimension, seed, and config provenance easy to audit without loading large tensors.

## 2026-05-26 - Use train-split class weights for Sprint 2 MLP baselines

Decision: enable class-weighted cross-entropy by default for frozen single-backbone MLP baselines, computing weights only from the cached train split.

Reason: HAM10000 is strongly imbalanced, and the evaluation protocol treats macro-F1 as the primary interpretation metric. Computing weights only from train data avoids validation/test leakage.
