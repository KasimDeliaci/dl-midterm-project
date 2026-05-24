# Decisions

## 2026-05-22 - Initialize uv-first repository

Decision: use `uv`, `pyproject.toml`, and `uv.lock` as the dependency source of truth.

Reason: the project needs a reproducible workflow that works locally and in Colab without manually maintained dependency files.

## 2026-05-22 - Keep Colab as a runner

Decision: keep Colab notebooks as thin launchers around scripts.

Reason: model logic, data handling, and evaluation should live in `src/dl_fusion/` and `scripts/` so experiments are reproducible and maintainable.

## 2026-05-22 - Use a flat package for the student-project scope

Decision: keep the reusable package flat under `src/dl_fusion/` rather than creating many nested subpackages.

Reason: this project needs clear boundaries without turning a midterm assignment into an over-engineered ML platform.

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
