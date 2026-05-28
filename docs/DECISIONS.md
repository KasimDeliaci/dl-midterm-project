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

## 2026-05-26 - Track MLP search runs under isolated search IDs

Decision: run MLP-only hyperparameter searches under `artifacts/runs/mlp_hparam_search/<search_id>/`, with every backbone/candidate pair writing a complete independent run folder.

Reason: Sprint 2b is exploratory. Isolated search IDs keep baseline results, hyperparameter candidates, plots, and aggregate tables auditable without mixing them into the canonical Sprint 2 baseline folders.

## 2026-05-27 - Keep local Sprint results canonical

Decision: treat the local Sprint 1 and Sprint 2 artifact set as the canonical result source unless explicitly requested otherwise. Colab smoke-test outputs are reproducibility checks and should not be mixed into the local Sprint result interpretation.

Reason: local artifacts were generated and analyzed as the project baseline. Colab runs are useful for verifying runner notebooks and future GPU workflows, but they can create extra run IDs and smoke-test artifacts that would make the result narrative harder to audit.

## 2026-05-27 - Select final report assets later

Decision: keep generated candidate tables/figures under `artifacts/report_assets/` during experimentation, then copy only the figures and tables actually referenced by the final LaTeX report into `reports/final_report/figures/` and `reports/final_report/tables/`.

Reason: `artifacts/` is the generated experiment-output area; `reports/final_report/` should stay focused on final selected deliverables. Choosing report assets late avoids clutter and stale figures in the final report folder.

## 2026-05-27 - Run Sprint 3 fusion only from cached frozen features

Decision: implement Sprint 3 fusion by loading and aligning the existing Sprint 2 `.pt` feature
caches, concatenating cached tensors for the dataloader input, and placing projected weighted
fusion inside the trainable classifier model.

Reason: Sprint 3 is meant to isolate feature-fusion behavior from CNN feature extraction and
fine-tuning. Reusing cached features avoids raw-image or CNN forward work, keeps the run matrix
local and reproducible, and preserves the Sprint 2 evaluation protocol.

## 2026-05-27 - Use global weighted-fusion weights for interpretability

Decision: weighted fusion uses one learned global softmax weight per backbone, after projecting
each backbone feature vector to the shared 512-dimensional space.

Reason: global weights satisfy the assignment's weighted-fusion requirement while producing a
small, reportable learned-weight table. Sample-specific attention would add complexity and make
the learned contribution analysis harder to explain for this midterm scope.

## 2026-05-27 - Analyze complementarity with sample-similarity structure

Decision: measure frozen backbone representation complementarity by comparing sample-by-sample
cosine similarity matrices rather than raw feature dimensions.

Reason: ResNet50, MobileNetV2, and EfficientNetB0 feature vectors have different widths, so direct
feature-feature correlation is not well-defined. Comparing the similarity structure over the same
test examples gives a dimension-agnostic signal for whether two backbones organize the dataset in
similar or complementary ways.

## 2026-05-28 - Fine-tune only final meaningful CNN blocks

Decision: Sprint 4 fine-tuning keeps most ImageNet-pretrained CNN parameters frozen and unfreezes
only the final meaningful stage plus the temporary classification head:

- ResNet50: `layer4` and `fc`.
- MobileNetV2: `features[16]`, `features[17]`, `features[18]`, and `classifier`.
- EfficientNetB0: `features[7]`, `features[8]`, and `classifier`.

Reason: HAM10000 is modest in size and imbalanced. Conservative partial fine-tuning reduces
overfitting risk, keeps Colab runtime manageable, and still satisfies the transfer-learning
comparison against frozen feature extraction.

## 2026-05-28 - Use `finetuned` as the canonical Sprint 4 feature source

Decision: store fine-tuned feature caches under
`artifacts/features/ham10000/finetuned/<backbone>/` and use `feature_source=finetuned` in MLP and
fusion matrix configs.

Reason: this path clearly distinguishes Sprint 4 checkpoints-derived feature caches from Sprint 2
`frozen` caches and avoids the older `ft_last_blocks` placeholder name leaking into report assets.

## 2026-05-28 - Select fine-tuned checkpoints by validation macro-F1

Decision: fine-tuned backbone checkpoints are selected by validation macro-F1, with class weights
computed only from the train split.

Reason: macro-F1 is the project primary metric because HAM10000 is imbalanced. Using validation
macro-F1 preserves the evaluation protocol; test metrics remain held out for final reporting only.
