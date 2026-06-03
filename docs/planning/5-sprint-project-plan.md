# Five-Sprint Project Plan

This plan is designed around the assignment requirements: public multi-class image classification dataset, ResNet50/MobileNetV2/EfficientNetB0, feature extraction, concatenation and weighted fusion, MLP classifier, single/two/three-CNN ablations, frozen vs fine-tuned transfer learning, and accuracy/precision/recall/F1 reporting.

The plan assumes HAM10000 as the primary dataset. EuroSAT remains the backup dataset if HAM10000 creates too much operational risk.

## Sprint 1: Dataset, Repository Foundation, and Research Framing

### Goal

Build a trustworthy project base before training anything: acquire HAM10000 from official or reliable sources, validate metadata, confirm the seven-class classification setup, create a leakage-aware split strategy using lesion IDs when possible, and establish the uv/PyTorch/Colab/LaTeX project skeleton.

This sprint should end with a reproducible dataset manifest and a short research framing note, not with a model.

### Main Workstreams

- Set up the uv-managed repo with the agreed structure: `src/dl_midterm/`, `scripts/`, `configs/`, `notebooks/`, `artifacts/`, and `reports/final_report/`.
- Acquire HAM10000 from the official ISIC Archive or a reliable mirrored release, not an unclear Kaggle copy.
- Download and inspect metadata: image IDs, diagnosis labels, lesion IDs, file paths, duplicate/missing images, and class counts.
- Define the classification target: standard HAM10000 seven-class diagnosis labels.
- Create a leakage-aware split plan: stratified by class and grouped by `lesion_id` where possible.
- Create initial configs: `configs/default.yaml`, `configs/dataset/selected_dataset.yaml`, `configs/backbones/*.yaml`, and `configs/experiments/*.yaml`.
- Create the LaTeX report skeleton with dataset, method, experiments, results, discussion, and conclusion sections.

### Decisions

- Use the standard 10,015-image HAM10000 metadata version for comparability, while documenting official ISIC collection availability.
- Confirm whether `lesion_id` is available and reliable enough for grouped splitting.
- Prefer a `70/15/15` train/validation/test split for more stable validation on minority classes.
- Decide whether to use class weighting after class distribution analysis. This is likely useful for HAM10000.
- Keep EuroSAT as a documented fallback, but do not prepare it in parallel unless HAM10000 becomes impractical.

### Verification Gates

Sprint 1 is done only if:

- Every image referenced in metadata is found, or missing files are explicitly listed.
- A class distribution table exists.
- Split files exist:
  - `data/splits/train.csv`
  - `data/splits/val.csv`
  - `data/splits/test.csv`
- No `lesion_id` appears in more than one split, if lesion IDs are available.
- A small dataloader smoke test can load one batch from train/validation/test.
- Dataset notes include source, license/usage constraints, class labels, and non-clinical benchmark disclaimer.
- Colab setup notebook can clone/install the repo and resolve dataset/output paths.

### Implementation Status - 2026-05-27

Sprint 1 is complete locally.

Completed:

- Dataset config defines HAM10000 labels, source URL, default `70/15/15` split ratios, and seed.
- `scripts/prepare_dataset.py` audits metadata/images, exports audited metadata and class distribution assets, and writes splits only when blocking audit errors are absent.
- `scripts/create_splits.py` supports split generation from an already audited metadata CSV.
- `src/dl_midterm/data/datasets.py` normalizes HAM10000 metadata and resolves image paths.
- `src/dl_midterm/data/splits.py` creates lesion-aware splits and checks leakage.
- `notebooks/01_dataset_prepare.ipynb` remains a thin runner around the script.
- Unit tests cover metadata normalization, missing image detection, unknown-label failure, and lesion leakage prevention.
- Real HAM10000 audit passed: 10,015 images, 10,015 unique image IDs, 0 duplicate image IDs, 0 missing images, and 0 missing labels.
- `lesion_id` is available with 7,470 unique lesion IDs.
- Real lesion-aware split files were generated:
  - train: 6,981 images
  - validation: 1,532 images
  - test: 1,502 images
- Cross-split lesion leakage is 0 for train/validation, train/test, and validation/test.
- Class distribution and split class distribution CSV/PNG assets were exported.
- The `df` class ratio deviation is documented as an acceptable lesion-aware grouping tradeoff.

### Expected Artifacts

- `configs/dataset/selected_dataset.yaml`
- `configs/default.yaml`
- `configs/backbones/resnet50.yaml`
- `configs/backbones/mobilenet_v2.yaml`
- `configs/backbones/efficientnet_b0.yaml`
- `configs/experiments/frozen_feature_matrix.yaml`
- `configs/experiments/finetune_backbones.yaml`
- `configs/experiments/finetuned_feature_matrix.yaml`
- `configs/report_assets.yaml`
- `src/dl_midterm/data/datasets.py`
- `src/dl_midterm/data/splits.py`
- `src/dl_midterm/data/dataloaders.py`
- `src/dl_midterm/utils/io.py`
- `scripts/prepare_dataset.py`
- `scripts/create_splits.py`
- `notebooks/00_colab_setup.ipynb`
- `notebooks/01_dataset_prepare.ipynb`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `artifacts/report_assets/tables/class_distribution.csv`
- `artifacts/report_assets/figures/class_distribution.png`
- `reports/final_report/main.tex`
- `reports/final_report/references.bib`
- `reports/final_report/sections/01_introduction.tex`
- `reports/final_report/sections/03_dataset.tex`

### Interim Sprint Report

Write a short note:

> We selected HAM10000 as the primary dataset because it is a public seven-class dermoscopic image benchmark available through ISIC. We validated the metadata, inspected class imbalance, and created leakage-aware train/validation/test splits using `lesion_id` grouping where possible. The class distribution shows strong imbalance, so later experiments will report macro-F1, weighted-F1, per-class F1, and confusion matrices instead of relying only on accuracy.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| HAM10000 download is awkward through ISIC. | Use a reliable academic mirror or Kaggle mirror only after confirming it matches the standard metadata, while citing the original ISIC/Nature source. |
| `lesion_id` grouping makes minority classes too small in validation/test. | Use grouped stratification as much as possible and document any compromise. |
| Too many missing images. | Use the standard 10,015-image release; if still messy, switch to EuroSAT. |
| Medical framing becomes too broad. | Use “benchmark dermoscopic image classification,” not diagnosis. |

### Git Hygiene

Expected commits:

- `chore: initialize uv project structure`
- `chore: add config templates and path conventions`
- `feat(data): add HAM10000 metadata loading and split generation`
- `test(data): add dataloader and split leakage checks`
- `docs: add dataset source and project framing notes`
- `report: add initial LaTeX report skeleton`

Do not commit raw images, large datasets, or generated feature files.

## Sprint 2: Frozen Feature Extraction and Single-Backbone MLP Baselines

### Goal

Get the first real model results quickly but scientifically: extract frozen ImageNet-pretrained features from ResNet50, MobileNetV2, and EfficientNetB0, cache those features, and train simple MLP classifiers for the three single-backbone baselines.

This sprint creates the minimum useful result before fusion and fine-tuning complexity.

### Main Workstreams

- Implement backbone wrappers that remove final classifier heads and output feature vectors.
- Implement consistent preprocessing: resize/crop to `224x224`, ImageNet normalization, and no heavy augmentation during feature extraction.
- Extract frozen features for ResNet50, MobileNetV2, and EfficientNetB0.
- Design feature cache format with split name, image ID, label, feature vector, backbone name, and feature source.
- Train MLP baselines for frozen ResNet50, MobileNetV2, and EfficientNetB0 features.
- Log runtime for feature extraction and MLP training.
- Export accuracy, macro precision, macro recall, macro F1, weighted F1, per-class F1, and confusion matrix.

### Decisions

- Use `.pt` files for features plus a small `.csv` manifest.
- Verify feature dimensions:
  - ResNet50: usually 2048
  - MobileNetV2: usually 1280
  - EfficientNetB0: usually 1280
- Start with a modest MLP. Avoid a large classifier that hides feature-quality differences.
- Decide whether class weights are needed in the loss. For HAM10000, this is likely useful.

### Verification Gates

Sprint 2 is done only if:

- Feature files exist for all three frozen backbones and all three splits.
- Feature dimensions match expected dimensions.
- Labels and image IDs align exactly with split CSVs.
- A small test verifies cached train/validation/test features do not mix samples.
- Three single-backbone MLP runs complete.
- Each run has `config_resolved.yaml`, `metrics.json`, `history.csv`, `classification_report.csv`, and `confusion_matrix.png`.
- Results are plausible: not random, not suspiciously perfect, and not interpreted only through accuracy.

### Expected Artifacts

- `src/dl_midterm/models/backbones.py`
- `src/dl_midterm/models/feature_extractors.py`
- `src/dl_midterm/features/extract.py`
- `src/dl_midterm/features/cache.py`
- `src/dl_midterm/features/normalize.py`
- `src/dl_midterm/models/mlp.py`
- `src/dl_midterm/training/loops.py`
- `src/dl_midterm/evaluation/metrics.py`
- `src/dl_midterm/evaluation/reports.py`
- `src/dl_midterm/evaluation/plots.py`
- `scripts/extract_features.py`
- `scripts/train_mlp.py`
- `scripts/evaluate_runs.py`
- `notebooks/02_extract_frozen_features.ipynb`
- `artifacts/features/ham10000/frozen/resnet50/`
- `artifacts/features/ham10000/frozen/mobilenet_v2/`
- `artifacts/features/ham10000/frozen/efficientnet_b0/`
- `artifacts/runs/*_frozen_r50_none_mlp_s42/`
- `artifacts/runs/*_frozen_mnv2_none_mlp_s42/`
- `artifacts/runs/*_frozen_effb0_none_mlp_s42/`
- `artifacts/report_assets/tables/single_backbone_frozen_results.csv`
- `artifacts/report_assets/figures/frozen_single_backbone_f1.png`

### Implementation Status - 2026-05-27

Implemented:

- Real image loading from Sprint 1 split CSVs through `HAM10000ImageDataset`.
- Deterministic RGB `224x224` ImageNet preprocessing for feature extraction.
- Frozen classifier-free ResNet50, MobileNetV2, and EfficientNetB0 feature extractors.
- Per-backbone/per-split `.pt` feature caches with CSV and JSON manifests.
- MLP training from cached features for the three single-backbone frozen baselines.
- Train-only class weighting for weighted cross-entropy.
- Accuracy, macro precision/recall/F1, weighted F1, per-class metrics, confusion matrices, training curves, and summary macro-F1 plot export.
- Thin Colab runner in `notebooks/02_extract_frozen_features.ipynb`.
- Local Sprint 2 result interpretation is recorded in `docs/report_notes/sprint-2-results-tr.md`.
- Local MLP hyperparameter-search notes are available through the `mlp_hparam_v1_full` report assets.

Generated feature caches and run artifacts remain intentionally gitignored.

Baseline result summary:

- Default class-weighted single-backbone baseline: ResNet50 is best by macro-F1 (`0.531`), followed by EfficientNetB0 (`0.506`) and MobileNetV2 (`0.468`).
- Local MLP hyperparameter search found the best tested macro-F1 with EfficientNetB0 + class-weighted SGD (`0.560`).
- Accuracy and weighted-F1 are higher than macro-F1 because the `nv` majority class dominates the test set; macro-F1 remains the main interpretation metric.

### Interim Sprint Report

Actual local note:

> We extracted frozen ImageNet-pretrained features from ResNet50, MobileNetV2, and EfficientNetB0 and trained MLP classifiers on each feature representation. In the default class-weighted baseline, ResNet50 achieved the strongest single-backbone macro-F1. A focused MLP hyperparameter search showed that EfficientNetB0 with class-weighted SGD can produce a higher macro-F1 among the tested MLP settings. Minority and visually similar classes, especially `df` and `mel`, remain difficult, so Sprint 3 will test whether fusion improves per-class behavior.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Feature extraction is too slow. | Use mixed precision where safe, increase batch size, and save after each backbone. |
| MLP overfits quickly. | Add dropout, early stopping, lower hidden size, or stronger weight decay. |
| Accuracy looks fine but macro-F1 is poor. | Treat this as expected HAM10000 behavior and make it part of the discussion. |
| Cached features become inconsistent. | Add manifest checks and config/hash metadata to cache files. |

### Git Hygiene

Expected commits:

- `feat(models): add pretrained backbone feature extractors`
- `feat(features): add frozen feature cache pipeline`
- `feat(train): add MLP training from cached features`
- `feat(eval): add classification metrics and confusion matrix export`
- `test(features): verify cache shapes and split alignment`
- `docs: add sprint 2 baseline notes`

Large feature files should stay out of Git.

## Sprint 3: Fusion Experiments

### Goal

Complete the core feature-fusion requirement using frozen features first. Run pairwise and three-CNN fusion experiments with both concatenation and projected learnable weighted fusion, then analyze whether fusion improves performance overall and per class.

Starting point for Sprint 3: frozen feature caches and single-backbone MLP baselines exist locally.

### Main Workstreams

- Implement concatenation fusion for pairwise and three-backbone combinations.
- Implement projected learnable weighted fusion:
  - project each backbone feature to a shared dimension, such as 512,
  - learn softmax-normalized weights,
  - feed fused vector into the MLP.
- Run frozen pairwise experiments:
  - ResNet50 + MobileNetV2
  - ResNet50 + EfficientNetB0
  - MobileNetV2 + EfficientNetB0
- Run frozen three-CNN experiments:
  - ResNet50 + MobileNetV2 + EfficientNetB0
- Export learned fusion weights for weighted runs.
- Compare single vs pairwise vs three-CNN, concatenation vs weighted fusion, and overall vs per-class metrics.
- Track runtime for each experiment.

### Decisions

- Use 512 as the default projection dimension for weighted fusion.
- Use global learnable softmax weights rather than sample-specific weights. This is simpler, interpretable, and report-friendly.
- Keep training settings similar across concatenation and weighted fusion, while allowing input dimensions to differ naturally.
- If time allows, repeat the best few experiments with three seeds. If not, report single-seed limitations honestly.

### Verification Gates

Sprint 3 is done only if:

- All frozen fusion combinations run:
  - 3 pairwise combinations x 2 fusion methods = 6 runs
  - 1 three-CNN combination x 2 fusion methods = 2 runs
- Total frozen matrix is complete:
  - 3 single-backbone runs from Sprint 2
  - 8 fusion runs from Sprint 3
- Weighted fusion weights are saved and sum to 1 after softmax.
- Concatenated feature dimensions are correct:
  - ResNet50 + MobileNetV2: about 3328
  - ResNet50 + EfficientNetB0: about 3328
  - MobileNetV2 + EfficientNetB0: about 2560
  - all three: about 4608
- A report table exists comparing all frozen experiments.
- Confusion matrix exists for the best frozen model.

### Implementation Status - 2026-05-27

Sprint 3 is complete locally.

Implemented:

- `src/dl_midterm/models/fusion.py` now contains concatenation fusion, projected learnable
  weighted fusion, expected concat dimension helpers, and a weighted-fusion MLP wrapper.
- `scripts/run_experiment_matrix.py` expands the configured frozen matrix and runs the 8 Sprint 3
  fusion experiments from cached Sprint 2 `.pt` features only.
- Cache alignment checks verify image ID, label name, label index, and split order before fusion.
- Weighted fusion projects all backbone vectors to 512 dimensions and saves learned global
  softmax weights.
- `scripts/make_report_assets.py` exports frozen matrix report assets without rerunning training.
- Unit tests cover fusion output shapes, softmax weight normalization, expected concat dimensions,
  matrix expansion, and cache alignment.

Generated local results:

- Best frozen fusion run by test macro-F1: ResNet50 + EfficientNetB0 concat (`0.595`).
- Best single-backbone Sprint 2 baseline by test macro-F1: ResNet50 (`0.531`).
- Best fusion gain over the default single-backbone baseline: `+0.064` macro-F1.
- Per-class fusion gain heatmap shows the largest class-level improvements on `df`, followed by
  gains on `bcc`, `bkl`, `akiec`, and `mel`.
- Representation complementarity analysis shows ResNet50 + EfficientNetB0 has the lowest
  sample-similarity correlation (`0.692`) and highest complementarity (`0.308`) among the tested
  backbone pairs.
- Weighted fusion weights sum to 1 for every weighted run after softmax.
- The frozen comparison table contains 11 runs: 3 single-backbone baselines and 8 fusion runs.

Verified:

- `uv run pytest`
- `uv run ruff check src scripts tests`
- `uv run python scripts/run_experiment_matrix.py --config configs/experiments/frozen_feature_matrix.yaml --default-config configs/default.yaml --dataset-config configs/dataset/selected_dataset.yaml --feature-source frozen`

### Expected Artifacts

- `src/dl_midterm/models/fusion.py`
- `scripts/run_experiment_matrix.py`
- `scripts/make_report_assets.py`
- `artifacts/runs/*_frozen_r50-mnv2_concat_mlp_s42/`
- `artifacts/runs/*_frozen_r50-mnv2_weighted_mlp_s42/`
- `artifacts/runs/*_frozen_r50-effb0_concat_mlp_s42/`
- `artifacts/runs/*_frozen_r50-effb0_weighted_mlp_s42/`
- `artifacts/runs/*_frozen_mnv2-effb0_concat_mlp_s42/`
- `artifacts/runs/*_frozen_mnv2-effb0_weighted_mlp_s42/`
- `artifacts/runs/*_frozen_r50-mnv2-effb0_concat_mlp_s42/`
- `artifacts/runs/*_frozen_r50-mnv2-effb0_weighted_mlp_s42/`
- `artifacts/report_assets/tables/frozen_all_results.csv`
- `artifacts/report_assets/tables/fusion_weight_summary.csv`
- `artifacts/report_assets/tables/per_class_f1_frozen.csv`
- `artifacts/report_assets/tables/fusion_gain_summary.csv`
- `artifacts/report_assets/tables/per_class_fusion_gain.csv`
- `artifacts/report_assets/tables/representation_complementarity_summary.csv`
- `artifacts/report_assets/tables/fusion_complementarity_summary.csv`
- `artifacts/report_assets/figures/frozen_fusion_comparison.png`
- `artifacts/report_assets/figures/concat_vs_weighted.png`
- `artifacts/report_assets/figures/fusion_gain_macro_f1.png`
- `artifacts/report_assets/figures/per_class_f1_frozen_heatmap.png`
- `artifacts/report_assets/figures/per_class_fusion_gain_heatmap.png`
- `artifacts/report_assets/figures/frozen_best_confusion_matrix.png`
- `artifacts/report_assets/figures/learned_fusion_weights.png`
- `artifacts/report_assets/figures/representation_similarity_heatmap.png`
- `artifacts/report_assets/figures/fusion_gain_vs_complementarity.png`
- `artifacts/report_assets/figures/fusion_runs/`

### Interim Sprint Report

Actual local note:

> We completed the frozen-feature fusion experiments using both concatenation and projected
> learnable weighted fusion. Fusion improved macro-F1 compared with the default Sprint 2
> single-backbone baseline: ResNet50 + EfficientNetB0 concat reached `0.595` test macro-F1 versus
> `0.531` for the best default single-backbone run. Weighted fusion produced interpretable
> softmax weights, but concat was stronger in this single-seed frozen matrix. Per-class analysis
> showed notable improvement on `df`, `bcc`, `bkl`, `akiec`, and `mel`, while `vasc` remained
> strongest in the original ResNet50 baseline.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Weighted fusion underperforms concatenation. | This is still useful: discuss that concatenation preserves more information while weighted sum compresses features. |
| Weighted fusion weights collapse to one backbone. | Report this as evidence that one representation dominated. |
| MLP overfits concatenated features because input dimension is large. | Add dropout, weight decay, or reduce hidden size. |
| Too many runs take time. | Prioritize three-CNN concat/weighted and the best two pairwise combinations first, then fill remaining pairwise runs. |

### Git Hygiene

Expected commits:

- `feat(fusion): add concatenation and projected weighted fusion modules`
- `feat(experiments): add frozen fusion experiment matrix`
- `feat(eval): export learned fusion weights and per-class F1`
- `feat(report): generate frozen fusion comparison tables and plots`
- `test(fusion): verify fusion output shapes and weight normalization`
- `docs: add sprint 3 fusion observations`

## Sprint 4: Fine-Tuning and Frozen-vs-Fine-Tuned Comparison

### Goal

Satisfy the transfer-learning requirement by fine-tuning the last meaningful blocks of ResNet50, MobileNetV2, and EfficientNetB0, extracting fine-tuned features, and rerunning the same MLP comparison matrix.

The goal is not to chase the highest possible score. It is to compare whether fine-tuned representations improve HAM10000 classification and which classes benefit.

### Main Workstreams

- Define architecture-specific unfreezing policies:
  - ResNet50: last stage/block, such as `layer4`
  - MobileNetV2: last meaningful feature blocks
  - EfficientNetB0: last meaningful feature blocks
- Fine-tune each backbone with a temporary classification head.
- Use conservative augmentation during fine-tuning:
  - resize/crop,
  - horizontal/vertical flips if acceptable for dermoscopy,
  - mild color jitter,
  - random rotation,
  - no extreme transforms that distort lesions unrealistically.
- Save best checkpoint by validation macro-F1 or validation loss.
- Extract fine-tuned features from the best checkpoint for train/validation/test.
- Rerun required MLP matrix using fine-tuned features:
  - 3 single-backbone runs,
  - 6 pairwise fusion runs,
  - 2 three-CNN fusion runs.
- Compare frozen vs fine-tuned results and runtime.

### Decisions

- Fine-tune the last meaningful blocks per architecture and document exactly what is unfrozen.
- Start with 5-10 fine-tuning epochs and rely on early stopping.
- Save both validation macro-F1 and validation loss.
- Ideally run the full fine-tuned matrix. If time is tight, the emergency reduction is fine-tuned singles plus three-CNN concat/weighted, but this should be marked as a fallback.

### Implementation Status - 2026-05-29

Sprint 4 implementation and full Colab execution are complete. The Drive artifact bundle was
downloaded locally, and frozen-vs-fine-tuned report assets were regenerated against the canonical
Sprint 3 frozen run set.

Implemented:

- Architecture-specific partial fine-tuning policies:
  - ResNet50: unfreeze `layer4` and `fc`.
  - MobileNetV2: unfreeze `features[16]`, `features[17]`, `features[18]`, and `classifier`.
  - EfficientNetB0: unfreeze `features[7]`, `features[8]`, and `classifier`.
- `scripts/finetune_backbone.py` now fine-tunes backbones, saves best checkpoints by validation
  macro-F1, writes fine-tuning histories/confusion matrices, and extracts classifier-free
  fine-tuned feature caches.
- Fine-tuned feature source is canonicalized as `finetuned`, with caches under
  `artifacts/features/ham10000/finetuned/<backbone>/`.
- `scripts/train_mlp.py` and `scripts/run_experiment_matrix.py` support cached `finetuned`
  features in addition to `frozen`.
- Report asset export now supports fine-tuned matrix tables/plots and frozen-vs-fine-tuned
  comparison summaries.
- `notebooks/03_finetune_backbones.ipynb` is a thin Colab runner around script commands.
- Unit tests cover freeze/unfreeze policy, fine-tuned cache shape/alignment, matrix expansion,
  weighted-fusion weight normalization, and cached-feature MLP tensor-only input.

Verified locally:

- `uv run ruff check src scripts tests`
- `uv run pytest`
- non-canonical `/tmp` smoke fine-tune/cache extraction for MobileNetV2 with
  `--limit-per-split 2`, `--epochs 1`, and `--no-pretrained`
- non-canonical `/tmp` smoke MLP run from the tiny fine-tuned cache using matching mini split CSVs

Canonical Sprint 4 results:

- Full ImageNet-pretrained fine-tuning completed for ResNet50, MobileNetV2, and EfficientNetB0.
- Fine-tuned feature caches exist for all three backbones with train/val/test alignment verified.
- Full 11-run fine-tuned matrix completed: 3 single-backbone runs and 8 fusion runs.
- Best fine-tuned run: `r50+mnv2+effb0 concat`, macro-F1 `0.706`, accuracy `0.811`,
  weighted-F1 `0.813`.
- Best frozen Sprint 3 run: `r50+effb0 concat`, macro-F1 `0.595`.
- Best matched frozen-vs-fine-tuned gain: `r50+mnv2+effb0 weighted`, macro-F1 gain `+0.168`.
- Report-ready fine-tuned and frozen-vs-fine-tuned CSV/PNG assets were generated under
  `artifacts/report_assets/`.

### Sprint 4D Extension Status - 2026-06-02

Sprint 4D completed a validation-gated test-time augmentation audit for already selected fine-tuned
fusion models. It did not retrain CNNs or MLPs and did not change the lesion-aware split.

- Validation screened `identity`, `tta_flip4`, and `tta_rot4` for the regenerated Sprint 4 concat
  checkpoint and the Sprint 4C weighted checkpoint.
- Validation selected `tta_rot4` for both test-eligible models.
- Test was evaluated once after the validation gate:
  - regenerated Sprint 4 concat: macro-F1 `0.685 -> 0.690`;
  - Sprint 4C weighted: macro-F1 `0.699 -> 0.733`.
- Sprint 4D uses a regenerated local Sprint 4 concat MLP checkpoint because the exact May 29
  `model.pt` was not persisted in the Drive artifact mirror. This is documented as an inference
  audit artifact, not a replacement for the canonical Sprint 4 result.

### Sprint 4E Extension Status - 2026-06-02

Sprint 4E completed a local cached-feature fusion diagnostic. It used the existing fine-tuned
feature caches, did not fine-tune CNNs, and did not read raw images during the fusion-stage MLP
experiments.

- Fourteen three-backbone fusion candidates were trained across concat, global weighted, and
  per-class weighted fusion variants.
- The diagnostic tested train-fitted per-backbone standardization, per-backbone L2 normalization,
  projection capacity, focal loss, and label smoothing.
- Best validation macro-F1 was `0.683` for `perclass_l2_p512`.
- The validation gate selected `concat_standardize_base` and `concat_l2_base` for test.
- Test macro-F1 was `0.691` for `concat_standardize_base` and `0.683` for `concat_l2_base`.
- Sprint 4E did not beat canonical Sprint 4 concat (`0.706`) or Sprint 4D weighted + `tta_rot4`
  (`0.733`), but it showed that feature normalization can improve concat validation behavior.
- Sprint 5 should frame Sprint 4E as a controlled diagnostic: simple fusion-stage normalization and
  capacity changes were not enough to surpass the stronger fine-tuned/TTA baselines.

### Sprint 4F Extension Status - 2026-06-02

Sprint 4F completed a Colab T4 augmented fine-tuning extension. It returned to image-level
fine-tuning because Sprint 4D suggested transformation robustness matters and Sprint 4E showed that
fusion-stage MLP changes alone were low-impact.

- Feature source: `finetuned_augmented`.
- Checkpoint directory: `artifacts/checkpoints/finetuned_augmented_backbones/`.
- Colab runner: `notebooks/04_sprint4f_augmented_finetuning.ipynb`.
- Backbone policy kept Sprint 4's conservative last-block unfreezing for all three
  backbones.
- Augmentation policy added controlled random resized crop, flips, rotation, mild affine, and mild
  color jitter.
- The fusion matrix was intentionally small: 3 single-backbone MLP runs plus three-backbone concat
  and weighted runs.
- Drive mirror policy now preserves checkpoints, feature cache `.pt` files, and MLP `model.pt`
  files; these large artifacts remain gitignored locally.
- Test macro-F1 results were below the stronger baselines:
  - augmented ResNet50 single: `0.589`;
  - augmented MobileNetV2 single: `0.575`;
  - augmented EfficientNetB0 single: `0.579`;
  - augmented three-backbone concat: `0.645`;
  - augmented three-backbone weighted: `0.615`.
- Sprint 4F should be reported as a controlled negative result: train-time augmentation did not
  reproduce Sprint 4D's inference-time TTA gain.

### Sprint 4G Extension Status - 2026-06-03

Sprint 4G completed a local cached-feature autoresearch ensemble over existing checkpointed
MLP/fusion models from Sprint 4B, 4C, 4E, and 4F. It did not fine-tune CNNs, did not create new
feature caches, and did not read raw images.

- Candidate discovery required existing `model.pt` and `config_resolved.yaml`.
- Candidate filtering and ensemble selection used validation macro-F1 only.
- Random Dirichlet weight search was disabled after a diagnostic run showed validation-overfit
  risk; the final search uses deterministic uniform/rank-weighted soft voting.
- Selected ensemble: uniform soft-vote of 3 existing models.
- Validation macro-F1: `0.725`.
- Test macro-F1: `0.707`, accuracy `0.806`, weighted-F1 `0.812`.
- This is a near-tie with canonical Sprint 4 concat (`0.706`) and remains below Sprint 4D weighted
  + `tta_rot4` (`0.733`).
- Sprint 4G should be framed as evidence that simple post-hoc model averaging has limited impact
  compared with the validation-gated TTA result.

### Sprint 4H Extension Status - 2026-06-03

Sprint 4H completed a Colab T4 targeted fine-tuning extension. It targeted macro-F1 more directly
than Sprint 4F by combining deeper ResNet50 fine-tuning with class-balanced focal loss while
removing the random crop/affine augmentation that likely hurt Sprint 4F.

- Feature source: `finetuned_targeted`.
- Checkpoint directory: `artifacts/checkpoints/finetuned_targeted_backbones/`.
- Colab runner: `notebooks/04_sprint4h_targeted_finetuning.ipynb`.
- ResNet50 policy: `layer3_layer4`.
- MobileNetV2 and EfficientNetB0 policy: Sprint 4 last feature blocks.
- Loss: class-balanced focal with sqrt-smoothed clipped train-split class weights.
- Augmentation: flips, rotation, mild color jitter; no random resized crop or affine in the first
  pass.
- Completed matrix: full 11-run fine-tuned cached-feature MLP matrix.
- Drive/local artifact check: 3 backbone checkpoints, 9 feature cache `.pt` files, 11 MLP
  `model.pt` files, and 14 metrics files.
- Best Sprint 4H cached-feature matrix result: three-backbone concat, test macro-F1 `0.643`,
  accuracy `0.768`, weighted-F1 `0.779`.
- Best image-level fine-tuned head was ResNet50, test macro-F1 `0.647`; this signal did not carry
  through the cached-feature MLP/fusion matrix.
- Sprint 4H remains below canonical Sprint 4 concat (`0.706`) and Sprint 4D weighted + `tta_rot4`
  (`0.733`), so it should be reported as an informative negative targeted-training result rather
  than a new best.

### Sprint 4I Extension Status - 2026-06-03

Sprint 4I completed a local inference-only geometry-safe TTA refinement. It reused the strongest
Sprint 4C weighted fusion model and the canonical Sprint 4 fine-tuned backbone checkpoints.

- No new CNN training, feature caches, or data splits were created.
- Policy selection remained validation-gated; test was not used to choose the TTA policy.
- Tested policy: full-frame deterministic D4 TTA (`identity`, rotations, and mirrored rotations).
- Validation macro-F1 improved from `0.678` with identity to `0.717` with D4 TTA.
- Test macro-F1 improved from `0.699` with identity to `0.727` with D4 TTA.
- Sprint 4I remained slightly below Sprint 4D weighted + `tta_rot4` (`0.733`), so the best result
  remains Sprint 4D. Sprint 4I is useful evidence that geometry-safe averaging helps, while also
  showing that more TTA views are not necessarily better than the simpler rot4 policy.

### Verification Gates

Sprint 4 is done only if:

- Three fine-tuned backbone checkpoints exist.
- Fine-tuned feature caches exist for all three backbones.
- Single-backbone fine-tuned MLP results exist.
- At minimum, three-CNN fine-tuned concat and weighted results exist.
- Ideally, the full 11-run fine-tuned matrix exists.
- A table compares frozen vs fine-tuned by backbone, fusion type, and runtime.
- Training logs show fine-tuning did not catastrophically overfit.
- Report notes explain exact unfreezing policy.

### Expected Artifacts

- `scripts/finetune_backbone.py`
- `src/dl_midterm/training/finetune.py`
- `src/dl_midterm/training/checkpointing.py`
- `notebooks/03_finetune_backbones.ipynb`
- `notebooks/04_extract_finetuned_features.ipynb`
- `artifacts/checkpoints/finetuned_backbones/resnet50_best.pt`
- `artifacts/checkpoints/finetuned_backbones/mobilenet_v2_best.pt`
- `artifacts/checkpoints/finetuned_backbones/efficientnet_b0_best.pt`
- `artifacts/features/ham10000/finetuned/resnet50/`
- `artifacts/features/ham10000/finetuned/mobilenet_v2/`
- `artifacts/features/ham10000/finetuned/efficientnet_b0/`
- `artifacts/runs/*_finetuned_*_mlp_s42/`
- `artifacts/report_assets/tables/finetuned_all_results.csv`
- `artifacts/report_assets/tables/frozen_vs_finetuned_summary.csv`
- `artifacts/report_assets/tables/finetuning_gain_summary.csv`
- `artifacts/report_assets/tables/finetuned_per_class_f1.csv`
- `artifacts/report_assets/figures/frozen_vs_finetuned_macro_f1.png`
- `artifacts/report_assets/figures/runtime_vs_macro_f1.png`
- `artifacts/report_assets/figures/finetuned_best_confusion_matrix.png`

### Interim Sprint Report

Write:

> We fine-tuned the final meaningful blocks of ResNet50, MobileNetV2, and EfficientNetB0 and extracted fine-tuned feature representations. Compared with frozen feature extraction, fine-tuning improved the best overall macro-F1 from 0.595 to 0.706. In the matched frozen-vs-fine-tuned table, the largest macro-F1 gain was +0.168 for the three-backbone weighted fusion run. The best fine-tuned concat model improved minority-class F1 most for vasc, bcc, df, akiec, and mel, while the additional runtime and single-seed limitation remain important final-report caveats.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Fine-tuning overfits minority classes or collapses. | Reduce unfrozen depth, lower learning rate, increase augmentation, and use early stopping. |
| Fine-tuning all three backbones takes too long. | Fine-tune one backbone at a time; prioritize EfficientNetB0 and ResNet50, then MobileNetV2. |
| Full fine-tuned matrix cannot finish. | Minimum fallback: fine-tuned singles plus three-CNN concat and three-CNN weighted. Clearly document scope limitation. |
| Colab disconnects. | Save checkpoints every epoch and features per split/backbone. |
| HAM10000 becomes operationally too risky. | Switch to EuroSAT before this sprint begins, not after half the fine-tuning is done. |

### Git Hygiene

Expected commits:

- `feat(finetune): add architecture-specific unfreezing policies`
- `feat(finetune): add backbone fine-tuning script`
- `feat(features): support feature extraction from fine-tuned checkpoints`
- `feat(experiments): add fine-tuned experiment matrix`
- `feat(report): add frozen-vs-finetuned comparison assets`
- `docs: document fine-tuning policy and sprint 4 observations`

## Sprint 5: Analysis, Report, Presentation, and Final Packaging

### Goal

Turn experiments into a strong scientific story. This sprint is about evidence, interpretation, report quality, and submission safety: aggregate all metrics, produce final tables/plots, write the LaTeX report, prepare the YouTube presentation, and package code, report, and video link according to the assignment.

The assignment explicitly emphasizes the discussion section and asks which model learned better features, whether fusion helped, which combination was best, model strengths/weaknesses, and which transfer-learning approach was better and how long it took.

### Main Workstreams

- Aggregate all results into final tables:
  - single-backbone comparison,
  - pairwise fusion comparison,
  - three-CNN fusion comparison,
  - concat vs weighted,
  - frozen vs fine-tuned,
  - runtime summary.
- Produce final plots:
  - macro-F1 comparison,
  - per-class F1 heatmap/table,
  - confusion matrix for best model,
  - learned fusion weights,
  - runtime vs performance.
- Write the LaTeX report:
  - introduction,
  - dataset,
  - method,
  - experiments,
  - results,
  - discussion,
  - conclusion.
- Add citations for the assignment, HAM10000, ISIC source, CNN architectures, and recent HAM10000 studies if included.
- Prepare YouTube presentation script:
  - problem,
  - dataset,
  - method,
  - experiment matrix,
  - key results,
  - discussion and limitations.
- Package final submission:
  - source code,
  - PDF report,
  - selected logs/graphs,
  - YouTube link,
  - archive with required naming.

### Decisions

- Use macro-F1 as the primary ranking metric because HAM10000 is imbalanced.
- Still report accuracy, precision, recall, and F1 as assignment-required metrics.
- Include weighted-F1, but do not let it hide minority-class weaknesses.
- Claim fusion works only if it improves macro-F1 or per-class behavior meaningfully.
- Include Grad-CAM only as an optional stretch if it is clean and useful.
- If time allows, rerun the best two or three configurations with three seeds. If not, state the single-seed limitation honestly.

### Verification Gates

Sprint 5 is done only if:

- Every assignment-required comparison appears in at least one final table.
- The report includes dataset explanation, training details, hyperparameters, tables/graphs, and discussion answering required questions.
- No report section makes clinical/diagnostic claims beyond benchmark classification.
- All generated figures are readable and referenced in text.
- PDF compiles cleanly.
- Git repo can be cloned and basic commands run.
- Submission archive contains source code, PDF report, relevant artifacts/logs/plots, and YouTube link.
- YouTube presentation video exists before final packaging.

### Expected Artifacts

- `artifacts/report_assets/tables/all_results.csv`
- `artifacts/report_assets/tables/best_models.csv`
- `artifacts/report_assets/tables/fusion_gain_summary.csv`
- `artifacts/report_assets/tables/frozen_vs_finetuned_summary.csv`
- `artifacts/report_assets/tables/runtime_summary.csv`
- `artifacts/report_assets/tables/per_class_f1_comparison.csv`
- `artifacts/report_assets/figures/macro_f1_all_experiments.png`
- `artifacts/report_assets/figures/concat_vs_weighted.png`
- `artifacts/report_assets/figures/frozen_vs_finetuned.png`
- `artifacts/report_assets/figures/learned_fusion_weights.png`
- `artifacts/report_assets/figures/runtime_vs_macro_f1.png`
- `artifacts/report_assets/figures/best_confusion_matrix.png`
- `reports/final_report/main.tex`
- `reports/final_report/references.bib`
- `reports/final_report/sections/*.tex`
- `reports/final_report/main.pdf`
- `submission/youtube_link.txt`
- `submission/archive/<required_name>.rar`

### Interim Sprint Report

Write:

> We aggregated the full experiment matrix and completed the final report assets. The best model was [configuration] under macro-F1. Fusion [helped/did not help] compared with the best single-backbone baseline, and weighted fusion [did/did not] outperform concatenation. Fine-tuning [improved/did not improve] performance but required [runtime cost]. The final discussion emphasizes class imbalance, visually similar lesion categories, learned fusion weights, and the limits of benchmark classification.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Results are messy or fusion is not clearly better. | This is acceptable if discussed scientifically; do not force a false conclusion. |
| Report takes longer than expected. | Use generated tables directly and write discussion around the assignment’s required questions. |
| Some pairwise fine-tuned runs are missing. | Mark emergency reduction clearly, but keep all frozen matrix plus fine-tuned singles and fine-tuned three-CNN fusion. |
| Video is forgotten. | Record after report draft, not at the last hour; video is mandatory for evaluation. |
| PDF formatting breaks. | Keep LaTeX simple and use standard packages. |

### Git Hygiene

Expected commits:

- `feat(report): generate final result tables and plots`
- `report: write dataset and method sections`
- `report: write results and discussion sections`
- `report: add references and final PDF`
- `docs: add presentation script and reproducibility notes`
- `chore(submission): stage final package and youtube link placeholder`

Final Git state should be clean: no accidental datasets, no checkpoints, and no huge feature files unless intentionally excluded from Git and included only in the submission package.

## Recommended Order and Decision Checkpoints

| Point | Decision |
|---|---|
| End of Sprint 1 | If HAM10000 acquisition/splitting is too messy, switch to EuroSAT immediately. |
| End of Sprint 2 | If frozen single-backbone results are random or unstable, fix preprocessing/splits before fusion. |
| Middle of Sprint 3 | If weighted fusion fails, keep concat complete and debug weighted fusion with one pair first. |
| Start of Sprint 4 | If time is short, fine-tune fewer epochs but still complete the transfer-learning comparison. |
| Middle of Sprint 5 | If the report is weak, prioritize discussion over optional extra experiments. |

## Minimal Required Experiment Matrix

### Frozen Feature Extraction

| Type | Runs |
|---|---:|
| Single CNN | 3 |
| Pairwise concat | 3 |
| Pairwise weighted | 3 |
| Three-CNN concat | 1 |
| Three-CNN weighted | 1 |
| Total | 11 |

### Fine-Tuned Feature Extraction

| Type | Runs |
|---|---:|
| Single CNN | 3 |
| Pairwise concat | 3 |
| Pairwise weighted | 3 |
| Three-CNN concat | 1 |
| Three-CNN weighted | 1 |
| Total | 11 |

Additional required runs:

- 3 backbone fine-tuning runs
- 3 frozen feature extraction runs
- 3 fine-tuned feature extraction runs

Emergency reduced matrix, only if necessary:

- Frozen: complete 11 runs
- Fine-tuned: 3 singles, best pair concat/weighted, and three-CNN concat/weighted

The full matrix is preferred and should be feasible because MLP training on cached features is cheap.

## Final Definition of Done

The project is done when all of these are true:

### Dataset Credibility

- HAM10000 source, license, metadata, class distribution, and split strategy are documented.
- Lesion-level leakage prevention is attempted and documented.

### Implementation Correctness

- ResNet50, MobileNetV2, and EfficientNetB0 final classifiers are removed.
- Feature vectors are extracted and cached.
- Concatenation and weighted fusion both work.
- MLP classifier is used consistently.
- Frozen and fine-tuned transfer-learning settings are both tested.

### Experiment Completeness

- Single CNN, pairwise fusion, and three-CNN fusion comparisons are present.
- Concatenation vs weighted fusion is present.
- Frozen vs fine-tuned comparison is present.
- Accuracy, precision, recall, and F1-score are reported.

### Scientific Discussion

The report answers:

- Which backbone learned better features?
- Did fusion help?
- Which combination was best?
- What are the model strengths and weaknesses?
- Did fine-tuning help, and how much time did it cost?

The report includes per-class analysis, confusion matrix, learned fusion weights, and runtime tradeoff. It avoids medical diagnosis claims.

### Reproducibility

- Main experiments are config-driven.
- Run folders contain resolved configs, metrics, histories, predictions, and plots.
- Colab notebooks are launchers, not hidden implementations.
- Git history shows progressive, understandable work.

### Submission

- PDF report is compiled.
- Code is included.
- Relevant logs/graphs are included.
- YouTube video link exists.
- Final archive follows the required naming format.

## Strategic Rule

Do not start by fine-tuning. Start with clean metadata, leakage-aware splits, frozen features, and MLP baselines. That gives the project a working scientific baseline early, then fusion and fine-tuning become controlled extensions instead of chaos.
