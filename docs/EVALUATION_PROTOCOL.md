# Evaluation Protocol

This document defines evaluation rules for the HAM10000 multi-CNN feature-fusion project.

## Primary Metric

Macro-F1 is the primary interpretation metric.

Reason: HAM10000 is class-imbalanced. Macro-F1 gives each class equal weight and better reflects minority-class behavior than accuracy alone.

## Required Metrics

The assignment requires reporting:

- Accuracy
- Precision
- Recall
- F1-score

For this project, report macro and weighted variants where useful:

- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Per-class precision/recall/F1
- Confusion matrix

## Split Rules

Primary split:

- Stratified train/validation/test split.
- Target ratio: `70/15/15`.
- Group by `lesion_id` where metadata supports it.
- No lesion ID should appear in more than one split when grouping is available.

If perfect stratified grouping is impossible because of minority-class constraints, document the compromise in `docs/DATASET_AUDIT.md` and the report.

## Leakage Rules

- Do not allow the same lesion ID in both train and validation/test when lesion IDs are available.
- Fit normalization, imputation, class weights, and any preprocessing statistics on training data only.
- Do not select hyperparameters using test metrics.
- Do not compare results from different split policies as if they are directly equivalent.

## Experiment Comparison Rules

Compare models only when they use the same:

- dataset version,
- split files,
- image preprocessing policy,
- feature cache source,
- MLP training policy,
- metrics.

The full target matrix is:

- 3 frozen single-backbone runs.
- 6 frozen pairwise fusion runs.
- 2 frozen three-CNN fusion runs.
- 3 fine-tuned single-backbone runs.
- 6 fine-tuned pairwise fusion runs.
- 2 fine-tuned three-CNN fusion runs.

## Reporting Rules

Every model result should state:

- run ID,
- feature source (`frozen` or `finetuned`),
- backbone combination,
- fusion method,
- seed,
- split files,
- primary metric,
- runtime,
- feature dimensions,
- whether class weighting was used.

The final report must answer:

- Which backbone learned better features?
- Did fusion help?
- Which combination was best?
- What are the model strengths and weaknesses?
- Did fine-tuning help, and how much time did it cost?

## Medical Framing Rule

Do not claim diagnosis, clinical usefulness, or deployment readiness. Use language such as “benchmark dermoscopic image classification on HAM10000.”
