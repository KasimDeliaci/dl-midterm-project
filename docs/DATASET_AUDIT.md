# Dataset Audit

This document is the canonical Sprint 1 dataset audit location for HAM10000.

The audit is not complete yet. It should be filled by `scripts/prepare_dataset.py` and Sprint 1 manual review.

## Source And Access

- Primary dataset: HAM10000.
- Canonical paper: Tschandl, Rosendahl, and Kittler, 2018.
- Official source: ISIC Archive / Harvard Dataverse.
- Expected standard benchmark size: 10,015 images.
- Classes: `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.
- License note: verify and record the exact license text from the downloaded source before final reporting.

## Audit Checklist

- [ ] Download source URL recorded.
- [ ] License/usage terms recorded.
- [ ] Metadata file path recorded.
- [ ] Image directory path recorded.
- [ ] All metadata image IDs checked against files.
- [ ] Missing images listed, if any.
- [ ] Duplicate image IDs checked.
- [ ] Class counts generated.
- [ ] `lesion_id` availability checked.
- [ ] Split policy selected and recorded.
- [ ] Train/validation/test split files generated.
- [ ] Lesion-level leakage check passed or compromise documented.
- [ ] Class distribution table exported.
- [ ] Class distribution figure exported.

## Expected Class Labels

| Label | Meaning |
|---|---|
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `nv` | Melanocytic nevi |
| `mel` | Melanoma |
| `vasc` | Vascular lesions |

## Modeling Caveats

- The dataset is class-imbalanced; accuracy alone is insufficient.
- Per-class F1 and macro-F1 should be prominent in the report.
- If metadata includes multiple images of the same lesion, splits should group by lesion ID.
- This project evaluates benchmark classification, not clinical diagnosis.
