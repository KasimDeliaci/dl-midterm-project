# Data Directory

Large dataset files stay out of Git.

Expected layout:

```text
data/
├── raw/         # downloaded HAM10000 files
├── metadata/    # HAM10000_metadata.csv and small metadata inputs
├── processed/   # normalized/reorganized local dataset view
└── splits/      # train/val/test CSV files
```

Tracked files in this directory should be documentation, split metadata when small enough, or `.gitkeep` placeholders. Do not commit raw images.

For Sprint 1, place the standard HAM10000 metadata at:

```text
data/metadata/HAM10000_metadata.csv
```

Place extracted images anywhere under `data/raw/`; the preparation script searches recursively
for `.jpg`, `.jpeg`, and `.png` files whose filename stem matches `image_id`.
