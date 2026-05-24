# Data Directory

Large dataset files stay out of Git.

Expected layout:

```text
data/
├── raw/         # downloaded HAM10000 files
├── processed/   # normalized/reorganized local dataset view
└── splits/      # train/val/test CSV files
```

Tracked files in this directory should be documentation, split metadata when small enough, or `.gitkeep` placeholders. Do not commit raw images.
