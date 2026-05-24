# Outputs Directory

Generated artifacts live here and are mostly ignored by Git.

Expected layout:

```text
outputs/
├── features/        # cached CNN feature tensors
├── checkpoints/     # backbone and MLP checkpoints
├── runs/            # one folder per experiment run
└── report_assets/   # tables and figures used by the report
```

Commit small, final report assets only when there is a deliberate reason. Do not commit feature caches, checkpoints, or full run folders.
