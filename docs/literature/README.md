# Literature Notes

Use this folder for paper notes that help write the report and interpret results.

Recommended pattern:

```text
docs/literature/
├── literature_index.md
├── paper_registry.yaml
├── papers/       # local PDF cache, ignored by Git
├── notes/        # paper-reading notes and synthesis notes
└── axes/         # theme-specific literature notes
```

Literature notes are project context. They should not be parsed by training code.

PDF files under `docs/literature/papers/` are local convenience copies. Keep
source URLs and citation metadata in `paper_registry.yaml` so the report remains
traceable without committing downloaded papers.
