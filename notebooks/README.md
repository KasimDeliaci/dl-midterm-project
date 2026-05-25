# Notebooks

Colab notebooks are launchers, not the main implementation.

They should:

- mount Google Drive,
- clone or update the repository,
- install dependencies with `uv`,
- run scripts from `scripts/`,
- copy artifacts back to Drive when needed.

Core logic belongs in `src/dl_midterm/`.
