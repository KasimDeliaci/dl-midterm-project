# Notebooks

Colab notebooks are launchers, not the main implementation.

They should:

- mount Google Drive,
- clone or update the repository,
- install dependencies with `uv`,
- run scripts from `scripts/`,
- copy artifacts back to Drive when needed.

Core logic belongs in `src/dl_midterm/`.

## Colab Order

Use `00_colab_setup.ipynb` first. It mounts Drive, clones/enters the repo, installs `uv`, and
extracts `ham10000_colab_bundle.tar` into the repo root.

The runner notebooks force-remount Drive and check these bundle locations:

- `/content/drive/MyDrive/ham10000_colab_bundle.tar`
- `/content/drive/MyDrive/Colab Notebooks/ham10000_colab_bundle.tar`
- `/content/drive/MyDrive/dl-assignment/ham10000_colab_bundle.tar`

All notebooks request a Colab GPU runtime with T4 metadata. Colab may still assign hardware based
on account limits and availability; if needed, use Runtime > Change runtime type and choose T4 GPU.
Runner notebooks also bootstrap the repo/data if Colab opens them in a fresh runtime.

Skip `01_dataset_prepare.ipynb` when preserving the Sprint 1 split CSVs from the uploaded bundle.
Run it only if you intentionally want to regenerate dataset audit/split artifacts.

Run `02_extract_frozen_features.ipynb` for Sprint 2 frozen feature extraction, MLP training, and
evaluation. It syncs frozen feature caches and report-ready assets to
`/content/drive/MyDrive/dl-midterm-artifacts/`.

Run `02b_mlp_hyperparam_search.ipynb` only after frozen feature caches exist locally or under that
Drive artifact folder. The full grid is opt-in; the default verification path runs a small smoke
test.
