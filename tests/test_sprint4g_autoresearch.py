"""Sprint 4G autoresearch ensemble tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from dl_midterm.config.load_config import load_yaml


def _load_autoresearch_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_autoresearch_ensemble.py"
    spec = importlib.util.spec_from_file_location("run_autoresearch_ensemble", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sprint4g_config_is_validation_gated_local_search() -> None:
    config = load_yaml("configs/experiments/sprint4g_autoresearch_ensemble.yaml")["sprint4g"]

    assert config["selection"]["metric"] == "macro_f1"
    assert config["selection"]["test_top_k"] == 1
    assert "finetuned_augmented" in config["eligible_feature_sources"]
    assert config["candidate_pool"]["top_n_by_validation"] <= 14
    assert config["ensemble_search"]["max_size"] <= 5
    assert config["ensemble_search"]["random_weight_top_n"] == 0
    assert config["ensemble_search"]["random_weight_samples"] == 0
    assert config["report"]["max_ensemble_rows"] <= 100


def test_sprint4g_ensemble_search_weights_sum_to_one() -> None:
    module = _load_autoresearch_module()
    labels = torch.tensor([0, 1, 2, 0])
    candidates = []
    for index in range(3):
        candidate = module.Candidate(
            key=f"candidate_{index}",
            run_dir=Path("/tmp"),
            config={"best_val_macro_f1": 0.6 + index * 0.01},
        )
        logits = torch.zeros(4, 3)
        logits[:, index % 3] = 1.0
        metrics = {"macro_f1": 0.6 + index * 0.01}
        candidates.append(
            module.PredictionBundle(
                candidate=candidate,
                labels=labels,
                logits=logits,
                metrics=metrics,
            )
        )
    config = {
        "ensemble_search": {
            "min_size": 2,
            "max_size": 2,
            "include_uniform": True,
            "include_rank_weighted": True,
            "random_weight_top_n": 0,
            "random_weight_samples": 0,
        }
    }

    rows, _ = module.search_ensembles(
        candidates,
        class_names=["a", "b", "c"],
        config=config,
        seed=42,
    )

    assert rows
    for row in rows:
        weights = torch.tensor(__import__("json").loads(row["weights"]))
        assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_sprint4g_script_does_not_load_raw_images() -> None:
    source = Path("scripts/run_autoresearch_ensemble.py").read_text(encoding="utf-8")

    assert "HAM10000ImageDataset" not in source
    assert "PIL" not in source
    assert "image_path" not in source
