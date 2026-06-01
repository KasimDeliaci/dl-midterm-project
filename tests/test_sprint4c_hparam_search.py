"""Sprint 4C hyperparameter-search planning tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from dl_midterm.config.load_config import load_yaml


def _load_sprint4c_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_sprint4c_hparam_search.py"
    )
    scripts_dir = str(module_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_sprint4c_hparam_search", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sprint4c_screen_config_expands_to_72_runs() -> None:
    module = _load_sprint4c_module()
    config = load_yaml("configs/experiments/sprint4c_finetuned_mlpfusion_screen.yaml")[
        "mlp_fusion_hparam_search"
    ]

    runs = module.expand_search_runs(config)

    assert len(runs) == 72
    assert sum(run["fusion_method"] == "none" for run in runs) == 24
    assert sum(run["fusion_method"] == "concat" for run in runs) == 24
    assert sum(run["fusion_method"] == "weighted" for run in runs) == 24
    assert {run["projection_dim"] for run in runs if run["fusion_method"] == "weighted"} == {
        256,
        512,
        1024,
    }


def test_sprint4c_three_run_smoke_selects_single_concat_and_weighted() -> None:
    module = _load_sprint4c_module()
    config = load_yaml("configs/experiments/sprint4c_finetuned_mlpfusion_screen.yaml")[
        "mlp_fusion_hparam_search"
    ]
    runs = module.expand_search_runs(config)

    smoke_runs = module._limit_planned_runs(runs, 3)

    assert [run["fusion_method"] for run in smoke_runs] == ["none", "concat", "weighted"]


def test_sprint4c_full_config_expands_to_22_runs_with_candidate_projection_dims() -> None:
    module = _load_sprint4c_module()
    config = load_yaml("configs/experiments/sprint4c_finetuned_mlpfusion_full.yaml")[
        "mlp_fusion_hparam_search"
    ]

    runs = module.expand_search_runs(config)

    assert len(runs) == 22
    assert sum(run["fusion_method"] == "none" for run in runs) == 6
    assert sum(run["fusion_method"] == "concat" for run in runs) == 8
    assert sum(run["fusion_method"] == "weighted" for run in runs) == 8
    assert {run["candidate_name"] for run in runs if run["fusion_method"] == "weighted"} == {
        "w_cw_adamw_base_p1024",
        "w_cw_adamw_low_lr_p512",
    }
