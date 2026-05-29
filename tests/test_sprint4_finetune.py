"""Sprint 4 fine-tuning policy and fine-tuned cache tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from dl_midterm.features.cache import (
    load_feature_cache,
    save_feature_cache,
    verify_cache_matches_split,
)
from dl_midterm.models.backbones import (
    apply_finetuning_policy,
    build_classification_backbone,
    build_finetuned_feature_extractor,
    expected_feature_dim,
)


def _load_run_matrix_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment_matrix.py"
    spec = importlib.util.spec_from_file_location("run_experiment_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finetuning_policy_unfreezes_only_final_blocks() -> None:
    policies = {
        "resnet50": ("layer4", "layer4.0.conv1.weight", "layer3.0.conv1.weight", "fc.weight"),
        "mobilenet_v2": (
            "last_feature_blocks",
            "features.18.0.weight",
            "features.0.0.weight",
            "classifier.1.weight",
        ),
        "efficientnet_b0": (
            "last_feature_blocks",
            "features.8.0.weight",
            "features.0.0.weight",
            "classifier.1.weight",
        ),
    }

    for backbone, (policy, trainable_name, frozen_name, head_name) in policies.items():
        model = build_classification_backbone(backbone, num_classes=7, pretrained=False)
        summary = apply_finetuning_policy(model, backbone, policy=policy)
        params = dict(model.named_parameters())

        assert summary["policy"] == policy
        assert params[trainable_name].requires_grad
        assert params[head_name].requires_grad
        assert not params[frozen_name].requires_grad
        assert 0 < summary["trainable_params"] < summary["total_params"]


def test_finetuned_checkpoint_feature_extractor_shape(tmp_path: Path) -> None:
    model = build_classification_backbone("mobilenet_v2", num_classes=7, pretrained=False)
    checkpoint_path = tmp_path / "mobilenet_v2_best.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    extractor = build_finetuned_feature_extractor(
        "mobilenet_v2",
        checkpoint_path=checkpoint_path,
        num_classes=7,
    )
    output = extractor(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, expected_feature_dim("mobilenet_v2"))


def test_finetuned_feature_cache_shape_and_split_alignment(tmp_path: Path) -> None:
    split_csv = tmp_path / "train.csv"
    split_csv.write_text(
        "image_id,label,image_path\nimg1,nv,/tmp/img1.jpg\nimg2,mel,/tmp/img2.jpg\n",
        encoding="utf-8",
    )
    cache_path = tmp_path / "train.pt"
    save_feature_cache(
        cache_path,
        features=torch.ones(2, 1280),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="train",
        backbone="efficientnet_b0",
        class_names=["nv", "mel"],
        feature_source="finetuned",
        seed=42,
    )

    cache = load_feature_cache(cache_path)

    assert cache.metadata["feature_source"] == "finetuned"
    assert cache.features.shape == (2, 1280)
    verify_cache_matches_split(cache, split_csv)


def test_finetuned_fusion_matrix_expansion_uses_requested_source() -> None:
    run_matrix = _load_run_matrix_module()
    config = {
        "combinations": [
            {"backbones": ["resnet50"], "fusion_methods": ["none"]},
            {
                "backbones": ["resnet50", "efficientnet_b0"],
                "fusion_methods": ["concat", "weighted"],
            },
        ]
    }

    runs = run_matrix.expand_fusion_run_matrix(config, feature_source="finetuned")

    assert len(runs) == 2
    assert {run["feature_source"] for run in runs} == {"finetuned"}
    assert {run["fusion_method"] for run in runs} == {"concat", "weighted"}


def test_cached_feature_mlp_stage_does_not_expose_raw_image_paths(tmp_path: Path) -> None:
    run_matrix = _load_run_matrix_module()
    first = save_feature_cache(
        tmp_path / "a.pt",
        features=torch.ones(2, 4),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="train",
        backbone="resnet50",
        class_names=["nv", "mel"],
        feature_source="finetuned",
        seed=42,
    )
    second = save_feature_cache(
        tmp_path / "b.pt",
        features=torch.zeros(2, 3),
        labels=torch.tensor([0, 1]),
        image_ids=["img1", "img2"],
        label_names=["nv", "mel"],
        split="train",
        backbone="efficientnet_b0",
        class_names=["nv", "mel"],
        feature_source="finetuned",
        seed=42,
    )

    dataset = run_matrix._build_concat_tensor_dataset([first, second])
    sample = dataset[0]

    assert isinstance(dataset, TensorDataset)
    assert len(sample) == 2
    assert all(isinstance(item, torch.Tensor) for item in sample)


def test_frozen_vs_finetuned_export_reports_missing_frozen_runs(tmp_path: Path) -> None:
    from dl_midterm.evaluation.reports import export_frozen_vs_finetuned_report_assets

    run_dir = tmp_path / "20260529_finetuned_r50_none_mlp_s42"
    run_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text(
        "\n".join(
            [
                "run_id: 20260529_finetuned_r50_none_mlp_s42",
                "feature_source: finetuned",
                "backbone: resnet50",
                "backbones:",
                "  - resnet50",
                "backbone_combination: resnet50",
                "backbone_count: 1",
                "fusion_method: none",
                "feature_dim: 2048",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        """
{
  "accuracy": 0.1,
  "macro_precision": 0.1,
  "macro_recall": 0.1,
  "macro_f1": 0.1,
  "weighted_f1": 0.1
}
""".strip(),
        encoding="utf-8",
    )

    try:
        export_frozen_vs_finetuned_report_assets(
            tmp_path,
            tmp_path / "tables",
            tmp_path / "figures",
        )
    except FileNotFoundError as exc:
        assert "requires frozen run metrics" in str(exc)
    else:
        raise AssertionError("Expected missing frozen run folders to be reported clearly.")
