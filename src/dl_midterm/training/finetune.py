"""Backbone fine-tuning routines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_midterm.evaluation.metrics import compute_classification_metrics
from dl_midterm.evaluation.reports import write_run_report
from dl_midterm.features.cache import FeatureCache
from dl_midterm.features.extract import extract_and_cache_backbone
from dl_midterm.models.backbones import (
    apply_finetuning_policy,
    backbone_alias,
    build_classification_backbone,
    build_finetuned_feature_extractor,
)
from dl_midterm.training.early_stopping import EarlyStopping
from dl_midterm.training.losses import (
    build_classification_loss,
)
from dl_midterm.training.losses import (
    class_weights_from_labels as build_class_weights_from_labels,
)


def class_weights_from_labels(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from train labels only."""

    return build_class_weights_from_labels(labels, num_classes)


def labels_from_loader(loader: DataLoader) -> list[int]:
    """Read integer labels from a HAM10000 image dataloader without loading images."""

    dataset = loader.dataset
    frame = getattr(dataset, "frame", None)
    label_to_index = getattr(dataset, "label_to_index", None)
    if frame is None or label_to_index is None:
        raise ValueError("Expected a HAM10000ImageDataset-backed dataloader.")
    return [int(label_to_index[str(label)]) for label in frame["label"].astype(str).tolist()]


def finetune_backbone(
    *,
    backbone: str,
    loaders: dict[str, DataLoader],
    class_names: list[str],
    device: torch.device,
    checkpoint_dir: str | Path,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    policy: str | None,
    mixed_precision: bool,
    feature_source: str = "finetuned",
    pretrained: bool = True,
    class_weighting: bool = True,
    loss_config: dict[str, Any] | None = None,
    backbone_learning_rate: float | None = None,
    head_learning_rate: float | None = None,
    output_run_root: str | Path = "artifacts/runs",
    experiment_name: str = "sprint4_finetune_backbones",
    limit_per_split: int | None = None,
    augmentation_config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Fine-tune one backbone and write a best-validation checkpoint plus run report."""

    model = build_classification_backbone(
        backbone,
        num_classes=len(class_names),
        pretrained=pretrained,
    )
    policy_summary = apply_finetuning_policy(model, backbone, policy=policy)
    model.to(device)

    train_labels = labels_from_loader(loaders["train"])
    criterion, loss_metadata = build_classification_loss(
        labels=train_labels,
        num_classes=len(class_names),
        class_weighting=class_weighting,
        loss_config=loss_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        _finetuning_parameter_groups(
            model,
            backbone=backbone,
            learning_rate=learning_rate,
            backbone_learning_rate=backbone_learning_rate,
            head_learning_rate=head_learning_rate,
        ),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    started = perf_counter()
    model, history, best_metrics = train_image_classifier(
        model,
        loaders["train"],
        loaders["val"],
        class_names=class_names,
        device=device,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        mixed_precision=mixed_precision,
    )
    runtime_seconds = perf_counter() - started
    test_metrics = evaluate_image_classifier(model, loaders["test"], class_names, device)

    feature_tag = "" if feature_source == "finetuned" else f"_{feature_source}"
    run_id = f"{backbone_alias(backbone)}{feature_tag}_finetune_s{seed}"
    run_dir = Path(output_run_root) / "finetune_backbones" / run_id
    checkpoint_path = Path(checkpoint_dir) / f"{backbone}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": backbone,
            "class_names": class_names,
            "seed": seed,
            "policy": policy_summary,
            "best_val_macro_f1": best_metrics.get("macro_f1"),
            "best_val_epoch": best_metrics.get("best_epoch"),
            "runtime_seconds": runtime_seconds,
            "feature_source": feature_source,
            "loss": loss_metadata,
            "augmentation": augmentation_config,
        },
        checkpoint_path,
    )

    resolved_config = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "seed": seed,
        "feature_source": feature_source,
        "backbone": backbone,
        "backbones": [backbone],
        "backbone_combination": backbone,
        "backbone_count": 1,
        "fusion_method": "finetune_head",
        "class_weighting": class_weighting,
        "loss": loss_metadata,
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
        "head_learning_rate": head_learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "mixed_precision": mixed_precision,
        "pretrained": pretrained,
        "limit_per_split": limit_per_split,
        "augmentation": augmentation_config,
        "checkpoint_path": str(checkpoint_path),
        "unfreeze_policy": policy_summary,
        "best_val_macro_f1": best_metrics.get("macro_f1"),
        "best_val_epoch": best_metrics.get("best_epoch"),
        "runtime_seconds": runtime_seconds,
    }
    test_metrics.update(
        {
            "run_id": run_id,
            "feature_source": feature_source,
            "backbone": backbone,
            "backbones": [backbone],
            "backbone_combination": backbone,
            "backbone_count": 1,
            "fusion_method": "finetune_head",
            "class_weighting": class_weighting,
            "loss": loss_metadata,
            "optimizer": "adamw",
            "learning_rate": learning_rate,
            "backbone_learning_rate": backbone_learning_rate,
            "head_learning_rate": head_learning_rate,
            "weight_decay": weight_decay,
            "best_val_macro_f1": best_metrics.get("macro_f1"),
            "best_val_epoch": best_metrics.get("best_epoch"),
            "runtime_seconds": runtime_seconds,
            "augmentation": augmentation_config,
        }
    )
    write_run_report(
        run_dir,
        resolved_config=resolved_config,
        history=history,
        metrics=test_metrics,
        class_names=class_names,
        backbone=backbone,
    )
    return checkpoint_path, run_dir


def extract_finetuned_feature_cache(
    *,
    backbone: str,
    checkpoint_path: str | Path,
    loaders: dict[str, DataLoader],
    output_dir: str | Path,
    class_names: list[str],
    seed: int,
    device: torch.device,
    mixed_precision: bool,
    feature_source: str = "finetuned",
    config: dict[str, Any] | None = None,
) -> list[FeatureCache]:
    """Extract classifier-free features from the best fine-tuned checkpoint."""

    model = build_finetuned_feature_extractor(
        backbone,
        checkpoint_path=checkpoint_path,
        num_classes=len(class_names),
    )
    return extract_and_cache_backbone(
        model=model,
        backbone=backbone,
        loaders=loaders,
        output_dir=output_dir,
        class_names=class_names,
        feature_source=feature_source,
        seed=seed,
        device=device,
        mixed_precision=mixed_precision,
        config=config,
    )


def train_image_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    early_stopping_patience: int,
    mixed_precision: bool,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    """Train an image classifier and keep the best validation macro-F1 state."""

    stopper = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_state = deepcopy(model.state_dict())
    best_metrics: dict[str, Any] = {}
    history_rows: list[dict[str, float | int]] = []
    device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"
    autocast_enabled = mixed_precision and device.type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred = _run_image_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            device_type=device_type,
            autocast_enabled=autocast_enabled,
            scaler=scaler,
        )
        val_loss, val_true, val_pred = predict_images_with_loss(
            model,
            val_loader,
            criterion,
            device,
        )
        train_metrics = compute_classification_metrics(train_true, train_pred, class_names)
        val_metrics = compute_classification_metrics(val_true, val_pred, class_names)
        current_lr = float(optimizer.param_groups[0]["lr"])
        current_head_lr = (
            float(optimizer.param_groups[1]["lr"]) if len(optimizer.param_groups) > 1 else None
        )
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "learning_rate": current_lr,
                "head_learning_rate": current_head_lr,
            }
        )
        if stopper.best_score is None or val_metrics["macro_f1"] > stopper.best_score:
            best_state = deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)
        if scheduler is not None:
            scheduler.step()
        if stopper.step(val_metrics["macro_f1"], epoch):
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    best_metrics["best_epoch"] = stopper.best_epoch
    return model, history, best_metrics


def _finetuning_parameter_groups(
    model: nn.Module,
    *,
    backbone: str,
    learning_rate: float,
    backbone_learning_rate: float | None,
    head_learning_rate: float | None,
) -> list[dict[str, Any]] | list[nn.Parameter]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if backbone_learning_rate is None and head_learning_rate is None:
        return trainable

    backbone_lr = float(
        backbone_learning_rate if backbone_learning_rate is not None else learning_rate
    )
    classifier_lr = float(head_learning_rate if head_learning_rate is not None else learning_rate)
    head_prefixes = _head_parameter_prefixes(backbone)
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(name.startswith(prefix) for prefix in head_prefixes):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        groups.append({"params": head_params, "lr": classifier_lr})
    return groups


def _head_parameter_prefixes(backbone: str) -> tuple[str, ...]:
    normalized = backbone.lower()
    if normalized == "resnet50":
        return ("fc.",)
    if normalized in {"mobilenet_v2", "efficientnet_b0"}:
        return ("classifier.",)
    raise ValueError(f"Unsupported backbone: {backbone}")


@torch.no_grad()
def evaluate_image_classifier(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate an image classifier."""

    true, pred = predict_images(model, loader, device)
    return compute_classification_metrics(true, pred, class_names)


@torch.no_grad()
def predict_images(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return integer labels and predictions from image batches."""

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


@torch.no_grad()
def predict_images_with_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Return average loss plus labels/predictions for image batches."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / max(total_samples, 1), y_true, y_pred


def _run_image_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    device_type: str,
    autocast_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> tuple[float, list[int], list[int]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / max(total_samples, 1), y_true, y_pred
