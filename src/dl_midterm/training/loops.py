"""Training and validation loops, early stopping, and checkpoint helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_midterm.evaluation.metrics import compute_classification_metrics
from dl_midterm.training.early_stopping import EarlyStopping


def train_mlp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping_patience: int,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    """Train an MLP and keep the best validation macro-F1 state."""

    model.to(device)
    stopper = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_state = deepcopy(model.state_dict())
    best_metrics: dict[str, Any] = {}
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred = _run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_true, val_pred = predict_with_loss(model, val_loader, criterion, device)
        train_metrics = compute_classification_metrics(train_true, train_pred, class_names)
        val_metrics = compute_classification_metrics(val_true, val_pred, class_names)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
        )
        if stopper.best_score is None or val_metrics["macro_f1"] > stopper.best_score:
            best_state = deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)
        if stopper.step(val_metrics["macro_f1"], epoch):
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    best_metrics["best_epoch"] = stopper.best_epoch
    return model, history, best_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a classifier and return classification metrics."""

    true, pred = predict(model, loader, device)
    return compute_classification_metrics(true, pred, class_names)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Return integer labels and predictions for a dataloader."""

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for features, labels in loader:
        logits = model(features.to(device))
        predictions = logits.argmax(dim=1).cpu()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.tolist())
    return y_true, y_pred


@torch.no_grad()
def predict_with_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    """Return average loss plus labels/predictions for a dataloader."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / max(total_samples, 1), y_true, y_pred


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / max(total_samples, 1), y_true, y_pred
