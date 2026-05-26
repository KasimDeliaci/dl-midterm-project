"""Torch device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an experiment device string to a PyTorch device."""

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(normalized)


def supports_mixed_precision(device: torch.device) -> bool:
    """Return whether autocast is useful for the selected device."""

    return device.type in {"cuda", "mps"}
