"""Standard MSE loss — no auxiliary targets."""

from __future__ import annotations

import torch
import torch.nn as nn

mse = nn.MSELoss()


def compute(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    """MSE between predictions and per-glacier annual mass balance targets."""
    return mse(pred, target)
