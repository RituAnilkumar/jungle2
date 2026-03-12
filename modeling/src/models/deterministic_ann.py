"""
Deterministic ANN — base architecture and trainer.

Used standalone (absolute-residual conformal only) or as the backbone
class for MC Dropout when dropout_p > 0.
"""

from __future__ import annotations

import copy
import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ─────────────────────────── Architecture ────────────────────────────────────

class DeterministicANN(nn.Module):
    """Feed-forward ANN with optional per-layer dropout.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    hidden_dims : Sequence[int]
        Width of each hidden layer, e.g. ``[64, 64]``.
    dropout_p : float
        Dropout probability after every hidden ReLU (0.0 = no dropout).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            if dropout_p > 0.0:
                layers.append(nn.Dropout(dropout_p))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # shape [N]


# ─────────────────────────── Factory ─────────────────────────────────────────

def build_model(cfg: DictConfig, in_dim: int) -> DeterministicANN:
    """Build from the ``model`` Hydra config node.

    Expected YAML (``configs/model/deterministic_ann.yaml``):

    .. code-block:: yaml

        _target_: src.models.deterministic_ann.DeterministicANN
        hidden_dims: [64, 64]
        dropout_p: 0.0
    """
    return DeterministicANN(
        in_dim=in_dim,
        hidden_dims=list(cfg.hidden_dims),
        dropout_p=float(cfg.get("dropout_p", 0.0)),
    )


# ─────────────────────────── Training ────────────────────────────────────────

def train(
    model: DeterministicANN,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: DictConfig,
) -> tuple[DeterministicANN, float]:
    """MSE training with best-val-loss checkpointing.

    Expected keys in ``cfg`` (the ``training`` node):

    .. code-block:: yaml

        lr: 1.0e-3
        weight_decay: 0.0
        epochs: 500

    Returns
    -------
    model : DeterministicANN
        Loaded with the best-validation checkpoint.
    best_val_mse : float
        Best validation MSE in *standardised* y-space.
    """
    device = X_val.device
    model = model.to(device)

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None

    for epoch in range(1, int(cfg.get("epochs", 500)) + 1):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb.squeeze(-1)).backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(X_val), y_val.squeeze(-1)).item()

        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 50 == 0 or epoch == 1:
            log.info("Epoch %03d | val MSE (std-space): %.4f", epoch, val_mse)

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


# ─────────────────────────── Inference ───────────────────────────────────────

@torch.no_grad()
def predict_mean(
    model: DeterministicANN,
    X: torch.Tensor,
    y_scaler,
) -> np.ndarray:
    """Point predictions in original (un-scaled) units."""
    model.eval()
    preds_std = model(X).cpu().numpy()
    return y_scaler.inverse_transform(preds_std.reshape(-1, 1)).ravel()