"""
Deep Ensemble model for geoUQnet.

Trains M independent DeterministicANN members with different random seeds
and aggregates their predictions to obtain a mean and epistemic uncertainty
estimate (member spread).  Supports both absolute-residual and normalised
conformal prediction.
"""

from __future__ import annotations

import logging
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .deterministic_ann import DeterministicANN, train as _train_ann

log = logging.getLogger(__name__)


# ─────────────────────────── Factory ─────────────────────────────────────────

def build_models(cfg: DictConfig, in_dim: int) -> list[DeterministicANN]:
    """Build ``cfg.n_members`` un-trained ensemble members.

    Expected YAML (``configs/model/deep_ensemble_ann.yaml``):

    .. code-block:: yaml

        _target_: src.models.deep_ensemble_ann
        hidden_dims: [64, 64]
        dropout_p: 0.0      # optionally add within-member dropout
        n_members: 5
        base_seed: 999
    """
    n = int(cfg.get("n_members", 5))
    return [
        DeterministicANN(
            in_dim=in_dim,
            hidden_dims=list(cfg.hidden_dims),
            dropout_p=float(cfg.get("dropout_p", 0.0)),
        )
        for _ in range(n)
    ]


# ─────────────────────────── Training ────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    models: list[DeterministicANN],
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg_model: DictConfig,
    cfg_training: DictConfig,
) -> tuple[list[DeterministicANN], list[float]]:
    """Train each ensemble member independently with a unique seed.

    Parameters
    ----------
    models : list[DeterministicANN]
        Un-trained members returned by :func:`build_models`.
    train_loader, X_val, y_val :
        Standard data inputs.
    cfg_model :
        The ``model`` config node (for ``base_seed``).
    cfg_training :
        The ``training`` config node (for ``lr``, ``epochs``, etc.).

    Returns
    -------
    trained_models : list[DeterministicANN]
    val_mses : list[float]
        Best validation MSE per member (standardised y-space).
    """
    base_seed = int(cfg_model.get("base_seed", 999))
    trained, val_mses = [], []

    for i, member in enumerate(models):
        seed = base_seed + i
        _set_seeds(seed)
        log.info("Training ensemble member %d / %d (seed=%d) …", i + 1, len(models), seed)
        member, best_val = _train_ann(member, train_loader, X_val, y_val, cfg_training)
        trained.append(member)
        val_mses.append(best_val)
        log.info("  → best val MSE (std-space): %.4f", best_val)

    return trained, val_mses


# ─────────────────────────── Inference ───────────────────────────────────────

@torch.no_grad()
def predict_mean_std(
    models: list[nn.Module],
    X: torch.Tensor,
    y_scaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate member predictions into ensemble mean and spread.

    Parameters
    ----------
    models : list[nn.Module]
        Trained ensemble members.
    X : torch.Tensor
        Input tensor on the correct device.
    y_scaler :
        Fitted ``sklearn.preprocessing.StandardScaler`` for the target.

    Returns
    -------
    mean : np.ndarray [N]   – ensemble mean in original units.
    std  : np.ndarray [N]   – member spread (epistemic proxy) in original units.
    """
    for m in models:
        m.eval()

    # [M, N] in standardised y-space
    preds = torch.stack([m(X) for m in models], dim=0)

    mean_std = preds.mean(0).cpu().numpy()
    std_std = preds.std(0).cpu().numpy()

    mean = y_scaler.inverse_transform(mean_std.reshape(-1, 1)).ravel()
    std = std_std * float(y_scaler.scale_[0])
    return mean, std


@torch.no_grad()
def predict_mean(
    models: list[nn.Module],
    X: torch.Tensor,
    y_scaler,
) -> np.ndarray:
    """Convenience wrapper — mean only."""
    return predict_mean_std(models, X, y_scaler)[0]