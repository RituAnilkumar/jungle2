"""
MC Dropout model for geoUQnet.

Wraps any torch nn.Module that contains Dropout layers and exposes
stochastic forward passes at inference time — "Bayes by Backprop-lite".

Supports both absolute-residual and normalised conformal prediction
because it produces a per-sample predictive *std* as well as a mean.

The default backbone is DeterministicANN with dropout_p > 0, but you
can supply any custom torch class via the Hydra config (custom_torch).
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .deterministic_ann import DeterministicANN, train as _train_ann

log = logging.getLogger(__name__)


# ─────────────────────────── Architecture ────────────────────────────────────

# MCDropoutANN *is* a DeterministicANN with dropout_p > 0.
# We alias it here so the registry and configs are self-documenting.
MCDropoutANN = DeterministicANN


# ─────────────────────────── Factory ─────────────────────────────────────────

def build_model(cfg: DictConfig, in_dim: int) -> nn.Module:
    """Build an MC-Dropout-capable model from the Hydra ``model`` node.

    You can plug in your own torch class by setting ``custom_class`` in the
    config.  The class must accept ``in_dim``, ``hidden_dims``, and
    ``dropout_p`` as constructor arguments and contain ``nn.Dropout`` layers.

    Expected YAML (``configs/model/mc_dropout_ann.yaml``):

    .. code-block:: yaml

        _target_: src.models.mc_dropout_ann.MCDropoutANN
        hidden_dims: [64, 64]
        dropout_p: 0.1
        # optional: point at your own class
        # custom_class: mypackage.mymodule.MyNet
    """
    custom_cls_path: str | None = cfg.get("custom_class", None)

    if custom_cls_path:
        import importlib
        module_path, cls_name = custom_cls_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        log.info("MC Dropout: using custom backbone '%s'", custom_cls_path)
    else:
        cls = DeterministicANN

    dropout_p = float(cfg.get("dropout_p", 0.1))
    if dropout_p <= 0.0:
        log.warning(
            "MC Dropout model built with dropout_p=%.2f — stochastic "
            "inference will be identical to a deterministic forward pass.",
            dropout_p,
        )

    return cls(
        in_dim=in_dim,
        hidden_dims=list(cfg.hidden_dims),
        dropout_p=dropout_p,
    )


# ─────────────────────────── Training ────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: DictConfig,
) -> tuple[nn.Module, float]:
    """Delegate to the standard MSE trainer (dropout is transparent here).

    Expected keys in ``cfg`` (the ``training`` node):

    .. code-block:: yaml

        lr: 1.0e-3
        weight_decay: 0.0
        epochs: 500
    """
    return _train_ann(model, train_loader, X_val, y_val, cfg)


# ─────────────────────────── Inference helpers ───────────────────────────────

def _enable_dropout(model: nn.Module) -> None:
    """Force all Dropout layers into train-mode for stochastic inference."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def predict_mean_std(
    model: nn.Module,
    X: torch.Tensor,
    y_scaler,
    T: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Run *T* stochastic forward passes and return mean & std in original units.

    Parameters
    ----------
    model : nn.Module
        Trained model with Dropout layers.
    X : torch.Tensor
        Input tensor on the correct device.
    y_scaler :
        Fitted ``sklearn.preprocessing.StandardScaler`` for the target.
    T : int
        Number of MC samples (default 200).

    Returns
    -------
    mean : np.ndarray, shape [N]
    std  : np.ndarray, shape [N]
        Both in *original* (un-scaled) units.
    """
    model.eval()
    _enable_dropout(model)

    preds = torch.stack([model(X) for _ in range(T)], dim=0)  # [T, N]

    mean_std = preds.mean(0).cpu().numpy()
    std_std = preds.std(0).cpu().numpy()

    mean = y_scaler.inverse_transform(mean_std.reshape(-1, 1)).ravel()
    std = std_std * float(y_scaler.scale_[0])
    return mean, std


@torch.no_grad()
def predict_mean(
    model: nn.Module,
    X: torch.Tensor,
    y_scaler,
    T: int = 200,
) -> np.ndarray:
    """Convenience wrapper — mean only."""
    return predict_mean_std(model, X, y_scaler, T=T)[0]