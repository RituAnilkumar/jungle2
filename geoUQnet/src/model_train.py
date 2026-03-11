"""
Model construction and training dispatch for geoUQnet.

Selects the correct src.models.* module based on ``cfg.model.name``
and delegates build + train to that module, keeping main.py clean.
"""

from __future__ import annotations

import importlib
import logging

import torch
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from .data_prep import DataBundle

log = logging.getLogger(__name__)

# Registry maps config name → module path inside src.models
_MODEL_REGISTRY: dict[str, str] = {
    "deterministic_ann": "src.models.deterministic_ann",
    "mc_dropout_ann":    "src.models.mc_dropout_ann",
    "deep_ensemble_ann": "src.models.deep_ensemble_ann",
    "bayesian_ann":      "src.models.bayesian_ann",
}


def _get_module(name: str):
    """Import and return the model module for *name*."""
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return importlib.import_module(_MODEL_REGISTRY[name])


def build_and_train(
    cfg: DictConfig,
    bundle: DataBundle,
    device: torch.device,
):
    """Build, train and return a model (or list of models for ensembles).

    Expected root config shape:

    .. code-block:: yaml

        model:
          name: mc_dropout_ann   # key into _MODEL_REGISTRY
          hidden_dims: [64, 64]
          dropout_p: 0.1

        training:
          lr: 1.0e-3
          weight_decay: 0.0
          epochs: 500
          batch_size: 256        # used in data_prep, not here

    The Hydra run directory is logged so every experiment is traceable.

    Returns
    -------
    result : model | (model, guide) | list[model]
        • ``deterministic_ann``  → ``DeterministicANN``
        • ``mc_dropout_ann``     → ``MCDropoutANN``  (same class, dropout_p>0)
        • ``deep_ensemble_ann``  → ``list[DeterministicANN]``
        • ``bayesian_ann``       → ``(BayesianANN, AutoDiagonalNormal)``
    """
    hydra_cfg = HydraConfig.get()
    log.info("Hydra output dir: %s", hydra_cfg.runtime.output_dir)

    name: str = cfg.model.name
    log.info("Building model: '%s'", name)
    mod = _get_module(name)

    if name == "deep_ensemble_ann":
        models = mod.build_models(cfg.model, bundle.in_dim)
        trained, val_mses = mod.train(
            models,
            bundle.train_loader,
            bundle.X_val_t,
            bundle.y_val_t,
            cfg_model=cfg.model,
            cfg_training=cfg.training,
        )
        log.info("Ensemble val MSEs: %s", [f"{v:.4f}" for v in val_mses])
        return trained

    if name == "bayesian_ann":
        model, guide = mod.build_model(cfg.model, bundle.in_dim, device)
        model, guide, best_val = mod.train(
            model,
            guide,
            bundle.train_loader,
            bundle.X_val_t,
            bundle.y_val_t,
            cfg.training,
        )
        log.info("BNN (VI) best val MSE (std-space): %.4f", best_val)
        return model, guide

    # deterministic_ann and mc_dropout_ann share the same train() API
    model = mod.build_model(cfg.model, bundle.in_dim)
    model, best_val = mod.train(
        model,
        bundle.train_loader,
        bundle.X_val_t,
        bundle.y_val_t,
        cfg.training,
    )
    log.info("Best val MSE (std-space): %.4f", best_val)
    return model