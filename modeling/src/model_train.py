"""
Model construction and training dispatch for modeling.

Routes to the correct model module based on cfg.model.name.
Handles pretrained checkpoint loading for constrained losses.
BayesNF is dispatched separately (JAX-based, uses DataFrames).
"""

from __future__ import annotations

import importlib
import logging

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .data_prep import DataBundle

log = logging.getLogger(__name__)

_PYTORCH_MODELS = {
    "deterministic_ann": "src.models.deterministic_ann",
    "mc_dropout_ann":    "src.models.mc_dropout_ann",
    "deep_ensemble_ann": "src.models.deep_ensemble_ann",
    "bayesian_ann":      "src.models.bayesian_ann",
}


def _load_pretrained(result, name: str, path: str, device: torch.device) -> object:
    """Load a pretrained PyTorch checkpoint into the model.

    Raises a clear error if the architecture does not match.
    """
    log.info("Loading pretrained checkpoint: %s", path)
    state = torch.load(path, map_location=device)

    if name == "bayesian_ann":
        import pyro
        model, guide = result
        try:
            pyro.clear_param_store()
            for k, v in state.items():
                pyro.param(k, v.to(device))
        except Exception as e:
            raise RuntimeError(
                f"Architecture mismatch loading Bayesian ANN checkpoint from '{path}'. "
                f"Ensure the model config matches the pretrained model exactly.\n{e}"
            ) from e
        return model, guide

    if name == "deep_ensemble_ann":
        models = result
        try:
            for i, (m, s) in enumerate(zip(models, state)):
                m.load_state_dict(s)
        except Exception as e:
            raise RuntimeError(
                f"Architecture mismatch loading ensemble checkpoint from '{path}'. "
                f"Ensure hidden_dims and n_members match.\n{e}"
            ) from e
        return models

    # deterministic_ann / mc_dropout_ann
    try:
        result.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"Architecture mismatch loading checkpoint from '{path}'. "
            f"Ensure hidden_dims match the pretrained model.\n{e}"
        ) from e
    return result


def build_and_train(cfg: DictConfig, bundle: DataBundle, device: torch.device):
    """Build, optionally load pretrained weights, and train the model.

    Returns
    -------
    For PyTorch models: model | (model, guide) | list[model]
    For BayesNF: fitted BayesNF model (stored on bundle.df_train)
    """
    log.info("Hydra output dir: %s", HydraConfig.get().runtime.output_dir)

    name          = cfg.model.name
    pretrained    = cfg.loss.get("pretrained_model_path", None)

    # ── BayesNF (JAX) ─────────────────────────────────────────────────────
    if name == "bayesnf":
        from src.models import bayesnf as bnf_mod

        if not hasattr(bundle, "df_train"):
            raise ValueError(
                "BayesNF requires validation=blocked (blocked split DataFrames). "
                "Set validation=blocked in your config."
            )

        if pretrained:
            model = bnf_mod.load_checkpoint(pretrained)
            log.info("BayesNF: loaded pretrained checkpoint, skipping training.")
        else:
            model = bnf_mod.build_model(cfg.model, bundle.feature_cols, bundle.target_col)
            model = bnf_mod.train(model, bundle.df_train, cfg.model)

        return model

    # ── PyTorch models ────────────────────────────────────────────────────
    if name not in _PYTORCH_MODELS:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(_PYTORCH_MODELS.keys()) + ['bayesnf']}"
        )
    mod = importlib.import_module(_PYTORCH_MODELS[name])

    if name == "deep_ensemble_ann":
        models = mod.build_models(cfg.model, bundle.in_dim)
        if pretrained:
            models = _load_pretrained(models, name, pretrained, device)
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
        if pretrained:
            model, guide = _load_pretrained((model, guide), name, pretrained, device)
        # Use model-level training config if present, else fall back to shared
        train_cfg = cfg.model.get("training", cfg.training)
        model, guide, best_val = mod.train(
            model, guide,
            bundle.train_loader,
            bundle.X_val_t,
            bundle.y_val_t,
            train_cfg,
        )
        log.info("BNN best val MSE (std-space): %.4f", best_val)
        return model, guide

    # deterministic_ann / mc_dropout_ann
    model = mod.build_model(cfg.model, bundle.in_dim)
    if pretrained:
        model = _load_pretrained(model, name, pretrained, device)
    model, best_val = mod.train(
        model,
        bundle.train_loader,
        bundle.X_val_t,
        bundle.y_val_t,
        cfg.training,
    )
    log.info("Best val MSE (std-space): %.4f", best_val)
    return model
