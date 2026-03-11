"""
main.py — entry-point for geoUQnet.

Run with Hydra:

    python main.py                                    # uses defaults
    python main.py model=mc_dropout_ann               # swap model
    python main.py conformal=normalized_residuals     # swap conformal type
    python main.py data.path=my_data.csv              # override any key
    python main.py -m model=mc_dropout_ann,bayesian_ann conformal=absolute_residuals,normalized_residuals
                                                      # multirun sweep
"""

from __future__ import annotations

import logging
import random

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.data_prep import load_and_split
from src.model_train import build_and_train
from src.model_conformal import run_conformal
from src.model_eval import compute_metrics, build_results_df, save_outputs

log = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_predict_fn(cfg: DictConfig, result, bundle, device):
    """Return a unified predict_fn(X_tensor) → mean or (mean, std).

    Wraps the model-specific inference calls so model_conformal.py
    stays model-agnostic.
    """
    import importlib

    name: str = cfg.model.name
    mod = importlib.import_module(
        f"src.models.{name}"
    )
    y_scaler = bundle.y_scaler

    if name == "deterministic_ann":
        def predict_fn(X):
            return mod.predict_mean(result, X, y_scaler)

    elif name == "mc_dropout_ann":
        T = int(cfg.model.get("mc_samples", 200))
        if cfg.conformal.type == "normalized_residuals":
            def predict_fn(X):
                return mod.predict_mean_std(result, X, y_scaler, T=T)
        else:
            def predict_fn(X):
                return mod.predict_mean(result, X, y_scaler, T=T)

    elif name == "deep_ensemble_ann":
        if cfg.conformal.type == "normalized_residuals":
            def predict_fn(X):
                return mod.predict_mean_std(result, X, y_scaler)
        else:
            def predict_fn(X):
                return mod.predict_mean(result, X, y_scaler)

    elif name == "bayesian_ann":
        model, guide = result
        n = int(cfg.model.get("vi_samples", 500))
        if cfg.conformal.type == "normalized_residuals":
            def predict_fn(X):
                return mod.predict_mean_std(model, guide, X, y_scaler, n_samples=n)
        else:
            def predict_fn(X):
                return mod.predict_mean(model, guide, X, y_scaler, n_samples=n)

    else:
        raise ValueError(f"No predict_fn defined for model '{name}'")

    return predict_fn


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    log.info("Output dir: %s", hydra_cfg.runtime.output_dir)
    log.info("Model: %s | Conformal: %s", cfg.model.name, cfg.conformal.type)

    seed = int(cfg.get("seed", 42))
    _set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────────────
    bundle = load_and_split(cfg, device)

    # ── Model ─────────────────────────────────────────────────────────────────
    result = build_and_train(cfg, bundle, device)

    # ── Predict + Conformal ───────────────────────────────────────────────────
    predict_fn = _make_predict_fn(cfg, result, bundle, device)
    conformal_results = run_conformal(cfg, predict_fn, bundle)

    # ── Evaluate + Save ───────────────────────────────────────────────────────
    y_true   = bundle.y_test.ravel()
    metrics  = compute_metrics(y_true, conformal_results)

    pred_std = None
    if cfg.conformal.type == "normalized_residuals":
        _, pred_std = predict_fn(bundle.X_test_t)

    results_df = build_results_df(y_true, conformal_results, pred_std=pred_std)
    save_outputs(cfg, metrics, results_df)


if __name__ == "__main__":
    main()