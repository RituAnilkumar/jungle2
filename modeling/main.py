"""
modeling — entry point.

Trains and evaluates neural network models with uncertainty quantification
on glacier mass balance data produced by gla_prep.

Run with Hydra:

    python main.py                                                  # defaults
    python main.py model=bayesnf validation=blocked conformal=none
    python main.py model=mc_dropout_ann conformal=normalized_residuals
    python main.py loss=glambie data=my_region                      # constrained loss
    python main.py -m model=deterministic_ann,mc_dropout_ann conformal=none,absolute_residuals

NOTE: For loss=glambie / temporal_avg / glambie_and_temporal_avg,
set pretrained_model_path in the loss config to an OGGM-pretrained checkpoint.
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
from src.conformal.conformal import run_conformal
from src.model_eval import (
    compute_metrics,
    build_results_df,
    build_bayesnf_results_df,
    compute_bayesnf_metrics,
    save_outputs,
)

log = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_predict_fn(cfg: DictConfig, result, bundle, device):
    """Return a unified predict_fn(X_tensor) → mean or (mean, std)."""
    import importlib
    name = cfg.model.name
    mod  = importlib.import_module(f"src.models.{name}")
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
    log.info("Output dir: %s", HydraConfig.get().runtime.output_dir)
    log.info(
        "Model: %s | Loss: %s | Validation: %s | Conformal: %s",
        cfg.model.name, cfg.loss.type, cfg.validation.type, cfg.conformal.type,
    )

    seed = int(cfg.get("seed", 42))
    _set_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────
    bundle = load_and_split(cfg, device)

    # ── Model ─────────────────────────────────────────────────────────────
    result = build_and_train(cfg, bundle, device)

    # ── BayesNF blocked evaluation path ──────────────────────────────────
    if cfg.model.name == "bayesnf":
        from src.models import bayesnf as bnf_mod
        import os
        out_dir = HydraConfig.get().runtime.output_dir

        bnf_mod.save(result, os.path.join(out_dir, "model.pkl"))

        all_metrics, all_dfs, split_names = [], [], []
        for split_name, df_split in [
            ("train",  bundle.df_train),
            ("logo",   bundle.df_logo),
            ("loyo",   bundle.df_loyo),
            ("loygo",  bundle.df_loygo),
        ]:
            if len(df_split) == 0:
                continue
            median, lower, upper = bnf_mod.predict(result, df_split, cfg.model)
            df_res = build_bayesnf_results_df(df_split, median, lower, upper, bundle.target_col)
            metrics = compute_bayesnf_metrics(df_res, bundle.target_col)
            all_metrics.append(metrics)
            all_dfs.append(df_res)
            split_names.append(split_name)
            log.info("BayesNF %s — RMSE: %.4f | R2: %.4f | Coverage: %.1f%%",
                     split_name, metrics["rmse"], metrics["r2"], metrics["coverage_pct"])

        save_outputs(cfg, all_metrics, all_dfs, split_names)
        return

    # ── PyTorch models: conformal + eval ─────────────────────────────────
    predict_fn        = _make_predict_fn(cfg, result, bundle, device)
    conformal_results = run_conformal(cfg, predict_fn, bundle)
    y_true            = bundle.y_test.ravel()
    metrics           = compute_metrics(y_true, conformal_results)

    pred_std = None
    if cfg.conformal.type == "normalized_residuals":
        _, pred_std = predict_fn(bundle.X_test_t)
    elif cfg.conformal.type == "none":
        out = conformal_results.get("std")
        pred_std = out if out is not None else None

    results_df = build_results_df(y_true, conformal_results, pred_std=pred_std)

    # For blocked validation, also evaluate on logo/loyo/loygo test sets
    if cfg.validation.type == "blocked":
        all_metrics, all_dfs, split_names = [], [], []
        for split_name, df_split in [
            ("logo",  bundle.df_logo),
            ("loyo",  bundle.df_loyo),
            ("loygo", bundle.df_loygo),
        ]:
            if len(df_split) == 0:
                continue
            X_s = bundle.x_scaler.transform(
                df_split[bundle.feature_cols].to_numpy(dtype=np.float32)
            )
            X_t = torch.from_numpy(X_s).to(device)
            y_s = df_split[bundle.target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
            # Temporarily swap test tensors for conformal evaluation
            _orig_x, _orig_y = bundle.X_test_t, bundle.y_test
            bundle.X_test_t = X_t
            bundle.y_test   = y_s
            cr = run_conformal(cfg, predict_fn, bundle)
            bundle.X_test_t, bundle.y_test = _orig_x, _orig_y

            m   = compute_metrics(y_s.ravel(), cr)
            df  = build_results_df(y_s.ravel(), cr)
            all_metrics.append(m)
            all_dfs.append(df)
            split_names.append(split_name)
            log.info("%s — RMSE: %.4f | R2: %.4f", split_name, m["rmse"], m["r2"])

        save_outputs(cfg, all_metrics, all_dfs, split_names)
    else:
        save_outputs(cfg, metrics, results_df)


if __name__ == "__main__":
    main()
