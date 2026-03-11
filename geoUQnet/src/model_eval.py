"""
Evaluation and results persistence for geoUQnet.

Computes metrics, builds a results DataFrame and writes it to the
Hydra output directory so every run is automatically logged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import r2_score

log = logging.getLogger(__name__)


# ─────────────────────────── Metrics ─────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    conformal_results: dict,
) -> dict:
    """Derive evaluation metrics from conformal results dict.

    Parameters
    ----------
    y_true : np.ndarray [N]
        Ground-truth test targets in original units.
    conformal_results : dict
        Output of :func:`~src.model_conformal.run_conformal`.

    Returns
    -------
    dict with keys:
        ``rmse``, ``mae``, ``coverage``, ``avg_width``,
        ``interval_score``, ``qhat``
    """
    mean     = conformal_results["mean"]
    lo       = conformal_results["lo"]
    hi       = conformal_results["hi"]
    alpha    = 1.0 - conformal_results["coverage"]   # approximate

    rmse     = float(np.sqrt(np.mean((mean - y_true) ** 2)))
    mae      = float(np.mean(np.abs(mean - y_true)))
    r2       = float(r2_score(y_true, mean))
    coverage = conformal_results["coverage"]
    width    = conformal_results["avg_width"]

    # Winkler interval score (penalises width + miscoverage)
    penalty = (
        (lo - y_true) * (y_true < lo)
        + (y_true - hi) * (y_true > hi)
    )
    interval_score = float(np.mean(width + (2.0 / max(alpha, 1e-8)) * penalty))

    metrics = {
        "rmse":           rmse,
        "mae":            mae,
        "r2":             r2,
        "coverage":       coverage,
        "avg_width":      width,
        "interval_score": interval_score,
        "qhat":           conformal_results["qhat"],
    }

    log.info("── Evaluation metrics ──────────────────────")
    for k, v in metrics.items():
        log.info("  %-20s %.4f", k, v)

    return metrics


# ─────────────────────────── Results DataFrame ───────────────────────────────

def build_results_df(
    y_true: np.ndarray,
    conformal_results: dict,
    pred_std: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assemble a per-sample results DataFrame.

    Parameters
    ----------
    y_true : np.ndarray [N]
    conformal_results : dict
    pred_std : np.ndarray [N] | None
        Predictive std (mc_dropout, ensemble, bnn); None for deterministic.
    """
    df = pd.DataFrame(
        {
            "y_true":       y_true,
            "pred_mean":    conformal_results["mean"],
            "conformal_lo": conformal_results["lo"],
            "conformal_hi": conformal_results["hi"],
            "in_interval":  (
                (y_true >= conformal_results["lo"])
                & (y_true <= conformal_results["hi"])
            ).astype(int),
        }
    )
    if pred_std is not None:
        df["pred_std"] = pred_std
    return df


# ─────────────────────────── Persistence ─────────────────────────────────────

def save_outputs(
    cfg: DictConfig,
    metrics: dict,
    results_df: pd.DataFrame,
) -> None:
    """Write metrics YAML and results CSV to the Hydra output directory.

    Output directory is resolved automatically from HydraConfig so each
    run gets its own timestamped folder when using the default sweeper.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config (saved alongside results for reproducibility).
    metrics : dict
        From :func:`compute_metrics`.
    results_df : pd.DataFrame
        From :func:`build_results_df`.
    """
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── metrics.yaml ─────────────────────────────────────────────────────────
    metrics_path = out_dir / "metrics.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {k: float(v) for k, v in metrics.items()}
        ),
        metrics_path,
    )
    log.info("Metrics saved → %s", metrics_path)

    # ── results.csv ──────────────────────────────────────────────────────────
    results_path = out_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Results saved → %s", results_path)

    # ── config snapshot ───────────────────────────────────────────────────────
    config_path = out_dir / "config_snapshot.yaml"
    OmegaConf.save(cfg, config_path)
    log.info("Config snapshot saved → %s", config_path)