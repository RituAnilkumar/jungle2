"""
Conformal prediction for modeling.

Three modes selected via cfg.conformal.type:

  absolute_residuals   — score = |y - ŷ|; constant-width intervals
  normalized_residuals — score = |y - ŷ| / σ̂; adaptive-width intervals
  none                 — no CP; model outputs passed through directly

For conformal=none, probabilistic models return (mean, std) or quantiles
directly. Deterministic models return point predictions only.
"""

from __future__ import annotations

import logging

import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)


# ─────────────────────────── Core functions ──────────────────────────────────

def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = scores.shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k - 1, n - 1)])


def absolute_residuals(
    y_cal: np.ndarray,
    cal_mean: np.ndarray,
    qry_mean: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = np.abs(y_cal.ravel() - cal_mean.ravel())
    qhat   = _conformal_quantile(scores, alpha)
    lo, hi = qry_mean - qhat, qry_mean + qhat
    log.info("Absolute-residual CP | alpha=%.2f | q̂=%.4f", alpha, qhat)
    return qry_mean, lo, hi, qhat


def normalized_residuals(
    y_cal: np.ndarray,
    cal_mean: np.ndarray,
    cal_std: np.ndarray,
    qry_mean: np.ndarray,
    qry_std: np.ndarray,
    alpha: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    scores = np.abs(y_cal.ravel() - cal_mean.ravel()) / (cal_std.ravel() + eps)
    qhat   = _conformal_quantile(scores, alpha)
    lo     = qry_mean - qhat * (qry_std + eps)
    hi     = qry_mean + qhat * (qry_std + eps)
    log.info("Normalized-residual CP | alpha=%.2f | q̂=%.4f", alpha, qhat)
    return qry_mean, lo, hi, qhat


# ─────────────────────────── Dispatch ────────────────────────────────────────

def run_conformal(cfg: DictConfig, predict_fn, bundle) -> dict:
    """Run conformal prediction (or passthrough for type=none).

    Parameters
    ----------
    cfg : DictConfig
        Root config; uses cfg.conformal.
    predict_fn : callable
        predict_fn(X_tensor) -> mean_array   (absolute_residuals / none)
        predict_fn(X_tensor) -> (mean, std)  (normalized_residuals / none for prob. models)
    bundle : DataBundle
        Populated DataBundle from data_prep.py.

    Returns
    -------
    dict with keys: mean, lo, hi, qhat, coverage, avg_width
        For type=none: lo=hi=None, qhat=None, coverage=None, avg_width=None.
    """
    conformal_type = cfg.conformal.type
    alpha          = float(cfg.conformal.get("alpha", 0.1))

    y_cal_orig  = bundle.y_cal.ravel()
    y_test_orig = bundle.y_test.ravel()

    # ── None: passthrough ────────────────────────────────────────────────
    if conformal_type == "none":
        test_out = predict_fn(bundle.X_test_t)
        if isinstance(test_out, tuple):
            mean, std = test_out
            log.info("Conformal=none | returning native model mean + std.")
        else:
            mean, std = test_out, None
            log.info("Conformal=none | returning point predictions only.")
        return {
            "mean": mean, "std": std,
            "lo": None, "hi": None,
            "qhat": None, "coverage": None, "avg_width": None,
        }

    # ── Absolute residuals ───────────────────────────────────────────────
    if conformal_type == "absolute_residuals":
        cal_out  = predict_fn(bundle.X_cal_t)
        test_out = predict_fn(bundle.X_test_t)
        cal_mean  = cal_out[0]  if isinstance(cal_out,  tuple) else cal_out
        test_mean = test_out[0] if isinstance(test_out, tuple) else test_out

        mean, lo, hi, qhat = absolute_residuals(y_cal_orig, cal_mean, test_mean, alpha)

    # ── Normalized residuals ─────────────────────────────────────────────
    elif conformal_type == "normalized_residuals":
        eps = float(cfg.conformal.get("eps", 1e-6))
        cal_mean,  cal_std  = predict_fn(bundle.X_cal_t)
        test_mean, test_std = predict_fn(bundle.X_test_t)
        mean, lo, hi, qhat = normalized_residuals(
            y_cal_orig, cal_mean, cal_std, test_mean, test_std, alpha, eps
        )

    else:
        raise ValueError(
            f"Unknown conformal type '{conformal_type}'. "
            "Choose 'absolute_residuals', 'normalized_residuals', or 'none'."
        )

    coverage  = float(np.mean((y_test_orig >= lo) & (y_test_orig <= hi)))
    avg_width = float(np.mean(hi - lo))

    log.info(
        "Coverage: %.2f%% (target %.2f%%) | avg width: %.4f",
        coverage * 100, (1 - alpha) * 100, avg_width,
    )
    return {
        "mean": mean, "std": None,
        "lo": lo, "hi": hi,
        "qhat": qhat, "coverage": coverage, "avg_width": avg_width,
    }
