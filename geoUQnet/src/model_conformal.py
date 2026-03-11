"""
Split conformal prediction for geoUQnet.

Two nonconformity score types are supported and selected via
``cfg.conformal.type``:

  absolute_residuals  — score = |y - ŷ|
                        works for any model (mean-only output)
  normalized_residuals — score = |y - ŷ| / (σ̂ + ε)
                         requires a predictive std estimate;
                         valid for mc_dropout_ann, deep_ensemble_ann,
                         bayesian_ann.

Both follow the standard split-conformal recipe:
  1. Fit model on train split.
  2. Score calibration set.
  3. Take the ceil((n+1)(1-α))/n quantile as q̂.
  4. Form intervals: [ŷ - q̂, ŷ + q̂]  (or scaled by σ̂ for normalised).
"""

from __future__ import annotations

import logging

import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)


# ─────────────────────────── Score functions ─────────────────────────────────

def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the finite-sample corrected (1-α) quantile of *scores*."""
    n = scores.shape[0]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k - 1, n - 1)])


def absolute_residuals(
    y_cal: np.ndarray,
    cal_mean: np.ndarray,
    qry_mean: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Absolute-residual conformal intervals (model-agnostic).

    Parameters
    ----------
    y_cal : np.ndarray [n_cal]
        True calibration targets in *original* units.
    cal_mean : np.ndarray [n_cal]
        Model mean predictions on the calibration set.
    qry_mean : np.ndarray [n_query]
        Model mean predictions on the query (test) set.
    alpha : float
        Miscoverage level, e.g. 0.1 → 90 % marginal coverage.

    Returns
    -------
    qry_mean, lo, hi : np.ndarray [n_query]
    qhat : float
    """
    scores = np.abs(y_cal.ravel() - cal_mean.ravel())
    qhat   = _conformal_quantile(scores, alpha)
    lo     = qry_mean - qhat
    hi     = qry_mean + qhat
    log.info(
        "Absolute-residual conformal | alpha=%.2f | q̂=%.4f", alpha, qhat
    )
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
    """Normalised-residual conformal intervals (uncertainty-adaptive width).

    Parameters
    ----------
    y_cal, cal_mean, cal_std : np.ndarray [n_cal]
        Calibration targets, means and predictive stds (original units).
    qry_mean, qry_std : np.ndarray [n_query]
        Query means and stds (original units).
    alpha : float
        Miscoverage level.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    qry_mean, lo, hi : np.ndarray [n_query]
    qhat : float
    """
    scores = np.abs(y_cal.ravel() - cal_mean.ravel()) / (cal_std.ravel() + eps)
    qhat   = _conformal_quantile(scores, alpha)
    lo     = qry_mean - qhat * (qry_std + eps)
    hi     = qry_mean + qhat * (qry_std + eps)
    log.info(
        "Normalised-residual conformal | alpha=%.2f | q̂=%.4f", alpha, qhat
    )
    return qry_mean, lo, hi, qhat


# ─────────────────────────── Dispatch ────────────────────────────────────────

def run_conformal(
    cfg: DictConfig,
    predict_fn,
    bundle,
) -> dict:
    """Select conformal type from config and return interval results.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.  Uses ``cfg.conformal`` and ``cfg.model``.
    predict_fn : callable
        A function with signature:

            predict_fn(X_tensor) -> np.ndarray           # mean only
        or
            predict_fn(X_tensor) -> (mean, std)          # mean + std

        For ``normalized_residuals``, the function **must** return a tuple.
        For ``absolute_residuals``, either form is accepted (std ignored).

    bundle : DataBundle
        Populated :class:`~src.data_prep.DataBundle`.

    Returns
    -------
    dict with keys:
        ``mean``, ``lo``, ``hi``, ``qhat``,
        ``coverage``, ``avg_width``, ``rmse``
    """
    hydra_cfg = HydraConfig.get()
    log.info("Hydra output dir: %s", hydra_cfg.runtime.output_dir)

    conformal_type: str = cfg.conformal.type   # absolute_residuals | normalized_residuals
    alpha: float        = float(cfg.conformal.get("alpha", 0.1))

    y_cal_orig  = bundle.y_cal.ravel()
    y_test_orig = bundle.y_test.ravel()

    if conformal_type == "absolute_residuals":
        cal_out  = predict_fn(bundle.X_cal_t)
        test_out = predict_fn(bundle.X_test_t)
        cal_mean  = cal_out[0]  if isinstance(cal_out,  tuple) else cal_out
        test_mean = test_out[0] if isinstance(test_out, tuple) else test_out

        mean, lo, hi, qhat = absolute_residuals(
            y_cal_orig, cal_mean, test_mean, alpha
        )

    elif conformal_type == "normalized_residuals":
        cal_mean,  cal_std  = predict_fn(bundle.X_cal_t)
        test_mean, test_std = predict_fn(bundle.X_test_t)
        eps = float(cfg.conformal.get("eps", 1e-6))

        mean, lo, hi, qhat = normalized_residuals(
            y_cal_orig, cal_mean, cal_std,
            test_mean,  test_std,
            alpha, eps=eps,
        )

    else:
        raise ValueError(
            f"Unknown conformal type '{conformal_type}'. "
            "Choose 'absolute_residuals' or 'normalized_residuals'."
        )

    coverage  = float(np.mean((y_test_orig >= lo) & (y_test_orig <= hi)))
    avg_width = float(np.mean(hi - lo))
    rmse      = float(np.sqrt(np.mean((mean - y_test_orig) ** 2)))

    log.info(
        "Coverage: %.2f%% (target %.2f%%) | avg width: %.4f | RMSE: %.4f",
        coverage * 100, (1 - alpha) * 100, avg_width, rmse,
    )

    return {
        "mean":      mean,
        "lo":        lo,
        "hi":        hi,
        "qhat":      qhat,
        "coverage":  coverage,
        "avg_width": avg_width,
        "rmse":      rmse,
    }