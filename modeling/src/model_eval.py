"""
Evaluation and results persistence for modeling.

Handles both random and blocked validation outputs.
For blocked splits, metrics are computed and reported for
logo, loyo, and loygo test sets independently.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from sklearn.metrics import r2_score
from scipy.stats import linregress

log = logging.getLogger(__name__)


# ─────────────────────────── Metrics ─────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, conformal_results: dict) -> dict:
    """Compute evaluation metrics from prediction results.

    Works for both conformal (lo/hi present) and non-conformal (lo/hi=None) modes.
    """
    mean = conformal_results["mean"]
    lo   = conformal_results.get("lo")
    hi   = conformal_results.get("hi")

    rmse = float(np.sqrt(np.mean((mean - y_true) ** 2)))
    mae  = float(np.mean(np.abs(mean - y_true)))
    r2   = float(r2_score(y_true, mean))

    slope, intercept, rval, pval, std_err = linregress(mean, y_true)

    metrics = {
        "rmse": rmse, "mae": mae, "r2": r2,
        "slope": float(slope), "intercept": float(intercept),
        "rval": float(rval), "pval": float(pval),
    }

    if lo is not None and hi is not None:
        alpha    = 1.0 - (conformal_results.get("coverage") or 0.9)
        coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
        width    = float(np.mean(hi - lo))
        penalty  = (lo - y_true) * (y_true < lo) + (y_true - hi) * (y_true > hi)
        interval_score = float(np.mean(width + (2.0 / max(alpha, 1e-8)) * penalty))

        metrics.update({
            "coverage":       coverage,
            "avg_width":      width,
            "interval_score": interval_score,
            "qhat":           conformal_results.get("qhat"),
        })

    log.info("── Metrics ──────────────────────────")
    for k, v in metrics.items():
        log.info("  %-22s %.4f", k, v if v is not None else float("nan"))

    return metrics


# ─────────────────────────── Results DataFrame ───────────────────────────────

def build_results_df(
    y_true: np.ndarray,
    conformal_results: dict,
    pred_std: np.ndarray | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true, "pred_mean": conformal_results["mean"]})

    if conformal_results.get("lo") is not None:
        df["conformal_lo"] = conformal_results["lo"]
        df["conformal_hi"] = conformal_results["hi"]
        df["in_interval"]  = (
            (y_true >= conformal_results["lo"]) & (y_true <= conformal_results["hi"])
        ).astype(int)

    if pred_std is not None:
        df["pred_std"] = pred_std

    return df


# ─────────────────────────── BayesNF blocked results ─────────────────────────

def build_bayesnf_results_df(
    df_split: pd.DataFrame,
    yhat_median: np.ndarray,
    yhat_lower: np.ndarray,
    yhat_upper: np.ndarray,
    target_col: str,
) -> pd.DataFrame:
    """Assemble results DataFrame for a BayesNF blocked-split evaluation."""
    result = df_split[["rgi_id", "year", target_col]].copy().reset_index(drop=True)
    result["pred_median"] = yhat_median
    result["pred_lower"]  = yhat_lower
    result["pred_upper"]  = yhat_upper
    result["in_interval"] = (
        (result[target_col] >= yhat_lower) & (result[target_col] <= yhat_upper)
    ).astype(int)
    return result


def compute_bayesnf_metrics(df_result: pd.DataFrame, target_col: str) -> dict:
    """Compute metrics for a BayesNF predictions DataFrame."""
    y_true = df_result[target_col].to_numpy()
    y_pred = df_result["pred_median"].to_numpy()
    lo     = df_result["pred_lower"].to_numpy()
    hi     = df_result["pred_upper"].to_numpy()

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    r2   = float(r2_score(y_true, y_pred))
    slope, intercept, rval, pval, _ = linregress(y_pred, y_true)

    coverage = float(np.mean((y_true >= lo) & (y_true <= hi)) * 100.0)
    widths   = hi - lo
    median_width = float(np.median(widths))

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "slope": float(slope), "intercept": float(intercept),
        "rval": float(rval), "pval": float(pval),
        "coverage_pct": coverage,
        "median_interval_width": median_width,
    }


# ─────────────────────────── Persistence ─────────────────────────────────────

def save_outputs(
    cfg: DictConfig,
    metrics: dict | list[dict],
    results_df: pd.DataFrame | list[pd.DataFrame],
    split_names: list[str] | None = None,
) -> None:
    """Write metrics YAML and results CSV(s) to the Hydra output directory.

    For blocked validation, pass a list of metrics dicts and DataFrames
    along with split_names (e.g. ["logo", "loyo", "loygo"]).
    """
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Config snapshot ───────────────────────────────────────────────────
    OmegaConf.save(cfg, out_dir / "config_snapshot.yaml")

    # ── Single split ──────────────────────────────────────────────────────
    if not isinstance(metrics, list):
        OmegaConf.save(
            OmegaConf.create({k: float(v) for k, v in metrics.items() if v is not None}),
            out_dir / "metrics.yaml",
        )
        results_df.to_csv(out_dir / "results.csv", index=False)
        log.info("Saved metrics.yaml and results.csv → %s", out_dir)
        return

    # ── Blocked split — one file per test set ────────────────────────────
    names = split_names or [f"split_{i}" for i in range(len(metrics))]
    all_metrics = {}
    for name, m, df in zip(names, metrics, results_df):
        all_metrics[name] = {k: float(v) for k, v in m.items() if v is not None}
        df.to_csv(out_dir / f"results_{name}.csv", index=False)
        log.info("Saved results_%s.csv → %s", name, out_dir)

    OmegaConf.save(OmegaConf.create(all_metrics), out_dir / "metrics.yaml")
    log.info("Saved metrics.yaml (all splits) → %s", out_dir)
