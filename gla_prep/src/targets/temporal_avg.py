"""
Temporal moving-window average mass balance target.

Reads per-glacier annual mass balance (from a previously generated
main_features.csv or any CSV with rgi_id, year, mass_balance columns),
applies a rolling window average per glacier, and produces the
temporal_avg_targets.csv with columns:
    rgi_id, start_date, end_date, avg_mb, [uncertainty]

Uncertainty is propagated as the standard deviation across the window
if no source uncertainty column is provided, otherwise as the mean of
source uncertainties within the window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load(target_cfg: DictConfig, rgi_code: str) -> pd.DataFrame:
    """Compute temporal moving-window averages for a region.

    Parameters
    ----------
    target_cfg : DictConfig
        The ``target`` config node (conf/target/temporal_avg.yaml).
    rgi_code : str
        Two-digit RGI region code string (used for logging only;
        the source CSV should already be filtered to the region).

    Returns
    -------
    pd.DataFrame
        Columns: rgi_id, start_date, end_date, avg_mb, uncertainty.
        One row per (glacier, window).
    """
    source_path = target_cfg.source_mb_path
    mb_col      = target_cfg.source_mb_col
    unc_col     = target_cfg.get("source_uncertainty_col", None)
    window      = int(target_cfg.window_years)

    log.info("Loading source MB for temporal averaging: %s", source_path)
    try:
        df = pd.read_csv(source_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(source_path, encoding="latin1")

    df = df[["rgi_id", "year", mb_col] + ([unc_col] if unc_col and unc_col in df.columns else [])].copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["rgi_id", "year"]).reset_index(drop=True)

    records = []
    for gla_id, grp in df.groupby("rgi_id"):
        grp = grp.sort_values("year").reset_index(drop=True)
        years = grp["year"].to_numpy()
        mb    = grp[mb_col].to_numpy()
        unc   = grp[unc_col].to_numpy() if (unc_col and unc_col in grp.columns) else None

        for i in range(len(grp) - window + 1):
            window_years = years[i : i + window]
            window_mb    = mb[i : i + window]
            avg_mb       = float(np.nanmean(window_mb))

            if unc is not None:
                uncertainty = float(np.nanmean(unc[i : i + window]))
            else:
                uncertainty = float(np.nanstd(window_mb))

            records.append({
                "rgi_id":     gla_id,
                "start_date": int(window_years[0]),
                "end_date":   int(window_years[-1]),
                "avg_mb":     avg_mb,
                "uncertainty": uncertainty,
            })

    result = pd.DataFrame(records)
    log.info(
        "Temporal averages: %d windows across %d glaciers (window=%d yr)",
        len(result), result["rgi_id"].nunique(), window,
    )
    return result
