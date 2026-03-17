"""
Temporal moving-window average mass balance target using Hugonnet et al. (2021)
per-glacier geodetic mass balance data.

Reads the Hugonnet cumulative mass balance CSV (rgiid, time DD/MM/YYYY,
area m², dh mwe cumulative, err_dh mwe), converts to per-glacier annual
mass balance (mwe/yr), applies a rolling window average, and produces
temporal_avg_targets.csv with columns:
    rgi_id, start_date, end_date, avg_mb_mwe, avg_mb_gt,
    uncertainty_mwe, uncertainty_gt

Also returns an annual DataFrame (rgi_id, year, annual_mb_mwe, area_m2,
uncertainty) for use as the climate sampling index in main.py.

Notes
-----
- dh is treated as cumulative mass balance in meters water equivalent (mwe),
  as provided by the Hugonnet et al. pergla_cumul dataset.
- Annual mass balance is derived as the difference between the last dh
  observation of consecutive calendar years.
- Gt conversion: avg_mb_gt = avg_mb_mwe [m/yr] * area_m2 [m²] * 1e-9
  (1 mwe = 1000 kg m⁻², 1 Gt = 1e12 kg).
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load(
    target_cfg: DictConfig, rgi_code: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute temporal moving-window averages from Hugonnet cumulative dh.

    Parameters
    ----------
    target_cfg : DictConfig
        The ``target`` config node (conf/target/temporal_avg.yaml).
    rgi_code : str
        Two-digit RGI region code string (e.g. "06").

    Returns
    -------
    temporal_avg_df : pd.DataFrame
        Columns: rgi_id, start_date, end_date, avg_mb_mwe, avg_mb_gt,
        uncertainty_mwe, uncertainty_gt.  One row per (glacier, window).
    annual_df : pd.DataFrame
        Columns: rgi_id, year, annual_mb_mwe, area_m2, uncertainty.
        Per-glacier annual mass balance; used as climate sampling index.
    """
    hugonnet_path = target_cfg.hugonnet_path
    window = int(target_cfg.window_years)

    log.info("Loading Hugonnet data: %s", hugonnet_path)
    try:
        raw = pd.read_csv(hugonnet_path, encoding="utf-8")
    except UnicodeDecodeError:
        raw = pd.read_csv(hugonnet_path, encoding="latin1")

    raw = raw.rename(columns={"rgiid": "rgi_id"})

    # Filter to the requested region
    region_prefix = f"RGI60-{rgi_code}."
    raw = raw[raw["rgi_id"].str.startswith(region_prefix)].copy()
    log.info(
        "Region r%s: %d observations for %d glaciers",
        rgi_code, len(raw), raw["rgi_id"].nunique(),
    )

    if raw.empty:
        log.warning("No Hugonnet data found for region %s", rgi_code)
        empty_ta = pd.DataFrame(
            columns=["rgi_id", "start_date", "end_date",
                     "avg_mb_mwe", "avg_mb_gt", "uncertainty_mwe", "uncertainty_gt"]
        )
        empty_an = pd.DataFrame(
            columns=["rgi_id", "year", "annual_mb_mwe", "area_m2", "uncertainty"]
        )
        return empty_ta, empty_an

    raw["date"] = pd.to_datetime(raw["time"], format="%d/%m/%Y")
    raw["year"] = raw["date"].dt.year

    annual_df = _to_annual(raw)
    temporal_avg_df = _to_window_avg(annual_df, window)

    log.info(
        "Temporal averages: %d windows across %d glaciers (window=%d yr)",
        len(temporal_avg_df), temporal_avg_df["rgi_id"].nunique(), window,
    )
    return temporal_avg_df, annual_df


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_annual(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative dh to per-glacier annual mass balance (mwe/yr).

    Annual mass balance for year Y is the difference between the last
    observation of year Y and the last observation of year Y-1.
    Uncertainty is propagated in quadrature from the two endpoint errors.
    """
    records = []
    for gla_id, grp in raw.groupby("rgi_id"):
        grp = grp.sort_values("date").reset_index(drop=True)
        area_m2 = float(grp["area"].iloc[0])

        # Select last observation within each calendar year
        annual = (
            grp.groupby("year", group_keys=False)
            .apply(lambda x: x.loc[x["date"].idxmax()])
            [["year", "dh", "err_dh"]]
            .sort_values("year")
            .reset_index(drop=True)
        )

        for i in range(1, len(annual)):
            prev = annual.iloc[i - 1]
            curr = annual.iloc[i]
            annual_mb = float(curr["dh"] - prev["dh"])   # mwe
            unc = float(np.sqrt(prev["err_dh"] ** 2 + curr["err_dh"] ** 2))
            records.append({
                "rgi_id":        gla_id,
                "year":          int(curr["year"]),
                "annual_mb_mwe": annual_mb,
                "area_m2":       area_m2,
                "uncertainty":   unc,
            })

    return pd.DataFrame(records)


def _to_window_avg(annual_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling window and convert to mwe/yr and Gt/yr."""
    records = []
    for gla_id, grp in annual_df.groupby("rgi_id"):
        grp = grp.sort_values("year").reset_index(drop=True)
        if len(grp) < window:
            continue

        years  = grp["year"].to_numpy()
        mb     = grp["annual_mb_mwe"].to_numpy()
        unc    = grp["uncertainty"].to_numpy()
        area_m2 = float(grp["area_m2"].iloc[0])

        for i in range(len(grp) - window + 1):
            w_years = years[i: i + window]
            w_mb    = mb[i: i + window]
            w_unc   = unc[i: i + window]

            avg_mb_mwe = float(np.nanmean(w_mb))
            # Gt/yr: mwe/yr × m² × 1e-9  (1 mwe = 1000 kg/m², 1 Gt = 1e12 kg)
            avg_mb_gt       = avg_mb_mwe * area_m2 * 1e-9
            uncertainty_mwe = float(np.nanmean(w_unc))
            uncertainty_gt  = uncertainty_mwe * area_m2 * 1e-9

            records.append({
                "rgi_id":          gla_id,
                "start_date":      int(w_years[0]),
                "end_date":        int(w_years[-1]),
                "avg_mb_mwe":      avg_mb_mwe,
                "avg_mb_gt":       avg_mb_gt,
                "uncertainty_mwe": uncertainty_mwe,
                "uncertainty_gt":  uncertainty_gt,
            })

    return pd.DataFrame(records)
