"""
Temporal aggregation routines for met_prep.

Four aggregation modes, all driven by Hydra config:

  monthly          — resample to monthly statistics
  seasonal         — aggregate into ablation / accumulation seasons
  annual_calendar  — aggregate over Jan-Dec calendar years
  annual_hydro     — aggregate over hydrological years (NH: Oct-Sep, SH: Apr-Mar)

Conventions:
  - Temperature  : stats as configured (mean, min, max, median, std)
  - Precipitation: always summed (mm per period)
  - Radiation    : always summed (J m-2 or W m-2 × seconds per period)
  - Output time coordinate is labelled with the *start* of each period
    (monthly → first day of month; seasonal → first day of ablation/accum season;
     annual → first day of the year / hydrological year)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

from .hemisphere import (
    get_ablation_months,
    get_accumulation_months,
    get_hydro_start_month,
)

log = logging.getLogger(__name__)


# ─────────────────────────── Helpers ─────────────────────────────────────────

def _apply_stats(
    da: xr.DataArray,
    grouper,
    stats: list[str],
    suffix: str,
) -> xr.Dataset:
    """Apply a list of named statistics to a grouped DataArray."""
    results = {}
    for stat in stats:
        fn: Callable = getattr(grouper, stat)
        results[f"{da.name}_{suffix}_{stat}"] = fn("time")
    return xr.Dataset(results)


def _precip_to_mm_per_day(da: xr.DataArray, units: str) -> xr.DataArray:
    """Convert precipitation to mm day-1 if in kg m-2 s-1."""
    if units == "kg m-2 s-1":
        return da * 86400.0
    return da   # assume already mm/day or mm/month


# ─────────────────────────── Monthly ─────────────────────────────────────────

def aggregate_monthly(ds: xr.Dataset, dataset_cfg: DictConfig, agg_cfg: DictConfig) -> xr.Dataset:
    """Resample to monthly stats for temperature; monthly sums for precip and radiation."""
    vars_cfg = dataset_cfg.variables
    stats    = list(agg_cfg.temp_stats)
    results  = []

    for role, var_cfg in vars_cfg.items():
        vname = var_cfg.name
        if vname not in ds:
            continue
        da = ds[vname]
        g  = da.resample(time="MS")

        if role == "temperature":
            results.append(_apply_stats(da, g, stats, "monthly"))
        elif role == "precipitation":
            da_mm = _precip_to_mm_per_day(da, var_cfg.units)
            monthly_sum = da_mm.resample(time="MS").sum("time").rename(f"{vname}_monthly_sum")
            monthly_sum.attrs["units"] = "mm month-1"
            results.append(monthly_sum.to_dataset())
        elif role == "radiation":
            monthly_sum = da.resample(time="MS").sum("time").rename(f"{vname}_monthly_sum")
            results.append(monthly_sum.to_dataset())

    return xr.merge(results, compat="override")


# ─────────────────────────── Seasonal ────────────────────────────────────────

def _season_mask(times: pd.DatetimeIndex, months: list[int]) -> np.ndarray:
    return np.isin(times.month, months)


def _season_year_label(times: pd.DatetimeIndex, start_months: list[int]) -> np.ndarray:
    """
    Label each timestep with the year of the season it belongs to.
    For seasons that span a year boundary (e.g. Oct-Apr), timestamps in the
    early months (before the start month of next occurrence) are labelled
    with the *previous* calendar year so all months of one season share
    one label.
    """
    labels = times.year.copy()
    start  = min(start_months)
    if max(start_months) > min(start_months):
        # Season is contiguous within one year — label is straightforward
        return labels
    # Season wraps around year boundary (e.g. Oct=10 … Apr=4)
    # Months < start that are in the season belong to the previous season-year
    for i, t in enumerate(times):
        if t.month < start:
            labels[i] -= 1
    return labels


def aggregate_seasonal(ds: xr.Dataset, dataset_cfg: DictConfig, agg_cfg: DictConfig) -> xr.Dataset:
    """
    Aggregate into ablation and accumulation seasons.

    The hemisphere is inferred from the latitude coordinate of each grid
    cell (lat > 0 → NH). If the dataset has no latitude coordinate or
    hemisphere is overridden in config, a single hemisphere is applied.

    Output coordinate ``season_year`` is an integer year label.
    Output variable names follow the pattern:
        <varname>_abl_<stat>  (ablation)
        <varname>_acc_<stat>  (accumulation)
        <varname>_abl_sum / <varname>_acc_sum  (precip and radiation)
    """
    vars_cfg   = dataset_cfg.variables
    stats      = list(agg_cfg.temp_stats)
    hemisphere = agg_cfg.get("hemisphere", None)
    results    = []

    # Use a representative latitude for hemisphere inference if global dataset
    # For per-cell inference, aggregation must be done after spatial subsetting.
    # Here we apply a single hemisphere based on config or default NH.
    hem = hemisphere if hemisphere and hemisphere.upper() in ("NH", "SH") else "NH"
    if hemisphere is None:
        log.warning(
            "hemisphere=null in config: applying NH season definition globally. "
            "For SH glaciers, set hemisphere: SH or run with -m hemisphere=NH,SH."
        )

    abl_months = get_ablation_months(hem, agg_cfg)
    acc_months = get_accumulation_months(hem, agg_cfg)

    times = pd.DatetimeIndex(ds.time.values)

    for role, var_cfg in vars_cfg.items():
        vname = var_cfg.name
        if vname not in ds:
            continue
        da = ds[vname]

        if role == "precipitation":
            da = _precip_to_mm_per_day(da, var_cfg.units)

        for season, months in [("abl", abl_months), ("acc", acc_months)]:
            mask   = _season_mask(times, months)
            da_sel = da.isel(time=mask)
            t_sel  = times[mask]

            # Assign season_year coordinate
            yr_labels = _season_year_label(t_sel, months)
            da_sel    = da_sel.assign_coords(
                season_year=("time", yr_labels)
            )
            grp = da_sel.groupby("season_year")

            if role == "temperature":
                for stat in stats:
                    name = f"{vname}_{season}_{stat}"
                    results.append(getattr(grp, stat)("time").rename(name).to_dataset())
            else:
                name = f"{vname}_{season}_sum"
                results.append(grp.sum("time").rename(name).to_dataset())

    return xr.merge(results, compat="override")


# ─────────────────────────── Annual calendar ──────────────────────────────────

def aggregate_annual_calendar(ds: xr.Dataset, dataset_cfg: DictConfig, agg_cfg: DictConfig) -> xr.Dataset:
    """Aggregate over Jan-Dec calendar years."""
    vars_cfg = dataset_cfg.variables
    stats    = list(agg_cfg.temp_stats)
    results  = []

    for role, var_cfg in vars_cfg.items():
        vname = var_cfg.name
        if vname not in ds:
            continue
        da = ds[vname]
        g  = da.resample(time="YS")

        if role == "temperature":
            results.append(_apply_stats(da, g, stats, "annual"))
        elif role == "precipitation":
            da_mm = _precip_to_mm_per_day(da, var_cfg.units)
            annual_sum = da_mm.resample(time="YS").sum("time").rename(f"{vname}_annual_sum")
            annual_sum.attrs["units"] = "mm year-1"
            results.append(annual_sum.to_dataset())
        elif role == "radiation":
            annual_sum = da.resample(time="YS").sum("time").rename(f"{vname}_annual_sum")
            results.append(annual_sum.to_dataset())

    return xr.merge(results, compat="override")


# ─────────────────────────── Annual hydrological ─────────────────────────────

def aggregate_annual_hydro(ds: xr.Dataset, dataset_cfg: DictConfig, agg_cfg: DictConfig) -> xr.Dataset:
    """
    Aggregate over hydrological years.

    NH (start_month=10): Oct(Y) – Sep(Y+1), labelled with Y+1 (the year Sep falls in).
    SH (start_month=4) : Apr(Y) – Mar(Y+1), labelled with Y+1 (the year Mar falls in).

    Output coordinate is ``hydro_year`` (integer).
    """
    vars_cfg   = dataset_cfg.variables
    stats      = list(agg_cfg.temp_stats)
    hemisphere = agg_cfg.get("hemisphere", None)
    hem        = hemisphere if hemisphere and hemisphere.upper() in ("NH", "SH") else "NH"

    start_month = get_hydro_start_month(hem, agg_cfg)
    times       = pd.DatetimeIndex(ds.time.values)

    # Assign hydro_year label: if month >= start_month → label = year+1; else label = year
    hydro_years = np.where(times.month >= start_month, times.year + 1, times.year)

    results = []

    for role, var_cfg in vars_cfg.items():
        vname = var_cfg.name
        if vname not in ds:
            continue
        da = ds[vname].assign_coords(hydro_year=("time", hydro_years))

        if role == "precipitation":
            da = _precip_to_mm_per_day(da, var_cfg.units)

        grp = da.groupby("hydro_year")

        if role == "temperature":
            for stat in stats:
                name = f"{vname}_hydro_{stat}"
                results.append(getattr(grp, stat)("time").rename(name).to_dataset())
        else:
            name = f"{vname}_hydro_sum"
            results.append(grp.sum("time").rename(name).to_dataset())

    return xr.merge(results, compat="override")


# ─────────────────────────── Dispatcher ──────────────────────────────────────

_AGGREGATORS = {
    "monthly":          aggregate_monthly,
    "seasonal":         aggregate_seasonal,
    "annual_calendar":  aggregate_annual_calendar,
    "annual_hydro":     aggregate_annual_hydro,
}


def aggregate(ds: xr.Dataset, dataset_cfg: DictConfig, agg_cfg: DictConfig) -> xr.Dataset:
    """Dispatch to the correct aggregator based on ``agg_cfg.name``."""
    name = agg_cfg.name
    if name not in _AGGREGATORS:
        raise ValueError(
            f"Unknown aggregation '{name}'. "
            f"Choose from: {list(_AGGREGATORS.keys())}"
        )
    log.info("Running aggregation: %s", name)
    return _AGGREGATORS[name](ds, dataset_cfg, agg_cfg)
