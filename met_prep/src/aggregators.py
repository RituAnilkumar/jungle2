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
  - Season-year label = the calendar year in which the season *ends*.
    e.g. NH accumulation Oct 2000 – Apr 2001 → season_year 2001.
  - For seasonal and annual_hydro, NH and SH definitions are applied
    per grid cell (lat >= 0 → NH) and combined into a single output file.
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

    return xr.merge(results, compat="override", join="outer")


# ─────────────────────────── Seasonal ────────────────────────────────────────

def _detect_lat_dim(ds: xr.Dataset) -> str:
    """Return the name of the latitude dimension."""
    for name in ("latitude", "lat"):
        if name in ds.dims:
            return name
    raise ValueError(f"No latitude dimension found. Available dims: {list(ds.dims)}")


def _season_mask(times: pd.DatetimeIndex, months: list[int]) -> np.ndarray:
    return np.isin(times.month, months)


def _season_year_label(times: pd.DatetimeIndex, months: list[int]) -> np.ndarray:
    """
    Label each timestep with the season-year it belongs to.
    Season-year = the calendar year in which the season ends.

    For non-wrapping seasons (e.g. May-Sep) this is just times.year.
    For wrapping seasons (e.g. Oct-Apr), months >= the season-start month
    are incremented by 1:
        Oct 2000 → 2001, Jan 2001 → 2001  ⟹ season_year 2001 = Oct 2000 – Apr 2001

    Wrapping is detected by a gap in the month list
    (e.g. [10,11,12,1,2,3,4] is non-consecutive).
    The season-start month is always in Jul-Dec for glaciological definitions.
    """
    wraps = (max(months) - min(months)) != (len(months) - 1)
    if not wraps:
        return times.year.to_numpy().copy()
    start = min(m for m in months if m > 6)
    return np.where(times.month >= start, times.year + 1, times.year)


def _agg_seasonal_hem(
    ds: xr.Dataset,
    dataset_cfg: DictConfig,
    agg_cfg: DictConfig,
    hem: str,
) -> xr.Dataset:
    """Compute seasonal aggregation using one hemisphere's month definitions."""
    vars_cfg   = dataset_cfg.variables
    stats      = list(agg_cfg.temp_stats)
    abl_months = get_ablation_months(hem, agg_cfg)
    acc_months = get_accumulation_months(hem, agg_cfg)
    times      = pd.DatetimeIndex(ds.time.values)
    results    = []

    for role, var_cfg in vars_cfg.items():
        vname = var_cfg.name
        if vname not in ds:
            continue
        da = ds[vname]
        if role == "precipitation":
            da = _precip_to_mm_per_day(da, var_cfg.units)

        for season, months in [("abl", abl_months), ("acc", acc_months)]:
            mask      = _season_mask(times, months)
            da_sel    = da.isel(time=mask)
            yr_labels = _season_year_label(times[mask], months)
            da_sel    = da_sel.assign_coords(season_year=("time", yr_labels))
            grp       = da_sel.groupby("season_year")

            if role == "temperature":
                for stat in stats:
                    name = f"{vname}_{season}_{stat}"
                    results.append(getattr(grp, stat)("time").rename(name).to_dataset())
            else:
                name = f"{vname}_{season}_sum"
                results.append(grp.sum("time").rename(name).to_dataset())

    return xr.merge(results, compat="override", join="outer")


def aggregate_seasonal(ds: xr.Dataset, dataset_cfg: DictConfig, agg_cfg: DictConfig) -> xr.Dataset:
    """
    Aggregate into ablation and accumulation seasons.

    NH and SH season definitions are applied per grid cell using the
    latitude sign (lat >= 0 → NH, lat < 0 → SH), then combined into a
    single output — matching the notebook pattern:
        nh.where(mask_nh).fillna(sh.where(~mask_nh))

    Output coordinate ``season_year`` is an integer labelled by the year
    the season ends in (e.g. NH acc Oct 2000 – Apr 2001 → season_year 2001).
    Output variable names: <varname>_abl_<stat> / <varname>_acc_<stat>
    (precip and radiation use _sum in place of a stat name).
    """
    lat_dim = _detect_lat_dim(ds)
    nh      = _agg_seasonal_hem(ds, dataset_cfg, agg_cfg, "NH")
    sh      = _agg_seasonal_hem(ds, dataset_cfg, agg_cfg, "SH")
    mask_nh = ds[lat_dim] >= 0
    log.info("Combining NH/SH seasonal aggregates per grid-cell latitude")
    return nh.where(mask_nh).fillna(sh.where(~mask_nh)).sortby("season_year")


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

    return xr.merge(results, compat="override", join="outer")


# ─────────────────────────── Annual hydrological ─────────────────────────────

def _agg_hydro_hem(
    ds: xr.Dataset,
    dataset_cfg: DictConfig,
    agg_cfg: DictConfig,
    hem: str,
) -> xr.Dataset:
    """Compute hydrological-year aggregation for one hemisphere."""
    vars_cfg    = dataset_cfg.variables
    stats       = list(agg_cfg.temp_stats)
    start_month = get_hydro_start_month(hem, agg_cfg)
    times       = pd.DatetimeIndex(ds.time.values)
    # month >= start → label year+1 (season ends in that year)
    hydro_years = np.where(times.month >= start_month, times.year + 1, times.year)
    results     = []

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

    return xr.merge(results, compat="override", join="outer")


def aggregate_annual_hydro(ds: xr.Dataset, dataset_cfg: DictConfig, agg_cfg: DictConfig) -> xr.Dataset:
    """
    Aggregate over hydrological years, combining NH and SH per grid cell.

    NH (start Oct): hydro_year N = Oct(N-1) – Sep(N).
    SH (start Apr): hydro_year N = Apr(N-1) – Mar(N).
    Label = the year the hydrological year ends in.
    """
    lat_dim = _detect_lat_dim(ds)
    nh      = _agg_hydro_hem(ds, dataset_cfg, agg_cfg, "NH")
    sh      = _agg_hydro_hem(ds, dataset_cfg, agg_cfg, "SH")
    mask_nh = ds[lat_dim] >= 0
    log.info("Combining NH/SH hydrological-year aggregates per grid-cell latitude")
    return nh.where(mask_nh).fillna(sh.where(~mask_nh)).sortby("hydro_year")


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
