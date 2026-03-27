"""
Climate feature sampler for gla_prep.

For each (glacier, year) pair, samples the nearest-neighbour climate
value from a met_prep NetCDF output using xarray .sel(method='nearest').

Longitude is normalised to 0–360 to match ERA5/W5E5 conventions before
sampling, then dropped from the output (CenLon from RGI is kept instead).

met_prep produces a single combined file (NH and SH aggregated per grid cell),
so no hemisphere routing is needed here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Maps met_input_freq → the time coordinate name used in met_prep outputs
_TIME_COORD = {
    "monthly":         "time",
    "seasonal":        "season_year",
    "annual_calendar": "time",
    "annual_hydro":    "hydro_year",
}

# For time-coordinate types that store full timestamps, we extract the year
_TIMESTAMP_FREQS = {"monthly", "annual_calendar"}


def _lon_to_360(lon: np.ndarray) -> np.ndarray:
    """Convert longitudes from [-180, 180] to [0, 360]."""
    return (lon + 360) % 360


def _detect_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
    """Find the latitude and longitude dimension names in a dataset."""
    for lat_name in ("latitude", "lat"):
        for lon_name in ("longitude", "lon"):
            if lat_name in ds.dims and lon_name in ds.dims:
                return lat_name, lon_name
    raise ValueError(
        f"Cannot find lat/lon dimensions in dataset. "
        f"Available dims: {list(ds.dims)}"
    )


def _sample_from_file(
    gla_group: pd.DataFrame,
    met_path: str,
    met_freq: str,
) -> pd.DataFrame:
    """Open one met_prep NetCDF file and sample climate for a glacier group."""
    log.info("Opening met_prep file: %s (%d glaciers)", met_path, len(gla_group))
    clim = xr.open_dataset(met_path, chunks="auto")

    lat_name, lon_name = _detect_lat_lon_names(clim)
    time_coord = _TIME_COORD.get(met_freq, "time")

    if time_coord not in clim.coords:
        raise ValueError(
            f"Expected coordinate '{time_coord}' not found in {met_path}. "
            f"Available coords: {list(clim.coords)}"
        )

    gla_lats = xr.DataArray(gla_group["CenLat"].to_numpy(), dims="glacier_year")
    gla_lons = xr.DataArray(
        _lon_to_360(gla_group["CenLon"].to_numpy()), dims="glacier_year"
    )

    if met_freq in _TIMESTAMP_FREQS:
        gla_times = xr.DataArray(
            pd.to_datetime(gla_group["year"].astype(str) + "-01-01"),
            dims="glacier_year",
        )
        sel_kwargs = {lat_name: gla_lats, lon_name: gla_lons, "time": gla_times}
    else:
        gla_years = xr.DataArray(gla_group["year"].to_numpy(), dims="glacier_year")
        sel_kwargs = {lat_name: gla_lats, lon_name: gla_lons, time_coord: gla_years}

    log.info("Sampling climate for %d (glacier, year) pairs …", len(gla_group))
    sampled = clim.sel(method="nearest", **sel_kwargs)

    clim_df = sampled.to_dataframe().reset_index(drop=True)

    drop_cols = [c for c in clim_df.columns if c in (lat_name, lon_name, time_coord,
                                                       "time", "season_year", "hydro_year",
                                                       "number", "step", "surface",
                                                       "valid_time")]
    clim_df = clim_df.drop(columns=[c for c in drop_cols if c in clim_df.columns])

    log.info("Climate columns sampled: %s", list(clim_df.columns))
    log.info("Missing values per column:\n%s", clim_df.isnull().sum().to_string())

    return pd.concat([gla_group.reset_index(drop=True), clim_df], axis=1)


def get_met_years(cfg: DictConfig) -> list[int]:
    """Return the integer years present in the met_prep NetCDF.

    The time coordinate name and type depend on met_input_freq:
      seasonal / annual_hydro  → integer year coordinate
      monthly / annual_calendar → cftime/datetime coordinate (extract unique years)
    """
    met_path = f"{cfg.met_input_base_path}.nc"
    met_freq = cfg.met_input_freq
    time_coord = _TIME_COORD.get(met_freq, "time")

    with xr.open_dataset(met_path) as ds:
        if time_coord not in ds.coords:
            raise ValueError(
                f"Expected coordinate '{time_coord}' not found in {met_path}. "
                f"Available coords: {list(ds.coords)}"
            )
        raw = ds[time_coord].values

    if met_freq in _TIMESTAMP_FREQS:
        years = sorted(set(int(pd.Timestamp(t).year) for t in raw))
    else:
        years = sorted(int(y) for y in raw)

    log.info("Met file covers %d years: %d – %d", len(years), years[0], years[-1])
    return years


def sample_climate(
    gla_df: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Sample met_prep climate variables at each (glacier, year) location.

    Parameters
    ----------
    gla_df : pd.DataFrame
        DataFrame with columns: rgi_id, year, CenLat, CenLon.
        year must be integer.
    cfg : DictConfig
        Root Hydra config. Uses cfg.met_input_base_path and cfg.met_input_freq.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with climate feature columns appended.
        NaN reporting happens here; dropping happens in joiner.py.
    """
    met_path = f"{cfg.met_input_base_path}.nc"
    return _sample_from_file(gla_df.copy(), met_path, cfg.met_input_freq)
