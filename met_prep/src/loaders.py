"""
Dataset loaders for met_prep.

Opens locally stored NetCDF or GRIB files with xarray, handling
per-variable file layouts (as used by W5E5) and single multi-variable
files (as used by ERA5/CMIP).

Supported formats: NetCDF (.nc), GRIB (.grib, .grb) — both via xarray
with the appropriate engine (netcdf4 / cfgrib).
"""

from __future__ import annotations

import logging
from pathlib import Path

import glob as _glob
import xarray as xr
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Map config file_format to xarray engine
_ENGINE_MAP = {
    "netcdf": "netcdf4",
    "grib":   "cfgrib",
}


def _resolve_paths(pattern_or_list) -> list[str]:
    """Expand glob patterns or return a list of literal paths."""
    if isinstance(pattern_or_list, str):
        paths = sorted(_glob.glob(pattern_or_list))
    else:
        paths = []
        for item in pattern_or_list:
            paths.extend(sorted(_glob.glob(item)))
    if not paths:
        raise FileNotFoundError(
            f"No files matched: {pattern_or_list}"
        )
    return paths


def _open_mfdataset(paths: list[str], engine: str, chunks: dict) -> xr.Dataset:
    return xr.open_mfdataset(
        paths,
        combine="by_coords",
        parallel=True,
        chunks=chunks,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        engine=engine,
    )


def load_dataset(cfg: DictConfig) -> xr.Dataset:
    """Load the full dataset described by *cfg* (a dataset config node).

    Handles two layouts:

    1. **Single multi-variable files** (ERA5, CMIP):
       ``cfg.input_paths`` is a glob string or list → one Dataset returned.

    2. **Per-variable files** (W5E5):
       ``cfg.input_paths`` is a dict keyed by variable role
       (``temperature``, ``precipitation``, ``radiation``) → each is opened
       separately and merged into one Dataset.

    Parameters
    ----------
    cfg : DictConfig
        A dataset config node (e.g. from ``conf/dataset/era5.yaml``).

    Returns
    -------
    xr.Dataset
        Merged dataset with standardised variable names as given in
        ``cfg.variables``.
    """
    engine  = _ENGINE_MAP.get(cfg.file_format, "netcdf4")
    chunks  = OmegaConf.to_container(cfg.chunks, resolve=True)
    inp     = cfg.input_paths

    # ── Per-variable layout (W5E5 style) ──────────────────────────────────
    if isinstance(inp, DictConfig):
        datasets = []
        for role, pattern in inp.items():
            paths = _resolve_paths(pattern)
            log.info("Loading %s (%d file(s)): %s …", role, len(paths), paths[0])
            ds = _open_mfdataset(paths, engine, chunks)
            # Rename to standard name from config
            raw_name = cfg.variables[role].name
            if raw_name in ds:
                datasets.append(ds[[raw_name]])
            else:
                log.warning("Variable '%s' not found in files for role '%s'.", raw_name, role)
        return xr.merge(datasets, compat="override")

    # ── Single-file layout (ERA5/CMIP style) ──────────────────────────────
    paths = _resolve_paths(inp)
    log.info("Loading dataset (%d file(s)): %s …", len(paths), paths[0])
    ds = _open_mfdataset(paths, engine, chunks)

    # Keep only the configured variables
    keep = [v.name for v in cfg.variables.values()]
    available = [v for v in keep if v in ds]
    missing   = [v for v in keep if v not in ds]
    if missing:
        log.warning("Variables not found in dataset: %s", missing)
    return ds[available]
