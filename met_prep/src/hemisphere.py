"""
Hemisphere-aware season and hydrological-year definitions.

Hemisphere is inferred from latitude (lat > 0 → NH, lat <= 0 → SH) unless
explicitly overridden in config via ``hemisphere: NH`` or ``hemisphere: SH``.

All month lists are 1-indexed (January = 1).
"""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig


def get_hemisphere(lat: float, cfg_hemisphere: str | None) -> str:
    """Return 'NH' or 'SH' for a given latitude and optional config override.

    Parameters
    ----------
    lat : float
        Latitude of the point of interest.
    cfg_hemisphere : str | None
        Value of ``hemisphere`` from config. If not null, overrides inference.
        Accepts 'NH', 'SH', or None/'auto'.
    """
    if cfg_hemisphere and cfg_hemisphere.upper() in ("NH", "SH"):
        return cfg_hemisphere.upper()
    return "NH" if lat > 0 else "SH"


def get_ablation_months(hemisphere: str, agg_cfg: DictConfig) -> list[int]:
    """Return ablation-season month list for the given hemisphere."""
    key = "nh" if hemisphere == "NH" else "sh"
    return list(agg_cfg[key].ablation_months)


def get_accumulation_months(hemisphere: str, agg_cfg: DictConfig) -> list[int]:
    """Return accumulation-season month list for the given hemisphere."""
    key = "nh" if hemisphere == "NH" else "sh"
    return list(agg_cfg[key].accumulation_months)


def get_hydro_start_month(hemisphere: str, agg_cfg: DictConfig) -> int:
    """Return the starting month of the hydrological year for the given hemisphere."""
    key = "nh" if hemisphere == "NH" else "sh"
    return int(agg_cfg[key].start_month)
