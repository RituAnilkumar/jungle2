"""
Season and hydrological-year month definitions for met_prep.

Provides helpers to look up NH/SH month lists from the aggregation config.
All month lists are 1-indexed (January = 1).
"""

from __future__ import annotations

from omegaconf import DictConfig


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
