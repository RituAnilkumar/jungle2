"""
Custom mass balance target loader.

Reads a user-provided CSV with configurable column name mappings.
Supports both per-glacier annual targets and region-level targets.
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load(target_cfg: DictConfig, rgi_code: str) -> pd.DataFrame:
    """Load a custom mass balance CSV.

    Parameters
    ----------
    target_cfg : DictConfig
        The ``target`` config node (conf/target/custom.yaml).
    rgi_code : str
        Two-digit RGI region code string.

    Returns
    -------
    pd.DataFrame
        Per-glacier: columns year, rgi_id, mass_balance, [mb_uncertainty]
        Regional:    columns region, year, regional_sum, uncertainty
    """
    path = target_cfg.data_path
    log.info("Loading custom MB data: %s", path)

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    is_regional = bool(target_cfg.get("is_regional", False))

    if is_regional:
        rename = {
            target_cfg.region_col: "region",
            target_cfg.year_col:   "year",
            target_cfg.mb_col:     "regional_sum",
        }
        unc_col = target_cfg.get("mb_uncertainty_col", None)
        if unc_col and unc_col in df.columns:
            rename[unc_col] = "uncertainty"
        df = df.rename(columns=rename)
        keep = [c for c in ["region", "year", "regional_sum", "uncertainty"] if c in df.columns]
        df = df[keep]
        df["year"] = df["year"].astype(int)
        df = df[df["region"].astype(str).str.contains(rgi_code)]
    else:
        rename = {
            target_cfg.rgi_id_col: "rgi_id",
            target_cfg.year_col:   "year",
            target_cfg.mb_col:     "mass_balance",
        }
        unc_col = target_cfg.get("mb_uncertainty_col", None)
        if unc_col and unc_col in df.columns:
            rename[unc_col] = "mb_uncertainty"
        df = df.rename(columns=rename)
        keep = [c for c in ["year", "rgi_id", "mass_balance", "mb_uncertainty"] if c in df.columns]
        df = df[keep].dropna(subset=["rgi_id", "year", "mass_balance"])
        df["year"] = df["year"].astype(int)

    log.info("Custom: %d records loaded for region %s", len(df), rgi_code)
    return df
