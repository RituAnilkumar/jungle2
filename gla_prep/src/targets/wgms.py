"""
WGMS annual mass balance target loader.

Reads a WGMS FoG CSV, applies column name mappings from config, and
returns a tidy DataFrame with columns:
    year, rgi_id, mass_balance, [mb_uncertainty]
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load(target_cfg: DictConfig, rgi_code: str) -> pd.DataFrame:
    """Load WGMS mass balance records for a single RGI region.

    Parameters
    ----------
    target_cfg : DictConfig
        The ``target`` config node (conf/target/wgms.yaml).
    rgi_code : str
        Two-digit RGI region code string — used to filter records by region.

    Returns
    -------
    pd.DataFrame
        Columns: year (int), rgi_id (str), mass_balance (float),
        and optionally mb_uncertainty (float).
    """
    path = target_cfg.data_path
    log.info("Loading WGMS data: %s", path)

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # Rename columns to standard names
    rename = {
        target_cfg.rgi_id_col: "rgi_id",
        target_cfg.year_col:   "year",
        target_cfg.mb_col:     "mass_balance",
    }
    df = df.rename(columns=rename)

    unc_col = target_cfg.get("mb_uncertainty_col", None)
    if unc_col and unc_col in df.columns:
        df = df.rename(columns={unc_col: "mb_uncertainty"})
        keep = ["year", "rgi_id", "mass_balance", "mb_uncertainty"]
    else:
        keep = ["year", "rgi_id", "mass_balance"]

    df = df[keep].dropna(subset=["rgi_id", "year", "mass_balance"])
    df["year"] = df["year"].astype(int)

    # Filter to this RGI region by matching rgi_id prefix
    df = df[df["rgi_id"].str.startswith(f"RGI60-{rgi_code}")]
    log.info("WGMS: %d records loaded for region %s", len(df), rgi_code)
    return df
