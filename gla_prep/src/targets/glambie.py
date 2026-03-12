"""
GLAMBIE regional mass balance target loader.

Reads annual regional mass balance sums from a GLAMBIE CSV and returns
a tidy DataFrame for use as an auxiliary target in the modeling directory.

Output file: glambie_targets.csv
Columns: region, year, regional_sum, uncertainty
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load(target_cfg: DictConfig, rgi_code: str) -> pd.DataFrame:
    """Load GLAMBIE regional mass balance for a single RGI region.

    Parameters
    ----------
    target_cfg : DictConfig
        The ``target`` config node (conf/target/glambie.yaml).
    rgi_code : str
        Two-digit RGI region code string — used to filter records.

    Returns
    -------
    pd.DataFrame
        Columns: region (str), year (int), regional_sum (float), uncertainty (float).
    """
    path = target_cfg.data_path
    log.info("Loading GLAMBIE data: %s", path)

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    rename = {
        target_cfg.region_col:       "region",
        target_cfg.year_col:         "year",
        target_cfg.regional_sum_col: "regional_sum",
        target_cfg.uncertainty_col:  "uncertainty",
    }
    df = df.rename(columns=rename)
    df = df[["region", "year", "regional_sum", "uncertainty"]].copy()
    df["year"] = df["year"].astype(int)

    # Filter to this region
    df = df[df["region"].astype(str).str.contains(rgi_code)]
    log.info("GLAMBIE: %d annual records loaded for region %s", len(df), rgi_code)
    return df
