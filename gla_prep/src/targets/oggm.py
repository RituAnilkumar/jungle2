"""
OGGM fixed-geometry mass balance target loader.

Reads OGGM L5 summary CSVs (one per RGI region), melts from wide to long
format, filters by min_year, and returns a tidy DataFrame with columns:
    year, rgi_id, mass_balance
"""

from __future__ import annotations

import logging

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def load(target_cfg: DictConfig, rgi_code: str) -> pd.DataFrame:
    """Load OGGM mass balance for a single RGI region.

    Parameters
    ----------
    target_cfg : DictConfig
        The ``target`` config node (conf/target/oggm.yaml).
    rgi_code : str
        Two-digit RGI region code string, e.g. "06".

    Returns
    -------
    pd.DataFrame
        Columns: year (int), rgi_id (str), mass_balance (float).
    """
    summary_dir = target_cfg.summary_dir
    min_year    = int(target_cfg.get("min_year", 1979))
    fn = f"{summary_dir}/fixed_geometry_mass_balance_{rgi_code}.csv"

    log.info("Loading OGGM summary: %s", fn)
    try:
        df = pd.read_csv(fn, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(fn, encoding="latin1")

    # Wide → long: columns are rgi_ids, rows are years
    df = df.melt(id_vars=["Unnamed: 0"], var_name="rgi_id", value_name="mass_balance")
    df = df.rename(columns={"Unnamed: 0": "year"})
    df = df[df["year"] >= min_year].reset_index(drop=True)
    df["year"] = df["year"].astype(int)

    log.info("OGGM: %d (glacier, year) pairs loaded (year >= %d)", len(df), min_year)
    return df
