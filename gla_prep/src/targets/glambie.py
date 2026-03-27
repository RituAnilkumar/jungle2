"""
GLAMBIE regional mass balance target loader.

Reads annual regional mass balance from a per-region GLAMBIE CSV and returns
combined, altimetry, and gravimetry columns in the configured unit (gt or mwe).
demdiff/glaciological columns are excluded.

GLAMBIE files are named {N}_{region_name}.csv where N is the non-zero-padded
RGI region number (e.g. 6_iceland.csv for r06). The file for the requested
region is located by globbing {data_dir}/{N}_*.csv.

Output file: glambie_targets_rNN.csv
Columns: region, start_date, end_date,
         combined_{u}, combined_{u}_errors,
         altimetry_{u}, altimetry_{u}_errors,
         gravimetry_{u}, gravimetry_{u}_errors
         (where {u} is gt or mwe; columns are NaN if method absent for region)
"""

from __future__ import annotations

import glob
import logging

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

_METHODS = ["combined", "altimetry", "gravimetry"]


def load(target_cfg: DictConfig, rgi_code: str) -> pd.DataFrame:
    """Load GLAMBIE regional mass balance for a single RGI region.

    Parameters
    ----------
    target_cfg : DictConfig
        The ``target`` config node (conf/target/glambie.yaml).
    rgi_code : str
        Two-digit RGI region code string, e.g. "06".

    Returns
    -------
    pd.DataFrame
        Columns: region, start_date, end_date,
                 combined_{u}, combined_{u}_errors,
                 altimetry_{u}, altimetry_{u}_errors,
                 gravimetry_{u}, gravimetry_{u}_errors.
    """
    rgi_num = str(int(rgi_code))  # "06" → "6"
    pattern = f"{target_cfg.data_dir}/{rgi_num}_*.csv"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No GLAMBIE file found matching {pattern}\n"
            f"Expected a file named {{N}}_{{region_name}}.csv in {target_cfg.data_dir}"
        )
    path = matches[0]
    log.info("Loading GLAMBIE data: %s", path)

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    u = target_cfg.units  # "gt" or "mwe"

    # Build list of columns to extract: time/region + method value + error
    keep = {
        target_cfg.region_col:     "region",
        target_cfg.start_date_col: "start_date",
        target_cfg.end_date_col:   "end_date",
    }
    for method in _METHODS:
        keep[f"{method}_{u}"]        = f"{method}_{u}"
        keep[f"{method}_{u}_errors"] = f"{method}_{u}_errors"

    # Only rename columns that exist; absent method columns become NaN
    available = {src: dst for src, dst in keep.items() if src in df.columns}
    missing   = [src for src in keep if src not in df.columns]
    if missing:
        log.warning("GLAMBIE columns not found (will be NaN): %s", missing)

    df = df.rename(columns=available)
    out_cols = list(keep.values())
    df = df.reindex(columns=out_cols)  # adds NaN cols for any missing methods

    log.info("GLAMBIE: %d records loaded for region %s (units=%s)", len(df), rgi_code, u)
    return df
