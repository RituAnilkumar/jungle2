"""
RGI static attribute loader for gla_prep.

Loads RGI glacier inventory (CSV or SHP), keeps the columns configured
in the root config, and always includes rgi_id, CenLat, CenLon.
"""

from __future__ import annotations

import logging

import pandas as pd
import geopandas as gpd
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Columns always included regardless of rgi_cols config
_ALWAYS_INCLUDE = {"RGIId", "GLIMSId", "CenLon", "CenLat"}


def load_rgi(region_cfg: DictConfig, rgi_cols: list[str]) -> pd.DataFrame:
    """Load RGI attributes for a region and return a tidy DataFrame.

    Parameters
    ----------
    region_cfg : DictConfig
        Region config node (contains rgi_attr_path, rgi_format).
    rgi_cols : list[str]
        Additional static columns to extract (from root config rgi_cols).

    Returns
    -------
    pd.DataFrame
        Columns: rgi_id, GLIMSId, CenLon, CenLat, + rgi_cols.
        rgi_id is the cleaned RGIId string.
    """
    path   = region_cfg.rgi_attr_path
    fmt    = region_cfg.get("rgi_format", "csv").lower()

    if fmt == "shp":
        gdf = gpd.read_file(path)
        df  = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin1")

    keep = list(_ALWAYS_INCLUDE | set(rgi_cols))
    available = [c for c in keep if c in df.columns]
    missing   = [c for c in keep if c not in df.columns]
    if missing:
        log.warning("RGI columns not found and will be skipped: %s", missing)

    df = df[available].copy()
    df = df.rename(columns={"RGIId": "rgi_id"})

    log.info("Loaded RGI: %d glaciers, columns: %s", len(df), list(df.columns))
    return df
