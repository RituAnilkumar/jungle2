"""
Joins climate features, RGI static attributes, and mass balance targets
into the final output CSVs for the modeling directory.

Output files written to cfg.output_path:
  main_features.csv        — always
  glambie_targets.csv      — if target produces regional sums
  temporal_avg_targets.csv — if target produces temporal averages
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def join_and_save(
    rgi_df: pd.DataFrame,
    mb_df: pd.DataFrame | None,
    clim_df: pd.DataFrame,
    glambie_df: pd.DataFrame | None,
    temporal_avg_df: pd.DataFrame | None,
    cfg: DictConfig,
) -> None:
    """Merge all components and write output CSVs.

    Parameters
    ----------
    rgi_df : pd.DataFrame
        RGI static attributes. Columns include rgi_id, CenLat, CenLon.
    mb_df : pd.DataFrame | None
        Per-glacier annual mass balance (from OGGM/WGMS/custom).
        Columns: rgi_id, year, mass_balance, [mb_uncertainty].
        None if no per-glacier annual MB is being used.
    clim_df : pd.DataFrame
        Climate features sampled from met_prep. Already merged with RGI coords
        by climate_sampler.py. Columns include rgi_id, year, + climate vars.
    glambie_df : pd.DataFrame | None
        GLAMBIE regional sums. Columns: region, year, regional_sum, uncertainty.
        None if not applicable.
    temporal_avg_df : pd.DataFrame | None
        Temporal moving-window averages.
        Columns: rgi_id, start_date, end_date, avg_mb, uncertainty.
        None if not applicable.
    cfg : DictConfig
        Root gla_prep config (used for output_path and rgi_cols).
    """
    out_dir  = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgi_code = cfg.region.rgi_code

    # ── Build main_features.csv ───────────────────────────────────────────
    # clim_df already has RGI coords; merge in remaining static RGI columns
    rgi_static_cols = ["rgi_id"] + list(cfg.rgi_cols)
    rgi_static_cols = [c for c in rgi_static_cols if c in rgi_df.columns]
    main_df = clim_df.merge(rgi_df[rgi_static_cols], on="rgi_id", how="left")

    if mb_df is not None:
        merge_cols = ["rgi_id", "year"]
        mb_keep    = [c for c in ["rgi_id", "year", "mass_balance", "mb_uncertainty"] if c in mb_df.columns]
        main_df    = main_df.merge(mb_df[mb_keep], on=merge_cols, how="inner")
        log.info("Merged per-glacier MB: %d rows remaining after inner join", len(main_df))

    # Drop rows with any NaN in key columns
    n_before = len(main_df)
    main_df  = main_df.dropna()
    log.info("Dropped %d rows with NaN values; %d rows remaining", n_before - len(main_df), len(main_df))

    main_path = out_dir / f"main_features_r{rgi_code}.csv"
    main_df.to_csv(main_path, index=False)
    log.info("Wrote %s", main_path)

    # ── Write glambie_targets.csv ─────────────────────────────────────────
    if glambie_df is not None:
        glambie_path = out_dir / f"glambie_targets_r{rgi_code}.csv"
        glambie_df.to_csv(glambie_path, index=False)
        log.info("Wrote %s", glambie_path)

    # ── Write temporal_avg_targets.csv ────────────────────────────────────
    if temporal_avg_df is not None:
        temporal_path = out_dir / f"temporal_avg_targets_r{rgi_code}.csv"
        temporal_avg_df.to_csv(temporal_path, index=False)
        log.info("Wrote %s", temporal_path)
