"""
Joins climate features and RGI static attributes into main_features.csv,
then writes any target-specific CSVs alongside it.

Output files written to cfg.output_path:
  main_features_rNN.csv        — always; all glaciers × all met years
  {target}_targets_rNN.csv     — per-glacier annual MB (oggm / wgms / custom)
  glambie_targets_rNN.csv      — regional sums (glambie or temporal_avg+glambie)
  temporal_avg_targets_rNN.csv — multi-year window averages (temporal_avg)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def join_and_save(
    rgi_df: pd.DataFrame,
    clim_df: pd.DataFrame,
    target_name: str,
    per_glacier_df: pd.DataFrame | None,
    glambie_df: pd.DataFrame | None,
    temporal_avg_df: pd.DataFrame | None,
    out_dir: str,
) -> None:
    """Merge features and write all output CSVs.

    Parameters
    ----------
    rgi_df : pd.DataFrame
        RGI static attributes. Columns include rgi_id, CenLat, CenLon.
    clim_df : pd.DataFrame
        Climate features sampled from met_prep for all glaciers × all years.
        Columns: rgi_id, year, CenLat, CenLon, + climate vars.
    target_name : str
        Name of the active target (used for per-glacier output filename).
    per_glacier_df : pd.DataFrame | None
        Per-glacier annual mass balance (oggm / wgms / custom).
        Columns: rgi_id, year, mass_balance, [mb_uncertainty].
    glambie_df : pd.DataFrame | None
        GLAMBIE regional sums. Columns: region, start_date, end_date,
        combined_{u}, combined_{u}_errors, altimetry_{u}, altimetry_{u}_errors,
        gravimetry_{u}, gravimetry_{u}_errors  (u = gt or mwe).
    temporal_avg_df : pd.DataFrame | None
        Multi-year window averages.
        Columns: rgi_id, start_date, end_date, avg_mb_mwe, avg_mb_gt,
                 uncertainty_mwe, uncertainty_gt.
    out_dir : str
        Output directory (Hydra runtime.output_dir — already includes target subdir).
    """
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgi_code = rgi_df["rgi_id"].str.extract(r"RGI60-(\d+)\.")[0].iloc[0]

    # ── main_features.csv — features only, no MB columns ─────────────────
    # rgi_df already contains only configured columns (filtered by load_rgi).
    # Exclude CenLat/CenLon — already present in clim_df from climate sampling.
    rgi_static_cols = [c for c in rgi_df.columns if c not in {"CenLat", "CenLon"}]
    main_df = clim_df.merge(rgi_df[rgi_static_cols], on="rgi_id", how="left")

    n_before = len(main_df)
    main_df  = main_df.dropna()
    if n_before - len(main_df):
        log.warning(
            "Dropped %d rows with NaN climate values; %d rows remaining",
            n_before - len(main_df), len(main_df),
        )

    main_path = out_dir / f"main_features_r{rgi_code}.csv"
    main_df.to_csv(main_path, index=False)
    log.info("Wrote %s  (%d rows)", main_path, len(main_df))

    # ── per-glacier annual MB (oggm / wgms / custom) ──────────────────────
    if per_glacier_df is not None:
        pg_path = out_dir / f"{target_name}_targets_r{rgi_code}.csv"
        per_glacier_df.to_csv(pg_path, index=False)
        log.info("Wrote %s  (%d rows)", pg_path, len(per_glacier_df))

    # ── glambie_targets.csv ───────────────────────────────────────────────
    if glambie_df is not None:
        glambie_path = out_dir / f"glambie_targets_r{rgi_code}.csv"
        glambie_df.to_csv(glambie_path, index=False)
        log.info("Wrote %s  (%d rows)", glambie_path, len(glambie_df))

    # ── temporal_avg_targets.csv ──────────────────────────────────────────
    if temporal_avg_df is not None:
        temporal_path = out_dir / f"temporal_avg_targets_r{rgi_code}.csv"
        temporal_avg_df.to_csv(temporal_path, index=False)
        log.info("Wrote %s  (%d rows)", temporal_path, len(temporal_avg_df))
