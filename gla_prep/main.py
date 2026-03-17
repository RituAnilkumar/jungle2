"""
gla_prep — entry point.

Prepares glacier mass balance targets, samples climate features from
met_prep NetCDF outputs, merges RGI static attributes, and writes
ready-to-use CSV files for the modeling directory.

Run with Hydra:

    python main.py                                    # defaults: oggm, r06
    python main.py target=wgms region=r06
    python main.py target=temporal_avg region=r01
    python main.py -m region=r01,r02,r03              # multirun over regions

IMPORTANT: met_prep must be run twice (hemisphere=NH and hemisphere=SH)
before running gla_prep. The output files are referenced via
met_input_base_path in config.yaml. gla_prep appends _NH.nc or _SH.nc
automatically per glacier based on CenLat.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from src.rgi import load_rgi
from src.climate_sampler import sample_climate
from src.joiner import join_and_save

log = logging.getLogger(__name__)

# Target loader dispatch
_TARGET_LOADERS = {
    "oggm":         "src.targets.oggm",
    "wgms":         "src.targets.wgms",
    "glambie":      "src.targets.glambie",
    "temporal_avg": "src.targets.temporal_avg",
    "custom":       "src.targets.custom",
}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    import importlib

    target_name = cfg.target.name
    rgi_code    = cfg.region.rgi_code
    log.info("Target: %s | Region: r%s", target_name, rgi_code)

    # ── Load RGI static attributes ────────────────────────────────────────
    rgi_df = load_rgi(cfg.region, list(cfg.rgi_cols))

    # ── Load target mass balance ──────────────────────────────────────────
    if target_name not in _TARGET_LOADERS:
        raise ValueError(
            f"Unknown target '{target_name}'. "
            f"Choose from: {list(_TARGET_LOADERS.keys())}"
        )
    mod = importlib.import_module(_TARGET_LOADERS[target_name])
    target_data = mod.load(cfg.target, rgi_code)

    # Separate per-glacier MB, GLAMBIE, and temporal avg outputs
    mb_df           = None
    glambie_df      = None
    temporal_avg_df = None

    annual_df = None  # per-glacier annual table used for climate sampling index

    if target_name in ("oggm", "wgms"):
        mb_df = target_data
    elif target_name == "glambie":
        glambie_df = target_data
    elif target_name == "temporal_avg":
        temporal_avg_df, annual_df = target_data
        # Also load GLAMBIE if a path is provided
        if cfg.target.get("glambie_path", None):
            import src.targets.glambie as glambie_mod
            from omegaconf import OmegaConf
            glambie_cfg = OmegaConf.create({
                "data_path": cfg.target.glambie_path,
                "region_col": "region",
                "year_col": "year",
                "regional_sum_col": "regional_sum",
                "uncertainty_col": "uncertainty",
            })
            glambie_df = glambie_mod.load(glambie_cfg, rgi_code)
    elif target_name == "custom":
        if cfg.target.get("is_regional", False):
            glambie_df = target_data
        else:
            mb_df = target_data

    # ── Build (glacier, year) index for climate sampling ─────────────────
    import pandas as pd
    if mb_df is not None:
        gla_df = mb_df[["rgi_id", "year"]].merge(
            rgi_df[["rgi_id", "CenLat", "CenLon"]], on="rgi_id", how="inner"
        )
    elif annual_df is not None:
        # temporal_avg: derive index from the Hugonnet annual table
        gla_df = annual_df[["rgi_id", "year"]].merge(
            rgi_df[["rgi_id", "CenLat", "CenLon"]], on="rgi_id", how="inner"
        )
    else:
        raise ValueError(
            f"Target '{target_name}' does not provide a per-glacier annual "
            "table for climate sampling. For glambie targets, run gla_prep "
            "with target=oggm or target=wgms first."
        )

    # ── Sample climate features ───────────────────────────────────────────
    clim_df = sample_climate(gla_df, cfg)

    # ── Join and save ─────────────────────────────────────────────────────
    join_and_save(rgi_df, mb_df, clim_df, glambie_df, temporal_avg_df, cfg)
    log.info("gla_prep complete for region r%s.", rgi_code)


if __name__ == "__main__":
    main()
