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

IMPORTANT: Run met_prep before gla_prep. met_prep produces a single
combined NetCDF (NH and SH aggregated per grid cell). Set
met_input_base_path in config.yaml to point to that file (without .nc).
"""

from __future__ import annotations

import logging

import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.rgi import load_rgi
from src.climate_sampler import sample_climate, get_met_years
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

    # ── Build full glacier × year index from met file ─────────────────────
    years   = get_met_years(cfg)
    rgi_coords = rgi_df[["rgi_id", "CenLat", "CenLon"]].copy()
    rgi_coords["_key"] = 1
    years_df = pd.DataFrame({"year": years, "_key": 1})
    gla_df = rgi_coords.merge(years_df, on="_key").drop(columns="_key")
    log.info(
        "Feature index: %d glaciers × %d years = %d rows",
        rgi_df["rgi_id"].nunique(), len(years), len(gla_df),
    )

    # ── Sample climate features for all glaciers × all years ─────────────
    clim_df = sample_climate(gla_df, cfg)

    # ── Load target ───────────────────────────────────────────────────────
    if target_name not in _TARGET_LOADERS:
        raise ValueError(
            f"Unknown target '{target_name}'. "
            f"Choose from: {list(_TARGET_LOADERS.keys())}"
        )
    mod = importlib.import_module(_TARGET_LOADERS[target_name])

    per_glacier_df  = None   # oggm / wgms / custom per-glacier annual MB
    glambie_df      = None   # regional sums
    temporal_avg_df = None   # multi-year window averages

    if target_name in ("oggm", "wgms"):
        per_glacier_df = mod.load(cfg.target, rgi_code)
    elif target_name == "glambie":
        glambie_df = mod.load(cfg.target, rgi_code)
    elif target_name == "temporal_avg":
        temporal_avg_df, _ = mod.load(cfg.target, rgi_code)
        if cfg.target.get("glambie_path", None):
            import src.targets.glambie as glambie_mod
            from omegaconf import OmegaConf
            glambie_cfg = OmegaConf.create({
                "data_path":        cfg.target.glambie_path,
                "region_col":       "region",
                "year_col":         "year",
                "regional_sum_col": "regional_sum",
                "uncertainty_col":  "uncertainty",
            })
            glambie_df = glambie_mod.load(glambie_cfg, rgi_code)
    elif target_name == "custom":
        if cfg.target.get("is_regional", False):
            glambie_df = mod.load(cfg.target, rgi_code)
        else:
            per_glacier_df = mod.load(cfg.target, rgi_code)

    # ── Join and save ─────────────────────────────────────────────────────
    out_dir = HydraConfig.get().runtime.output_dir
    join_and_save(
        rgi_df, clim_df,
        target_name, per_glacier_df, glambie_df, temporal_avg_df,
        out_dir,
    )
    log.info("gla_prep complete for region r%s.", rgi_code)


if __name__ == "__main__":
    main()
