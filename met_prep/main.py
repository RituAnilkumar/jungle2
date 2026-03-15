"""
met_prep — entry point.

Aggregates locally stored climate data (ERA5, W5E5, CMIP) into monthly,
seasonal, or annual NetCDF outputs.

Run with Hydra:

    python main.py                                          # defaults: era5, seasonal
    python main.py dataset=w5e5 aggregation=annual_hydro
    python main.py dataset=era5 aggregation=monthly output_path=/my/outputs
    python main.py -m aggregation=monthly,seasonal,annual_hydro   # multirun

IMPORTANT: Run met_prep before gla_prep.
gla_prep samples from met_prep NetCDF outputs via xarray nearest-neighbour.
Ensure the aggregation level here matches met_input_freq in your gla_prep region config.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.loaders import load_dataset
from src.aggregators import aggregate

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Dataset : %s", cfg.dataset.name)
    log.info("Aggregation: %s", cfg.aggregation.name)

    # ── Load ──────────────────────────────────────────────────────────────
    ds = load_dataset(cfg.dataset)
    log.info("Loaded dataset with variables: %s", list(ds.data_vars))

    # ── Aggregate ─────────────────────────────────────────────────────────
    ds_agg = aggregate(ds, cfg.dataset, cfg.aggregation)
    log.info("Aggregated variables: %s", list(ds_agg.data_vars))

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(cfg.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = cfg.dataset.get("scenario", "")
    scenario_tag = f"_{scenario}" if scenario else ""
    hemisphere = cfg.aggregation.get("hemisphere", None)
    hem_tag = f"_{hemisphere.upper()}" if hemisphere and hemisphere.upper() in ("NH", "SH") else ""
    out_fn = out_dir / f"{cfg.dataset.name}{scenario_tag}_{cfg.aggregation.name}{hem_tag}.nc"

    log.info("Writing → %s", out_fn)
    ds_agg.to_netcdf(out_fn)
    log.info("Done.")


if __name__ == "__main__":
    main()
