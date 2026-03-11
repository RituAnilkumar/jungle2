import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

def run_target_prep(targ_cfg: DictConfig) -> pd.DataFrame:
    rgi_reg=targ_cfg.rgi_reg
    out_folder=targ_cfg.out_folder

    # Boolean flags for inclusion of data
    oggm_compute=targ_cfg.oggm_compute
    glam_alt_compute=targ_cfg.glam_alt_compute
    glam_grav_compute=targ_cfg.glam_grav_compute
    hug_20yr_compute=targ_cfg.hug_20yr_compute
    hug_roll_compute=targ_cfg.hug_roll_compute
    glam_inputs_compute=targ_cfg.glam_inputs_compute

    