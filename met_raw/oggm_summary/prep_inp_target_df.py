import numpy as np
import pandas as pd
import xarray as xr
import os
import glob

# gla_path='/scratch/b5at/ranil.b5at/met_raw/oggm_summary/L5/target_fxgeom_oggm/'
# reg='r06'
# gla_fn=f'{gla_path}{reg}.csv'
# gla_file=pd.read_csv(gla_fn)

# clim_file=xr.open_dataset('/scratch/b5at/ranil.b5at/met_raw/era5aggregates/era5_seasonal_accum_ablat.nc',chunks="auto")

# out_path='/scratch/b5at/ranil.b5at/met_raw/oggm_summary/L5/oggm_with_clim/'
# out_fn=f'{out_path}{reg}.csv'

def extract_clim_to_gla(gla_file, clim_file, out_fn):
    # --- Convert glacier longitudes to 0–360 ---
    gla_lons = (gla_file["CenLon"].to_numpy() + 360) % 360
    gla_lats = gla_file["CenLat"].to_numpy()
    gla_years = gla_file["year"].to_numpy()

    # --- Create index dimension for glaciers for merging with csv later---
    glacier_ids = gla_file["rgi_id"].to_numpy()

    lats = xr.DataArray(gla_lats, dims="rgi_id")
    lons = xr.DataArray(gla_lons, dims="rgi_id")
    yrs = xr.DataArray(gla_years,dims="rgi_id")

    # --- Nearest neighbour sampling ---
    sampled = clim_file.sel(
        latitude=lats,
        longitude=lons,
        season_year=yrs,
        method="nearest"
    )

    # Add glacier_id as coordinate
    sampled = sampled.assign_coords(rgi_id=("rgi_id", glacier_ids))

    # --- Convert to dataframe ---
    df = sampled.to_dataframe().reset_index()
    # Rename season year to year
    df = df.rename(columns={"season_year": "year"})

    # Optional: merge back glacier metadata if needed
    df = df.merge(gla_file[['year', 'rgi_id', 'mass_balance', 'GLIMSId', 'CenLon', 'CenLat','Slope', 'Area', 'Aspect', 'Zmed', 'Zmin', 'Zmax']], on=["rgi_id", "year"], how="left")
    # Drop 'CenLon','CenLat','GLIMSId','number','step','surface'

    df=df.drop(columns=['CenLon','CenLat','GLIMSId','number','step','surface'])
    print(f'--------- Missing values for Region {reg} ---------')
    print(df.isnull().sum())
    # Drop rows with missing values
    df = df.dropna()

    df.to_csv(out_fn, index=False)

clim_file=xr.open_dataset('/scratch/b5at/ranil.b5at/met_raw/era5aggregates/era5_seasonal_accum_ablat.nc',chunks="auto")
gla_path='/scratch/b5at/ranil.b5at/met_raw/oggm_summary/L5/target_fxgeom_oggm/'
out_path='/scratch/b5at/ranil.b5at/met_raw/oggm_summary/L5/oggm_with_clim/'

for i in range(1, 20):#Change to i,i+1 for getting only the ith region. Or select 1,20 to get all rgi
    reg=f'r{i:02}' 
    gla_fn=f'{gla_path}{reg}.csv'
    gla_file=pd.read_csv(gla_fn)
    out_fn=f'{out_path}{reg}.csv'
    extract_clim_to_gla(gla_file, clim_file, out_fn)
    # print(f'Wrote {out_fn}')