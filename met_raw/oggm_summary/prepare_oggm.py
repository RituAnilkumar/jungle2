import os
import numpy as np
import pandas as pd
import glob

in_name = '/scratch/b5at/ranil.b5at/met_raw/oggm_summary/L5/summary/fixed_geometry_mass_balance_' # Everything except xx.csv
out_name = '/scratch/b5at/ranil.b5at/met_raw/oggm_summary/L5/target_fxgeom_oggm/r'

# Create a list of str with 0 to 19 and 2 places before the decimal
name_end=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']
# name_end=['11','12','13','14','15','16','17','18','19']


# Open file as csv
for end in name_end:
    in_filename = in_name + end+'.csv'
    out_filename = out_name + end+'.csv'
    try:
        fxgeom_df = pd.read_csv(in_filename, encoding="utf-8")
    except UnicodeDecodeError:
        fxgeom_df = pd.read_csv(in_filename, encoding="latin1")  # fallback
    # fxgeom_df=pd.read_csv(in_filename)
    fxgeom_melted=fxgeom_df.melt(id_vars=['Unnamed: 0'], var_name='rgi_id', value_name='mass_balance')
    sel_df=fxgeom_melted.reset_index(drop=True).rename(columns={'Unnamed: 0': 'year'})
    
    # Combine with RGI
    rgi_fn=glob.glob(f'/scratch/b5at/ranil.b5at/met_raw/rgi/nsidc0770_00.rgi60.attribs/{end}_rgi60*.csv')[0]
    # rgi_df=pd.read_csv(rgi_fn)
    try:
        rgi_df = pd.read_csv(rgi_fn, encoding="utf-8")
    except UnicodeDecodeError:
        rgi_df = pd.read_csv(rgi_fn, encoding="latin1")  # fallback
    merged_df=sel_df.merge(rgi_df[['RGIId', 'GLIMSId', 'CenLon', 'CenLat', 'Slope', 'Area', 'Aspect', 'Zmed', 'Zmin', 'Zmax']], left_on='rgi_id', right_on='RGIId', how='left').drop(columns=['RGIId'])
    print(f'Wrote {out_filename}')
    # Save
    merged_df.to_csv(out_filename, index=False)