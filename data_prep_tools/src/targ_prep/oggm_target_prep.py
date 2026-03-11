# Preprocess by using wget to pick fixed_geom oggm data from summary file into a tmp_data folder (cd into tmp_data and then wget to create a summary folder)
# wget -r -np -nH --cut-dirs=10 -A "*fixed_geometry_mass_balance_*.csv" https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.3/elev_bands/W5E5_spinup/RGI62/b_160/L4/summary/
# Needn't do a hydra run here. Can do a direct vscode run

import os
import numpy as np
import pandas as pd

in_name = '/home/ritu/data_prep_tools/tmp_data/summary/fixed_geometry_mass_balance_' # Everything except xx.csv
out_name = 'tmp_data/target_fxgeom_oggm/r'

# Create a list of str with 0 to 19 and 2 places before the decimal
name_end=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19']
# name_end=['11','12','13','14','15','16','17','18','19']


# Open file as csv
for end in name_end:
    in_filename = in_name + end+'.csv'
    out_filename = out_name + end+'.csv'
    fxgeom_df=pd.read_csv(in_filename)
    fxgeom_melted=fxgeom_df.melt(id_vars=['Unnamed: 0'], var_name='rgi_id', value_name='mass_balance')
    sel_df=fxgeom_melted[fxgeom_melted['Unnamed: 0'] >=1979].reset_index(drop=True).rename(columns={'Unnamed: 0': 'year'})
    sel_df.to_csv(out_filename, index=False)