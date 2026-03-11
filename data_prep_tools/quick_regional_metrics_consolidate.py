# Run standalone changing input directory to create consolidated metrics for all regions
import numpy as np
import pandas as pd
import os
from glob import glob

inp_dir='/scratch/b5at/ranil.b5at/data_prep_tools/fixed_64hn_2l_10ens_50000ep_allreg/'
out_fn='/scratch/b5at/ranil.b5at/data_prep_tools/fixed_64hn_2l_10ens_50000ep_allreg/consolidated_metrics.csv'

# Pick a metrics.csv file from 1 subdirectory down
for i,fn in enumerate(glob(inp_dir+'*/metrics.csv')):
    reg=fn.split('/')[-2]
    df=pd.read_csv(fn)
    df['region']=reg
    if i==0:
        df_out=df
    else:
        df_out=pd.concat([df_out, df], axis=0)
df_out.to_csv(out_fn, index=False)