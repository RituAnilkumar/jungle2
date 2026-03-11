# Extracts seasonal from monthly data and saves to new csv files. For hydrological yr and calendar yr
import os
import pandas as pd
import numpy as np
from glob import glob

monthly_fld='/scratch/b5at/ranil.b5at/data_prep_tools/data_for_model/w5e5runs/monthly/'

reg = [f"r{i:02d}" for i in range(1,20)]  # Example region subdirectories r01, r02, ..., r19
reg_subdir=[os.path.join(monthly_fld, r) for r in reg]  # Just use one region to get the file list
csv_list=glob(os.path.join(reg_subdir[0],'*.csv'))
fin_files=[os.path.join(monthly_fld,r,os.path.basename(f)) for r in reg for f in csv_list]

non_seasonal_vars=['year','rgi_id','GLIMSId','CenLat','CenLon','Slope','Aspect','Area','mass_balance','Zmed','Zmin','Zmax']
non_seasonal_nomb=['year','rgi_id','GLIMSId','CenLat','CenLon','Slope','Aspect','Area','Zmed','Zmin','Zmax']

for fin in fin_files:
    df=pd.read_csv(fin)
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    if fin.endswith('for_preds.csv'):
        fin_df=df[non_seasonal_nomb]
    else:
        fin_df=df[non_seasonal_vars]

    # # Hydrological year to be done: Oct to Sep for regions 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15. calendar yr for 16 and 1apr to 31 march for 17, 18 and 19
    ## Calendar year seasons extracted here
    if ("r17" in fin) or ("r18" in fin) or ("r19" in fin):
        abl_suffix=['10','11','12','01','02','03']
        acc_suffix=['04','05','06','07','08','09']
    else:
        abl_suffix=['05','06','07','08','09']
        acc_suffix=['10','11','12','01','02','03','04']
    tas_sel=df[[col for col in df.columns if 'tas_mean' in col]]
    rsds_sel=df[[col for col in df.columns if 'rsds_mean' in col]]
    pr_sel=df[[col for col in df.columns if 'pr_monthly_total' in col]]
    tas_abl=tas_sel[[col for col in tas_sel.columns if any(suf in col for suf in abl_suffix)]]
    tas_acc=tas_sel[[col for col in tas_sel.columns if any(suf in col for suf in acc_suffix)]]
    rsds_abl=rsds_sel[[col for col in rsds_sel.columns if any(suf in col for suf in abl_suffix)]]
    rsds_acc=rsds_sel[[col for col in rsds_sel.columns if any(suf in col for suf in acc_suffix)]]
    pr_abl=pr_sel[[col for col in pr_sel.columns if any(suf in col for suf in abl_suffix)]]
    pr_acc=pr_sel[[col for col in pr_sel.columns if any(suf in col for suf in acc_suffix)]]
    fin_df['pr_abl_sum']=pr_abl.sum(axis=1)
    fin_df['pr_acc_sum']=pr_acc.sum(axis=1)
    fin_df['tas_abl_mean']=tas_abl.mean(axis=1)
    fin_df['tas_acc_mean']=tas_acc.mean(axis=1)
    fin_df['rsds_abl_mean']=rsds_abl.mean(axis=1)
    fin_df['rsds_acc_mean']=rsds_acc.mean(axis=1)
    outfn=fin.replace('monthly','seasonal')
    if os.path.exists(os.path.dirname(outfn))==False:
        os.makedirs(os.path.dirname(outfn))
    fin_df.to_csv(outfn,index=False)
    print(f"Saved seasonal data to {outfn}")
    