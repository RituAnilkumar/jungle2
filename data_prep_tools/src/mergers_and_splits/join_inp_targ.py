# can run standalone. No requirement for main as cfg setup not complete
import numpy as np
import pandas as pd
import os
from glob import glob

met_path='/home/ritu/met_raw/pergla_monthlymetw5e5_sampled_csv/'
targ_path='tmp_data/target_fxgeom_oggm'

out_path='data_for_model/w5e5runs/monthly'

# reg_list=['r01','r02','r03','r04','r05','r06','r07','r08','r09','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19'] 
reg_list=['r14','r15','r16','r17','r18','r19']
reg_metfn=[f'{i:02}*.csv' for i in range(14,20)]
# reg_list=['r06']
# reg_metfn=['06*.csv']

for reg,metfn in zip(reg_list,reg_metfn):
    # Check if output path exists, if not create
    if not os.path.exists(os.path.join(out_path,f'{reg}')):
        os.makedirs(os.path.join(out_path,f'{reg}'))
    met_fn_pattern = os.path.join(met_path, metfn)
    met_fn=glob(met_fn_pattern)[0]
    targ_fn=os.path.join(targ_path,reg+'.csv')
    met_df=pd.read_csv(met_fn)
    targ_df=pd.read_csv(targ_fn)
    df_full_null=pd.merge(met_df, targ_df, on=['rgi_id','year'])
    df_full=df_full_null.dropna()
    df_full.to_csv(os.path.join(out_path,f'{reg}/full.csv'), index=False)
    # split into train and test: loyo+logo
    yr_un=df_full['year'].unique()
    gla_un=df_full['rgi_id'].unique()
    # select random 10% of years
    yr_loyo=np.random.choice(yr_un, size=int(np.ceil(len(yr_un)*0.1)), replace=False)
    gla_logo=np.random.choice(gla_un, size=int(np.ceil(len(gla_un)*0.1)), replace=False)
    df_loyo_logo=df_full[(df_full['year'].isin(yr_loyo)) & (df_full['rgi_id'].isin(gla_logo))]
    df_loyo_logo.to_csv(os.path.join(out_path,f'{reg}/loygo.csv'), index=False)
    df_loyo=df_full[(df_full['year'].isin(yr_loyo)) & ~(df_full['rgi_id'].isin(gla_logo))]
    df_loyo.to_csv(os.path.join(out_path,f'{reg}/loyo.csv'), index=False)
    df_logo=df_full[~(df_full['year'].isin(yr_loyo)) & (df_full['rgi_id'].isin(gla_logo))]
    df_logo.to_csv(os.path.join(out_path,f'{reg}/logo.csv'), index=False)
    df_train=df_full[~(df_full['year'].isin(yr_loyo)) & ~(df_full['rgi_id'].isin(gla_logo))]
    df_train.to_csv(os.path.join(out_path,f'{reg}/train.csv'), index=False)
    # save each version
    for_preds=df_full.drop('mass_balance', axis=1)
    for_preds.to_csv(os.path.join(out_path,f'{reg}/for_preds.csv'), index=False)
    print(f"{reg} done with {df_full.shape} shape")