# To be run standalone only one time per region to create folders r01..r19 and aggregate data from data_for_model (monthly)
import numpy as np
import pandas as pd
import os

reg_list=['r01','r02','r03','r04','r05','r06','r07','r08','r09','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19']
metvarnames=['surface_solar_radiation_downwards_sum','temperature_2m', 'total_precipitation_sum'] # Change here if required
for reg in reg_list:
    reg_fld=f"data_for_model/{reg}/"
    # outreg_fldvar=f"data_for_model_annvar/{reg}/"
    outreg_fld=f"data_for_model_annvar/{reg}/" # Change here if required
    if not os.path.exists(outreg_fld):
        print(f"Creating {outreg_fld}")
        os.makedirs(outreg_fld)

    # Read csvs
    for_preds=pd.read_csv(reg_fld+'for_preds.csv')
    full=pd.read_csv(reg_fld+'full.csv')
    logo=pd.read_csv(reg_fld+'logo.csv')
    loyo=pd.read_csv(reg_fld+'loyo.csv')
    loygo=pd.read_csv(reg_fld+'loygo.csv')
    train=pd.read_csv(reg_fld+'train.csv')

    for metvar in metvarnames:
        met_list=[f"{metvar}_{m:02}" for m in range(1,13)]
        sel_forpreds=for_preds[met_list]
        sel_full=full[met_list]
        sel_logo=logo[met_list]
        sel_loyo=loyo[met_list]
        sel_loygo=loygo[met_list]
        sel_train=train[met_list]

        if metvar in ['surface_solar_radiation_downwards_sum', 'total_precipitation_sum']:
            mean_forpreds=sel_forpreds.sum(axis=1)
            mean_full=sel_full.sum(axis=1)
            mean_logo=sel_logo.sum(axis=1)
            mean_loyo=sel_loyo.sum(axis=1)
            mean_loygo=sel_loygo.sum(axis=1)
            mean_train=sel_train.sum(axis=1)
        else:
            mean_forpreds=sel_forpreds.mean(axis=1)
            mean_full=sel_full.mean(axis=1)
            mean_logo=sel_logo.mean(axis=1)
            mean_loyo=sel_loyo.mean(axis=1)
            mean_loygo=sel_loygo.mean(axis=1)
            mean_train=sel_train.mean(axis=1)

        std_forpreds=sel_forpreds.std(axis=1)
        std_full=sel_full.std(axis=1)
        std_logo=sel_logo.std(axis=1)
        std_loyo=sel_loyo.std(axis=1)
        std_loygo=sel_loygo.std(axis=1)
        std_train=sel_train.std(axis=1)

        for_preds[f"{metvar}_agg"] = mean_forpreds
        for_preds[f"{metvar}_std"] = std_forpreds

        full[f"{metvar}_agg"] = mean_full
        full[f"{metvar}_std"] = std_full

        logo[f"{metvar}_agg"] = mean_logo
        logo[f"{metvar}_std"] = std_logo

        loyo[f"{metvar}_agg"] = mean_loyo
        loyo[f"{metvar}_std"] = std_loyo

        loygo[f"{metvar}_agg"] = mean_loygo
        loygo[f"{metvar}_std"] = std_loygo

        train[f"{metvar}_agg"] = mean_train
        train[f"{metvar}_std"] = std_train

        # Drop met_list columns
        for_preds=for_preds.drop(met_list, axis=1)
        full=full.drop(met_list, axis=1)
        logo=logo.drop(met_list, axis=1)
        loyo=loyo.drop(met_list, axis=1)
        loygo=loygo.drop(met_list, axis=1)
        train=train.drop(met_list, axis=1)

    for_preds.to_csv(outreg_fld+'for_preds.csv', index=False)
    full.to_csv(outreg_fld+'full.csv', index=False)
    logo.to_csv(outreg_fld+'logo.csv', index=False)
    loyo.to_csv(outreg_fld+'loyo.csv', index=False)
    loygo.to_csv(outreg_fld+'loygo.csv', index=False)
    train.to_csv(outreg_fld+'train.csv', index=False)