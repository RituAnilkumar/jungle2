# To be run standalone only one time per region to create folders r01..r19 and aggregate data from data_for_model (monthly)
import numpy as np
import pandas as pd
import os

reg_list=['r01','r02','r03','r04','r05','r06','r07','r08','r09','r10','r11','r12','r13','r14','r15','r16','r17','r18','r19']
metvarnames=['surface_solar_radiation_downwards_sum','temperature_2m', 'total_precipitation_sum'] # Change here if required
for reg in reg_list:
    reg_fld=f"data_for_model/{reg}/"
    # outreg_fldvar=f"data_for_model_annvar/{reg}/"
    outreg_fld=f"data_for_model_seasonal/{reg}/" # Change here if required
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
    print('starting loop with size',train.shape)

    for metvar in metvarnames:
        jjas_list=[f"{metvar}_{m:02}" for m in [6,7,8,9]]
        oth_list=[f"{metvar}_{m:02}" for m in [1,2,3,4,5,10,11,12]]
        sel_forpreds_jjas=for_preds[jjas_list]
        sel_full_jjas=full[jjas_list]
        sel_logo_jjas=logo[jjas_list]
        sel_loyo_jjas=loyo[jjas_list]
        sel_loygo_jjas=loygo[jjas_list]
        sel_train_jjas=train[jjas_list]
        sel_forpreds_oth=for_preds[oth_list]
        sel_full_oth=full[oth_list]
        sel_logo_oth=logo[oth_list]
        sel_loyo_oth=loyo[oth_list]
        sel_loygo_oth=loygo[oth_list]
        sel_train_oth=train[oth_list]

        if metvar in ['surface_solar_radiation_downwards_sum', 'total_precipitation_sum']:
            mean_forpreds_jjas=sel_forpreds_jjas.sum(axis=1)
            mean_full_jjas=sel_full_jjas.sum(axis=1)
            mean_logo_jjas=sel_logo_jjas.sum(axis=1)
            mean_loyo_jjas=sel_loyo_jjas.sum(axis=1)
            mean_loygo_jjas=sel_loygo_jjas.sum(axis=1)
            mean_train_jjas=sel_train_jjas.sum(axis=1)
            mean_forpreds_oth=sel_forpreds_oth.sum(axis=1)
            mean_full_oth=sel_full_oth.sum(axis=1)
            mean_logo_oth=sel_logo_oth.sum(axis=1)
            mean_loyo_oth=sel_loyo_oth.sum(axis=1)
            mean_loygo_oth=sel_loygo_oth.sum(axis=1)
            mean_train_oth=sel_train_oth.sum(axis=1)
        else:
            mean_forpreds_jjas=sel_forpreds_jjas.mean(axis=1)
            mean_full_jjas=sel_full_jjas.mean(axis=1)
            mean_logo_jjas=sel_logo_jjas.mean(axis=1)
            mean_loyo_jjas=sel_loyo_jjas.mean(axis=1)
            mean_loygo_jjas=sel_loygo_jjas.mean(axis=1)
            mean_train_jjas=sel_train_jjas.mean(axis=1)
            mean_forpreds_oth=sel_forpreds_oth.mean(axis=1)
            mean_full_oth=sel_full_oth.mean(axis=1)
            mean_logo_oth=sel_logo_oth.mean(axis=1)
            mean_loyo_oth=sel_loyo_oth.mean(axis=1)
            mean_loygo_oth=sel_loygo_oth.mean(axis=1)
            mean_train_oth=sel_train_oth.mean(axis=1)

        for_preds[f"{metvar}_jjas"] = mean_forpreds_jjas
        full[f"{metvar}_jjas"] = mean_full_jjas
        logo[f"{metvar}_jjas"] = mean_logo_jjas
        loyo[f"{metvar}_jjas"] = mean_loyo_jjas
        loygo[f"{metvar}_jjas"] = mean_loygo_jjas
        train[f"{metvar}_jjas"] = mean_train_jjas

        for_preds[f"{metvar}_oth"] = mean_forpreds_oth
        full[f"{metvar}_oth"] = mean_full_oth
        logo[f"{metvar}_oth"] = mean_logo_oth
        loyo[f"{metvar}_oth"] = mean_loyo_oth
        loygo[f"{metvar}_oth"] = mean_loygo_oth
        train[f"{metvar}_oth"] = mean_train_oth
        
        met_list=jjas_list+oth_list
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
    print(f"{reg} done with {train.shape} shape")
    print('------------------')