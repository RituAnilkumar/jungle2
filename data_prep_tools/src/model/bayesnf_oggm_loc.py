import warnings
warnings.filterwarnings("ignore")                # blanket ignore
warnings.simplefilter("ignore", FutureWarning)   # belt-and-suspenders
warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np
import xarray as xr
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

import geopandas as gpd
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
import jax
from jax.extend import backend
from bayesnf.spatiotemporal import BayesianNeuralFieldMAP
from bayesnf.spatiotemporal import BayesianNeuralFieldVI
from bayesnf.spatiotemporal import BayesianNeuralFieldMLE
import time
import cloudpickle

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

def train_oggm_bayesnf(input_cfg: DictConfig) -> None:
    # print device being used by jax
    print(jax.devices())
    # Set input and  output folders and files
    inp_fld=input_cfg.inp_dir
    out_folder=HydraConfig.get().runtime.output_dir

    train_fn=inp_fld+'/train.csv'
    logo_fn=inp_fld+'/logo.csv'
    loyo_fn=inp_fld+'/loyo.csv'
    logyo_fn=inp_fld+'/loygo.csv'
    for_preds_fn=inp_fld+'/for_preds.csv'

    # Read data
    train_df=pd.read_csv(train_fn)
    logo_df=pd.read_csv(logo_fn)
    loyo_df=pd.read_csv(loyo_fn)
    logyo_df=pd.read_csv(logyo_fn)
    for_preds_df=pd.read_csv(for_preds_fn)

    # Convert the column Year in date time format
    train_df['year'] = pd.to_datetime(train_df['year'], format='%Y')
    logo_df['year'] = pd.to_datetime(logo_df['year'], format='%Y')
    loyo_df['year'] = pd.to_datetime(loyo_df['year'], format='%Y')
    logyo_df['year'] = pd.to_datetime(logyo_df['year'], format='%Y')
    for_preds_df['year'] = pd.to_datetime(for_preds_df['year'], format='%Y')

    # Set up model configuration from config
    model_nlayers=input_cfg.model_nlayers
    model_nhidden=input_cfg.model_nhidden
    model_freq=input_cfg.model_freq
    model_nepochs=input_cfg.model_nepochs
    model_nensemble=input_cfg.model_nensemble
    model_ftcols=input_cfg.model_ftcols
    model_targcols=input_cfg.model_targcols
    model_obsmode=input_cfg.model_obsmode
    standardize_fts=input_cfg.standardize_fts
    seasonality_periods=input_cfg.seasonality_periods
    num_seasonal_harmonics=input_cfg.num_seasonal_harmonics
    model_traintype=input_cfg.model_traintype

    print('model_traintype',model_traintype)
    print('model_nlayers',model_nlayers)
    print('model_nhidden',model_nhidden)
    print('model_freq',model_freq)
    print('model_nepochs',model_nepochs)
    print('model_nensemble',model_nensemble)
    print('model_ftcols',model_ftcols)
    print('model_targcols',model_targcols)
    print('model_obsmode',model_obsmode)
    print('standardize_fts',standardize_fts)
    print('seasonality_periods',seasonality_periods)
    print('num_seasonal_harmonics',num_seasonal_harmonics)

    if model_traintype=='MAP':
        model=BayesianNeuralFieldMAP(width=model_nhidden,depth=model_nlayers,freq=model_freq,feature_cols=model_ftcols,target_col=model_targcols,observation_model=model_obsmode,standardize=standardize_fts,seasonality_periods=seasonality_periods,num_seasonal_harmonics=num_seasonal_harmonics)
    elif model_traintype=='VI':
        model=BayesianNeuralFieldVI(width=model_nhidden,depth=model_nlayers,freq=model_freq,feature_cols=model_ftcols,target_col=model_targcols,observation_model=model_obsmode,standardize=standardize_fts,seasonality_periods=seasonality_periods,num_seasonal_harmonics=num_seasonal_harmonics)
    elif model_traintype=='MLE':
        model=BayesianNeuralFieldMLE(width=model_nhidden,depth=model_nlayers,freq=model_freq,feature_cols=model_ftcols,target_col=model_targcols,observation_model=model_obsmode,standardize=standardize_fts,seasonality_periods=seasonality_periods,num_seasonal_harmonics=num_seasonal_harmonics)
    else:
        raise ValueError(f"Unsuported train type: {model_traintype}")

    # Train the model
    start = time.time()
    model=model.fit(train_df,ensemble_size=model_nensemble,num_epochs=model_nepochs,seed=jax.random.PRNGKey(0))
    jax.block_until_ready(model)  # waits for GPU work to finish
    end = time.time()
    print(f"Training time: {end - start}")

    # Save the model
    with open(os.path.join(out_folder,"model.pkl"), "wb") as f:
        cloudpickle.dump(model, f)

    # Inspect training losses
    losses=np.row_stack(model.losses_) # type: ignore
    fig, ax = plt.subplots(figsize=(5, 3), tight_layout=True)
    ax.plot(losses.T)
    ax.plot(np.mean(losses, axis=0), color='k', linewidth=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Negative Joint Probability')
    ax.set_yscale('log', base=10)
    fig.savefig(os.path.join(out_folder,'training_loss.png'),dpi=300)

    # Check whether training was successful by checking performance of the training set alone
    # Run the prediction
    yhat_train, yhat_quantiles_train = model.predict(train_df, quantiles=(0.025, 0.5, 0.975))
    # Write training outputs to csv files
    yhat_quantiles_train_df=pd.DataFrame({'yhat_025':yhat_quantiles_train[0],'yhat_50':yhat_quantiles_train[1],'yhat_975':yhat_quantiles_train[2],'true_mass_balance':train_df.mass_balance,'rgi_id':train_df.rgi_id,'year':train_df.year})
    yhat_quantiles_train_df.to_csv(os.path.join(out_folder,'preds_training.csv'),index=False)

    # Run validation for loyo_df
    yhat_logo, yhat_quantiles_logo = model.predict(logo_df, quantiles=(0.025, 0.5, 0.975))
    yhat_quantiles_logo_df=pd.DataFrame({'yhat_025':yhat_quantiles_logo[0],'yhat_50':yhat_quantiles_logo[1],'yhat_975':yhat_quantiles_logo[2],'true_mass_balance':logo_df.mass_balance,'rgi_id':logo_df.rgi_id,'year':logo_df.year})
    yhat_quantiles_logo_df.to_csv(os.path.join(out_folder,'preds_logo.csv'),index=False)

    # Run validation for loyo_df
    yhat_loyo, yhat_quantiles_loyo = model.predict(loyo_df, quantiles=(0.025, 0.5, 0.975))
    yhat_quantiles_loyo_df=pd.DataFrame({'yhat_025':yhat_quantiles_loyo[0],'yhat_50':yhat_quantiles_loyo[1],'yhat_975':yhat_quantiles_loyo[2],'true_mass_balance':loyo_df.mass_balance,'rgi_id':loyo_df.rgi_id,'year':loyo_df.year})
    yhat_quantiles_loyo_df.to_csv(os.path.join(out_folder,'preds_loyo.csv'),index=False)

    # Run validation for logyo_df
    yhat_logyo, yhat_quantiles_logyo = model.predict(logyo_df, quantiles=(0.025, 0.5, 0.975))
    yhat_quantiles_logyo_df=pd.DataFrame({'yhat_025':yhat_quantiles_logyo[0],'yhat_50':yhat_quantiles_logyo[1],'yhat_975':yhat_quantiles_logyo[2],'true_mass_balance':logyo_df.mass_balance,'rgi_id':logyo_df.rgi_id,'year':logyo_df.year})
    yhat_quantiles_logyo_df.to_csv(os.path.join(out_folder,'preds_logoy.csv'),index=False)

    # run preds for for_preds_df
    yhat_preds,yhat_quantiles_preds=model.predict(for_preds_df, quantiles=(0.025, 0.5, 0.975))
    yhat_quantiles_preds_df=pd.DataFrame({'yhat_025':yhat_quantiles_preds[0],'yhat_50':yhat_quantiles_preds[1],'yhat_975':yhat_quantiles_preds[2],'rgi_id':for_preds_df.rgi_id,'year':for_preds_df.year})
    yhat_quantiles_preds_df.to_csv(os.path.join(out_folder,'preds_full.csv'),index=False)

    # Compute rmse, median absolute error, coefficient of determination and linear regression slope and intercept for the training
    rmse_train=mean_squared_error(y_pred=yhat_quantiles_train_df['yhat_50'],y_true=yhat_quantiles_train_df['true_mass_balance'],squared=False)
    mae_train=median_absolute_error(y_pred=yhat_quantiles_train_df['yhat_50'],y_true=yhat_quantiles_train_df['true_mass_balance'])
    r2_train=r2_score(y_pred=yhat_quantiles_train_df['yhat_50'],y_true=yhat_quantiles_train_df['true_mass_balance'])
    slope_train,intercept_train,rval_train,pval_train,std_err_train=linregress(yhat_quantiles_train_df['yhat_50'],yhat_quantiles_train_df['true_mass_balance'])

    rmse_logo=mean_squared_error(y_pred=yhat_quantiles_logo_df['yhat_50'],y_true=yhat_quantiles_logo_df['true_mass_balance'],squared=False)
    mae_logo=median_absolute_error(y_pred=yhat_quantiles_logo_df['yhat_50'],y_true=yhat_quantiles_logo_df['true_mass_balance'])
    r2_logo=r2_score(y_pred=yhat_quantiles_logo_df['yhat_50'],y_true=yhat_quantiles_logo_df['true_mass_balance'])
    slope_logo,intercept_logo,rval_logo,pval_logo,std_err_logo=linregress(yhat_quantiles_logo_df['yhat_50'],yhat_quantiles_logo_df['true_mass_balance'])

    rmse_loyo=mean_squared_error(y_pred=yhat_quantiles_loyo_df['yhat_50'],y_true=yhat_quantiles_loyo_df['true_mass_balance'],squared=False)
    mae_loyo=median_absolute_error(y_pred=yhat_quantiles_loyo_df['yhat_50'],y_true=yhat_quantiles_loyo_df['true_mass_balance'])
    r2_loyo=r2_score(y_pred=yhat_quantiles_loyo_df['yhat_50'],y_true=yhat_quantiles_loyo_df['true_mass_balance'])
    slope_loyo,intercept_loyo,rval_loyo,pval_loyo,std_err_loyo=linregress(yhat_quantiles_loyo_df['yhat_50'],yhat_quantiles_loyo_df['true_mass_balance'])

    rmse_logyo=mean_squared_error(y_pred=yhat_quantiles_logyo_df['yhat_50'],y_true=yhat_quantiles_logyo_df['true_mass_balance'],squared=False)
    mae_logyo=median_absolute_error(y_pred=yhat_quantiles_logyo_df['yhat_50'],y_true=yhat_quantiles_logyo_df['true_mass_balance'])
    r2_logyo=r2_score(y_pred=yhat_quantiles_logyo_df['yhat_50'],y_true=yhat_quantiles_logyo_df['true_mass_balance'])
    slope_logyo,intercept_logyo,rval_logyo,pval_logyo,std_err_logyo=linregress(yhat_quantiles_logyo_df['yhat_50'],yhat_quantiles_logyo_df['true_mass_balance'])

    metrics_df=pd.DataFrame({'Test':['Train','LOGO','LOYO','LOGYO'],
                             'RMSE':[rmse_train,rmse_logo,rmse_loyo,rmse_logyo],
                             'MAE':[mae_train,mae_logo,mae_loyo,mae_logyo],
                             'R2':[r2_train,r2_logo,r2_loyo,r2_logyo],
                             'Slope':[slope_train,slope_logo,slope_loyo,slope_logyo],
                             'Intercept':[intercept_train,intercept_logo,intercept_loyo,intercept_logyo],
                             'Rval':[rval_train,rval_logo,rval_loyo,rval_logyo],
                             'Pval':[pval_train,pval_logo,pval_loyo,pval_logyo],
                             'Std_err':[std_err_train,std_err_logo,std_err_loyo,std_err_logyo]
                            })
    metrics_df.to_csv(os.path.join(out_folder,'metrics.csv'),index=False)