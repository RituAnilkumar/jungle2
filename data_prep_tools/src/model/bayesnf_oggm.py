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
    inp_dir=input_cfg.inp_dir
    reg_subdir=input_cfg.reg_subdir

    inp_fld=os.path.join(inp_dir,reg_subdir) #inp_dir+reg_subdir
    out_folder=HydraConfig.get().runtime.output_dir

    train_fn=inp_fld+'/train.csv'
    logo_fn=inp_fld+'/logo.csv'
    loyo_fn=inp_fld+'/loyo.csv'
    logyo_fn=inp_fld+'/loygo.csv'
    for_preds_fn=inp_fld+'/for_preds.csv'
    full_fn=inp_fld+'/full.csv'

    # Read data
    train_df=pd.read_csv(train_fn)
    logo_df=pd.read_csv(logo_fn)
    loyo_df=pd.read_csv(loyo_fn)
    logyo_df=pd.read_csv(logyo_fn)
    for_preds_df=pd.read_csv(for_preds_fn)
    full_df = pd.read_csv(full_fn)

    # # If any of the values are 0, replace with 0.1
    # num_cols_train = train_df.select_dtypes(include='number').columns
    # train_df[num_cols_train] = train_df[num_cols_train].replace(0, 0.1)
    # num_cols_logo = logo_df.select_dtypes(include='number').columns
    # logo_df[num_cols_logo]=logo_df[num_cols_logo].replace(0,0.1)
    # num_cols_loyo = loyo_df.select_dtypes(include='number').columns
    # loyo_df[num_cols_loyo]=loyo_df[num_cols_loyo].replace(0,0.1)
    # num_cols_logyo = logyo_df.select_dtypes(include='number').columns
    # logyo_df[num_cols_logyo]=logyo_df[num_cols_logyo].replace(0,0.1)
    # num_cols_preds = for_preds_df.select_dtypes(include='number').columns
    # for_preds_df[num_cols_preds]=for_preds_df[num_cols_preds].replace(0,0.1)
    # num_cols_full = full_df.select_dtypes(include='number').columns
    # full_df[num_cols_full]=full_df[num_cols_full].replace(0,0.1)


    if 'Year' in train_df.columns:
        train_df['year'] = train_df['Year']
        logo_df['year'] = logo_df['Year']
        logyo_df['year'] = logyo_df['Year']
        loyo_df['year'] = loyo_df['Year']
        for_preds_df['year'] = for_preds_df['Year']
        full_df['year'] = full_df['Year']



    # Convert the column Year in date time format if not already in pandas datetime format
    if type(train_df['year'][0]) == np.int64:
        train_df['year'] = pd.to_datetime(train_df['year'], format='%Y')#annual and monthly format
    else:
        train_df['year'] = pd.to_datetime(train_df['year'], format="%Y-%m-%d")#seasonal format
    if type(logo_df['year'][0]) == np.int64:
        logo_df['year'] = pd.to_datetime(logo_df['year'], format='%Y')
    else:
        logo_df['year'] = pd.to_datetime(logo_df['year'], format="%Y-%m-%d")
    if type(loyo_df['year'][0]) == np.int64:
        loyo_df['year'] = pd.to_datetime(loyo_df['year'], format='%Y')
    else:
        loyo_df['year'] = pd.to_datetime(loyo_df['year'], format="%Y-%m-%d")
    if type(logyo_df['year'][0]) == np.int64:
        logyo_df['year'] = pd.to_datetime(logyo_df['year'], format='%Y')
    else:
        logyo_df['year'] = pd.to_datetime(logyo_df['year'], format="%Y-%m-%d")
    if type(for_preds_df['year'][0]) == np.int64:
        for_preds_df['year'] = pd.to_datetime(for_preds_df['year'], format='%Y')
    else:
        for_preds_df['year'] = pd.to_datetime(for_preds_df['year'], format="%Y-%m-%d")
    if type(full_df['year'][0]) == np.int64:
        full_df['year'] = pd.to_datetime(full_df['year'], format='%Y')
    else:
        full_df['year'] = pd.to_datetime(full_df['year'], format="%Y-%m-%d")

    ## Comment the following if not running the backup run
    #train_df=full_df


    # Set up model configuration from config
    model_nlayers=input_cfg.model_nlayers
    model_nhidden=input_cfg.model_nhidden
    model_freq=input_cfg.model_freq
    model_nepochs=input_cfg.model_nepochs
    model_nensemble=input_cfg.model_nensemble
    model_ftcols1=input_cfg.model_ftcols
    model_rmcols=input_cfg.rm_fts
    model_ftcols=[col for col in model_ftcols1 if col not in model_rmcols]
    model_targcols=input_cfg.model_targcols
    model_obsmode=input_cfg.model_obsmode
    standardize_fts=[col for col in model_ftcols if col!='year']
    seasonality_periods=input_cfg.seasonality_periods # Will be included only in monthly runs
    num_seasonal_harmonics=input_cfg.num_seasonal_harmonics
    model_traintype=input_cfg.model_traintype


    if model_traintype=='MAP':
        model=BayesianNeuralFieldMAP(width=model_nhidden,depth=model_nlayers,freq=model_freq,feature_cols=model_ftcols,target_col=model_targcols,observation_model=model_obsmode,standardize=standardize_fts,timetype='index',seasonality_periods=['Y'],num_seasonal_harmonics=[0.1])
    elif model_traintype=='VI':
        model=BayesianNeuralFieldVI(width=model_nhidden,depth=model_nlayers,freq=model_freq,feature_cols=model_ftcols,target_col=model_targcols,observation_model=model_obsmode,timetype='index',standardize=standardize_fts,seasonality_periods=['Y'],num_seasonal_harmonics=[0.1])
    elif model_traintype=='MLE':
        model=BayesianNeuralFieldMLE(width=model_nhidden,depth=model_nlayers,freq=model_freq,feature_cols=model_ftcols,target_col=model_targcols,observation_model=model_obsmode,standardize=standardize_fts,timetype='index',seasonality_periods=['Y'],num_seasonal_harmonics=[0.1])
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

    # Compute metrics for all splits
    def compute_metrics(
        df: pd.DataFrame,
        y_true_col: str = "true_mass_balance",
        y_med_col: str = "yhat_50",
        y_lower_col: str = "yhat_025",
        y_upper_col: str = "yhat_975",
    ) -> dict:
        """Compute deterministic + uncertainty metrics from a prediction dataframe."""
        y_true = df[y_true_col].to_numpy()
        y_med  = df[y_med_col].to_numpy()

        # Point metrics
        rmse = float(np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_med)))
        mae  = float(median_absolute_error(y_true=y_true, y_pred=y_med))
        r2   = float(r2_score(y_true=y_true, y_pred=y_med))

        # Regression of truth vs prediction (matches your current linregress usage)
        slope, intercept, rval, pval, std_err = linregress(y_med, y_true)

        # Uncertainty metrics (95% PI)
        y_lower = df[y_lower_col].to_numpy()
        y_upper = df[y_upper_col].to_numpy()

        interval_widths = y_upper - y_lower
        median_interval_width95 = float(np.median(interval_widths))
        coverage_95 = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)) * 100.0)

        return {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Slope": float(slope),
            "Intercept": float(intercept),
            "Rval": float(rval),
            "Pval": float(pval),
            "Std_err": float(std_err),
            "Median_Interval_Width_95": median_interval_width95,
            "Coverage_95_pct": coverage_95,
        }


    # --- Use the same functional form for every split ---
    splits = {
        "Train": yhat_quantiles_train_df,
        "LOGO":  yhat_quantiles_logo_df,
        "LOYO":  yhat_quantiles_loyo_df,
        "LOGYO": yhat_quantiles_logyo_df,
    }

    rows = []
    for name, df in splits.items():
        row = {"Test": name}
        row.update(compute_metrics(df))
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    print(metrics_df)
    metrics_df.to_csv(os.path.join(out_folder, "metrics.csv"), index=False)
