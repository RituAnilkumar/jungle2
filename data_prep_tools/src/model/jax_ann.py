import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.random as random
import optax
from flax import serialization
import orbax.checkpoint as ocp
from flax.training.orbax_utils import save_args_from_target

from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

# from jax.experimental import datasets
def tabular_dataloader(X: jax.Array, y: jax.Array, if_batching: bool = True, batch_size: int = 32, drop_last: bool = True):
    """
    Create batches from X and y.

    Args:
        X: jax.Array of input features
        y: jax.Array of targets
        batch_size: batch size used in mini batch based training
        drop_last: bool, whether to drop the last incomplete batch. Use True for training and False for evaluation/testing

    Yields:
        tuple of jax.Array, containing the next batch of input features and targets
    """
    if not if_batching:
        yield X,y
        return#This is to stop the iteration if batching is not required
    n=len(X)
    for start_indx in range(0,n,batch_size):
        X_batch=X[start_indx:start_indx+batch_size]
        y_batch=y[start_indx:start_indx+batch_size]
        if drop_last and len(X_batch)<batch_size:
            break
        yield X_batch,y_batch#yield is used instead of return as it is a generator that saves the state of the iteration.

class ANN(nn.Module):
    features: list[int]#List of hidden layer and output layer sizes

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Forward pass of the ANN model.

        Args:
            inputs: jax.Array of input features for that data

        Returns:
            jax.Array output of the model forward pass
        """
        x=inputs
        for i,ft in enumerate(self.features):
            x=nn.Dense(features=ft)(x)
            if i<len(self.features)-1:
                x=nn.relu(x)#We dont apply activation on the output layer for regression
        return x
    
def lossfn_oggm(params,inpft,targ,model):
    """
    Compute the batch loss as the mean of the squared errors for a given set of parameters, input features, and targets.

    Args:
        params: Model parameters
        inpft: jax.Array of input features for the batch
        targ: jax.Array of targets for the batch

    Returns:
        jax.Array of loss values for the entire batch
    """
    pred=model.apply(params,inpft)
    mse=jnp.mean((pred-targ)**2)
    return mse

@jax.jit
def train_step(params, optimizer, optimizer_state, X, y, model):
    #Compute the value of the loss function and its gradient for each parameter (including bias)
    loss,grads = jax.value_and_grad(lossfn_oggm)(params, X, y, model)
    #Update the optimizer state so that momentum is accordingly updated
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    #Apply the updates on the parameters
    params = optax.apply_updates(params, updates)
    return loss, params, optimizer_state

def train_oggm(X_train: pd.DataFrame, y_train: pd.DataFrame,targ_cols: list, input_cols: list, neuron_list: list, optimizer: optax.GradientTransformationExtraArgs, max_iter: int, batch_size: int) -> dict:
    """
    Trains a model on the input data
    """
    # Merge/join/combine x and y to ensure they are matched by rgi_id and year
    merged_df=pd.merge(X_train,y_train,on=['rgi_id','year'],how='inner')
    targets_jax=jnp.array(merged_df[targ_cols],dtype=jnp.float32)
    inp_features_jax=jnp.array(merged_df[input_cols],dtype=jnp.float32)

    #Create model instance
    model=ANN(features=neuron_list)
    key=jax.random.PRNGKey(0)
    params=model.init(key,inp_features_jax[0])['params']
    opt_state=optimizer.init(params)

    #Write the epoch loops to max epochs.
    losses=[]
    for epoch in range(max_iter):
        epoch_losses=[]
        print(f'Starting Epoch: {epoch+1}')
        for xb,yb in tabular_dataloader(inp_features_jax,targets_jax,if_batching=True,batch_size=batch_size,drop_last=True):
            batch_loss,params,opt_state=train_step(params,optimizer,opt_state,xb,yb,model)
            epoch_losses.append(batch_loss)
        epoch_loss=jnp.mean(jnp.array(epoch_losses))
        print(f'Completed Epoch: {epoch+1}, Loss: {epoch_loss}')
        losses.append(epoch_loss)

    plt.plot(losses)



    # Placeholder function to comply with existing structure
    pass

def prep_model(config: DictConfig):
    """
    Extracts info from config and sets up model training accordingly.
    Args:
        config: Hydra DictConfig object containing model configuration parameters
    """
    ## ---- Model configuration ---- ##
    validation_mode=config.validation_mode
    if validation_mode not in ["random", "blocking", "kfold_random", "kfold_blocking"]:
        raise ValueError(f"Unsupported validation_mode: {validation_mode}")
    
    test_mode=config.test_mode
    if test_mode not in ["random", "blocking"]:
        raise ValueError(f"Unsupported test_mode: {test_mode}")
    
    if validation_mode.startswith('kfold'):
        num_folds=config.num_folds
        if num_folds<2 or not isinstance(num_folds,int):
            raise ValueError(f"Unsupported num_folds: {num_folds}. Ensure its an integer greater than 1")
    
    if validation_mode is "random":
        val_fraction=config.val_fraction
        if val_fraction<0 or val_fraction>1:
            raise ValueError(f"Unsupported val_fraction: {val_fraction}. Ensure its between 0 and 1")
    if validation_mode is "blocking":
        blocking_perc=config.blocking_perc
        if blocking_perc<0 or blocking_perc>1:
            raise ValueError(f"Unsupported blocking_perc: {blocking_perc}. Ensure its between 0 and 1")
    
    if test_mode is "random":
        test_fraction=config.test_fraction
        if test_fraction<0 or test_fraction>1:
            raise ValueError(f"Unsupported test_fraction: {test_fraction}. Ensure its between 0 and 1")
    if test_mode is "blocking":
        blocking_perc_test=config.blocking_perc_test
        if blocking_perc_test<0 or blocking_perc_test>1:
            raise ValueError(f"Unsupported blocking_perc_test: {blocking_perc_test}. Ensure its between 0 and 1")
        
    ## ---- Model hyperparameters ---- ##
    hidden_layer_size=config.hidden_layer_size
    if not isinstance(hidden_layer_size, list) or len(hidden_layer_size)<1:
        raise ValueError(f"Unsupported hidden_layer_size: {hidden_layer_size}. Ensure its a list of integers with length >=1")
    for item in hidden_layer_size:
        if not isinstance(item, int) or item<1:
            raise ValueError(f"Unsupported hidden_layer_size: {hidden_layer_size}. Ensure its a list of integers with length >=1")
    neuron_list=hidden_layer_size.append(1)  # Output layer size is 1 for regression

    solver=config.solver
    learning_rate_init=config.learning_rate_init

    solver=config.solver
    if solver not in ["adagrad", "sgd", "adam"]:
        raise ValueError(f"Unsupported solver: {solver}")
    if solver == "adam":
        optimizer=optax.adam(learning_rate=learning_rate_init)
    elif solver == "sgd":
        optimizer=optax.sgd(learning_rate=learning_rate_init)
    elif solver == "adagrad":
        optimizer=optax.adagrad(learning_rate=learning_rate_init)
    
    max_iter=config.max_iter
    batch_size=config.batch_size
    ## ---- Extract input info from config ----- ##
    input_mode=config.input_mode
    if input_mode not in ["monthly_hydrological_nh","monthly_hydrological_sh", "monthly_calendar", "annual_hydrological_nh", "annual_hydrological_sh", "annual_calendar"]:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    
    inp_ft_path=config.inp_ft_path
    if not os.path.exists(inp_ft_path):
        raise ValueError(f"Incorrect inp_ft_path: {inp_ft_path}")
    inp_ft_df_all=pd.read_csv(inp_ft_path)

    stat_ft_list=["year","area_km2","cenlon","cenlat","zmin_m","zmax_m","zmed_m","slope_deg","aspect_deg","NDSI"]
    met_ft_list=["temperature_2m","temperature_2m_max","temperature_2m_min","surface_solar_radiation_downwards_sum","surface_solar_radiation_downwards_min","surface_solar_radiation_downwards_max","snowfall_sum","total_precipitation_sum","u_component_of_wind_10m","v_component_of_wind_10m","surface_net_solar_radiation_sum","surface_net_solar_radiation_min","surface_net_solar_radiation_max"]
    inp_ft_list=stat_ft_list+met_ft_list
    inp_fts=config.input_features
    # If any of the inp_fts is not in inp_ft_list raise error
    for inp in inp_fts:
        if inp not in inp_ft_list:
            raise ValueError(f"Unsupported input feature: {inp}")
    
    fin_inp_list=['rgi_id','year','cenlon','cenlat']
    if input_mode == "monthly_calendar":
        for inp in inp_fts:
            if inp in stat_ft_list:
                fin_inp_list.append(inp)
            else:
                inp_monthly=[f"{inp}_{m:02d}" for m in range(1,13)]
                fin_inp_list.extend(inp_monthly)
        inp_ft_df=inp_ft_df_all[fin_inp_list]
    elif input_mode == "monthly_hydrological_nh":
        prev_yr_cols=[]
        for inp in inp_fts:
            if inp in stat_ft_list:
                fin_inp_list.append(inp)
            else:
                inp_monthly=[f"{inp}_{m:02d}" for m in range(1,13)]
                fin_inp_list.extend(inp_monthly)
                prev_yr_cols.extend([f"{inp}_{m:02d}" for m in [10,11,12]])
        inp_ft_df=inp_ft_df_all[fin_inp_list].sort_values(['rgi_id','year']).copy()
        inp_ft_df[prev_yr_cols]=inp_ft_df.groupby('rgi_id')[prev_yr_cols].shift(1)
    elif input_mode == "monthly_hydrological_sh":
        prev_yr_cols=[]
        for inp in inp_fts:
            if inp in stat_ft_list:
                fin_inp_list.append(inp)
            else:
                inp_monthly=[f"{inp}_{m:02d}" for m in range(1,13)]
                fin_inp_list.extend(inp_monthly)
                prev_yr_cols.extend([f"{inp}_{m:02d}" for m in range(4,13)])
        # For each rgi_id, pick y1 months from previous year
        inp_ft_df_sorted=inp_ft_df_all.groupby('rgi_id').apply(lambda x: x.sort_values('year')).reset_index(drop=True)
        inp_ft_df_list=[]
        for rgi_id in inp_ft_df_sorted['rgi_id'].unique():
            inp_ft_df_list.append(inp_ft_df_sorted[inp_ft_df_sorted['rgi_id']==rgi_id].iloc[-12:])
        inp_ft_df=pd.concat(inp_ft_df_list).reset_index(drop=True)
    elif input_mode == "annual_calendar":
        pass
    elif input_mode == "annual_hydrological_nh":
        pass
    elif input_mode == "annual_hydrological_sh":
        pass

    ## ---- Extract target info from config ---- ##
    training_mode=config.training_mode
    if training_mode not in ["pretraining_oggm", "finetuning_hug", "finetuning_glam_altgrav", "finetuning_combined", "standalone_hug", "standalone_glam_altgrav", "standalone_combined"]:
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    if training_mode == "pretraining_oggm":
        target_path=config.pretraining_oggm.target_path
    elif training_mode == "finetuning_hug":
        pretraining_path=config.finetuning_hug.pretraining_path
        target_path=config.finetuning_hug.target_path
    elif training_mode == "finetuning_glam_altgrav":
        pretraining_path=config.finetuning_glam_altgrav.pretraining_path
        target_path_alt=config.finetuning_glam_altgrav.target_path_alt
        target_path_grav=config.finetuning_glam_altgrav.target_path_grav
    elif training_mode == "finetuning_combined":
        pretraining_path=config.finetuning_combined.pretraining_path
        target_path_hug=config.finetuning_combined.target_path_hug
        target_path_alt=config.finetuning_combined.target_path_alt
        target_path_grav=config.finetuning_combined.target_path_grav
    elif training_mode == "standalone_hug":
        target_path=config.standalone_hug.target_path
    elif training_mode == "standalone_glam_altgrav":
        target_path_alt=config.standalone_glam_altgrav.target_path_alt
        target_path_grav=config.standalone_glam_altgrav.target_path_grav
    elif training_mode == "standalone_combined":
        target_path_hug=config.standalone_combined.target_path_hug
        target_path_alt=config.standalone_combined.target_path_alt
        target_path_grav=config.standalone_combined.target_path_grav

