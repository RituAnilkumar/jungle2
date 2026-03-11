import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import os
import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import flax.linen as nn
import jax.random as random
import optax
from flax import serialization
import orbax.checkpoint as ocp
from flax.training.orbax_utils import save_args_from_target
from flax.linen.initializers import lecun_normal, he_normal, glorot_normal

from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

class ANN(nn.Module):
    features: list[int]          # [hidden..., out]
    act: str = "relu"                # "relu","gelu","tanh","elu","leaky_relu","swish","sigmoid","selu"
    # use_batch_norm: bool = False
    # dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, inputs: jax.Array, is_training: bool) -> jax.Array:
        # pick activation
        acts = {
            "relu": nn.relu,
            "gelu": nn.gelu,
            "tanh": nn.tanh,
            "elu": nn.elu,
            "leaky_relu": nn.leaky_relu,
            "swish": nn.swish,
            "sigmoid": nn.sigmoid,
            "selu": nn.selu,
        }
        act_fn = acts.get(self.act, nn.relu) # Sets the activation in config and defaults to nn.relu as a fallback

        # derive local flags to avoid mutating self
        selu_mode = (self.act == "selu")
        sig_mode = ((self.act == "sigmoid") or (self.act == "tanh"))
        # use_bn = (self.use_batch_norm and not selu_mode)
        # drop_rate = 0.0 if selu_mode else self.dropout_rate

        # init choice (LeCun pairs best with SELU)
        if selu_mode:
            w_init = lecun_normal()
        elif sig_mode:
            w_init = glorot_normal()
        else:
            w_init = he_normal()

        x = inputs
        L = len(self.features)
        for i, ft in enumerate(self.features):
            x = nn.Dense(features=ft, kernel_init=w_init)(x)
            if i < L - 1:
                # if use_bn:
                #     x = nn.BatchNorm(
                #         use_running_average=not is_training,
                #         momentum=0.9, epsilon=1e-5
                #     )(x)
                x = act_fn(x)  # keep output layer linear for regression
                # if drop_rate > 0:
                #     x = nn.Dropout(rate=drop_rate, deterministic=not is_training)(x)
        return x
    
def tabular_dataloader(X:jax.Array,y:jax.Array,if_batching:bool,batch_size:int,drop_last:bool):
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

def fit_standard_scaler_np(X_np, eps=1e-8):
    mean = X_np.mean(axis=0, dtype=np.float64)
    std  = X_np.std(axis=0, dtype=np.float64)
    std  = np.where(std < eps, eps, std)
    return mean.astype(np.float32), std.astype(np.float32)

def transform_standard_np(X_np, mean, std):
    return (X_np - mean) / std

def inverse_standard_np(Xs_np, mean, std):
    return Xs_np * std + mean

def load_simplified_model_config(config: DictConfig) -> None:
    """Load simplified model configuration from a YAML file.

    Args:
        config (DictConfig): Configuration dictionary loaded from YAML.

    Returns:
        pd.DataFrame: DataFrame containing the model configuration.
    """
    out_dir = HydraConfig.get().runtime.output_dir
    training_mode=config.training_mode
    print(f"Training mode: {training_mode}")

    # Data info
    target_path=config.target_path_oggm
    inp_ft_mode= config.inp_ft_mode
    inp_ft_path= config.inp_ft_path
    inp_fts= config.input_features

    # Model info
    act= config.act
    hidden_layer_size= config.hidden_layer_size
    solver= config.solver
    use_scheduler= config.use_scheduler
    clip_grad= config.clip_grad
    learning_rate_init= config.learning_rate_init
    max_iter= config.max_iter
    batch_size= config.batch_size
    
    # Read Data
    target_df_full=pd.read_csv(target_path)
    input_features_df_full=pd.read_csv(inp_ft_path)

    # Pick input features from list specified
    stat_ft_list=["year","Area","cenlon","cenlat","Zmin","Zmax","Zmed","Slope","Aspect","NDSI","perc_deb"]
    met_ft_list=["temperature_2m","temperature_2m_max","temperature_2m_min","surface_solar_radiation_downwards_sum","surface_solar_radiation_downwards_min","surface_solar_radiation_downwards_max","snowfall_sum","total_precipitation_sum","u_component_of_wind_10m","v_component_of_wind_10m","surface_net_solar_radiation_sum","surface_net_solar_radiation_min","surface_net_solar_radiation_max"]
    # inp_ft_list=stat_ft_list+met_ft_list
    fin_inp_list=['rgi_id','year','cenlon','cenlat']
    if inp_ft_mode == "monthly_calendar":
        for inp in inp_fts:
            if inp in stat_ft_list:
                fin_inp_list.append(inp)
            else:
                inp_monthly=[f"{inp}_{m:02d}" for m in range(1,13)]
                fin_inp_list.extend(inp_monthly)
        inp_ft_df=input_features_df_full[fin_inp_list]

    # Drop nulls in targets and inputs
    target_df_full['year']=target_df_full['Year']#To have consistent column names for merging
    target_df_full=target_df_full.drop(columns=['Year'])
    merged_df=pd.merge(target_df_full,inp_ft_df,on=['rgi_id','year'],how='inner')
    merged_df=merged_df.dropna(subset=['mass_balance'])
    target_df_full=merged_df[['rgi_id','year','mass_balance']]
    inp_ft_df=merged_df.drop(columns=['mass_balance'])
    #Ensure inputs and targets are sorted by rgi_id and year
    target_df_full=target_df_full.sort_values(by=['rgi_id','year']).reset_index(drop=True)
    inp_ft_df=inp_ft_df.sort_values(by=['rgi_id','year']).reset_index(drop=True)

    # Scale inputs and targets
    X_np=inp_ft_df.drop(columns=['rgi_id','year']).to_numpy(dtype=np.float32)
    y_np=target_df_full['mass_balance'].to_numpy(dtype=np.float32)
    # Extract means and stds
    x_mean, x_std = fit_standard_scaler_np(X_np)
    y_mean, y_std = fit_standard_scaler_np(y_np)
    # Transform inputs and targets
    X_np = transform_standard_np(X_np, x_mean, x_std)
    y_np = transform_standard_np(y_np, y_mean, y_std)
    tot_dataset_size=X_np.shape[0]
    
    # Save means and stds
    np.save(os.path.join(out_dir, 'x_mean.npy'), x_mean)
    np.save(os.path.join(out_dir, 'x_std.npy'), x_std)
    np.save(os.path.join(out_dir, 'y_mean.npy'), y_mean)
    np.save(os.path.join(out_dir, 'y_std.npy'), y_std)
    # For inverse transform run the following after loading the std and means
    # y_mean=np.load(os.path.join(out_dir, 'y_mean.npy'))
    # y_std=np.load(os.path.join(out_dir, 'y_std.npy'))
    # y_np = inverse_standard_np(y_np, y_mean, y_std)
    # X_mean=np.load(os.path.join(out_dir, 'x_mean.npy'))
    # X_std=np.load(os.path.join(out_dir, 'x_std.npy'))
    # X_np = inverse_standard_np(X_np, X_mean, X_std)

    # Convert each to jax array
    targets_jnp=jnp.array(y_np,dtype=jnp.float32)
    input_features_jnp=jnp.array(X_np,dtype=jnp.float32)

    # Create a model instance
    hidden_layer_size.append(1)
    model=ANN(features=hidden_layer_size,act=act)#Output layer size is 1 for regression
    #Initialize parameters for the model
    key=jax.random.PRNGKey(0)
    params=model.init(key,inputs=input_features_jnp[0],is_training=True)

    if use_scheduler:
        schedule=optax.cosine_decay_schedule(init_value=learning_rate_init,decay_steps=max_iter*(tot_dataset_size/batch_size))
    else:
        schedule=learning_rate_init

    #Select an optimizer from optax
    if solver== "adam":
        optcore=optax.adam(learning_rate=schedule)
    elif solver== "sgd":
        optcore=optax.sgd(learning_rate=schedule)
    elif solver== "adamW":
        optcore=optax.adamw(learning_rate=schedule, weight_decay=1e-4)
    else:
        raise ValueError(f"Solver {solver} not recognized. Choose from 'adamW','adam' or 'sgd'.")
    
    if clip_grad:
        optimizer=optax.chain(
            optax.clip_by_global_norm(1.0),
            optcore
        )
    else:
        optimizer=optcore
    #Initialize the optimizer state for momentum
    optstate=optimizer.init(params)

    #Write a loss function in batched mode
    def lossfn(params,inpft,targ):
        """
        Compute the batch loss as the mean of the squared errors for a given set of parameters, input features, and targets.

        Args:
            params: Model parameters
            inpft: jax.Array of input features for the batch
            targ: jax.Array of targets for the batch
            model: ANN model instance

        Returns:
            jax.Array of loss values for the entire batch
        """
        pred=model.apply(params,inpft,is_training=True).squeeze(-1) #type: ignore
        mse=jnp.mean((pred-targ)**2)
        return mse

    @jax.jit
    def train_step(params, optimizer_state, X, y):
        #Compute the value of the loss function and its gradient for each parameter (including bias)
        loss,grads = jax.value_and_grad(lossfn)(params, X, y)
        #Update the optimizer state so that momentum is accordingly updated
        updates, optimizer_state = optimizer.update(grads, optimizer_state,params=params) # Params is passed for use in weight decay based optimizers such as adamW
        #Apply the updates on the parameters
        params = optax.apply_updates(params, updates)
        return loss, params, optimizer_state

    #Write the epoch loops to max epochs.
    losses=[]
    r2_list=[]
    rmse_list=[]
    best_loss=np.inf
    patience=50
    pat=0
    for epoch in range(max_iter):
        epoch_losses=[]
        for xb,yb in tabular_dataloader(input_features_jnp,targets_jnp,if_batching=True,batch_size=batch_size,drop_last=True):
            batch_loss,params,optstate=train_step(params,optstate,xb,yb)
            epoch_losses.append(batch_loss)
        epoch_loss=jnp.mean(jnp.array(epoch_losses))
        if (epoch+1)%10==0:
            with jax.disable_jit():
                N_eff=(tot_dataset_size//batch_size)*batch_size
                yhat_scaled = model.apply(params, input_features_jnp, is_training=False).squeeze(-1) #type: ignore
                yhat_np     = np.array(yhat_scaled)
                yhat_orig   = inverse_standard_np(yhat_np, y_mean, y_std)[:N_eff]
                y_orig      = inverse_standard_np(np.array(targets_jnp), y_mean, y_std)[:N_eff]
                rmse_from_loss_orig = float(np.sqrt(epoch_loss) * float(y_std))
                print('RMSE test', rmse_from_loss_orig)
                rmse = np.sqrt(np.mean((yhat_orig - y_orig)**2))
                r2 = r2_score(y_orig, yhat_orig)
                rmse_list.append(rmse)
                r2_list.append(r2)
            print(f"Epoch {epoch+1}: loss={float(epoch_loss):.6f}, RMSE(orig units)={rmse:.4f}. R2={r2:.4f}")
        losses.append(epoch_loss)
        # (Optional) naive early-stopping on training loss
        if float(epoch_loss) + 1e-7 < best_loss:
            best_loss = float(epoch_loss)
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                print(f"Early stop at epoch {epoch+1} (best loss {best_loss:.6f}).")
                break

    plt.plot(losses)
    # plt.xlim(0,max_iter)
    # make y on the logscale
    plt.yscale('log')
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'losses.png'))
    np.save(os.path.join(out_dir, 'losses.npy'),losses)
