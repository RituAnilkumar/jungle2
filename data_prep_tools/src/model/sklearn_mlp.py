import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import os

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes the mean absolute error, mean squared error and R2 score
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def train_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,config: DictConfig) -> dict:
    """
    Trains a model on the input data
    """
    num_neurons = config.hidden_layer_size
    # If num_neurons is <1 or not an integer or a list of length = num_layers, raise error
    
    activation = config.activation
    # If activation not in list, raise error
    activation_list=["identity", "logistic", "tanh", "relu"]
    if activation not in activation_list:
        raise ValueError(f"Unsuported activation: {activation}")
    
    solver = config.solver
    # If solver not in list, raise error
    solver_list=["lbfgs", "sgd", "adam"]
    if solver not in solver_list:
        raise ValueError(f"Unsuported solver: {solver}")
    
    learning_rate_init = config.learning_rate_init
    # If learning rate is negative, raise error
    if learning_rate_init < 0:
        raise ValueError(f"Unsuported learning_rate_init: {learning_rate_init}. Ensure its a positive number")
    
    alpha = config.alpha
    # If alpha is negative, raise error
    if alpha < 0:
        raise ValueError(f"Unsuported alpha: {alpha}. Ensure its a positive number")
    
    max_iter = config.max_iter
    # If max_iter is <1 and not an integer, raise error
    if max_iter < 1 or not isinstance(max_iter, int):
        raise ValueError(f"Unsuported max_iter: {max_iter}. Ensure its a natural number")
    random_state = config.random_state
    
    model = MLPRegressor(hidden_layer_sizes=num_neurons, activation=activation, solver=solver, learning_rate_init=learning_rate_init, alpha=alpha, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)

    # Get the hydra run directory
    out_dir = HydraConfig.get().runtime.output_dir
    plt.figure()
    plt.plot(model.loss_curve_)
    save_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(save_path) # Save loss_curve_
    y_pred = model.predict(X_test)
    metrics=compute_metrics(y_test, y_pred)
    return metrics
