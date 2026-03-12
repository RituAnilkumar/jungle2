"""
BayesNF model for modeling — JAX/Flax-based spatiotemporal Bayesian Neural Field.

Wraps the bayesnf library to follow the same build/train/predict interface
as the PyTorch models so model_train.py can dispatch uniformly.

Unlike PyTorch models, BayesNF:
  - Takes DataFrames as input (not tensors)
  - Produces distributional quantile outputs natively
  - Does not use the shared training.lr / training.epochs block
  - Is best used with conformal=none (native uncertainty)

Requires: pip install bayesnf cloudpickle jax
"""

from __future__ import annotations

import logging
import os
import time

import cloudpickle
import jax
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from bayesnf.spatiotemporal import (
    BayesianNeuralFieldMAP,
    BayesianNeuralFieldVI,
    BayesianNeuralFieldMLE,
)

log = logging.getLogger(__name__)

_TRAINTYPE_MAP = {
    "MAP": BayesianNeuralFieldMAP,
    "VI":  BayesianNeuralFieldVI,
    "MLE": BayesianNeuralFieldMLE,
}


def build_model(cfg: DictConfig, feature_cols: list[str], target_col: str):
    """Instantiate the BayesNF model from the Hydra model config.

    Parameters
    ----------
    cfg : DictConfig
        The ``model`` config node (configs/model/bayesnf.yaml).
    feature_cols : list[str]
        Feature column names. year must be first.
    target_col : str
        Target column name.

    Returns
    -------
    Unfitted BayesNF model instance.
    """
    traintype = cfg.model_traintype.upper()
    if traintype not in _TRAINTYPE_MAP:
        raise ValueError(f"Unsupported model_traintype '{traintype}'. Choose MAP, VI, or MLE.")

    cls = _TRAINTYPE_MAP[traintype]
    standardize_fts = [col for col in feature_cols if col != "year"]

    model = cls(
        width=int(cfg.model_nhidden),
        depth=int(cfg.model_nlayers),
        freq=cfg.model_freq,
        feature_cols=feature_cols,
        target_col=target_col,
        observation_model=cfg.get("model_obsmode", "NORMAL"),
        standardize=standardize_fts,
        timetype="index",
        seasonality_periods=list(cfg.seasonality_periods),
        num_seasonal_harmonics=list(cfg.num_seasonal_harmonics),
    )
    log.info("Built BayesNF (%s) | layers=%d width=%d", traintype, cfg.model_nlayers, cfg.model_nhidden)
    return model


def train(model, train_df: pd.DataFrame, cfg: DictConfig):
    """Fit BayesNF on the training DataFrame.

    Parameters
    ----------
    model : BayesNF model instance
        From :func:`build_model`.
    train_df : pd.DataFrame
        Training data with year, feature_cols, and target_col.
    cfg : DictConfig
        The ``model`` config node.

    Returns
    -------
    Fitted BayesNF model.
    """
    log.info("JAX devices: %s", jax.devices())
    start = time.time()

    model = model.fit(
        train_df,
        ensemble_size=int(cfg.model_nensemble),
        num_epochs=int(cfg.model_nepochs),
        seed=jax.random.PRNGKey(0),
    )
    jax.block_until_ready(model)
    log.info("BayesNF training time: %.1f s", time.time() - start)
    return model


def predict(model, df: pd.DataFrame, cfg: DictConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run BayesNF predictions and return (lower, median, upper) quantiles.

    Parameters
    ----------
    model : fitted BayesNF model
    df : pd.DataFrame
        Data to predict on (must contain feature_cols; target_col optional).
    cfg : DictConfig
        Model config node; uses output_quantiles.

    Returns
    -------
    yhat_mean : np.ndarray [N]   (median / 50th quantile)
    yhat_lower : np.ndarray [N]  (lower quantile, e.g. 2.5%)
    yhat_upper : np.ndarray [N]  (upper quantile, e.g. 97.5%)
    """
    quantiles = list(cfg.get("output_quantiles", [0.025, 0.5, 0.975]))
    _, yhat_quantiles = model.predict(df, quantiles=tuple(quantiles))

    q_idx = {q: i for i, q in enumerate(quantiles)}
    lower  = yhat_quantiles[q_idx[quantiles[0]]]
    median = yhat_quantiles[q_idx[0.5]] if 0.5 in q_idx else yhat_quantiles[q_idx[quantiles[len(quantiles)//2]]]
    upper  = yhat_quantiles[q_idx[quantiles[-1]]]

    return median, lower, upper


def save(model, path: str) -> None:
    """Serialize a fitted BayesNF model to disk with cloudpickle."""
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)
    log.info("BayesNF model saved → %s", path)


def load_checkpoint(path: str):
    """Load a fitted BayesNF model from a cloudpickle file."""
    with open(path, "rb") as f:
        model = cloudpickle.load(f)
    log.info("BayesNF checkpoint loaded ← %s", path)
    return model
