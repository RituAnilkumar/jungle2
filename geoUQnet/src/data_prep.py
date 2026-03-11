"""
Data loading, splitting and scaling for geoUQnet.

All behaviour is driven by Hydra config. No data logic lives in main.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


# ─────────────────────────── Public dataclass ────────────────────────────────

class DataBundle:
    """Holds every split as both numpy arrays and torch tensors.

    Attributes
    ----------
    Arrays (original scale):
        y_train, y_val, y_cal, y_test  — shape [N, 1]
    Tensors (standardised, on *device*):
        X_train_t, y_train_t, X_val_t, y_val_t,
        X_cal_t,   y_cal_t,   X_test_t, y_test_t
    Scalers:
        x_scaler, y_scaler  (fitted StandardScalers)
    Loader:
        train_loader  (shuffled DataLoader over the training split)
    Meta:
        in_dim        (number of input features)
        feature_cols  (list[str])
        target_col    (str)
    """

    def __init__(self) -> None:
        # populated by load_and_split()
        pass


# ─────────────────────────── Main entry-point ────────────────────────────────

def load_and_split(cfg: DictConfig, device: torch.device) -> DataBundle:
    """Load CSV, clean, split and scale data according to Hydra config.

    Expected keys in ``cfg`` (the root config):

    .. code-block:: yaml

        data:
          path: data/iceland_era5_sampled.csv
          feature_cols:
            - tp_accum_sum
            - ...
          target_col: annual_balance

        splits:
          test_size:  0.15
          cal_size:   0.20   # fraction of remaining after test split
          val_size:   0.15   # fraction of remaining after cal split
          seed:       42

        training:
          batch_size: 256

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.
    device : torch.device
        Torch device; all tensors are moved here.

    Returns
    -------
    DataBundle
        Fully populated bundle ready for training.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    path = cfg.data.path
    log.info("Loading data from '%s'", path)
    df = pd.read_csv(path)

    feature_cols: list[str] = list(cfg.data.feature_cols)
    target_col: str = cfg.data.target_col

    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=feature_cols + [target_col])
    log.info("Rows after cleaning: %d", len(df))

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    # ── 2. Split ─────────────────────────────────────────────────────────────
    seed      = int(cfg.splits.get("seed", 42))
    test_size = float(cfg.splits.get("test_size", 0.15))
    cal_size  = float(cfg.splits.get("cal_size", 0.20))
    val_size  = float(cfg.splits.get("val_size", 0.15))

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    X_trainval, X_cal, y_trainval, y_cal = train_test_split(
        X_temp, y_temp, test_size=cal_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=seed
    )

    log.info(
        "Split sizes — train: %d | val: %d | cal: %d | test: %d",
        len(X_train), len(X_val), len(X_cal), len(X_test),
    )

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    X_cal_s   = x_scaler.transform(X_cal)
    X_test_s  = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s   = y_scaler.transform(y_val)
    y_cal_s   = y_scaler.transform(y_cal)
    y_test_s  = y_scaler.transform(y_test)

    # ── 4. To torch ──────────────────────────────────────────────────────────
    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.float32)).to(device)

    batch_size = int(cfg.training.get("batch_size", 256))

    bundle = DataBundle()

    bundle.X_train_t = _t(X_train_s)
    bundle.y_train_t = _t(y_train_s)
    bundle.X_val_t   = _t(X_val_s)
    bundle.y_val_t   = _t(y_val_s)
    bundle.X_cal_t   = _t(X_cal_s)
    bundle.y_cal_t   = _t(y_cal_s)
    bundle.X_test_t  = _t(X_test_s)
    bundle.y_test_t  = _t(y_test_s)

    # keep un-scaled numpy targets for conformal scoring
    bundle.y_train   = y_train
    bundle.y_val     = y_val
    bundle.y_cal     = y_cal
    bundle.y_test    = y_test

    bundle.x_scaler  = x_scaler
    bundle.y_scaler  = y_scaler

    bundle.train_loader = DataLoader(
        TensorDataset(bundle.X_train_t, bundle.y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    bundle.in_dim       = X_train.shape[1]
    bundle.feature_cols = feature_cols
    bundle.target_col   = target_col

    return bundle