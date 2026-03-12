"""
Data loading, splitting, scaling, and startup validation for modeling.

Driven entirely by Hydra config. Handles both random and blocked splits,
and validates that required auxiliary CSV paths are present for constrained losses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .validation.splits import random_split, blocked_split

log = logging.getLogger(__name__)


# ─────────────────────────── DataBundle ──────────────────────────────────────

class DataBundle:
    """Holds all split data as numpy arrays, torch tensors, and DataFrames.

    Attributes (always present)
    ---------------------------
    Numpy arrays (original scale):
        y_train, y_val, y_cal, y_test  — shape [N, 1]  (random split)
    Torch tensors (standardised, on device):
        X_train_t, y_train_t, X_val_t, y_val_t,
        X_cal_t,   y_cal_t,   X_test_t, y_test_t
    Scalers:
        x_scaler, y_scaler
    Loader:
        train_loader
    Meta:
        in_dim, feature_cols, target_col, validation_type

    Additional (blocked split only)
    --------------------------------
    DataFrames:
        df_train, df_logo, df_loyo, df_loygo, df_full
    Auxiliary targets (if applicable):
        glambie_df, temporal_avg_df
    """
    pass


# ─────────────────────────── Startup validation ───────────────────────────────

def _validate_loss_data(cfg: DictConfig) -> None:
    """Raise clear errors if required auxiliary CSV paths are missing."""
    loss_type = cfg.loss.type

    if loss_type in ("glambie", "glambie_and_temporal_avg"):
        path = cfg.data.get("glambie_targets_path", None)
        if not path:
            raise ValueError(
                "loss.type='{}' requires data.glambie_targets_path to be set. "
                "Run gla_prep with target=glambie first.".format(loss_type)
            )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"glambie_targets_path not found: {path}"
            )

    if loss_type in ("temporal_avg", "glambie_and_temporal_avg"):
        path = cfg.data.get("temporal_avg_targets_path", None)
        if not path:
            raise ValueError(
                "loss.type='{}' requires data.temporal_avg_targets_path to be set. "
                "Run gla_prep with target=temporal_avg first.".format(loss_type)
            )
        if not Path(path).exists():
            raise FileNotFoundError(
                f"temporal_avg_targets_path not found: {path}"
            )


# ─────────────────────────── Main entry ──────────────────────────────────────

def load_and_split(cfg: DictConfig, device: torch.device) -> DataBundle:
    """Load CSVs, validate config, split, and scale data.

    Parameters
    ----------
    cfg : DictConfig
        Root Hydra config.
    device : torch.device

    Returns
    -------
    DataBundle
    """
    # ── Startup validation ────────────────────────────────────────────────
    _validate_loss_data(cfg)

    # ── Load main features CSV ────────────────────────────────────────────
    path = cfg.data.main_features_path
    log.info("Loading main features: %s", path)
    df = pd.read_csv(path)
    df = df.replace([float("inf"), float("-inf")], float("nan"))

    feature_cols:   list[str] = list(cfg.data.feature_cols)
    target_col:     str       = cfg.data.target_col
    glacier_id_col: str       = cfg.data.get("glacier_id_col", "rgi_id")
    year_col:       str       = cfg.data.get("year_col", "year")

    df = df.dropna(subset=feature_cols + [target_col])
    log.info("Rows after cleaning: %d", len(df))

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    # ── Load auxiliary targets if needed ─────────────────────────────────
    glambie_df      = None
    temporal_avg_df = None
    loss_type       = cfg.loss.type

    if loss_type in ("glambie", "glambie_and_temporal_avg"):
        glambie_df = pd.read_csv(cfg.data.glambie_targets_path)
        log.info("Loaded GLAMBIE targets: %d rows", len(glambie_df))

    if loss_type in ("temporal_avg", "glambie_and_temporal_avg"):
        temporal_avg_df = pd.read_csv(cfg.data.temporal_avg_targets_path)
        log.info("Loaded temporal avg targets: %d rows", len(temporal_avg_df))

    # ── Split ─────────────────────────────────────────────────────────────
    val_type = cfg.validation.type
    bundle   = DataBundle()
    bundle.feature_cols    = feature_cols
    bundle.target_col      = target_col
    bundle.in_dim          = X.shape[1]
    bundle.validation_type = val_type
    bundle.glambie_df      = glambie_df
    bundle.temporal_avg_df = temporal_avg_df

    if val_type == "random":
        splits = random_split(X, y, cfg)
        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val,   y_val   = splits["X_val"],   splits["y_val"]
        X_cal,   y_cal   = splits["X_cal"],   splits["y_cal"]
        bundle.y_test = splits["y_test"]
        X_test = splits["X_test"]

    elif val_type == "blocked":
        from hydra.core.hydra_config import HydraConfig
        out_dir = Path(HydraConfig.get().runtime.output_dir)
        splits = blocked_split(
            df, feature_cols, target_col,
            glacier_id_col, year_col,
            cfg.validation, out_dir,
        )
        bundle.df_train = splits["df_train"]
        bundle.df_logo  = splits["df_logo"]
        bundle.df_loyo  = splits["df_loyo"]
        bundle.df_loygo = splits["df_loygo"]
        bundle.df_full  = splits["df_full"]

        X_train, y_train = splits["X_train"], splits["y_train"]
        X_val,   y_val   = splits["X_val"],   splits["y_val"]
        X_cal,   y_cal   = splits["X_cal"],   splits["y_cal"]
        # For blocked, y_test is set per evaluation set in model_eval.py
        bundle.y_test = splits["y_train"]  # placeholder; eval uses df_logo etc.
        X_test = X_train  # placeholder

    else:
        raise ValueError(f"Unknown validation type '{val_type}'.")

    # ── Scale ─────────────────────────────────────────────────────────────
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    X_cal_s   = x_scaler.transform(X_cal)
    X_test_s  = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s   = y_scaler.transform(y_val)
    y_cal_s   = y_scaler.transform(y_cal)

    bundle.x_scaler = x_scaler
    bundle.y_scaler = y_scaler

    # ── To torch ──────────────────────────────────────────────────────────
    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.float32)).to(device)

    batch_size = int(cfg.training.get("batch_size", 256))

    bundle.X_train_t = _t(X_train_s)
    bundle.y_train_t = _t(y_train_s)
    bundle.X_val_t   = _t(X_val_s)
    bundle.y_val_t   = _t(y_val_s)
    bundle.X_cal_t   = _t(X_cal_s)
    bundle.y_cal_t   = _t(y_cal_s)
    bundle.X_test_t  = _t(X_test_s)

    # Keep un-scaled numpy arrays for conformal scoring
    bundle.y_train   = y_train
    bundle.y_val     = y_val
    bundle.y_cal     = y_cal

    bundle.train_loader = DataLoader(
        TensorDataset(bundle.X_train_t, bundle.y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    log.info(
        "Split — train: %d | val: %d | cal: %d",
        len(X_train), len(X_val), len(X_cal),
    )
    return bundle
