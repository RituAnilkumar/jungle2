"""
Train/validation/calibration/test split logic for modeling.

Two modes, selected via cfg.validation.type:

  random  — standard random split into train / val / cal / test
  blocked — LOGO / LOYO / LOYGO blocked split for glacier mass balance

For blocked splits, four partitions are produced from one run:
  train   : glaciers NOT held out AND years NOT held out
  logo    : held-out glaciers × non-held-out years
  loyo    : non-held-out glaciers × held-out years
  loygo   : held-out glaciers × held-out years (hardest case; removed from logo+loyo)

held_out_glaciers.txt and held_out_years.txt are saved to the Hydra output
directory for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


# ─────────────────────────── Random split ────────────────────────────────────

def random_split(
    X: np.ndarray,
    y: np.ndarray,
    cfg: DictConfig,
) -> dict[str, np.ndarray]:
    """Split arrays randomly into train / val / cal / test.

    Parameters
    ----------
    X : np.ndarray [N, F]
    y : np.ndarray [N, 1]
    cfg : DictConfig
        Root config; uses cfg.splits (test_size, cal_size, val_size, seed).

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_cal, y_cal, X_test, y_test
    """
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
        "Random split — train: %d | val: %d | cal: %d | test: %d",
        len(X_train), len(X_val), len(X_cal), len(X_test),
    )
    return dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_cal=X_cal,     y_cal=y_cal,
        X_test=X_test,   y_test=y_test,
    )


# ─────────────────────────── Blocked split ───────────────────────────────────

def blocked_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    glacier_id_col: str,
    year_col: str,
    val_cfg: DictConfig,
    output_dir: Path,
) -> dict[str, pd.DataFrame | np.ndarray]:
    """Produce LOGO / LOYO / LOYGO partitions from the full DataFrame.

    Partition logic:
      - Hold out `holdout_glacier_frac` of unique glaciers → LOGO
      - Hold out `holdout_year_frac`   of unique years    → LOYO
      - LOYGO = intersection (held-out glacier AND held-out year)
      - LOGO  = held-out glaciers × non-held-out years   (LOYGO removed)
      - LOYO  = non-held-out glaciers × held-out years   (LOYGO removed)
      - Train = remaining points not in any holdout set

    Saves held_out_glaciers.txt and held_out_years.txt to output_dir.

    Returns
    -------
    dict with keys:
      df_train, df_logo, df_loyo, df_loygo, df_full
      X_train, y_train, X_val, y_val, X_cal, y_cal
      (val and cal are random sub-splits of df_train for model selection)
    """
    frac_gla  = float(val_cfg.get("holdout_glacier_frac", 0.10))
    frac_yr   = float(val_cfg.get("holdout_year_frac", 0.10))
    seed      = int(val_cfg.get("split_seed", 42))
    rng       = np.random.default_rng(seed)

    glaciers = df[glacier_id_col].unique()
    years    = df[year_col].unique()

    n_gla = max(1, int(np.ceil(len(glaciers) * frac_gla)))
    n_yr  = max(1, int(np.ceil(len(years)    * frac_yr)))

    held_glaciers = set(rng.choice(glaciers, size=n_gla, replace=False))
    held_years    = set(rng.choice(years,    size=n_yr,  replace=False))

    log.info(
        "Blocked split — held glaciers: %d/%d | held years: %d/%d",
        len(held_glaciers), len(glaciers), len(held_years), len(years),
    )

    is_held_gla = df[glacier_id_col].isin(held_glaciers)
    is_held_yr  = df[year_col].isin(held_years)

    df_loygo = df[is_held_gla  &  is_held_yr].copy()
    df_logo  = df[is_held_gla  & ~is_held_yr].copy()
    df_loyo  = df[~is_held_gla &  is_held_yr].copy()
    df_train = df[~is_held_gla & ~is_held_yr].copy()
    df_full  = df.copy()

    log.info(
        "Partition sizes — train: %d | logo: %d | loyo: %d | loygo: %d",
        len(df_train), len(df_logo), len(df_loyo), len(df_loygo),
    )

    # ── Save artefacts ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "held_out_glaciers.txt").write_text(
        "\n".join(sorted(str(g) for g in held_glaciers))
    )
    (output_dir / "held_out_years.txt").write_text(
        "\n".join(sorted(str(y) for y in held_years))
    )

    # ── Build val/cal from train for model selection ────────────────────
    X_tr = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_tr = df_train[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    X_tv, X_cal, y_tv, y_cal = train_test_split(X_tr, y_tr, test_size=0.20, random_state=seed)
    X_train_s, X_val, y_train_s, y_val = train_test_split(X_tv, y_tv, test_size=0.15, random_state=seed)

    return dict(
        df_train=df_train,
        df_logo=df_logo,
        df_loyo=df_loyo,
        df_loygo=df_loygo,
        df_full=df_full,
        X_train=X_train_s, y_train=y_train_s,
        X_val=X_val,       y_val=y_val,
        X_cal=X_cal,       y_cal=y_cal,
    )
