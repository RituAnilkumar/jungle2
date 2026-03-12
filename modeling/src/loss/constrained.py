"""
Constrained multi-task loss for modeling.

Supports three modes (set via cfg.loss.type):

  glambie              — primary MSE + GLAMBIE regional sum term
  temporal_avg         — primary MSE + temporal moving-window average term
  glambie_and_temporal_avg — primary MSE + both auxiliary terms

The auxiliary terms aggregate per-glacier model predictions and compare
them to external observational targets:

  GLAMBIE term:
    - Groups predictions by RGI region
    - Sums predictions per region per year (weighted by glacier area if available)
    - Computes MSE against glambie_targets_df (regional_sum column)

  Temporal average term:
    - Groups predictions by (glacier, time_window)
    - Averages predictions over each window
    - Computes MSE against temporal_avg_targets_df (avg_mb column)

Usage:
    from src.loss.constrained import ConstrainedLoss
    loss_fn = ConstrainedLoss(cfg.loss, glambie_df, temporal_avg_df, region_map)
    loss = loss_fn.compute(pred, target, rgi_ids, years)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig

log = logging.getLogger(__name__)

mse = nn.MSELoss()


def _derive_region(rgi_id: str) -> str:
    """Derive RGI region code from rgi_id string (e.g. 'RGI60-06.12345' → '06')."""
    try:
        return rgi_id.split("-")[1].split(".")[0]
    except (IndexError, AttributeError):
        return "unknown"


class ConstrainedLoss:
    """Multi-task loss combining primary MSE with optional auxiliary terms.

    Parameters
    ----------
    loss_cfg : DictConfig
        The loss config node (configs/loss/*.yaml).
    glambie_df : pd.DataFrame | None
        GLAMBIE targets with columns: region, year, regional_sum.
    temporal_avg_df : pd.DataFrame | None
        Temporal average targets with columns: rgi_id, start_date, end_date, avg_mb.
    region_col : str | None
        Column name in the training DataFrame identifying the RGI region.
        If None, region is auto-derived from rgi_id.
    """

    def __init__(
        self,
        loss_cfg: DictConfig,
        glambie_df: pd.DataFrame | None = None,
        temporal_avg_df: pd.DataFrame | None = None,
        region_col: str | None = None,
    ) -> None:
        self.loss_type      = loss_cfg.type
        self.glambie_weight = float(loss_cfg.get("glambie_weight",  0.5))
        self.temporal_weight = float(loss_cfg.get("temporal_weight", 0.5))
        self.glambie_df     = glambie_df
        self.temporal_avg_df = temporal_avg_df
        self.region_col     = region_col

        if self.loss_type in ("glambie", "glambie_and_temporal_avg") and glambie_df is None:
            raise ValueError("loss_type requires glambie_df but it was not provided.")
        if self.loss_type in ("temporal_avg", "glambie_and_temporal_avg") and temporal_avg_df is None:
            raise ValueError("loss_type requires temporal_avg_df but it was not provided.")

    def _glambie_term(
        self,
        pred: torch.Tensor,
        rgi_ids: list[str],
        years: list[int],
    ) -> torch.Tensor:
        """Compute GLAMBIE regional sum auxiliary loss."""
        df = pd.DataFrame({"pred": pred.detach().cpu().numpy(), "rgi_id": rgi_ids, "year": years})

        if self.region_col:
            # region_col should already be in the dataframe — passed via rgi_ids metadata
            # For simplicity we derive from rgi_id here; caller can override region_col logic
            df["region"] = df["rgi_id"].apply(_derive_region)
        else:
            df["region"] = df["rgi_id"].apply(_derive_region)

        pred_sums = df.groupby(["region", "year"])["pred"].sum().reset_index()
        pred_sums.columns = ["region", "year", "pred_sum"]

        merged = pred_sums.merge(
            self.glambie_df[["region", "year", "regional_sum"]],
            on=["region", "year"], how="inner"
        )
        if len(merged) == 0:
            log.warning("GLAMBIE: no matching (region, year) pairs found in batch.")
            return torch.tensor(0.0, device=pred.device)

        pred_t = torch.tensor(merged["pred_sum"].to_numpy(dtype=np.float32), device=pred.device)
        true_t = torch.tensor(merged["regional_sum"].to_numpy(dtype=np.float32), device=pred.device)
        return mse(pred_t, true_t)

    def _temporal_avg_term(
        self,
        pred: torch.Tensor,
        rgi_ids: list[str],
        years: list[int],
    ) -> torch.Tensor:
        """Compute temporal moving-window average auxiliary loss."""
        df = pd.DataFrame({"pred": pred.detach().cpu().numpy(), "rgi_id": rgi_ids, "year": years})

        pred_avgs, true_avgs = [], []
        for _, row in self.temporal_avg_df.iterrows():
            window_mask = (
                (df["rgi_id"] == row["rgi_id"]) &
                (df["year"] >= row["start_date"]) &
                (df["year"] <= row["end_date"])
            )
            if window_mask.sum() == 0:
                continue
            pred_avgs.append(df.loc[window_mask, "pred"].mean())
            true_avgs.append(row["avg_mb"])

        if not pred_avgs:
            log.warning("Temporal avg: no matching windows found in batch.")
            return torch.tensor(0.0, device=pred.device)

        pred_t = torch.tensor(np.array(pred_avgs, dtype=np.float32), device=pred.device)
        true_t = torch.tensor(np.array(true_avgs, dtype=np.float32), device=pred.device)
        return mse(pred_t, true_t)

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        rgi_ids: list[str],
        years: list[int],
    ) -> torch.Tensor:
        """Compute total loss.

        Parameters
        ----------
        pred : torch.Tensor [N]
            Per-glacier annual predictions (standardised or original units —
            must be consistent with target and auxiliary target units).
        target : torch.Tensor [N]
            Per-glacier annual mass balance ground truth.
        rgi_ids : list[str]
            RGI IDs corresponding to each prediction.
        years : list[int]
            Years corresponding to each prediction.
        """
        primary = mse(pred, target.squeeze(-1))

        if self.loss_type == "glambie":
            aux = self.glambie_weight * self._glambie_term(pred, rgi_ids, years)
            return primary + aux

        if self.loss_type == "temporal_avg":
            aux = self.temporal_weight * self._temporal_avg_term(pred, rgi_ids, years)
            return primary + aux

        if self.loss_type == "glambie_and_temporal_avg":
            g_term = self.glambie_weight  * self._glambie_term(pred, rgi_ids, years)
            t_term = self.temporal_weight * self._temporal_avg_term(pred, rgi_ids, years)
            return primary + g_term + t_term

        raise ValueError(f"Unknown constrained loss type: {self.loss_type}")
