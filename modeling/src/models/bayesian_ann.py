"""
Bayesian ANN via Pyro SVI for geoUQnet.

Uses a mean-field (AutoDiagonalNormal) variational guide trained with
Stochastic Variational Inference (Trace_ELBO).  Posterior predictive
samples give both aleatoric and epistemic uncertainty, enabling either
absolute-residual or normalised conformal prediction.

Requires:  pip install pyro-ppl
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import Adam as PyroAdam

log = logging.getLogger(__name__)


# ─────────────────────────── Architecture ────────────────────────────────────

class BayesianANN(PyroModule):
    """Bayesian feed-forward network with Normal weight priors.

    Each weight and bias is treated as a random variable drawn from
    ``Normal(0, prior_scale)``.  Observation noise ``sigma`` has a
    ``Gamma(1, 1)`` prior.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    hidden_dims : Sequence[int]
        Width of each hidden layer.
    prior_scale : float
        Std-dev of the weight / bias priors (default ``1.0``).
    device : torch.device | None
        Device on which prior tensors are created.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        prior_scale: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        loc = torch.tensor(0.0, device=device)
        scale = torch.tensor(prior_scale, device=device)

        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            lin = PyroModule[nn.Linear](last, h)
            lin.weight = PyroSample(
                dist.Normal(loc, scale).expand([h, last]).to_event(2)
            )
            lin.bias = PyroSample(
                dist.Normal(loc, scale).expand([h]).to_event(1)
            )
            layers.extend([lin, nn.ReLU()])
            last = h

        out = PyroModule[nn.Linear](last, 1)
        out.weight = PyroSample(
            dist.Normal(loc, scale).expand([1, last]).to_event(2)
        )
        out.bias = PyroSample(
            dist.Normal(loc, scale).expand([1]).to_event(1)
        )
        layers.append(out)

        self.net = PyroModule[nn.Sequential](*layers)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mu = self.net(x).squeeze(-1)  # [N]

        sigma = pyro.sample(
            "sigma",
            dist.Gamma(
                torch.tensor(1.0, device=x.device),
                torch.tensor(1.0, device=x.device),
            ),
        )
        sigma = torch.clamp(sigma, min=1e-4)

        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(mu, sigma),
                obs=y.squeeze(-1) if y is not None else None,
            )
        return mu


# ─────────────────────────── Factory ─────────────────────────────────────────

def build_model(
    cfg: DictConfig,
    in_dim: int,
    device: torch.device,
) -> tuple[BayesianANN, AutoDiagonalNormal]:
    """Instantiate model + variational guide from the Hydra ``model`` node.

    Expected YAML (``configs/model/bayesian_ann.yaml``):

    .. code-block:: yaml

        _target_: src.models.bayesian_ann.BayesianANN
        hidden_dims: [64, 64]
        prior_scale: 1.0

    Returns
    -------
    model : BayesianANN
    guide : AutoDiagonalNormal
    """
    model = BayesianANN(
        in_dim=in_dim,
        hidden_dims=list(cfg.hidden_dims),
        prior_scale=float(cfg.get("prior_scale", 1.0)),
        device=device,
    ).to(device)

    guide = AutoDiagonalNormal(model).to(device)
    return model, guide


# ─────────────────────────── Training ────────────────────────────────────────

def _eval_mse_svi(
    model: BayesianANN,
    guide: AutoDiagonalNormal,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    n_samples: int = 50,
) -> float:
    model.eval()
    with torch.no_grad():
        predictive = Predictive(
            model, guide=guide, num_samples=n_samples, return_sites=("obs",)
        )
        pred_mean = predictive(X_val)["obs"].mean(0)  # [N]
        return torch.mean((pred_mean - y_val.squeeze(-1)) ** 2).item()


def train(
    model: BayesianANN,
    guide: AutoDiagonalNormal,
    train_loader: DataLoader,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: DictConfig,
) -> tuple[BayesianANN, AutoDiagonalNormal, float]:
    """Train via SVI (Trace_ELBO) with best-val-MSE checkpointing.

    Expected keys in ``cfg`` (the ``training`` node):

    .. code-block:: yaml

        lr: 1.0e-3
        epochs: 1000
        n_samples_eval: 50    # posterior samples used during validation

    Returns
    -------
    model, guide :
        Loaded with best-validation Pyro param store checkpoint.
    best_val_mse : float
    """
    pyro.clear_param_store()
    pyro.set_rng_seed(int(cfg.get("seed", 42)))

    lr = float(cfg.get("lr", 1e-3))
    epochs = int(cfg.get("epochs", 1000))
    n_samples_eval = int(cfg.get("n_samples_eval", 50))

    svi = SVI(model, guide, PyroAdam({"lr": lr}), loss=Trace_ELBO())

    best_val = float("inf")
    best_params: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            epoch_loss += svi.step(xb, yb)

        val_mse = _eval_mse_svi(model, guide, X_val, y_val, n_samples=n_samples_eval)

        if val_mse < best_val:
            best_val = val_mse
            best_params = {
                k: v.detach().cpu().clone()
                for k, v in pyro.get_param_store().items()
            }

        if epoch % 50 == 0 or epoch == 1:
            log.info(
                "Epoch %03d | ELBO/N: %.4f | val MSE (std-space): %.4f",
                epoch,
                epoch_loss / len(X_val),
                val_mse,
            )

    if best_params is not None:
        device = X_val.device
        pyro.clear_param_store()
        for k, v in best_params.items():
            pyro.param(k, v.to(device))

    return model, guide, best_val


# ─────────────────────────── Inference ───────────────────────────────────────

@torch.no_grad()
def predict_mean_std(
    model: BayesianANN,
    guide: AutoDiagonalNormal,
    X: torch.Tensor,
    y_scaler,
    n_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw posterior predictive samples and return mean & std in original units.

    Parameters
    ----------
    model, guide :
        Trained Pyro model/guide pair.
    X : torch.Tensor
        Input tensor on the correct device.
    y_scaler :
        Fitted ``sklearn.preprocessing.StandardScaler`` for the target.
    n_samples : int
        Number of posterior predictive samples (default 500).

    Returns
    -------
    mean : np.ndarray [N]
    std  : np.ndarray [N]
        Both in *original* (un-scaled) units.
    """
    model.eval()
    predictive = Predictive(
        model, guide=guide, num_samples=n_samples, return_sites=("obs",)
    )
    obs = predictive(X)["obs"]  # [S, N]

    mean_std = obs.mean(0).cpu().numpy()
    std_std = obs.std(0).cpu().numpy()

    mean = y_scaler.inverse_transform(mean_std.reshape(-1, 1)).ravel()
    std = std_std * float(y_scaler.scale_[0])
    return mean, std


@torch.no_grad()
def predict_mean(
    model: BayesianANN,
    guide: AutoDiagonalNormal,
    X: torch.Tensor,
    y_scaler,
    n_samples: int = 500,
) -> np.ndarray:
    """Convenience wrapper — mean only."""
    return predict_mean_std(model, guide, X, y_scaler, n_samples=n_samples)[0]