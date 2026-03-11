"""Model registry for geoUQnet."""

from .deterministic_ann import DeterministicANN
from .mc_dropout_ann import MCDropoutANN
# from .deep_ensemble_ann import DeepEnsembleANN
from .bayesian_ann import BayesianANN

__all__ = [
    "DeterministicANN",
    "MCDropoutANN",
    "DeepEnsembleANN",
    "BayesianANN",
]