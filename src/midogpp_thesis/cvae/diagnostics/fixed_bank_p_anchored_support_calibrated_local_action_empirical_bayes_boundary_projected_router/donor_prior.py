"""Backward-compatible facade for the modular SCALE-BP donor prior."""

from __future__ import annotations

from .donor_contracts import (
    DonorDeleteCenterFold,
    DonorObservation,
    DonorPriorModel,
    DonorPriorPrediction,
)
from .donor_fit import fit_donor_prior
from .donor_prediction import predict_donor_prior
from .donor_workflow import crossfit_donor_prediction, fit_final_donor_prior
from .identity import ACTION_IDS


# Historical name retained because admission and persisted pseudo rectangles use
# this public constant.  The canonical ownership now lives in identity.py.
CELL_IDS = ACTION_IDS


__all__ = (
    "CELL_IDS",
    "DonorObservation",
    "DonorDeleteCenterFold",
    "DonorPriorModel",
    "DonorPriorPrediction",
    "crossfit_donor_prediction",
    "fit_donor_prior",
    "fit_final_donor_prior",
    "predict_donor_prior",
)
