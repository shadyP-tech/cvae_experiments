"""Named final and pseudo donor workflows for SCALE-BP."""

from __future__ import annotations

from collections.abc import Sequence

from .donor_contracts import (
    DonorDeleteCenterFold,
    DonorObservation,
    DonorPriorModel,
    DonorPriorPrediction,
)
from .donor_fit import fit_donor_prior
from .donor_prediction import predict_donor_prior
from .influence.contracts import ActionDescriptor
from .replay_scope import FinalDonorScope, PseudoReplayScope


def crossfit_donor_prediction(
    observations: Sequence[DonorObservation],
    *,
    scope: PseudoReplayScope,
    descriptor: ActionDescriptor,
    delete_center_folds: Sequence[DonorDeleteCenterFold],
) -> DonorPriorPrediction:
    """Predict pseudo-held ``d`` only from scope-verified ``not H/J/d`` rows."""

    model = fit_donor_prior(
        observations,
        scope=scope,
        delete_center_folds=delete_center_folds,
    )
    return predict_donor_prior(model, descriptor, scope=scope)


def fit_final_donor_prior(
    observations: Sequence[DonorObservation],
    *,
    scope: FinalDonorScope,
    delete_center_folds: Sequence[DonorDeleteCenterFold],
) -> DonorPriorModel:
    """Named final-route entry point so pseudo and final regimes stay distinct."""

    return fit_donor_prior(
        observations,
        scope=scope,
        delete_center_folds=delete_center_folds,
    )


__all__ = ("crossfit_donor_prediction", "fit_final_donor_prior")
