"""Donor, route-local, empirical-Bayes, and descriptive uncertainty layers."""

from .contracts import DonorFitScope, DonorObservation, LocalResidualObservation, ScaleVector
from .donor import (
    DonorActionModel,
    DonorDeleteCenterFold,
    DonorPrediction,
    assess_donor_support,
    fit_donor_action_model,
    predict_donor_action,
)
from .empirical_bayes import ActionEstimate, combine_empirical_bayes
from .local import (
    LOCAL_FOLD_COUNT,
    LocalResidualModel,
    LocalResidualPrediction,
    assign_local_support_folds,
    build_local_residual_observations,
    fit_route_local_residual,
    predict_local_residual,
)
from .pipeline import estimate_action_rectangle
from .uncertainty import DescriptiveBounds, build_preargmax_bounds

__all__ = tuple(name for name in globals() if not name.startswith("_"))
