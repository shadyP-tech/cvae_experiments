"""Target-local case-block residual correction for SCALE-BP."""

from .contracts import (
    LocalCrossfitResult,
    LocalResidualModel,
    LocalResidualRecord,
    OOFResidualPrediction,
)
from .crossfit import (
    crossfit_local_residuals,
    fit_local_residual_model,
    predict_local_residual,
)

__all__ = (
    "LocalCrossfitResult",
    "LocalResidualModel",
    "LocalResidualRecord",
    "OOFResidualPrediction",
    "crossfit_local_residuals",
    "fit_local_residual_model",
    "predict_local_residual",
)
