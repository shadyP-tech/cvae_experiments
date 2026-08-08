"""Private scientific payload serialization for utility-aligned models."""

from __future__ import annotations

import numpy as np

from ..local_marginal_utility.ridge import ClusterWeightedRidgeModel
from ..residual_topup.hashing import array_sha256


def _ridge_payload(model: ClusterWeightedRidgeModel) -> dict[str, object]:
    return {
        "feature_names": list(model.feature_names),
        "alpha": model.alpha,
        "feature_mean": np.asarray(model.feature_mean).tolist(),
        "feature_scale": np.asarray(model.feature_scale).tolist(),
        "intercept": model.intercept,
        "coefficients": np.asarray(model.coefficients).tolist(),
        "coefficient_covariance_sha256": array_sha256(model.coefficient_covariance),
        "residual_variance": model.residual_variance,
        "training_query_clusters": list(model.training_query_clusters),
        "observation_count": model.observation_count,
        "effective_rank": model.effective_rank,
    }


__all__: tuple[str, ...] = ()
