"""Transform common-frame features with a pre-frozen shared Nyström map.

Map fitting is intentionally absent.  The landmarks, preprocessing state, and
gamma must be frozen from an equal-count, target-excluded candidate-source pool
before this router core is called.
"""

from __future__ import annotations

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    FrozenNystroemFeatureMap,
    TransformedKernelFeatures,
    readonly_matrix,
)


def transform_frozen_nystroem(
    feature_map: FrozenNystroemFeatureMap,
    preprocessed_common_features: object,
    *,
    common_frame_hash: str,
    preprocessing_hash: str,
) -> TransformedKernelFeatures:
    """Apply one shared source-pool-fitted RBF Nyström feature map."""

    features = readonly_matrix(
        preprocessed_common_features, "preprocessed common-frame features"
    )
    if (
        str(common_frame_hash) != feature_map.common_frame_hash
        or str(preprocessing_hash) != feature_map.preprocessing_hash
        or features.shape[1] != feature_map.components.shape[1]
    ):
        raise ProtocolError(
            "Nyström transform attempted to mix common frames or preprocessing states."
        )
    try:
        from sklearn.metrics.pairwise import rbf_kernel
    except ModuleNotFoundError as exc:  # pragma: no cover - project dependency
        raise RuntimeError(
            "MMD/KMM Nyström transformation requires scikit-learn."
        ) from exc
    kernel = rbf_kernel(
        features,
        feature_map.components,
        gamma=float(feature_map.gamma),
    )
    transformed = np.asarray(
        kernel @ feature_map.normalization.T,
        dtype=np.float64,
    )
    if (
        transformed.shape != (len(features), len(feature_map.components))
        or not np.isfinite(transformed).all()
    ):
        raise ProtocolError("Frozen Nyström transformation produced invalid features.")
    return TransformedKernelFeatures(
        values=transformed,
        common_frame_hash=feature_map.common_frame_hash,
        preprocessing_hash=feature_map.preprocessing_hash,
        candidate_pool_fit_hash=feature_map.candidate_pool_fit_hash,
        kernel_map_hash=feature_map.kernel_map_hash,
        map_fit_role=feature_map.fit_role,
        target_rows_used_to_fit=False,
        evaluation_rows_used_to_fit=False,
    )


__all__ = ("transform_frozen_nystroem",)
