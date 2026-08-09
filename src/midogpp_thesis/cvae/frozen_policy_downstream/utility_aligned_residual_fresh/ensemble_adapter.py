"""Stage-70 boundary to the neutral exact-nine probability aggregator.

This deliberately contains no labels, metric calculation, routing decision, or
state mutation.  Prediction sealing and the H×e terminal action geometry stay
in the Stage-70 package; the shared helper owns only arithmetic aggregation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from ...routing.utility_aligned.ensemble_contracts import ENSEMBLE_SEED_KEYS, SeedProbabilityVector
from ...routing.residual_topup.hashing import array_sha256, canonical_sha256


def mean_exact_nine_probabilities(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Delegate the pure nine-vector mean to the neutral routing core."""

    try:
        helper = getattr(
            import_module("...routing.utility_aligned.ensemble_endpoint", package=__package__),
            "mean_exact_nine_positive_class_probabilities",
        )
    except (ImportError, AttributeError) as exc:
        raise ProtocolError(
            "Neutral exact-nine probability aggregation API is unavailable."
        ) from exc
    arrays = tuple(np.asarray(vector, dtype=np.float64) for vector in vectors)
    if len(arrays) != 9 or not arrays or any(
        vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all()
        or bool(np.any(vector < 0.0) or np.any(vector > 1.0))
        for vector in arrays
    ) or len({vector.shape for vector in arrays}) != 1:
        raise ProtocolError("Stage-70 exact-nine probability geometry drifted.")
    row_identity_hash = canonical_sha256(
        {"shape": list(arrays[0].shape), "component_sha256": [array_sha256(item) for item in arrays]}
    )
    typed = tuple(
        SeedProbabilityVector(
            training_seed=training_seed,
            generation_seed=generation_seed,
            row_identity_hash=row_identity_hash,
            prediction_provenance_hash=array_sha256(vector),
            positive_class_probabilities=vector,
        )
        for (training_seed, generation_seed), vector in zip(ENSEMBLE_SEED_KEYS, arrays, strict=True)
    )
    result = np.asarray(helper(typed), dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all() or bool(
        np.any(result < 0.0) or np.any(result > 1.0)
    ):
        raise ProtocolError("Neutral exact-nine probability aggregation drifted.")
    result.setflags(write=False)
    return result


__all__ = ("mean_exact_nine_probabilities",)
