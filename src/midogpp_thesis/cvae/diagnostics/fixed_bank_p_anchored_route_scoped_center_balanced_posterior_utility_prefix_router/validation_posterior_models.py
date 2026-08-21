"""Rebuild all posterior DTO hashes and replay their held-case probabilities."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .constants import TARGET_POSTERIOR_MAX_ITER
from .hashing import canonical_hash
from .posterior_contracts import TargetLocalPosteriorModel
from .posterior_fit import predict_route_posterior
from .validation_origin import PhysicalOriginTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import fail


def validate_posterior_model_predictions(
    root: Path,
    *,
    origin: PhysicalOriginTopology,
    topology: PlanPosteriorTopology,
) -> Mapping[tuple[str, str, str], np.ndarray]:
    """Replay 436 model predictions from rebuilt physical fingerprints."""

    output: dict[tuple[str, str, str], np.ndarray] = {}
    model_hashes: set[str] = set()
    prediction_hashes: set[str] = set()
    with np.load(
        root / "arrays/target_local_posterior_probabilities.npz",
        allow_pickle=False,
    ) as store:
        for key, row in topology.models.items():
            try:
                model = TargetLocalPosteriorModel(
                    str(row["target_center"]),
                    str(row["held_case_id"]),
                    str(row["control_id"]),
                    tuple(str(value) for value in row["training_case_ids"]),  # type: ignore[index]
                    tuple(str(value) for value in row["feature_names"]),  # type: ignore[index]
                    tuple(float(value) for value in row["feature_mean"]),  # type: ignore[index]
                    tuple(float(value) for value in row["feature_scale"]),  # type: ignore[index]
                    tuple(float(value) for value in row["coefficients"]),  # type: ignore[index]
                    float(row["intercept"]),
                    int(row["training_row_count"]),
                    int(row["training_n_positive"]),
                    int(row["training_n_negative"]),
                    str(row["fingerprint_hash"]),
                    str(row["training_identity_hash"]),
                    int(row["iterations"]),
                    bool(row["converged"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProtocolError(
                    "CBPUPR persisted posterior model is malformed."
                ) from exc
            expected_model_row = {
                **model.to_payload(),
                "structural_reference_reuse_allowed": True,
            }
            if (
                dict(row) != expected_model_row
                or model.iterations >= TARGET_POSTERIOR_MAX_ITER
            ):
                fail("posterior model DTO/hash")
            fingerprint = origin.fingerprints[(model.target_center, model.control_id)]
            prediction = predict_route_posterior(fingerprint, model)
            persisted = topology.posteriors[key]
            expected_prediction_row = {
                "target_center": prediction.target_center,
                "held_case_id": prediction.held_case_id,
                "control_id": prediction.control_id,
                "sample_ids": list(prediction.sample_ids),
                "array_key": prediction.prediction_hash,
                "prediction_hash": prediction.prediction_hash,
                "model_hash": prediction.model_hash,
                "fingerprint_hash": prediction.fingerprint_hash,
                "sample_identity_hash": canonical_hash(
                    list(prediction.sample_ids)
                ),
            }
            observed = np.ascontiguousarray(
                np.asarray(store[prediction.prediction_hash], dtype=np.float32)
            )
            expected = np.ascontiguousarray(
                prediction.natural_probabilities, dtype=np.float32
            )
            if (
                dict(persisted) != expected_prediction_row
                or observed.tobytes(order="C") != expected.tobytes(order="C")
            ):
                fail("posterior model prediction replay")
            output[key] = observed
            model_hashes.add(model.model_hash)
            prediction_hashes.add(prediction.prediction_hash)
    if (
        len(model_hashes) != len(topology.models)
        or len(prediction_hashes) != len(topology.posteriors)
    ):
        fail("posterior model/prediction hash uniqueness")
    return MappingProxyType(output)


__all__ = ("validate_posterior_model_predictions",)
