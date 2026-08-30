"""One-way adapter from sealed HARP case surfaces into model observations."""

from __future__ import annotations

from ...protocol import ProtocolError
from ..harp_protocol.hashing import canonical_hash
from ..harp_action_surface import HarpActionFeatureSurface, HarpDirectionalResponseSurface
from .contracts import HarpTrainingObservation


def training_observations_from_surfaces(
    features: HarpActionFeatureSurface,
    responses: HarpDirectionalResponseSurface,
) -> tuple[HarpTrainingObservation, ...]:
    """Bind source outcomes to pre-sealed case features exactly once.

    Raw seed cells are structurally absent from both inputs.  The resulting
    observations retain only the exact-nine ensemble receipt and independent
    case identity used by the query/case-balanced model.
    """

    if not isinstance(features, HarpActionFeatureSurface) or not isinstance(responses, HarpDirectionalResponseSurface):
        raise ProtocolError("HARP model adaptation requires typed case surfaces.")
    if responses.feature_surface_hash != features.surface_hash:
        raise ProtocolError("HARP response surface is not bound to the feature surface.")
    response_by_key = {row.row_key: row for row in responses.rows}
    if set(response_by_key) != {row.row_key for row in features.rows}:
        raise ProtocolError("HARP feature and response case grids are not exact.")
    output: list[HarpTrainingObservation] = []
    for feature in features.rows:
        if feature.inner_donor is not None:
            raise ProtocolError("HARP policy fitting derives inner donor folds itself.")
        response = response_by_key[feature.row_key]
        if response.feature_hash != feature.feature_hash or response.ensemble_receipt_hash != feature.ensemble_receipt_hash or response.case_aggregation_receipt_hash != feature.case_aggregation_receipt_hash:
            raise ProtocolError("HARP response row escaped its sealed case feature.")
        sample_ids = (feature.sample_id,) if hasattr(feature, "sample_id") else tuple(feature.sample_ids)
        for sample_id in sample_ids:
            output.append(HarpTrainingObservation(
                outer_target_id=feature.outer_target,
                pseudo_query_id=feature.pseudo_query,
                candidate_source_id=feature.candidate_source,
                case_id=feature.case_id,
                sample_id=sample_id,
                lambda_value=feature.action_lambda,
                direction=feature.direction,
                feature_names=feature.feature_names,
                feature_values=feature.feature_values,
                weighted_correctness_surrogate=response.weighted_correctness_surrogate,
                brier_delta=response.brier_delta,
                log_loss_delta=response.log_loss_delta,
                truth_class=response.truth_class,
                ensemble_size=feature.seed_count,
                ensemble_receipt_hash=feature.ensemble_receipt_hash,
                case_aggregation_receipt_hash=feature.case_aggregation_receipt_hash,
                prediction_seal_hash=feature.prediction_seal_hash,
                # Retain the exact response-row identity in the fitted model
                # bank.  The exported surface hash separately binds the full
                # collection, while this receipt identifies the observation
                # that actually contributed to the normal equations.
                response_receipt_hash=response.response_hash,
            ))
    return tuple(sorted(output, key=lambda row: row.row_key))


def training_observation_surface_payload(
    rows: tuple[HarpTrainingObservation, ...],
    *,
    feature_surface_hash: str,
    response_surface_hash: str,
) -> dict[str, object]:
    """Canonical file boundary consumed by the Stage-60 policy adapter."""

    typed = tuple(sorted(rows, key=lambda row: (row.outer_target_id, row.row_key)))
    if not typed or len({(row.outer_target_id, row.row_key) for row in typed}) != len(typed):
        raise ProtocolError("HARP training observation export is empty or duplicated.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_harp_training_observation_surface_v1",
        "feature_surface_hash": feature_surface_hash,
        "response_surface_hash": response_surface_hash,
        "rows": [
            {
                "outer_target_id": row.outer_target_id,
                "pseudo_query_id": row.pseudo_query_id,
                "candidate_source_id": row.candidate_source_id,
                "case_id": row.case_id,
                "sample_id": row.sample_id,
                "lambda_value": row.lambda_value,
                "direction": row.direction,
                "feature_names": list(row.feature_names),
                "feature_values": list(row.feature_values),
                "weighted_correctness_surrogate": row.weighted_correctness_surrogate,
                "brier_delta": row.brier_delta,
                "log_loss_delta": row.log_loss_delta,
                "truth_class": row.truth_class,
                "ensemble_size": row.ensemble_size,
                "ensemble_receipt_hash": row.ensemble_receipt_hash,
                "case_aggregation_receipt_hash": row.case_aggregation_receipt_hash,
                "prediction_seal_hash": row.prediction_seal_hash,
                "response_receipt_hash": row.response_receipt_hash,
            }
            for row in typed
        ],
    }
    payload["training_surface_hash"] = canonical_hash(payload)
    return payload


__all__ = ("training_observation_surface_payload", "training_observations_from_surfaces")
