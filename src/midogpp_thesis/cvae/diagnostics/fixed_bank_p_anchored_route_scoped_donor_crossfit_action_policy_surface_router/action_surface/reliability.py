"""Nested-OOF family/direction reliability gates for P-DCAPS actions."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..identity import ACTION_STRATA, METRICS, canonical_hash
from .contracts import (
    ActionCalibrationModel,
    ActionPrediction,
    ActionResponse,
    ActionStratumReliability,
)
from .descriptors import build_action_descriptor
from .ridge import predict_weighted_ridge
from .weights import build_hierarchical_weights


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def finite_spearman(left: object, right: object) -> float | None:
    """Return deterministic midrank Spearman, undefined on constants/nonfinite."""

    x = np.ascontiguousarray(np.asarray(left, dtype=np.float64))
    y = np.ascontiguousarray(np.asarray(right, dtype=np.float64))
    if (
        x.ndim != 1
        or y.shape != x.shape
        or len(x) < 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.all(x == x[0])
        or np.all(y == y[0])
    ):
        return None
    x_rank = _midranks(x)
    y_rank = _midranks(y)
    x_centered = x_rank - np.mean(x_rank, dtype=np.float64)
    y_centered = y_rank - np.mean(y_rank, dtype=np.float64)
    denominator = float(
        np.sqrt(
            np.sum(x_centered * x_centered, dtype=np.float64)
            * np.sum(y_centered * y_centered, dtype=np.float64)
        )
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        return None
    value = float(np.sum(x_centered * y_centered, dtype=np.float64) / denominator)
    return value if np.isfinite(value) else None


def _model_family(
    models: Sequence[ActionCalibrationModel],
    *,
    outer_center: str,
    scored_center: str,
    reliability_scored_context: str | None,
) -> tuple[ActionCalibrationModel, ...]:
    rows = tuple(models)
    expected_training_centers = tuple(
        center
        for center in CENTERS
        if center != outer_center
        and center != scored_center
        and center != reliability_scored_context
    )
    if (
        tuple(model.metric for model in rows) != METRICS
        or any(model.excluded_outer_center != outer_center for model in rows)
        or any(model.excluded_scored_center != scored_center for model in rows)
        or any(model.training_centers != expected_training_centers for model in rows)
    ):
        raise ProtocolError("P-DCAPS OOF action-model family exclusion drifted.")
    return rows


def _calibrated_utility(
    prediction: ActionPrediction,
    models: Sequence[ActionCalibrationModel],
) -> FavorableUtility:
    family = tuple(models)
    values = tuple(
        float(predict_weighted_ridge(model, build_action_descriptor(prediction, model.metric)))
        for model in family
    )
    return FavorableUtility.from_array(values)


def evaluate_action_stratum_reliability(
    predictions: Sequence[ActionPrediction],
    responses: Sequence[ActionResponse],
    oof_models_by_scored_center: Mapping[str, Sequence[ActionCalibrationModel]],
    *,
    outer_center: str,
    family: str,
    direction: str,
    scored_center: str | None = None,
    minimum_center_count: int = 6,
) -> ActionStratumReliability:
    """Evaluate one gate using only K-scored models that excluded both H and K.

    ``scored_center`` is additionally removed when constructing a gate that
    will be applied to pseudo-center J.  Hence neither the action calibrator nor
    its reliability gate can see J's own realized response before routing J.
    """

    outer = str(outer_center)
    scored = None if scored_center is None else str(scored_center)
    stratum = (str(family), str(direction))
    if (
        outer not in CENTERS
        or scored is not None and (scored not in CENTERS or scored == outer)
        or stratum not in ACTION_STRATA
        or int(minimum_center_count) <= 0
    ):
        raise ProtocolError("P-DCAPS reliability identity drifted.")
    response_by_prediction = {row.prediction_hash: row for row in responses}
    if len(response_by_prediction) != len(tuple(responses)):
        raise ProtocolError("P-DCAPS reliability responses are duplicated.")
    selected: list[tuple[ActionPrediction, ActionResponse, FavorableUtility, tuple[str, ...]]] = []
    for prediction in predictions:
        route = prediction.key.route_key
        if (
            route.surface_role != "pseudo"
            or route.outer_center != outer
            or prediction.key.stratum != stratum
            or route.route_center == scored
        ):
            continue
        response = response_by_prediction.get(prediction.prediction_hash)
        if response is None or response.key.action_key_hash != prediction.key.action_key_hash:
            raise ProtocolError("P-DCAPS reliability prediction/response lineage drifted.")
        center = route.route_center
        if center not in oof_models_by_scored_center:
            raise ProtocolError("P-DCAPS reliability is missing a scored-center OOF model.")
        models = _model_family(
            oof_models_by_scored_center[center],
            outer_center=outer,
            scored_center=center,
            reliability_scored_context=scored,
        )
        selected.append(
            (
                prediction,
                response,
                _calibrated_utility(prediction, models),
                tuple(model.model_hash for model in models),
            )
        )

    if not selected:
        evidence_hash = canonical_hash(
            {
                "schema_version": "pdcaps_action_reliability_evidence_v1",
                "excluded_outer_center": outer,
                "excluded_scored_center": scored,
                "stratum": stratum,
                "rows": [],
            }
        )
        return ActionStratumReliability(
            outer,
            scored,
            stratum[0],
            stratum[1],
            (),
            (),
            FavorableUtility.zeros(),
            None,
            False,
            0,
            int(minimum_center_count),
            False,
            0,
            evidence_hash,
        )

    selected_responses = tuple(row[1] for row in selected)
    weight_audit = build_hierarchical_weights(selected_responses)
    weights = weight_audit.as_array()
    predicted_bacc = np.asarray(
        [row[2].bacc_gain for row in selected], dtype=np.float64
    )
    realized = np.asarray(
        [row[1].realized_utility.as_tuple() for row in selected], dtype=np.float64
    )
    equal_center = FavorableUtility.from_array(
        np.sum(weights[:, None] * realized, axis=0, dtype=np.float64)
    )
    centers = tuple(
        center
        for center in CENTERS
        if center in {row[0].key.route_key.route_center for row in selected}
    )
    center_means: list[tuple[str, float, float, float]] = []
    for center in centers:
        indices = np.asarray(
            [
                index
                for index, row in enumerate(selected)
                if row[0].key.route_key.route_center == center
            ],
            dtype=np.int64,
        )
        local_weights = weights[indices]
        local_weights = local_weights / np.sum(local_weights, dtype=np.float64)
        values = np.sum(
            local_weights[:, None] * realized[indices], axis=0, dtype=np.float64
        )
        center_means.append((center, *(float(value) for value in values)))
    rho = finite_spearman(predicted_bacc, realized[:, 0])
    bank_viable = all(
        row[0].bank_viability.row_preserving and row[0].bank_viability.passed
        for row in selected
    )
    evidence_hash = canonical_hash(
        {
            "schema_version": "pdcaps_action_reliability_evidence_v1",
            "excluded_outer_center": outer,
            "excluded_scored_center": scored,
            "stratum": stratum,
            "rows": [
                {
                    "prediction_hash": prediction.prediction_hash,
                    "response_hash": response.response_hash,
                    "oof_model_hashes": model_hashes,
                    "calibrated_utility": calibrated.to_payload(),
                }
                for prediction, response, calibrated, model_hashes in selected
            ],
            "weight_audit_hash": weight_audit.weight_audit_hash,
        }
    )
    return ActionStratumReliability(
        outer,
        scored,
        stratum[0],
        stratum[1],
        centers,
        tuple(center_means),
        equal_center,
        rho,
        rho is not None,
        sum(1 for _center, value, _brier, _log in center_means if value > 0.0),
        int(minimum_center_count),
        bank_viable,
        len(selected),
        evidence_hash,
    )


def build_action_reliability_by_stratum(
    predictions: Sequence[ActionPrediction],
    responses: Sequence[ActionResponse],
    oof_models_by_scored_center: Mapping[str, Sequence[ActionCalibrationModel]],
    *,
    outer_center: str,
    scored_center: str | None = None,
    minimum_center_count: int = 6,
) -> tuple[ActionStratumReliability, ...]:
    return tuple(
        evaluate_action_stratum_reliability(
            predictions,
            responses,
            oof_models_by_scored_center,
            outer_center=outer_center,
            family=family,
            direction=direction,
            scored_center=scored_center,
            minimum_center_count=minimum_center_count,
        )
        for family, direction in ACTION_STRATA
    )


__all__ = (
    "build_action_reliability_by_stratum",
    "evaluate_action_stratum_reliability",
    "finite_spearman",
)
