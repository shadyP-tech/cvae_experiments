"""Donor-cross-fitted action calibration and pre-argmax quarantine runtime."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..identity import (
    ACTION_FAMILIES,
    ACTION_STRATA,
    DIRECTIONS,
    METRICS,
    TIE_TOLERANCE,
    canonical_hash,
)
from .contracts import (
    ActionCalibrationModel,
    ActionPrediction,
    ActionResponse,
    ActionStratumReliability,
    CalibratedAction,
    CalibratedActionSelection,
)
from .descriptors import (
    action_feature_names,
    build_action_descriptor,
    build_action_descriptor_matrix,
)
from .ridge import fit_weighted_ridge, predict_weighted_ridge
from .weights import build_hierarchical_weights


def _metric_response(response: ActionResponse, metric: str) -> float:
    if metric == "bacc":
        return response.realized_utility.bacc_gain
    if metric == "brier":
        return response.realized_utility.brier_gain
    if metric == "log":
        return response.realized_utility.log_gain
    raise ProtocolError("P-DCAPS action response metric drifted.")


def _canonical_training_rows(
    predictions: Sequence[ActionPrediction],
    responses: Sequence[ActionResponse],
    *,
    outer_center: str,
    scored_center: str | None,
    additional_excluded_centers: tuple[str, ...],
) -> tuple[tuple[ActionPrediction, ActionResponse], ...]:
    response_by_prediction = {row.prediction_hash: row for row in responses}
    if len(response_by_prediction) != len(tuple(responses)):
        raise ProtocolError("P-DCAPS action calibration responses are duplicated.")
    selected: list[tuple[ActionPrediction, ActionResponse]] = []
    for prediction in predictions:
        route = prediction.key.route_key
        if route.surface_role != "pseudo" or route.outer_center != outer_center:
            continue
        if route.route_center == scored_center or route.route_center in additional_excluded_centers:
            continue
        response = response_by_prediction.get(prediction.prediction_hash)
        if response is None or response.key.action_key_hash != prediction.key.action_key_hash:
            raise ProtocolError("P-DCAPS action calibration row lineage drifted.")
        selected.append((prediction, response))
    selected.sort(key=lambda row: row[0].key.action_key_hash)
    if not selected or len({row[0].prediction_hash for row in selected}) != len(selected):
        raise ProtocolError("P-DCAPS action calibration rows are empty or duplicated.")
    expected_centers = tuple(
        center
        for center in CENTERS
        if center != outer_center
        and center != scored_center
        and center not in additional_excluded_centers
    )
    actual_centers = tuple(
        center
        for center in CENTERS
        if center in {row[0].key.route_key.route_center for row in selected}
    )
    if actual_centers != expected_centers:
        raise ProtocolError("P-DCAPS action calibration lacks a legal donor center.")
    return tuple(selected)


def fit_action_calibration_models(
    predictions: Sequence[ActionPrediction],
    responses: Sequence[ActionResponse],
    *,
    outer_center: str,
    scored_center: str | None = None,
    additional_excluded_centers: Sequence[str] = (),
) -> tuple[ActionCalibrationModel, ...]:
    """Fit the frozen three-metric model family for target H or pseudo J."""

    outer = str(outer_center)
    scored = None if scored_center is None else str(scored_center)
    additional = tuple(
        center
        for center in CENTERS
        if center in {str(value) for value in additional_excluded_centers}
    )
    if (
        outer not in CENTERS
        or scored is not None and (scored not in CENTERS or scored == outer)
        or len(additional) != len(tuple(additional_excluded_centers))
        or outer in additional
        or scored in additional
    ):
        raise ProtocolError("P-DCAPS action calibration H/J identity drifted.")
    rows = _canonical_training_rows(
        predictions,
        responses,
        outer_center=outer,
        scored_center=scored,
        additional_excluded_centers=additional,
    )
    selected_predictions = tuple(row[0] for row in rows)
    selected_responses = tuple(row[1] for row in rows)
    weights = build_hierarchical_weights(selected_responses)
    training_centers = tuple(
        center
        for center in CENTERS
        if center != outer and center != scored and center not in additional
    )
    models: list[ActionCalibrationModel] = []
    for metric in METRICS:
        response_values = np.ascontiguousarray(
            [_metric_response(row, metric) for row in selected_responses],
            dtype=np.float64,
        )
        response_hash = canonical_hash(
            {
                "schema_version": "pdcaps_action_model_response_v1",
                "metric": metric,
                "training_centers": training_centers,
                "rows": [
                    {
                        "prediction_hash": prediction.prediction_hash,
                        "response_hash": response.response_hash,
                        "value": float(value),
                    }
                    for prediction, response, value in zip(
                        selected_predictions,
                        selected_responses,
                        response_values,
                        strict=True,
                    )
                ],
            }
        )
        models.append(
            fit_weighted_ridge(
                build_action_descriptor_matrix(selected_predictions, metric),
                response_values,
                weights.as_array(),
                metric=metric,
                excluded_outer_center=outer,
                excluded_scored_center=scored,
                training_centers=training_centers,
                feature_names=action_feature_names(metric),
                training_response_hash=response_hash,
                weight_audit_hash=weights.weight_audit_hash,
            )
        )
    return tuple(models)


def build_nested_action_calibration_models(
    predictions: Sequence[ActionPrediction],
    responses: Sequence[ActionResponse],
    *,
    outer_center: str,
) -> tuple[
    tuple[ActionCalibrationModel, ...],
    tuple[tuple[str, tuple[ActionCalibrationModel, ...]], ...],
]:
    """Fit the target model plus every leave-J-out pseudo model serially."""

    outer = str(outer_center)
    target = fit_action_calibration_models(
        predictions, responses, outer_center=outer, scored_center=None
    )
    pseudo = tuple(
        (
            center,
            fit_action_calibration_models(
                predictions,
                responses,
                outer_center=outer,
                scored_center=center,
            ),
        )
        for center in CENTERS
        if center != outer
    )
    return target, pseudo


def build_reliability_oof_action_models(
    predictions: Sequence[ActionPrediction],
    responses: Sequence[ActionResponse],
    *,
    outer_center: str,
    scored_context: str | None = None,
) -> tuple[tuple[str, tuple[ActionCalibrationModel, ...]], ...]:
    """Fit K-scored models excluding H, K, and optional pseudo context J."""

    outer = str(outer_center)
    context = None if scored_context is None else str(scored_context)
    if (
        outer not in CENTERS
        or context is not None and (context not in CENTERS or context == outer)
    ):
        raise ProtocolError("P-DCAPS reliability OOF model context drifted.")
    return tuple(
        (
            center,
            fit_action_calibration_models(
                predictions,
                responses,
                outer_center=outer,
                scored_center=center,
                additional_excluded_centers=(() if context is None else (context,)),
            ),
        )
        for center in CENTERS
        if center != outer and center != context
    )


def calibrated_utility_for_prediction(
    prediction: ActionPrediction,
    models: Sequence[ActionCalibrationModel],
) -> FavorableUtility:
    rows = tuple(models)
    route = prediction.key.route_key
    if (
        tuple(model.metric for model in rows) != METRICS
        or any(model.excluded_outer_center != route.excluded_outer_center for model in rows)
        or any(model.excluded_scored_center != route.excluded_scored_center for model in rows)
    ):
        raise ProtocolError("P-DCAPS action calibration replay exclusion drifted.")
    return FavorableUtility.from_array(
        [
            predict_weighted_ridge(
                model,
                build_action_descriptor(prediction, model.metric),
            )
            for model in rows
        ]
    )


def calibrate_action(
    prediction: ActionPrediction,
    models: Sequence[ActionCalibrationModel],
    reliability: ActionStratumReliability,
) -> CalibratedAction:
    rows = tuple(models)
    utility = calibrated_utility_for_prediction(prediction, rows)
    return CalibratedAction(
        prediction,
        utility,
        tuple((model.metric, model.model_hash) for model in rows),
        rows[0].excluded_outer_center,
        rows[0].excluded_scored_center,
        reliability,
    )


def select_calibrated_action(
    actions: Sequence[CalibratedAction],
    *,
    empty_route_key: object | None = None,
) -> CalibratedActionSelection:
    """Quarantine first, then select by BACC with frozen deterministic ties."""

    rows = tuple(sorted(actions, key=lambda row: row.prediction.key.action_key_hash))
    if not rows:
        from ..contracts import RouteKey

        if not isinstance(empty_route_key, RouteKey):
            raise ProtocolError(
                "P-DCAPS empty action selection requires its sealed route key."
            )
        return CalibratedActionSelection(
            empty_route_key,
            (),
            (),
            None,
            FavorableUtility.zeros(),
            True,
            "EXACT_P_NO_CROSSING_ACTION",
        )
    route = rows[0].prediction.key.route_key
    if (
        any(row.prediction.key.route_key != route for row in rows)
        or len({row.prediction.key.action_key_hash for row in rows}) != len(rows)
    ):
        raise ProtocolError("P-DCAPS case-local action selection topology drifted.")
    eligible = tuple(row for row in rows if row.eligible)
    hashes = tuple(row.calibrated_action_hash for row in rows)
    quarantined = tuple(row.calibrated_action_hash for row in rows if row.quarantined)
    if not eligible:
        reason = (
            "EXACT_P_ALL_ACTIONS_QUARANTINED"
            if len(quarantined) == len(rows)
            else "EXACT_P_NO_JOINTLY_SAFE_POSITIVE_ACTION"
        )
        return CalibratedActionSelection(
            route,
            hashes,
            quarantined,
            None,
            FavorableUtility.zeros(),
            True,
            reason,
        )
    best_bacc = max(row.calibrated_utility.bacc_gain for row in eligible)
    tied = tuple(
        row
        for row in eligible
        if best_bacc - row.calibrated_utility.bacc_gain <= TIE_TOLERANCE
    )
    selected = min(
        tied,
        key=lambda row: (
            ACTION_FAMILIES.index(row.prediction.key.family),
            DIRECTIONS.index(row.prediction.key.direction),
            row.prediction.key.action_key_hash,
        ),
    )
    return CalibratedActionSelection(
        route,
        hashes,
        quarantined,
        selected.prediction.key,
        selected.calibrated_utility,
        False,
        "CALIBRATED_ACTION_SELECTED",
    )


def calibrate_and_select_actions(
    predictions: Sequence[ActionPrediction],
    models: Sequence[ActionCalibrationModel],
    reliabilities: Sequence[ActionStratumReliability],
    *,
    empty_route_key: object | None = None,
) -> tuple[tuple[CalibratedAction, ...], CalibratedActionSelection]:
    """Public case-local runtime from sealed action DTOs to one decision."""

    prediction_rows = tuple(predictions)
    reliability_rows = tuple(reliabilities)
    if not prediction_rows:
        return (), select_calibrated_action((), empty_route_key=empty_route_key)
    if (
        tuple(row.stratum for row in reliability_rows) != ACTION_STRATA
        or len({row.reliability_hash for row in reliability_rows}) != len(reliability_rows)
    ):
        raise ProtocolError("P-DCAPS reliability family is incomplete or misordered.")
    by_stratum: Mapping[tuple[str, str], ActionStratumReliability] = {
        row.stratum: row for row in reliability_rows
    }
    calibrated = tuple(
        calibrate_action(prediction, models, by_stratum[prediction.key.stratum])
        for prediction in sorted(
            prediction_rows, key=lambda row: row.key.action_key_hash
        )
    )
    return calibrated, select_calibrated_action(calibrated)


__all__ = (
    "build_nested_action_calibration_models",
    "build_reliability_oof_action_models",
    "calibrate_action",
    "calibrate_and_select_actions",
    "calibrated_utility_for_prediction",
    "fit_action_calibration_models",
    "select_calibrated_action",
)
