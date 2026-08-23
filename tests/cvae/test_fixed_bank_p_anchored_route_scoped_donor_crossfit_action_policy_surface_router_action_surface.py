from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    ActionCalibrationModel,
    ActionKey,
    ActionPrediction,
    ActionResponse,
    ActionStratumReliability,
    action_feature_names,
    build_action_descriptor,
    build_action_reliability_by_stratum,
    build_action_response,
    build_hierarchical_weights,
    calibrate_and_select_actions,
    canonical_probabilities,
    fit_weighted_ridge,
    predict_weighted_ridge,
    probability_sha256,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    BankViability,
    FavorableUtility,
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_STRATA,
    METRICS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.label_firewall import (
    LabelFirewall,
    PseudoResponseLabelCapability,
    pseudo_response_scope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.route_support import (
    BinaryLabel,
)
from midogpp_thesis.cvae.protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _viability(token: object = "bank") -> BankViability:
    return BankViability(True, True, (("0", 20.0), ("1", 20.0)), 5.0, _hash(token))


def _route(*, outer: str, center: str, case: str, target: bool = False) -> RouteKey:
    return RouteKey(
        "target" if target else "pseudo",
        outer,
        outer if target else center,
        case,
        outer,
        None if target else center,
        _hash((outer, center, case, target, "fit")),
    )


def _prediction(
    *,
    outer: str = "0",
    center: str = "1",
    case: str = "case-1",
    family: str = "B",
    direction: str = "zero_to_one",
    action_id: str = "a",
    utility: tuple[float, float, float] = (0.2, 0.1, 0.1),
    target: bool = False,
) -> ActionPrediction:
    probabilities = np.asarray([0.2, 0.8], dtype=np.float32)
    key = ActionKey(
        _route(outer=outer, center=center, case=case, target=target),
        family,
        direction,
        action_id,
        probability_sha256(probabilities),
        _hash((outer, "surface")),
    )
    return ActionPrediction(key, FavorableUtility(*utility), 0.25, _viability((center, case)))


def _response(
    prediction: ActionPrediction,
    utility: tuple[float, float, float],
) -> ActionResponse:
    return ActionResponse(
        prediction.key,
        prediction.prediction_hash,
        FavorableUtility(*utility),
        2,
        10,
        10,
        20,
        _hash("P"),
        _hash((prediction.key.route_key.route_center, "rows")),
    )


def _model_family(
    *,
    outer: str = "0",
    scored: str | None = None,
    bacc_intercept: float = 0.0,
    bacc_coefficients: tuple[float, ...] | None = None,
    proper_intercept: float = 0.1,
    additional_excluded: tuple[str, ...] = (),
) -> tuple[ActionCalibrationModel, ...]:
    training_centers = tuple(
        center
        for center in CENTERS
        if center != outer and center != scored and center not in additional_excluded
    )
    rows = []
    for metric in METRICS:
        names = action_feature_names(metric)
        coefficients = (
            bacc_coefficients
            if metric == "bacc" and bacc_coefficients is not None
            else tuple(0.0 for _ in names)
        )
        intercept = bacc_intercept if metric == "bacc" else proper_intercept
        rows.append(
            ActionCalibrationModel(
                metric,
                outer,
                scored,
                training_centers,
                names,
                tuple(0.0 for _ in names),
                tuple(1.0 for _ in names),
                intercept,
                coefficients,
                1.0,
                10,
                _hash((outer, scored, metric, "responses")),
                _hash((outer, scored, metric, "weights")),
                "solve",
            )
        )
    return tuple(rows)


def _reliability(
    family: str,
    direction: str,
    *,
    outer: str = "0",
    scored: str | None = None,
    passed: bool = True,
) -> ActionStratumReliability:
    centers = tuple(center for center in CENTERS if center != outer and center != scored)
    bacc = 0.1 if passed else -0.1
    proper = 0.05 if passed else -0.05
    means = tuple((center, bacc, proper, proper) for center in centers)
    return ActionStratumReliability(
        outer,
        scored,
        family,
        direction,
        centers,
        means,
        FavorableUtility(bacc, proper, proper),
        0.5 if passed else -0.5,
        True,
        len(centers) if passed else 0,
        6,
        True,
        len(centers),
        _hash((outer, scored, family, direction, passed)),
    )


def test_response_equations_use_fixed_denominators_and_do_not_persist_labels() -> None:
    prediction = _prediction(center="1")
    baseline = np.asarray([0.6, 0.4], dtype=np.float32)
    action = np.asarray([0.2, 0.8], dtype=np.float32)
    prediction = ActionPrediction(
        ActionKey(
            prediction.key.route_key,
            prediction.key.family,
            prediction.key.direction,
            prediction.key.action_id,
            probability_sha256(action),
            prediction.key.action_surface_seal_hash,
        ),
        prediction.predicted_utility,
        prediction.crossing_fraction,
        prediction.bank_viability,
    )
    scope = pseudo_response_scope(prediction.key.route_key)
    rows = (
        BinaryLabel("1", "case-1", "sample-a", 0, scope),
        BinaryLabel("1", "case-1", "sample-b", 1, scope),
    )
    with pytest.raises(ProtocolError, match="pseudo-response label capability"):
        PseudoResponseLabelCapability(prediction.key.route_key, rows, scope)
    firewall = LabelFirewall(lambda _keys, _scope: rows)
    firewall.advance_support()
    firewall.seal_action_surface(prediction.key.action_surface_seal_hash)
    firewall.advance_pseudo_response()
    capability = firewall.open_pseudo_response(
        route_key=prediction.key.route_key,
        sample_ids=("sample-a", "sample-b"),
    )
    response = build_action_response(
        prediction,
        baseline_probabilities=baseline,
        action_probabilities=action,
        label_capability=capability,
        positive_denominator=1,
        negative_denominator=1,
        row_denominator=2,
    )
    assert response.realized_utility.bacc_gain == pytest.approx(1.0)
    assert response.realized_utility.brier_gain == pytest.approx(0.32)
    assert response.realized_utility.log_gain == pytest.approx(np.log(2.0))
    payload = response.to_payload()
    assert payload["raw_labels_persisted"] is False
    assert "labels" not in payload
    canonical = canonical_probabilities(action)
    assert canonical.dtype == np.float32 and canonical.flags.c_contiguous
    assert canonical.flags.writeable is False


def test_action_descriptor_has_frozen_metric_stratum_and_interaction_columns() -> None:
    prediction = _prediction(
        family="R_NINE_ARM_ROBUST",
        direction="one_to_zero",
        utility=(0.3, 0.2, 0.1),
    )
    names = action_feature_names("bacc")
    values = build_action_descriptor(prediction, "bacc")
    assert len(names) == len(values) == 14
    assert names[:2] == ("predicted_favorable_bacc", "crossing_fraction")
    stratum_index = ACTION_STRATA.index(("R_NINE_ARM_ROBUST", "one_to_zero"))
    assert values[0] == pytest.approx(0.3)
    assert values[1] == pytest.approx(0.25)
    assert values[2 + stratum_index] == 1.0
    assert np.sum(values[2:8]) == 1.0
    assert values[8 + stratum_index] == pytest.approx(0.3)
    assert values.flags.c_contiguous and values.dtype == np.float64
    assert values.flags.writeable is False


def test_hierarchical_weights_are_equal_center_route_and_action() -> None:
    predictions = (
        _prediction(center="1", case="a", action_id="a1"),
        _prediction(center="1", case="a", action_id="a2", family="I_OPPORTUNITY_GATED"),
        _prediction(center="2", case="b", action_id="b1"),
        _prediction(center="2", case="c", action_id="c1"),
    )
    audit = build_hierarchical_weights(tuple(_response(row, (0.1, 0.1, 0.1)) for row in predictions))
    assert audit.row_weights == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert dict(audit.effective_total_by_center) == pytest.approx({"1": 0.5, "2": 0.5})
    assert audit.total_weight == pytest.approx(1.0)
    assert len(audit.weight_audit_hash) == 64


def test_manual_weighted_ridge_uses_fit_only_scaling_and_replays_serialized_model() -> None:
    features = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]])
    response = np.asarray([0.0, 1.0, 2.0, 3.0])
    weights = np.full(4, 0.25)
    model = fit_weighted_ridge(
        features,
        response,
        weights,
        metric="bacc",
        excluded_outer_center="0",
        excluded_scored_center=None,
        training_centers=("1", "2"),
        feature_names=("x", "z"),
        training_response_hash=_hash("ridge-y"),
        weight_audit_hash=_hash("ridge-w"),
    )
    assert model.feature_mean == pytest.approx((2.5, 0.5))
    assert model.feature_scale == pytest.approx((np.sqrt(1.25), 0.5))
    target = np.asarray([1000.0, 1.0])
    expected = model.intercept + np.sum(
        ((target - model.feature_mean) / model.feature_scale) * model.coefficients
    )
    assert predict_weighted_ridge(model, target) == pytest.approx(expected)
    assert model.feature_mean == pytest.approx((2.5, 0.5))
    assert model.to_payload()["estimator_persisted"] is False


def test_calibration_can_change_action_before_case_argmax_and_uses_frozen_ties() -> None:
    b_prediction = _prediction(
        target=True,
        center="0",
        family="B",
        action_id="b",
        utility=(0.8, 0.1, 0.1),
    )
    i_prediction = _prediction(
        target=True,
        center="0",
        family="I_OPPORTUNITY_GATED",
        action_id="i",
        utility=(0.4, 0.1, 0.1),
    )
    names = action_feature_names("bacc")
    coefficients = [0.0] * len(names)
    coefficients[0] = 1.0
    coefficients[2 + ACTION_STRATA.index(("B", "zero_to_one"))] = -1.0
    models = _model_family(bacc_coefficients=tuple(coefficients))
    reliabilities = tuple(_reliability(*stratum) for stratum in ACTION_STRATA)
    calibrated, selection = calibrate_and_select_actions(
        (b_prediction, i_prediction), models, reliabilities
    )
    assert b_prediction.predicted_utility.bacc_gain > i_prediction.predicted_utility.bacc_gain
    assert selection.selected_action_key is not None
    assert selection.selected_action_key.family == "I_OPPORTUNITY_GATED"
    assert selection.exact_p_fallback is False
    assert any(row.prediction.key.family == "B" and not row.eligible for row in calibrated)

    tie_models = _model_family(bacc_intercept=0.2)
    _rows, tie_selection = calibrate_and_select_actions(
        (i_prediction, b_prediction), tie_models, reliabilities
    )
    assert tie_selection.selected_action_key is not None
    assert tie_selection.selected_action_key.family == "B"


def test_fully_oof_anti_calibrated_robust_zero_to_one_stratum_is_quarantined() -> None:
    predictions = []
    responses = []
    donor_centers = tuple(center for center in CENTERS if center != "0")
    for index, center in enumerate(donor_centers, start=1):
        prediction = _prediction(
            center=center,
            case=f"case-{center}",
            family="R_NINE_ARM_ROBUST",
            direction="zero_to_one",
            utility=(index / 100.0, 0.1, 0.1),
        )
        predictions.append(prediction)
        responses.append(_response(prediction, (-index / 100.0, 0.1, 0.1)))
    identity_coefficients = [0.0] * len(action_feature_names("bacc"))
    identity_coefficients[0] = 1.0
    oof_models = {
        center: _model_family(
            scored=center,
            bacc_coefficients=tuple(identity_coefficients),
        )
        for center in donor_centers
    }
    gates = build_action_reliability_by_stratum(
        predictions,
        responses,
        oof_models,
        outer_center="0",
    )
    robust_gate = gates[ACTION_STRATA.index(("R_NINE_ARM_ROBUST", "zero_to_one"))]
    assert robust_gate.bacc_spearman == pytest.approx(-1.0)
    assert robust_gate.equal_center_utility.bacc_gain < 0.0
    assert robust_gate.passed is False
    assert "BACC_SPEARMAN_NOT_POSITIVE" in robust_gate.reason_codes

    target = _prediction(
        target=True,
        center="0",
        family="R_NINE_ARM_ROBUST",
        direction="zero_to_one",
    )
    target_models = _model_family(bacc_intercept=0.2)
    target_gates = tuple(
        robust_gate if stratum == robust_gate.stratum else _reliability(*stratum)
        for stratum in ACTION_STRATA
    )
    calibrated, selection = calibrate_and_select_actions((target,), target_models, target_gates)
    assert calibrated[0].quarantined is True
    assert selection.exact_p_fallback is True

    with pytest.raises(ProtocolError, match="OOF action-model family exclusion"):
        build_action_reliability_by_stratum(
            predictions,
            responses,
            oof_models,
            outer_center="0",
            scored_center="1",
        )
    context_models = {
        center: _model_family(
            scored=center,
            bacc_coefficients=tuple(identity_coefficients),
            additional_excluded=("1",),
        )
        for center in donor_centers
        if center != "1"
    }
    context_gates = build_action_reliability_by_stratum(
        predictions,
        responses,
        context_models,
        outer_center="0",
        scored_center="1",
    )
    context_robust = context_gates[
        ACTION_STRATA.index(("R_NINE_ARM_ROBUST", "zero_to_one"))
    ]
    assert "1" not in context_robust.represented_centers


def test_nonpositive_calibrated_action_falls_back_to_byte_exact_p() -> None:
    prediction = _prediction(target=True, center="0", utility=(0.5, 0.1, 0.1))
    models = _model_family(bacc_intercept=1.0e-13)
    reliabilities = tuple(_reliability(*stratum) for stratum in ACTION_STRATA)
    _calibrated, selection = calibrate_and_select_actions(
        (prediction,), models, reliabilities
    )
    assert selection.exact_p_fallback is True
    assert selection.selected_action_key is None
    assert selection.selected_utility == FavorableUtility.zeros()


def test_action_dtos_are_frozen_picklable_and_full_hashes_cover_identity() -> None:
    first = _prediction(action_id="first")
    repeated = _prediction(action_id="first")
    changed = _prediction(action_id="changed")
    assert first.prediction_hash == repeated.prediction_hash
    assert first.prediction_hash != changed.prediction_hash
    assert len(first.key.action_key_hash) == len(first.prediction_hash) == 64
    assert pickle.loads(pickle.dumps(first)) == first
    with pytest.raises(FrozenInstanceError):
        first.crossing_fraction = 0.9  # type: ignore[misc]
