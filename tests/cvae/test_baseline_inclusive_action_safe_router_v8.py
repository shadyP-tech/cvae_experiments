from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
import midogpp_thesis.cvae.routing.baseline_inclusive_action_safe_router_v8 as router_v8
from midogpp_thesis.cvae.routing.baseline_inclusive_action_safe_router_v8 import (
    ActionEstimate,
    AdmissionConfig,
    Direction,
    FitConfig,
    LabelFreeAction,
    ResidualCalibrationCell,
    RiskCoverageConfig,
    SourceActionOutcome,
    build_effective_menu,
    calibrate_policy_risk_coverage,
    certify_action,
    evaluate_outer_admission,
    fit_source_lodo,
    float32_probability_hex,
    predict_target_actions,
    select_exact_top1,
)


OUTER = "H"
SOURCES = ("A", "B", "C", "D")
BASELINE = float32_probability_hex((0.2, 0.7, 0.3))
D01_PROBABILITY = float32_probability_hex((0.8, 0.7, 0.3))
D10_PROBABILITY = float32_probability_hex((0.2, 0.3, 0.3))
HXE_PROBABILITY = float32_probability_hex((0.65, 0.7, 0.3))


def _action(
    *,
    query: str,
    case: str,
    action_id: str,
    direction: Direction,
    signal: float,
    preference: float,
    probability: tuple[str, ...],
    kind: str = "U",
    candidate: str | None = None,
) -> LabelFreeAction:
    return LabelFreeAction(
        outer_target_id=OUTER,
        query_center_id=query,
        case_id=case,
        action_id=action_id,
        action_kind=kind,
        direction=direction,
        candidate_source_id=candidate,
        feature_names=("activity_signal", "direction_signal"),
        feature_values=(signal, preference),
        baseline_probability_hex=BASELINE,
        action_probability_hex=probability,
    )


def _surface(*, include_hxe: bool = False) -> tuple[SourceActionOutcome, ...]:
    rows: list[SourceActionOutcome] = []
    for center_index, center in enumerate(SOURCES):
        other = tuple(value for value in SOURCES if value != center)
        for ordinal in range(6):
            active = ordinal < 4
            signal = 1.0 if active else -1.0
            preference = 1.0 if ordinal % 2 == 0 else -1.0
            case = f"case-{ordinal}"
            d01_gain = (0.12 if preference > 0.0 else 0.04) if active else -0.08
            d10_gain = (0.12 if preference < 0.0 else 0.04) if active else -0.08
            for action_id, direction, gain, probability in (
                ("d01", Direction.D01, d01_gain, D01_PROBABILITY),
                ("d10", Direction.D10, d10_gain, D10_PROBABILITY),
            ):
                rows.append(
                    SourceActionOutcome(
                        _action(
                            query=center,
                            case=case,
                            action_id=action_id,
                            direction=direction,
                            signal=signal,
                            preference=preference,
                            probability=probability,
                        ),
                        gain,
                        -gain / 2.0,
                        -gain / 3.0,
                    )
                )
            if include_hxe:
                candidate = other[(center_index + ordinal) % len(other)]
                gain = 0.09 if active else -0.06
                rows.append(
                    SourceActionOutcome(
                        _action(
                            query=center,
                            case=case,
                            action_id=f"hxe-{candidate}",
                            direction=Direction.D01,
                            signal=signal,
                            preference=0.25,
                            probability=HXE_PROBABILITY,
                            kind="HXE",
                            candidate=candidate,
                        ),
                        gain,
                        -gain / 2.0,
                        -gain / 3.0,
                    )
                )
    return tuple(rows)


def _fit_config() -> FitConfig:
    return FitConfig(
        harm_alpha=0.01,
        endpoint_alpha=0.01,
        residual_quantile=1.0,
        max_harm_probability=0.49,
        max_action_brier_delta=0.0,
        max_action_log_delta=0.0,
        max_harm_brier_risk=1.0,
        max_harm_log_loss_risk=10.0,
        min_calibration_centers=2,
        min_calibration_rows_per_group=2,
    )


def _policy(result, surface):
    admission = evaluate_outer_admission(
        result.oof_predictions,
        surface,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.25,
            min_delete_center_top1_excess=0.25,
            min_opportunity_top1_accuracy=0.75,
            min_opportunity_cases=8,
        ),
    )
    calibration = calibrate_policy_risk_coverage(
        result.oof_predictions,
        surface,
        nested_policy_folds=result.nested_policy_folds,
        config=RiskCoverageConfig(
            certificate_confidence_thresholds=(0.0,),
            rank_margin_thresholds=(0.0,),
            min_case_equal_bacc_gain=0.0,
            min_delete_center_bacc_gain=0.0,
            max_routed_harm_rate=0.0,
            max_case_equal_brier_delta=0.0,
            max_case_equal_log_delta=0.0,
            min_coverage=0.1,
            min_routed_cases=4,
        ),
    )
    return admission, calibration


def test_v8_science_package_is_independent_of_exhausted_v7() -> None:
    root = Path(router_v8.__file__).parent
    for path in root.glob("*.py"):
        assert "source_active_selective_router_v7" not in path.read_text(encoding="utf-8")


def test_label_free_contract_preserves_menu_behavior_and_blocks_target_outcomes() -> None:
    noop = _action(
        query="A",
        case="case",
        action_id="noop",
        direction=Direction.D01,
        signal=0.0,
        preference=0.0,
        probability=BASELINE,
    )
    first = _action(
        query="A",
        case="case",
        action_id="a",
        direction=Direction.D01,
        signal=1.0,
        preference=1.0,
        probability=D01_PROBABILITY,
    )
    duplicate = _action(
        query="A",
        case="case",
        action_id="z",
        direction=Direction.D01,
        signal=1.0,
        preference=1.0,
        probability=D01_PROBABILITY,
    )
    menu = build_effective_menu((duplicate, noop, first))
    assert tuple(row.action_id for row in menu.actions) == ("a",)
    assert menu.dropped_noop_action_ids == ("noop",)
    assert menu.duplicate_representatives == (("z", "a"),)
    target = _action(
        query=OUTER,
        case="target",
        action_id="d01",
        direction=Direction.D01,
        signal=1.0,
        preference=1.0,
        probability=D01_PROBABILITY,
    )
    with pytest.raises(ProtocolError, match="Target evaluation labels"):
        SourceActionOutcome(target, 0.1, -0.1, -0.1)


def test_lodo_excludes_outer_and_held_source_as_query_and_candidate() -> None:
    result = fit_source_lodo(_surface(include_hxe=True), config=_fit_config())
    assert OUTER in result.final_model.excluded_center_ids
    assert OUTER not in result.final_model.training_center_ids
    assert OUTER not in result.final_model.training_candidate_ids
    assert all(
        row.query_center_id in row.excluded_center_ids
        and row.query_center_id not in row.training_center_ids
        and row.query_center_id not in row.training_candidate_ids
        for row in result.oof_predictions
    )
    assert all(
        fold.heldout_center_id not in row.training_center_ids
        and fold.heldout_center_id not in row.training_candidate_ids
        for fold in result.nested_policy_folds
        for row in (*fold.predictions, *fold.heldout_predictions)
    )


def test_unavailable_calibration_is_finite_unsafe_and_cannot_route() -> None:
    action = _action(
        query=OUTER,
        case="target",
        action_id="d01",
        direction=Direction.D01,
        signal=1.0,
        preference=1.0,
        probability=D01_PROBABILITY,
    )
    estimate = ActionEstimate(
        action.action_id,
        action.action_hash,
        "U:D01",
        Direction.D01,
        0.2,
        0.01,
        -0.1,
        -0.1,
        True,
    )
    certificate = certify_action(
        estimate,
        None,
        max_harm_probability=0.25,
        max_brier_delta=0.0,
        max_log_delta=0.0,
        max_harm_brier_risk=0.25,
        max_harm_log_loss_risk=0.7,
    )
    assert not certificate.safe
    assert certificate.failed_gates == ("RESIDUAL_CALIBRATION_UNAVAILABLE",)
    assert certificate.brier_delta_ucb < float("inf")


def test_negative_gain_lcb_is_diagnostic_not_an_action_safety_gate() -> None:
    action = _action(
        query=OUTER,
        case="relative-ranking",
        action_id="d01",
        direction=Direction.D01,
        signal=1.0,
        preference=1.0,
        probability=D01_PROBABILITY,
    )
    estimate = ActionEstimate(
        action.action_id,
        action.action_hash,
        "U:D01",
        Direction.D01,
        0.1,
        0.05,
        -0.1,
        -0.1,
        True,
    )
    cell = ResidualCalibrationCell(
        action_group="U:D01",
        calibration_center_ids=("A", "B"),
        residual_quantile=1.0,
        gain_shortfall_radius=0.2,
        harm_excess_radius=0.0,
        brier_excess_radius=0.0,
        log_excess_radius=0.0,
        harm_brier_risk=0.01,
        harm_log_loss_risk=0.1,
        row_count=4,
        available=True,
    )
    certificate = certify_action(
        estimate,
        cell,
        max_harm_probability=0.25,
        max_brier_delta=0.0,
        max_log_delta=0.0,
        max_harm_brier_risk=0.25,
        max_harm_log_loss_risk=0.7,
    )
    assert certificate.gain_lcb < 0.0
    assert certificate.safe
    assert certificate.failed_gates == ()


def test_source_development_requires_exact_case_action_membership() -> None:
    from midogpp_thesis.cvae.runtime.harp_v8_execution.source_development import (
        SourceDevelopmentState,
    )

    surface = _surface()
    menus = router_v8.group_effective_menus(tuple(row.action for row in surface))
    SourceDevelopmentState(menus, surface)
    with pytest.raises(ProtocolError, match="crossed sealed menus"):
        SourceDevelopmentState(menus, surface[:-1])


def test_unsafe_action_is_rejected_and_exact_b_bytes_are_preserved() -> None:
    surface = _surface()
    result = fit_source_lodo(surface, config=_fit_config())
    admission, calibration = _policy(result, surface)
    target_actions = (
        _action(
            query=OUTER,
            case="unsafe-target",
            action_id="d01",
            direction=Direction.D01,
            signal=-1.0,
            preference=1.0,
            probability=D01_PROBABILITY,
        ),
        _action(
            query=OUTER,
            case="unsafe-target",
            action_id="d10",
            direction=Direction.D10,
            signal=-1.0,
            preference=-1.0,
            probability=D10_PROBABILITY,
        ),
    )
    menu, prediction = predict_target_actions(result.final_model, target_actions)
    assert prediction.safe_action_ids == ()
    assert all(not row.safe for row in prediction.action_certificates)
    decision = select_exact_top1(menu, prediction, admission, calibration)
    assert decision.selected_action_id == "B"
    assert decision.exact_b_fallback
    assert decision.probability_hex == BASELINE
    assert b"".join(bytes.fromhex(value) for value in decision.probability_hex) == b"".join(
        bytes.fromhex(value) for value in BASELINE
    )


def test_synthetic_positive_opportunity_routes_nonzero_safe_top1() -> None:
    surface = _surface()
    result = fit_source_lodo(surface, config=_fit_config())
    admission, calibration = _policy(result, surface)
    assert admission.admitted
    assert calibration.calibrated
    assert calibration.selected_replay.routed_cases > 0
    assert calibration.nested_replay.routed_cases > 0
    assert all(
        heldout not in training
        for heldout, _confidence, _margin in calibration.heldout_thresholds
        for training in (
            next(
                fold.training_center_ids
                for fold in result.nested_policy_folds
                if fold.heldout_center_id == heldout
            ),
        )
    )
    target_actions = (
        _action(
            query=OUTER,
            case="positive-target",
            action_id="d01",
            direction=Direction.D01,
            signal=1.0,
            preference=1.0,
            probability=D01_PROBABILITY,
        ),
        _action(
            query=OUTER,
            case="positive-target",
            action_id="d10",
            direction=Direction.D10,
            signal=1.0,
            preference=1.0,
            probability=D10_PROBABILITY,
        ),
    )
    menu, prediction = predict_target_actions(result.final_model, target_actions)
    assert "d01" in prediction.safe_action_ids
    decision = select_exact_top1(menu, prediction, admission, calibration)
    assert decision.selected_action_id == "d01"
    assert not decision.exact_b_fallback
    assert decision.probability_hex == D01_PROBABILITY
    assert decision.reason == "ROUTED_CERTIFIED_SAFE_EXACT_TOP1"


def test_production_artifacts_bind_nested_folds_thresholds_and_certificates() -> None:
    """Exercise the science-to-durable-validation seam with real v8 objects."""

    from types import SimpleNamespace

    from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash
    from midogpp_thesis.cvae.runtime.harp_v8_execution.contracts import ArtifactValue
    from midogpp_thesis.cvae.runtime.harp_v8_execution.model_adapter import (
        OuterRouterBundle,
        RouterFitState,
        model_manifest,
    )
    from midogpp_thesis.cvae.runtime.harp_v8_execution.source_development import (
        SourceDevelopmentState,
    )
    from midogpp_thesis.cvae.runtime.harp_v8_execution.source_model_artifacts import (
        _numeric_oof_arrays,
        build_source_admission_artifact,
    )
    from midogpp_thesis.cvae.runtime.harp_v8_execution.validation import (
        _validate_numeric_oof,
    )

    surface = _surface()
    result = fit_source_lodo(surface, config=_fit_config())
    fit_state = RouterFitState((OuterRouterBundle(OUTER, result),))
    model_body = model_manifest(fit_state)
    model = ArtifactValue(
        state=fit_state,
        manifest={**model_body, "model_hash": canonical_hash(model_body)},
        arrays=_numeric_oof_arrays(fit_state),
    )
    menus = router_v8.group_effective_menus(tuple(row.action for row in surface))
    development = ArtifactValue(
        state=SourceDevelopmentState(menus, surface),
        manifest={"surface_hash": "a" * 64},
    )
    config = SimpleNamespace(
        model={
            "certificate_confidence_threshold_grid": [0.0],
            "rank_margin_threshold_grid": [0.0],
            "admission": {
                "min_pooled_top1_excess_over_always_b": 0.25,
                "min_delete_center_top1_excess_over_always_b": 0.25,
                "min_opportunity_top1_accuracy": 0.75,
                "min_opportunity_cases": 8,
            },
            "policy": {
                "min_case_equal_bacc_gain": 0.0,
                "min_delete_center_bacc_gain": 0.0,
                "max_routed_harm_rate": 0.0,
                "max_case_equal_brier_delta": 0.0,
                "max_case_equal_log_loss_delta": 0.0,
                "min_coverage": 0.1,
                "min_routed_cases": 4,
            },
        }
    )
    admission = build_source_admission_artifact(
        model,
        development,
        config=config,
    )
    _validate_numeric_oof(model, admission)

    bad_rows = [dict(row) for row in admission.manifest["source_policy_oof_rows"]]
    bad_rows[0]["nested_prediction_hash"] = "f" * 64
    with pytest.raises(ProtocolError, match="threshold provenance"):
        _validate_numeric_oof(
            model,
            ArtifactValue(
                state=admission.state,
                manifest={
                    **dict(admission.manifest),
                    "source_policy_oof_rows": bad_rows,
                },
                arrays=admission.arrays,
            ),
        )
