from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.source_active_selective_router_v7 import (
    AdmissionConfig,
    DISABLED_OPPORTUNITY_THRESHOLD,
    Direction,
    FitConfig,
    LabelFreeAction,
    RiskCoverageConfig,
    SourceActionOutcome,
    build_effective_menu,
    calibrate_policy_risk_coverage,
    evaluate_outer_admission,
    fit_source_lodo,
    float32_probability_hex,
    predict_target_actions,
    probability_bytes_to_hex,
    probability_hex_to_bytes,
    select_exact_top1,
)


OUTER = "H"
SOURCES = ("A", "B", "C", "D")
BASELINE = float32_probability_hex((0.2, 0.7, 0.3))
D01_PROBABILITY = float32_probability_hex((0.8, 0.7, 0.3))
D10_PROBABILITY = float32_probability_hex((0.2, 0.3, 0.3))


def _action(
    *,
    query: str,
    case: str,
    action_id: str,
    direction: Direction,
    activity: float,
    preference: float,
    probability: tuple[str, ...],
) -> LabelFreeAction:
    return LabelFreeAction(
        outer_target_id=OUTER,
        query_center_id=query,
        case_id=case,
        action_id=action_id,
        action_kind="U",
        direction=direction,
        candidate_source_id=None,
        feature_names=("activity_signal", "direction_signal"),
        feature_values=(activity, preference),
        baseline_probability_hex=BASELINE,
        action_probability_hex=probability,
    )


def _surface() -> tuple[SourceActionOutcome, ...]:
    rows: list[SourceActionOutcome] = []
    for center in SOURCES:
        for ordinal in range(6):
            positive = ordinal < 4
            preference = 1.0 if ordinal % 2 == 0 else -1.0
            activity = 1.0 if positive else -1.0
            case = f"case-{ordinal}"
            d01_gain = (0.10 if preference > 0 else 0.02) if positive else -0.01
            d10_gain = (0.10 if preference < 0 else 0.02) if positive else -0.01
            rows.extend(
                (
                    SourceActionOutcome(
                        action=_action(
                            query=center,
                            case=case,
                            action_id="d01",
                            direction=Direction.D01,
                            activity=activity,
                            preference=preference,
                            probability=D01_PROBABILITY,
                        ),
                        bacc_gain=d01_gain,
                        brier_delta=-d01_gain / 2.0,
                        log_delta=-d01_gain / 3.0,
                    ),
                    SourceActionOutcome(
                        action=_action(
                            query=center,
                            case=case,
                            action_id="d10",
                            direction=Direction.D10,
                            activity=activity,
                            preference=preference,
                            probability=D10_PROBABILITY,
                        ),
                        bacc_gain=d10_gain,
                        brier_delta=-d10_gain / 2.0,
                        log_delta=-d10_gain / 3.0,
                    ),
                )
            )
    return tuple(rows)


def _menus_with_explicit_inactive_cases(
    surface: tuple[SourceActionOutcome, ...],
) -> tuple[object, ...]:
    grouped: dict[tuple[str, str], list[LabelFreeAction]] = {}
    for row in surface:
        grouped.setdefault(
            (row.action.query_center_id, row.action.case_id), []
        ).append(row.action)
    menus = [build_effective_menu(rows) for _, rows in sorted(grouped.items())]
    for center in SOURCES:
        noop = _action(
            query=center,
            case="inactive-case",
            action_id="noop",
            direction=Direction.D01,
            activity=-2.0,
            preference=0.0,
            probability=BASELINE,
        )
        menus.append(build_effective_menu((noop,)))
    return tuple(menus)


def test_effective_menu_filter_is_label_free_symmetric_and_deterministic() -> None:
    assert probability_bytes_to_hex(probability_hex_to_bytes(BASELINE)) == BASELINE
    for query in ("A", OUTER):
        noop = _action(
            query=query,
            case="case",
            action_id="noop",
            direction=Direction.D01,
            activity=0.0,
            preference=0.0,
            probability=BASELINE,
        )
        representative = _action(
            query=query,
            case="case",
            action_id="a-representative",
            direction=Direction.D01,
            activity=1.0,
            preference=1.0,
            probability=D01_PROBABILITY,
        )
        duplicate = _action(
            query=query,
            case="case",
            action_id="z-duplicate",
            direction=Direction.D01,
            activity=1.0,
            preference=1.0,
            probability=D01_PROBABILITY,
        )
        menu = build_effective_menu((duplicate, noop, representative))
        assert tuple(row.action_id for row in menu.actions) == ("a-representative",)
        assert menu.dropped_noop_action_ids == ("noop",)
        assert menu.duplicate_representatives == (("z-duplicate", "a-representative"),)
        assert menu.actions[0].action_probability_hex == D01_PROBABILITY


def test_contracts_block_target_outcomes_and_outcome_bearing_features() -> None:
    target = _action(
        query=OUTER,
        case="target",
        action_id="d01",
        direction=Direction.D01,
        activity=1.0,
        preference=1.0,
        probability=D01_PROBABILITY,
    )
    with pytest.raises(ProtocolError, match="Target evaluation labels"):
        SourceActionOutcome(target, 0.1, -0.1, -0.1)
    with pytest.raises(ProtocolError, match="outcome-bearing"):
        LabelFreeAction(
            outer_target_id=OUTER,
            query_center_id="A",
            case_id="case",
            action_id="bad",
            action_kind="U",
            direction=Direction.D01,
            candidate_source_id=None,
            feature_names=("oracle_bacc_gain",),
            feature_values=(1.0,),
            baseline_probability_hex=BASELINE,
            action_probability_hex=D01_PROBABILITY,
        )


def test_source_lodo_is_strict_and_persists_numeric_oof() -> None:
    result = fit_source_lodo(_surface(), config=FitConfig(rank_alpha=0.1))
    assert result.final_model.training_center_ids == SOURCES
    assert len(result.oof_predictions) == 24
    assert all(
        row.query_center_id not in row.training_center_ids
        and OUTER not in row.training_center_ids
        for row in result.oof_predictions
    )
    payload = result.numeric_oof_payload()
    assert len(payload["rows"]) == 24
    assert all("opportunity_probability" in row for row in payload["rows"])
    json.dumps(payload, sort_keys=True)
    json.dumps(asdict(result.final_model), sort_keys=True)


def test_complete_menu_inventory_retains_no_active_cases_and_nested_tuning() -> None:
    surface = _surface()
    menus = _menus_with_explicit_inactive_cases(surface)
    grid = (
        FitConfig(opportunity_alpha=0.1, rank_alpha=0.1),
        FitConfig(opportunity_alpha=1.0, rank_alpha=1.0),
    )
    result = fit_source_lodo(surface, effective_menus=menus, config_grid=grid)
    assert len(result.oof_predictions) == 28
    inactive = [row for row in result.oof_predictions if row.case_id == "inactive-case"]
    assert len(inactive) == len(SOURCES)
    assert all(row.opportunity_probability == 0.0 and not row.action_scores for row in inactive)
    assert len(result.config_selections) == len(SOURCES) + 1
    assert all(len(selection.scores) == len(grid) for selection in result.config_selections)
    assert all(
        heldout not in selection.training_center_ids
        for heldout, selection in zip(SOURCES, result.config_selections[:-1], strict=True)
    )
    admission = evaluate_outer_admission(
        result.oof_predictions,
        surface,
        effective_menus=menus,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.1,
            min_delete_center_top1_excess=0.1,
            min_opportunity_top1_accuracy=0.9,
            min_opportunity_cases=8,
        ),
    )
    assert admission.case_count == 28
    calibration = calibrate_policy_risk_coverage(
        result.oof_predictions,
        surface,
        effective_menus=menus,
        nested_policy_folds=result.nested_policy_folds,
        config=RiskCoverageConfig(
            opportunity_thresholds=(0.4, 0.5, 0.6),
            rank_margin_thresholds=(0.0,),
            min_case_equal_bacc_gain=0.01,
            min_delete_center_bacc_gain=0.01,
            max_routed_harm_rate=0.0,
            max_case_equal_brier_delta=0.0,
            max_case_equal_log_delta=0.0,
            min_coverage=0.25,
            min_routed_cases=4,
        ),
    )
    assert calibration.selected_replay.case_count == 28
    assert calibration.nested_replay.case_count == 28


def test_admission_exploits_conditional_rank_signal_without_prevalence_penalty() -> None:
    surface = _surface()
    result = fit_source_lodo(surface, config=FitConfig(rank_alpha=0.1))
    admission = evaluate_outer_admission(
        result.oof_predictions,
        surface,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.1,
            min_delete_center_top1_excess=0.1,
            min_opportunity_top1_accuracy=0.9,
            min_opportunity_cases=8,
        ),
    )
    assert admission.admitted
    assert admission.learned_top1_accuracy == pytest.approx(1.0)
    assert admission.always_b_top1_accuracy == pytest.approx(0.0)
    assert admission.pooled_top1_excess == pytest.approx(1.0)
    assert admission.opportunity_top1_accuracy == pytest.approx(1.0)


def test_sparse_opportunity_prevalence_does_not_erase_rank_skill() -> None:
    surface = _surface()
    menus = list(_menus_with_explicit_inactive_cases(surface))
    for center in SOURCES:
        for ordinal in range(5):
            noop = _action(
                query=center,
                case=f"extra-inactive-{ordinal}",
                action_id="noop",
                direction=Direction.D01,
                activity=-2.0,
                preference=0.0,
                probability=BASELINE,
            )
            menus.append(build_effective_menu((noop,)))
    result = fit_source_lodo(
        surface,
        effective_menus=tuple(menus),
        config=FitConfig(rank_alpha=0.1),
    )
    admission = evaluate_outer_admission(
        result.oof_predictions,
        surface,
        effective_menus=tuple(menus),
        config=AdmissionConfig(
            min_pooled_top1_excess=0.1,
            min_delete_center_top1_excess=0.1,
            min_opportunity_top1_accuracy=0.9,
            min_opportunity_cases=8,
        ),
    )
    assert admission.case_count == 48
    assert admission.opportunity_case_count == 16
    assert admission.admitted
    assert admission.pooled_top1_excess == pytest.approx(1.0)


def test_policy_level_calibration_and_target_top1_preserve_exact_bytes() -> None:
    surface = _surface()
    result = fit_source_lodo(surface, config=FitConfig(rank_alpha=0.1))
    admission = evaluate_outer_admission(
        result.oof_predictions,
        surface,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.1,
            min_delete_center_top1_excess=0.1,
            min_opportunity_top1_accuracy=0.9,
            min_opportunity_cases=8,
        ),
    )
    calibration = calibrate_policy_risk_coverage(
        result.oof_predictions,
        surface,
        nested_policy_folds=result.nested_policy_folds,
        config=RiskCoverageConfig(
            opportunity_thresholds=(0.4, 0.5, 0.6, 0.7),
            rank_margin_thresholds=(0.0, 0.01),
            min_case_equal_bacc_gain=0.01,
            min_delete_center_bacc_gain=0.01,
            max_routed_harm_rate=0.0,
            max_case_equal_brier_delta=0.0,
            max_case_equal_log_delta=0.0,
            min_coverage=0.25,
            min_routed_cases=4,
        ),
    )
    assert calibration.calibrated
    assert calibration.selected_replay.routed_harm_rate == 0.0
    target_actions = (
        _action(
            query=OUTER,
            case="target-case",
            action_id="d01",
            direction=Direction.D01,
            activity=1.0,
            preference=1.0,
            probability=D01_PROBABILITY,
        ),
        _action(
            query=OUTER,
            case="target-case",
            action_id="d10",
            direction=Direction.D10,
            activity=1.0,
            preference=1.0,
            probability=D10_PROBABILITY,
        ),
    )
    menu, prediction = predict_target_actions(result.final_model, target_actions)
    decision = select_exact_top1(menu, prediction, admission, calibration)
    assert decision.selected_action_id == "d01"
    assert decision.reason == "ROUTED_EXACT_TOP1"
    assert decision.probability_hex == D01_PROBABILITY
    assert not decision.exact_b_fallback

    failed = replace(admission, admitted=False, reasons=("TEST_FAILURE",))
    fallback = select_exact_top1(menu, prediction, failed, calibration)
    assert fallback.selected_action_id == "B"
    assert fallback.exact_b_fallback
    assert fallback.probability_hex == BASELINE


def test_zero_opportunity_folds_disable_ranker_and_keep_exact_b() -> None:
    surface = tuple(
        replace(row, bacc_gain=-0.1, brier_delta=0.1, log_delta=0.1)
        for row in _surface()
    )
    result = fit_source_lodo(surface)
    assert all(not row.action_scores for row in result.oof_predictions)
    assert not result.final_model.d01_rank_head.available
    assert not result.final_model.d10_rank_head.available
    admission = evaluate_outer_admission(result.oof_predictions, surface)
    assert not admission.admitted
    assert "INSUFFICIENT_SOURCE_OPPORTUNITY_CASES" in admission.reasons
    calibration = calibrate_policy_risk_coverage(
        result.oof_predictions,
        surface,
        nested_policy_folds=result.nested_policy_folds,
    )
    assert not calibration.calibrated
    assert calibration.opportunity_threshold == DISABLED_OPPORTUNITY_THRESHOLD
    assert all(
        opportunity == DISABLED_OPPORTUNITY_THRESHOLD and margin == 0.0
        for _center, opportunity, margin in calibration.heldout_thresholds
    )


def test_single_active_action_bypasses_rank_ambiguity_gate() -> None:
    surface = _surface()
    result = fit_source_lodo(surface, config=FitConfig(rank_alpha=0.1))
    admission = evaluate_outer_admission(
        result.oof_predictions,
        surface,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.1,
            min_delete_center_top1_excess=0.1,
            min_opportunity_top1_accuracy=0.9,
            min_opportunity_cases=8,
        ),
    )
    calibration = calibrate_policy_risk_coverage(
        result.oof_predictions,
        surface,
        nested_policy_folds=result.nested_policy_folds,
        config=RiskCoverageConfig(
            opportunity_thresholds=(0.4, 0.5, 0.6, 0.7),
            rank_margin_thresholds=(0.02,),
            min_case_equal_bacc_gain=0.01,
            min_delete_center_bacc_gain=0.01,
            max_routed_harm_rate=0.0,
            max_case_equal_brier_delta=0.0,
            max_case_equal_log_delta=0.0,
            min_coverage=0.25,
            min_routed_cases=4,
        ),
    )
    target = _action(
        query=OUTER,
        case="single-action-target",
        action_id="d01",
        direction=Direction.D01,
        activity=1.0,
        preference=1.0,
        probability=D01_PROBABILITY,
    )
    menu, prediction = predict_target_actions(result.final_model, (target,))
    assert len(prediction.action_scores) == 1
    assert prediction.rank_margin == 0.0
    assert prediction.passes_rank_margin(calibration.rank_margin_threshold)
    decision = select_exact_top1(menu, prediction, admission, calibration)
    assert decision.selected_action_id == "d01"
    assert not decision.exact_b_fallback
