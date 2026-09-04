from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v15 import (
    ActionFamily,
    ActionRiskCertificate,
    CasePrediction,
    Direction,
    EndpointPrediction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    MenuRiskCalibration,
    RouterFitConfig,
    SupportActionOutcome,
    SupportCaseClassProfile,
    SurfaceRole,
    build_effective_menu,
    case_balanced_weights,
    certify_case_prediction,
    fit_support_router,
    float32_probability_hex,
    leave_one_case_out_crossfit,
    select_hierarchical_certificate,
)


H = "H"
BASELINE = float32_probability_hex((0.2, 0.8, 0.4))


def _action(
    case_id: str,
    action_id: str,
    *,
    role: SurfaceRole,
    family: ActionFamily,
    direction: Direction,
    source: str | None,
    value: float,
    output_offset: float,
) -> LabelFreeAction:
    return LabelFreeAction(
        outer_target_id=H,
        case_id=case_id,
        surface_role=role,
        action_id=action_id,
        family=family,
        direction=direction,
        candidate_source_id=source,
        feature_names=("active_mask_fraction", "compatibility_mean_z"),
        feature_values=(value, value / 2.0),
        baseline_probability_hex=BASELINE,
        action_probability_hex=float32_probability_hex(
            (0.55 + output_offset, 0.8, 0.4)
        ),
    )


def _menu(case_id: str, *, role: SurfaceRole) -> LabelFreeCaseMenu:
    actions = (
        _action(
            case_id,
            "U:D01",
            role=role,
            family=ActionFamily.U,
            direction=Direction.D01,
            source=None,
            value=0.1,
            output_offset=0.01,
        ),
        _action(
            case_id,
            "U:D10",
            role=role,
            family=ActionFamily.U,
            direction=Direction.D10,
            source=None,
            value=0.2,
            output_offset=0.02,
        ),
        _action(
            case_id,
            "HXE:E1:D01",
            role=role,
            family=ActionFamily.HXE,
            direction=Direction.D01,
            source="E1",
            value=0.3,
            output_offset=0.03,
        ),
        _action(
            case_id,
            "HXE:E2:D10",
            role=role,
            family=ActionFamily.HXE,
            direction=Direction.D10,
            source="E2",
            value=0.4,
            output_offset=0.04,
        ),
    )
    return LabelFreeCaseMenu(
        outer_target_id=H,
        case_id=case_id,
        surface_role=role,
        baseline_probability_hex=BASELINE,
        actions=actions,
    )


def _support_surface(case_count: int = 12):
    menus = tuple(
        _menu(f"support-{index:02d}", role=SurfaceRole.TARGET_TRAIN_SUPPORT)
        for index in range(case_count)
    )
    outcomes = tuple(
        SupportActionOutcome(
            action=action,
            menu_hash=menu.menu_hash,
            bacc_gain=0.20,
            brier_delta=-0.01,
            log_loss_delta=-0.02,
            class_recall_deltas=(0.20, 0.20),
            class_support=(True, True),
        )
        for menu in menus
        for action in menu.actions
    )
    return menus, outcomes


def _profiles(
    menus: tuple[LabelFreeCaseMenu, ...],
    *,
    overrides: dict[str, tuple[bool, bool]] | None = None,
) -> tuple[SupportCaseClassProfile, ...]:
    selected = {} if overrides is None else overrides
    return tuple(
        SupportCaseClassProfile(
            outer_target_id=menu.outer_target_id,
            case_id=menu.case_id,
            supports_class_0=selected.get(menu.case_id, (True, True))[0],
            supports_class_1=selected.get(menu.case_id, (True, True))[1],
        )
        for menu in menus
    )


def _fit_router(
    menus: tuple[LabelFreeCaseMenu, ...],
    outcomes: tuple[SupportActionOutcome, ...],
    *,
    profiles: tuple[SupportCaseClassProfile, ...] | None = None,
    candidates: tuple[str, ...] = ("E1", "E2"),
):
    return fit_support_router(
        menus,
        outcomes,
        case_profiles=_profiles(menus) if profiles is None else profiles,
        candidate_source_ids=candidates,
    )


def test_support_router_crossfits_cases_and_routes_target_without_labels() -> None:
    menus, outcomes = _support_surface()
    router = _fit_router(menus, outcomes)

    assert router.admission.admitted
    assert router.admission.routed_case_count == len(menus)
    for record in router.support_crossfit.records:
        assert record.prediction.action.case_id not in record.prediction.training_case_ids

    target = _menu("test-001", role=SurfaceRole.TARGET_EVALUATION)
    route = router.route(target)
    assert not route.exact_b_fallback
    assert route.selected_action_id != "B"
    assert route.probability_hex == target.action_for(route.selected_action_id).action_probability_hex
    assert route.reason == "ROUTED_SUPPORT_CERTIFIED_HIERARCHICAL_EXACT_ACTION"


def test_router_fit_config_public_payload_round_trips_exactly() -> None:
    config = RouterFitConfig(
        ridge_alpha=2.5,
        maximum_numeric_features=7,
        minimum_support_cases=8,
        calibration_alpha=0.15,
    )
    assert RouterFitConfig(**config.public_payload()) == config


def test_router_is_deterministic_and_case_balances_every_menu() -> None:
    menus, outcomes = _support_surface()
    first = _fit_router(menus, outcomes)
    reversed_menus = tuple(reversed(menus))
    second = _fit_router(reversed_menus, tuple(reversed(outcomes)))
    assert first.router_hash == second.router_hash

    weights = case_balanced_weights(tuple(row.action for row in outcomes))
    by_case: dict[str, float] = {}
    for row, weight in zip(outcomes, weights, strict=True):
        by_case[row.action.case_id] = by_case.get(row.action.case_id, 0.0) + weight
    assert all(value == pytest.approx(1.0) for value in by_case.values())


def test_heldout_case_labels_do_not_change_its_model_or_risk_calibration() -> None:
    menus, outcomes = _support_surface()
    first = _fit_router(menus, outcomes)
    changed = tuple(
        replace(row, bacc_gain=-0.5, class_recall_deltas=(-0.5, -0.5))
        if row.action.case_id == "support-00" and row.action.action_id == "U:D01"
        else row
        for row in outcomes
    )
    second = _fit_router(menus, changed)

    first_predictions = tuple(
        row.prediction.prediction_hash
        for row in first.support_crossfit.records_for_case("support-00")
    )
    second_predictions = tuple(
        row.prediction.prediction_hash
        for row in second.support_crossfit.records_for_case("support-00")
    )
    assert first_predictions == second_predictions
    assert dict(first.admission.heldout_calibration_hashes)["support-00"] == dict(
        second.admission.heldout_calibration_hashes
    )["support-00"]


def test_exact_b_fallback_is_byte_identical_for_empty_effective_menu() -> None:
    menus, outcomes = _support_surface()
    router = _fit_router(menus, outcomes)
    empty = LabelFreeCaseMenu(
        outer_target_id=H,
        case_id="test-empty",
        surface_role=SurfaceRole.TARGET_EVALUATION,
        baseline_probability_hex=BASELINE,
        actions=(),
    )
    decision = router.route(empty)
    assert decision.exact_b_fallback
    assert decision.selected_action_id == "B"
    assert decision.probability_hex == BASELINE
    assert decision.reason == "EXACT_B_NO_ACTIVE_ACTION"


def test_admission_denominator_includes_every_exact_b_support_case() -> None:
    menus, outcomes = _support_surface()
    negative = tuple(
        replace(
            row,
            bacc_gain=-0.20,
            brier_delta=0.02,
            log_loss_delta=0.02,
            class_recall_deltas=(-0.20, -0.20),
        )
        for row in outcomes
    )
    router = _fit_router(menus, negative)
    assert router.admission.support_case_count == len(menus)
    assert router.admission.routed_case_count == 0
    assert router.admission.coverage == 0.0
    assert not router.admission.admitted


def test_target_labels_cannot_enter_support_outcome_or_overlap_support() -> None:
    target = _menu("test-001", role=SurfaceRole.TARGET_EVALUATION)
    with pytest.raises(ProtocolError, match="target-train support capability"):
        SupportActionOutcome(
            action=target.actions[0],
            menu_hash=target.menu_hash,
            bacc_gain=0.1,
            brier_delta=0.0,
            log_loss_delta=0.0,
        )

    menus, outcomes = _support_surface()
    router = _fit_router(menus, outcomes)
    overlapping = _menu(menus[0].case_id, role=SurfaceRole.TARGET_EVALUATION)
    with pytest.raises(ProtocolError, match="support/evaluation roles or cases"):
        router.route(overlapping)


def test_target_expert_is_never_a_candidate() -> None:
    with pytest.raises(ProtocolError, match="target expert crossed"):
        _action(
            "support-00",
            "HXE:H:D01",
            role=SurfaceRole.TARGET_TRAIN_SUPPORT,
            family=ActionFamily.HXE,
            direction=Direction.D01,
            source=H,
            value=0.1,
            output_offset=0.01,
        )


def test_surface_role_is_part_of_action_and_menu_identity() -> None:
    support = _menu("shared-case", role=SurfaceRole.TARGET_TRAIN_SUPPORT)
    target = _menu("shared-case", role=SurfaceRole.TARGET_EVALUATION)
    assert support.menu_hash != target.menu_hash
    assert tuple(row.action_hash for row in support.actions) != tuple(
        row.action_hash for row in target.actions
    )


def test_support_inventory_requires_exact_action_outcome_coverage() -> None:
    menus, outcomes = _support_surface()
    with pytest.raises(ProtocolError, match="exactly cover"):
        leave_one_case_out_crossfit(
            menus,
            outcomes[:-1],
            config=RouterFitConfig(),
            case_profiles=_profiles(menus),
            candidate_source_ids=("E1", "E2"),
        )


def test_scalar_only_legacy_outcome_cannot_enter_v15_fold_normalization() -> None:
    menu = _menu("support-legacy", role=SurfaceRole.TARGET_TRAIN_SUPPORT)
    legacy = SupportActionOutcome(
        action=menu.actions[0],
        menu_hash=menu.menu_hash,
        bacc_gain=0.1,
        brier_delta=0.0,
        log_loss_delta=0.0,
    )
    with pytest.raises(ProtocolError, match="primitive class-local"):
        legacy.with_fold_normalization(
            case_count=4,
            class_support_counts=(4, 4),
            normalization_hash="a" * 64,
        )


def _prediction_for(action: LabelFreeAction, gain: float) -> EndpointPrediction:
    return EndpointPrediction(
        action=action,
        menu_hash="a" * 64,
        predicted_gain=gain,
        predicted_harm_probability=0.0,
        predicted_brier_delta=-0.02,
        predicted_log_loss_delta=-0.02,
        training_case_ids=("support-a", "support-b"),
        feature_map_hash="b" * 64,
        model_hash="c" * 64,
        out_of_fold=False,
    )


def _certificate(prediction: EndpointPrediction, *, passed: bool = True):
    return ActionRiskCertificate(
        prediction=prediction,
        gain_lcb=prediction.predicted_gain,
        harm_ucb=0.0,
        brier_delta_ucb=-0.02,
        log_loss_delta_ucb=-0.02,
        gain_passed=passed,
        harm_passed=passed,
        brier_passed=passed,
        log_loss_passed=passed,
        calibration_hash="d" * 64,
    )


def test_hierarchy_chooses_route_then_direction_family_and_expert() -> None:
    menu = _menu("test-hierarchy", role=SurfaceRole.TARGET_EVALUATION)
    by_id = {row.action_id: row for row in menu.actions}
    certificates = (
        _certificate(_prediction_for(by_id["U:D10"], 0.20)),
        _certificate(_prediction_for(by_id["U:D01"], 0.10)),
        _certificate(_prediction_for(by_id["HXE:E1:D01"], 0.40)),
        _certificate(_prediction_for(by_id["HXE:E2:D10"], 0.30)),
    )
    selected, trace = select_hierarchical_certificate(certificates)
    assert selected is not None
    assert selected.prediction.action.action_id == "HXE:E1:D01"
    assert trace.selected_direction is Direction.D01
    assert trace.selected_family is ActionFamily.HXE


def test_direct_certificate_can_reject_proper_loss_despite_positive_gain() -> None:
    menu = _menu("test-risk", role=SurfaceRole.TARGET_EVALUATION)
    prediction = _prediction_for(menu.actions[0], 0.20)
    case_prediction = CasePrediction(
        menu_hash="a" * 64,
        action_predictions=(prediction,),
    )
    calibration = MenuRiskCalibration(
        outer_target_id=H,
        support_case_ids=("s1", "s2"),
        active_calibration_case_ids=("s1", "s2"),
        gain_lower_offset=0.01,
        harm_upper_offset=0.01,
        brier_upper_offset=0.03,
        log_loss_upper_offset=0.0,
        alpha=0.2,
        support_crossfit_hash="e" * 64,
    )
    certificate = certify_case_prediction(
        case_prediction,
        calibration,
        config=RouterFitConfig(minimum_support_cases=4),
    )[0]
    assert certificate.gain_passed
    assert not certificate.brier_passed
    assert not certificate.passed


def test_menu_rejects_noops_and_duplicate_physical_outputs() -> None:
    menu = _menu("support-00", role=SurfaceRole.TARGET_TRAIN_SUPPORT)
    no_op = replace(menu.actions[0], action_probability_hex=BASELINE)
    with pytest.raises(ProtocolError, match="retained a no-op"):
        LabelFreeCaseMenu(H, menu.case_id, menu.surface_role, BASELINE, (no_op,))

    duplicate = replace(
        menu.actions[1], action_probability_hex=menu.actions[0].action_probability_hex
    )
    with pytest.raises(ProtocolError, match="deduplicate physical outputs"):
        LabelFreeCaseMenu(
            H,
            menu.case_id,
            menu.surface_role,
            BASELINE,
            (menu.actions[0], duplicate),
        )

    compiled = build_effective_menu(
        outer_target_id=H,
        case_id=menu.case_id,
        surface_role=menu.surface_role,
        baseline_probability_hex=BASELINE,
        raw_actions=(menu.actions[0], duplicate, no_op),
    )
    assert tuple(row.action_id for row in compiled.actions) == (
        min(menu.actions[0].action_id, duplicate.action_id),
    )


def test_empty_support_menu_is_crossfit_and_retained_in_admission_denominator() -> None:
    menus, outcomes = _support_surface()
    empty_case = menus[0].case_id
    mixed = (
        replace(menus[0], actions=()),
        *menus[1:],
    )
    active_outcomes = tuple(
        row for row in outcomes if row.action.case_id != empty_case
    )
    router = _fit_router(mixed, active_outcomes)

    sealed = router.support_crossfit.prediction_for_case(empty_case)
    assert sealed.prediction.action_predictions == ()
    assert router.support_crossfit.records_for_case(empty_case) == ()
    assert empty_case not in router.risk_calibration.active_calibration_case_ids
    assert router.admission.support_case_count == len(mixed)
    assert router.admission.routed_case_count <= len(mixed) - 1
    assert router.admission.coverage == pytest.approx(
        router.admission.routed_case_count / len(mixed)
    )


def test_all_empty_support_surface_builds_deterministic_always_b_router() -> None:
    menus, _outcomes = _support_surface()
    empty = tuple(replace(menu, actions=()) for menu in menus)
    first = _fit_router(empty, ())
    second = _fit_router(tuple(reversed(empty)), ())
    assert first.router_hash == second.router_hash
    assert first.endpoint_model.is_null
    assert not first.admission.admitted
    assert first.admission.support_case_count == len(empty)
    assert first.admission.routed_case_count == 0
    assert first.admission.coverage == 0.0
    assert all(
        row.null_model and not row.prediction.action_predictions
        for row in first.support_crossfit.case_predictions
    )

    target = _menu("test-null-model", role=SurfaceRole.TARGET_EVALUATION)
    decision = first.route(target)
    assert decision.exact_b_fallback
    assert decision.probability_hex == target.baseline_probability_hex
    assert decision.reason == "EXACT_B_NULL_SUPPORT_MODEL"


def test_declared_candidate_universe_allows_target_only_active_expert() -> None:
    menus, outcomes = _support_surface()
    support = tuple(
        replace(
            menu,
            actions=tuple(
                action
                for action in menu.actions
                if action.candidate_source_id in (None, "E1")
            ),
        )
        for menu in menus
    )
    support_hashes = {
        action.action_hash for menu in support for action in menu.actions
    }
    support_menu_hashes = {menu.case_id: menu.menu_hash for menu in support}
    support_outcomes = tuple(
        replace(row, menu_hash=support_menu_hashes[row.action.case_id])
        for row in outcomes
        if row.action.action_hash in support_hashes
    )
    router = _fit_router(
        support,
        support_outcomes,
        candidates=("E1", "E3"),
    )
    assert router.endpoint_model.feature_map.candidate_source_ids == ("E1", "E3")

    target_action = _action(
        "test-target-only-expert",
        "HXE:E3:D01",
        role=SurfaceRole.TARGET_EVALUATION,
        family=ActionFamily.HXE,
        direction=Direction.D01,
        source="E3",
        value=0.25,
        output_offset=0.03,
    )
    target = LabelFreeCaseMenu(
        outer_target_id=H,
        case_id=target_action.case_id,
        surface_role=SurfaceRole.TARGET_EVALUATION,
        baseline_probability_hex=BASELINE,
        actions=(target_action,),
    )
    decision = router.route(target)
    assert decision.selected_action_id in {"B", "HXE:E3:D01"}


def test_single_class_heldout_profile_cannot_change_its_fold_fit() -> None:
    menus, outcomes = _support_surface()
    heldout = menus[0].case_id

    def surface_for(
        support: tuple[bool, bool], deltas: tuple[float, float]
    ) -> tuple[tuple[SupportActionOutcome, ...], tuple[SupportCaseClassProfile, ...]]:
        changed = tuple(
            replace(
                row,
                bacc_gain=0.5 * sum(
                    delta
                    for delta, present in zip(deltas, support, strict=True)
                    if present
                ),
                class_recall_deltas=deltas,
                class_support=support,
            )
            if row.action.case_id == heldout
            else row
            for row in outcomes
        )
        profiles = _profiles(menus, overrides={heldout: support})
        return changed, profiles

    first_outcomes, first_profiles = surface_for((True, False), (0.20, 0.0))
    second_outcomes, second_profiles = surface_for((False, True), (0.0, 0.20))
    first = _fit_router(menus, first_outcomes, profiles=first_profiles)
    second = _fit_router(menus, second_outcomes, profiles=second_profiles)

    first_seal = first.support_crossfit.prediction_for_case(heldout)
    second_seal = second.support_crossfit.prediction_for_case(heldout)
    assert first_seal.training_case_ids == second_seal.training_case_ids
    assert heldout not in first_seal.training_case_ids
    assert first_seal.normalizer_hash == second_seal.normalizer_hash
    assert first_seal.model_hash == second_seal.model_hash
    assert first_seal.prediction.prediction_hash == second_seal.prediction.prediction_hash
    assert dict(first.admission.heldout_calibration_hashes)[heldout] == dict(
        second.admission.heldout_calibration_hashes
    )[heldout]
