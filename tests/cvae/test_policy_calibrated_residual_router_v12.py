from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v12 import (
    ActionScore,
    AdmissionConfig,
    CaseInventoryKind,
    CasePrediction,
    Direction,
    LabelFreeAction,
    NestedPolicyFold,
    PolicyRiskConfig,
    SourceActionOutcome,
    SourceOutcomeUniverse,
    build_effective_menu,
    build_label_free_case_inventory,
    calibrate_selected_policy,
    evaluate_outer_admission,
    fit_source_lodo,
    float32_probability_hex,
    select_policy_action,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v12.source_crossfit_fold_store import (
    load_source_crossfit_fold,
    persist_source_crossfit_fold,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v12.source_crossfit_orchestration import (
    FoldFitExecution,
)
from midogpp_thesis.cvae.runtime.harp_v12_execution.model_adapter import (
    OuterPolicyState,
    RouterAdmissionState,
)
from midogpp_thesis.cvae.runtime.harp_v12_execution.model_artifacts import (
    _numeric_oof_arrays,
)
from midogpp_thesis.cvae.runtime.harp_v12_execution.policy_replay_artifacts import (
    _policy_oof_replay,
)
from midogpp_thesis.cvae.runtime.harp_v12_execution.validation import (
    _validate_numeric_oof,
)


OUTER = "H"
BASELINE = float32_probability_hex((0.2, 0.8))
ACTIVE = float32_probability_hex((0.8, 0.2))


def _action(query: str, case: str, *, active: bool = True) -> LabelFreeAction:
    return LabelFreeAction(
        outer_target_id=OUTER,
        query_center_id=query,
        case_id=case,
        action_id="U:D01",
        action_kind="U",
        direction=Direction.D01,
        candidate_source_id=None,
        feature_names=("source_shift",),
        feature_values=(1.0,),
        baseline_probability_hex=BASELINE,
        action_probability_hex=ACTIVE if active else BASELINE,
    )


def _menu(query: str, case: str, *, active: bool = True):
    return build_effective_menu((_action(query, case, active=active),))


def _prediction(
    menu,
    *,
    excluded: tuple[str, ...] | None = None,
    action_hash: str | None = None,
) -> CasePrediction:
    if menu.actions:
        action = menu.actions[0]
        scores = (
            ActionScore(
                action_id=action.action_id,
                action_hash=action.action_hash if action_hash is None else action_hash,
                action_group="U:D01",
                direction=Direction.D01,
                pairwise_score=1.0,
                predicted_budget_gain=1.0,
                predicted_allocation_gain=0.0,
                predicted_total_gain=1.0,
                predicted_harm_probability=0.0,
                predicted_brier_delta=-0.1,
                predicted_log_delta=-0.1,
                acceptance_probability=1.0,
                model_available=True,
            ),
        )
        top = action.action_id
        acceptance = 1.0
        margin = 1.0
    else:
        scores = ()
        top = "B"
        acceptance = 0.0
        margin = 0.0
    return CasePrediction(
        outer_target_id=menu.outer_target_id,
        query_center_id=menu.query_center_id,
        case_id=menu.case_id,
        action_scores=scores,
        raw_top_action_id=top,
        top_action_id=top,
        acceptance_probability=acceptance,
        rank_margin=margin,
        model_hash="a" * 64,
        ranker_hash="b" * 64,
        acceptor_hash="c" * 64,
        training_center_ids=("S",),
        training_candidate_ids=(),
        excluded_center_ids=(OUTER, menu.query_center_id)
        if excluded is None
        else excluded,
        menu_hash=menu.menu_hash,
    )


def _outcome(menu, gain: float = 0.2) -> SourceActionOutcome:
    return SourceActionOutcome(
        action=menu.actions[0],
        bacc_gain=gain,
        brier_delta=-0.01,
        log_delta=-0.02,
    )


def _surface():
    menus = (
        _menu("Q", "q-active"),
        _menu("Q", "q-control", active=False),
        _menu("R", "r-active"),
        _menu("R", "r-control", active=False),
    )
    predictions = tuple(_prediction(menu) for menu in menus)
    outcomes = tuple(_outcome(menu) for menu in menus if menu.actions)
    return menus, predictions, outcomes


def _nested_folds(menus) -> tuple[NestedPolicyFold, ...]:
    by_query = {
        query: tuple(menu for menu in menus if menu.query_center_id == query)
        for query in ("Q", "R")
    }
    folds = []
    for heldout, training in (("Q", "R"), ("R", "Q")):
        folds.append(
            NestedPolicyFold(
                heldout_center_id=heldout,
                training_center_ids=(training,),
                predictions=tuple(
                    _prediction(
                        menu,
                        excluded=(OUTER, heldout, training),
                    )
                    for menu in by_query[training]
                ),
                heldout_predictions=tuple(
                    _prediction(menu) for menu in by_query[heldout]
                ),
            )
        )
    return tuple(folds)


def test_empty_menu_is_explicit_label_free_exact_b_control() -> None:
    menu = _menu("Q", "control", active=False)
    inventory = build_label_free_case_inventory((_prediction(menu),), (menu,), require_complete=True)

    assert inventory.contexts[0].kind is CaseInventoryKind.EXACT_B_CONTROL
    assert inventory.contexts[0].prediction.action_scores == ()
    assert inventory.contexts[0].prediction.top_action_id == "B"


def test_prediction_action_hash_must_match_effective_menu() -> None:
    menu = _menu("Q", "active")
    prediction = _prediction(menu, action_hash="f" * 64)

    with pytest.raises(ProtocolError, match="exactly cover"):
        build_label_free_case_inventory((prediction,), (menu,), require_complete=True)


def test_active_menu_requires_exact_source_outcome_but_control_requires_none() -> None:
    menus, predictions, outcomes = _surface()
    universe = SourceOutcomeUniverse(menus, outcomes)
    inventory = universe.bind_predictions(predictions, require_complete=True)

    assert len(inventory.contexts) == 4
    assert sum(row.is_exact_b_control for row in inventory.contexts) == 2
    assert all(not row.outcomes for row in inventory.contexts if row.is_exact_b_control)

    with pytest.raises(ProtocolError, match="incomplete or drifted"):
        SourceOutcomeUniverse(menus, outcomes[:-1])


def test_extra_or_duplicate_source_outcome_fails_closed() -> None:
    menus, _predictions, outcomes = _surface()

    with pytest.raises(ProtocolError, match="extra or duplicated"):
        SourceOutcomeUniverse(menus, (*outcomes, outcomes[0]))


def test_source_fit_rejects_outcome_only_menu_reconstruction() -> None:
    _menus, _predictions, outcomes = _surface()

    with pytest.raises(ProtocolError, match="explicit effective-menu inventory"):
        fit_source_lodo(outcomes)


def test_admission_counts_controls_but_excludes_them_from_opportunity() -> None:
    menus, predictions, outcomes = _surface()
    inventory = SourceOutcomeUniverse(menus, outcomes).bind_predictions(
        predictions, require_complete=True
    )
    admission = evaluate_outer_admission(
        inventory,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.0,
            min_delete_center_top1_excess=0.0,
            min_opportunity_top1_accuracy=0.0,
            min_opportunity_cases=1,
        ),
    )

    assert admission.admitted
    assert admission.case_count == 4
    assert admission.opportunity_case_count == 2
    assert admission.opportunity_top1_accuracy == 1.0


def test_calibration_uses_all_cases_as_coverage_denominator() -> None:
    menus, predictions, outcomes = _surface()
    universe = SourceOutcomeUniverse(menus, outcomes)
    inventory = universe.bind_predictions(predictions, require_complete=True)
    calibration = calibrate_selected_policy(
        inventory,
        outcome_universe=universe,
        nested_policy_folds=_nested_folds(menus),
        config=PolicyRiskConfig(
            acceptance_thresholds=(0.0,),
            min_case_equal_bacc_gain=-1.0,
            min_delete_center_bacc_gain=-1.0,
            max_routed_harm_rate=1.0,
            max_case_equal_brier_delta=1.0,
            max_case_equal_log_delta=1.0,
            min_coverage=0.0,
            min_routed_cases=1,
        ),
    )

    assert calibration.calibrated
    assert calibration.selected_replay.case_count == 4
    assert calibration.selected_replay.routed_cases == 2
    assert calibration.selected_replay.coverage == 0.5
    assert calibration.nested_replay.case_count == 4


def test_empty_control_cannot_carry_acceptance_mass() -> None:
    menu = _menu("Q", "control", active=False)

    with pytest.raises(ProtocolError, match="virtual baseline"):
        replace(_prediction(menu), acceptance_probability=0.1)


def test_empty_target_menu_routes_to_byte_identical_exact_b() -> None:
    source_menus, source_predictions, source_outcomes = _surface()
    universe = SourceOutcomeUniverse(source_menus, source_outcomes)
    inventory = universe.bind_predictions(source_predictions, require_complete=True)
    admission = evaluate_outer_admission(
        inventory,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.0,
            min_delete_center_top1_excess=0.0,
            min_opportunity_top1_accuracy=0.0,
            min_opportunity_cases=1,
        ),
    )
    calibration = calibrate_selected_policy(
        inventory,
        outcome_universe=universe,
        nested_policy_folds=_nested_folds(source_menus),
        config=PolicyRiskConfig(
            acceptance_thresholds=(0.0,),
            min_case_equal_bacc_gain=-1.0,
            min_delete_center_bacc_gain=-1.0,
            max_routed_harm_rate=1.0,
            max_case_equal_brier_delta=1.0,
            max_case_equal_log_delta=1.0,
            min_coverage=0.0,
            min_routed_cases=1,
        ),
    )
    target_menu = _menu(OUTER, "target-control", active=False)
    target_prediction = _prediction(target_menu, excluded=(OUTER,))

    route = select_policy_action(
        target_menu,
        target_prediction,
        admission,
        calibration,
    )

    assert route.selected_action_id == "B"
    assert route.exact_b_fallback
    assert route.probability_hex == target_menu.baseline_probability_hex
    assert route.reason == "EXACT_B_NO_ACTIVE_ACTION"


def test_prelabel_fold_inventory_round_trips_before_outcome_join(tmp_path) -> None:
    menus, _predictions, _outcomes = _surface()
    fold = _nested_folds(menus)[0]
    seal = persist_source_crossfit_fold(
        tmp_path / "stores/source_crossfit_folds",
        nested_fold=fold,
        outer_target_id=OUTER,
        heldout_center_id="Q",
        source_surface_receipt_hash="1" * 64,
        source_surface_hash="2" * 64,
        effective_adapter_hash="3" * 64,
        prediction_surface_hash="4" * 64,
        fitting_surface_hash="5" * 64,
        label_capability_hash="6" * 64,
        isolation_receipt_hash="7" * 64,
        effective_menus=menus,
    )
    reconstructed = load_source_crossfit_fold(
        seal.path,
        effective_menus=menus,
    )

    assert reconstructed.label_free_case_inventory_hash == seal.label_free_case_inventory_hash
    assert reconstructed.exact_b_control_count == 2
    assert reconstructed.active_menu_count == 2


def test_worker_pid_cannot_change_fold_scientific_identity(tmp_path) -> None:
    menus, _predictions, _outcomes = _surface()
    fold = _nested_folds(menus)[0]
    first = FoldFitExecution(
        outer_target_id=OUTER,
        heldout_center_id="Q",
        task_scope_hash="8" * 64,
        nested_fold=fold,
        worker_process_id=101,
        cuda_visible_to_worker=False,
    )
    second = replace(first, worker_process_id=202)

    assert first.isolation_receipt_hash == second.isolation_receipt_hash

    seals = tuple(
        persist_source_crossfit_fold(
            tmp_path / f"pid-{execution.worker_process_id}",
            nested_fold=fold,
            outer_target_id=OUTER,
            heldout_center_id="Q",
            source_surface_receipt_hash="1" * 64,
            source_surface_hash="2" * 64,
            effective_adapter_hash="3" * 64,
            prediction_surface_hash="4" * 64,
            fitting_surface_hash="5" * 64,
            label_capability_hash="6" * 64,
            isolation_receipt_hash=execution.isolation_receipt_hash,
            effective_menus=menus,
        )
        for execution in (first, second)
    )
    assert seals[0].manifest_hash == seals[1].manifest_hash
    assert seals[0].seal_hash == seals[1].seal_hash


def test_oof_artifact_replay_reuses_the_admitted_inventory() -> None:
    menus, predictions, outcomes = _surface()
    folds = _nested_folds(menus)
    universe = SourceOutcomeUniverse(menus, outcomes)
    inventory = universe.bind_predictions(predictions, require_complete=True)
    admission = evaluate_outer_admission(
        inventory,
        config=AdmissionConfig(
            min_pooled_top1_excess=0.0,
            min_delete_center_top1_excess=0.0,
            min_opportunity_top1_accuracy=0.0,
            min_opportunity_cases=1,
        ),
    )
    calibration = calibrate_selected_policy(
        inventory,
        outcome_universe=universe,
        nested_policy_folds=folds,
        config=PolicyRiskConfig(
            acceptance_thresholds=(0.0,),
            min_case_equal_bacc_gain=-1.0,
            min_delete_center_bacc_gain=-1.0,
            max_routed_harm_rate=1.0,
            max_case_equal_brier_delta=1.0,
            max_case_equal_log_delta=1.0,
            min_coverage=0.0,
            min_routed_cases=1,
        ),
    )
    policy = OuterPolicyState(
        outer_target_id=OUTER,
        admission=admission,
        calibration=calibration,
        source_outcome_universe=universe,
        source_oof_inventory=inventory,
        source_outcome_universe_hash=universe.universe_hash,
        source_oof_inventory_hash=inventory.inventory_hash,
        exact_b_control_count=2,
        active_menu_count=2,
    )
    fitted = SimpleNamespace(
        bundles=(
            SimpleNamespace(
                outer_target_id=OUTER,
                lodo=SimpleNamespace(
                    oof_predictions=predictions,
                    nested_policy_folds=folds,
                ),
            ),
        )
    )

    rows, replay, nested = _policy_oof_replay(
        fitted,  # type: ignore[arg-type]
        RouterAdmissionState((policy,)),
    )

    assert len(rows) == 4
    assert replay.shape == (4, 8)
    assert nested.shape == (4, 8)
    assert sum(bool(row["is_exact_b_control"]) for row in rows) == 2
    assert {
        str(row["source_oof_inventory_hash"]) for row in rows
    } == {inventory.inventory_hash}

    _validate_numeric_oof(
        SimpleNamespace(arrays=_numeric_oof_arrays(fitted)),
        SimpleNamespace(
            manifest={
                "source_policy_oof_rows": rows,
                "outer_policies": [
                    {
                        "outer_target_id": OUTER,
                        "source_outcome_universe_hash": universe.universe_hash,
                        "source_oof_label_free_inventory_hash": (
                            inventory.label_free_inventory_hash
                        ),
                        "source_oof_inventory_hash": inventory.inventory_hash,
                        "exact_b_control_count": 2,
                        "active_menu_count": 2,
                        "admission": admission.public_payload(),
                    }
                ],
                "typed_case_outcome_inventory_replayed": True,
                "empty_effective_menus_retained_as_exact_B_controls": True,
            },
            arrays={
                "source_policy_oof_values": replay,
                "nested_source_policy_oof_values": nested,
            },
        ),
    )
