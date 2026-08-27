from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.admission import (
    evaluate_pseudo_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.case_inventory import (
    DatasetCaseInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.controls import (
    METHOD_IDS,
    P_PROTECTED,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.engine import (
    CaseRouteRequest,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.evidence.bundle import (
    AllOuterReplayEvidenceBundle,
    PseudoReplayEvidenceBundle,
    assemble_all_outer_replay_evidence,
    assemble_replay_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.evidence.contracts import (
    PseudoRouteActionEvidence,
    PseudoRoutePolicyEvidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    ACTION_IDS,
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.replay import (
    PseudoCaseReplayRequest,
    TerminalCaseLabelInput,
    load_terminal_case_label_receipt,
    method_menu_hash,
    replay_pseudo_case,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.replay_inventory import (
    build_pseudo_replay_inventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.replay_scope import (
    PseudoReplayScope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.route_identity import (
    RouteIdentityInventory,
    RouteScopeWitness,
    SampleIdentity,
    build_route_identity_inventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.support_folds import (
    SupportMember,
    build_support_fold_plan,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "d" * 64


@lru_cache(maxsize=1)
def _inventory() -> DatasetCaseInventory:
    return DatasetCaseInventory(
        SHA,
        SHA,
        SHA,
        tuple(
            (
                center,
                tuple(
                    f"case-{center}-{index:03d}"
                    for index in range(dict(EXPECTED_CASE_COUNTS_BY_CENTER)[center])
                ),
            )
            for center in CENTERS
        ),
    )


@lru_cache(maxsize=1)
def _route_identity_inventory() -> RouteIdentityInventory:
    inventory = _inventory()
    return build_route_identity_inventory(
        tuple(
            SampleIdentity(
                center,
                case,
                f"group-{case}",
                f"patient-{case}",
                f"slide-{case}",
                f"sample-{case}",
            )
            for center in CENTERS
            for case in inventory.cases(center)
        ),
        case_inventory=inventory,
    )


def _scope(outer_center: str, center: str, case_id: str) -> PseudoReplayScope:
    return PseudoReplayScope(
        outer_center,
        center,
        case_id,
        RouteScopeWitness(center, case_id, _route_identity_inventory()),
        _inventory(),
    )


@lru_cache(maxsize=None)
def _center_label_rows(center: str) -> tuple[TerminalCaseLabelInput, ...]:
    return tuple(
        TerminalCaseLabelInput(
            case,
            ((center, case, f"sample-{case}"),),
            (index % 2,),
        )
        for index, case in enumerate(_inventory().cases(center))
    )


def _no_action_replay(
    scope: PseudoReplayScope,
    *,
    center_label_rows: tuple[TerminalCaseLabelInput, ...] | None = None,
):
    witness = scope.route_witness
    members = tuple(
        SupportMember(
            f"member-{index}",
            binding.center,
            binding.case_id,
            binding.group_id,
            binding.patient_id,
            binding.slide_id,
            binding.sample_key_hash,
            binding.row_count,
        )
        for index, binding in enumerate(witness.support_bindings)
    )
    plan = build_support_fold_plan(members, route_witness=witness)
    route = CaseRouteRequest(
        scope.held_case_id,
        scope,
        (0.49,),
        plan,
        (),
        (),
    )
    return replay_pseudo_case(
        PseudoCaseReplayRequest(
            route,
            load_terminal_case_label_receipt(
                scope,
                center_label_rows or _center_label_rows(scope.pseudo_center),
            ),
        )
    )


@lru_cache(maxsize=None)
def _outer_bundle(outer_center: str) -> PseudoReplayEvidenceBundle:
    scopes = tuple(
        _scope(outer_center, center, case)
        for center in CENTERS
        if center != outer_center
        for case in _inventory().cases(center)
    )
    inventory = build_pseudo_replay_inventory(
        scopes,
        outer_center=outer_center,
        case_inventory=_inventory(),
    )
    return assemble_replay_evidence(
        inventory,
        tuple(_no_action_replay(scope) for scope in scopes),
    )


@lru_cache(maxsize=1)
def _bundle() -> AllOuterReplayEvidenceBundle:
    return assemble_all_outer_replay_evidence(
        tuple(_outer_bundle(outer_center) for outer_center in CENTERS)
    )


def _outer_bundle_with_poisoned_center_population(
    outer_center: str,
    poisoned_center: str,
) -> PseudoReplayEvidenceBundle:
    scopes = tuple(
        _scope(outer_center, center, case)
        for center in CENTERS
        if center != outer_center
        for case in _inventory().cases(center)
    )
    inventory = build_pseudo_replay_inventory(
        scopes,
        outer_center=outer_center,
        case_inventory=_inventory(),
    )
    poisoned_rows = tuple(
        replace(row, labels=(1 - row.labels[0],))
        for row in _center_label_rows(poisoned_center)
    )
    return assemble_replay_evidence(
        inventory,
        tuple(
            _no_action_replay(
                scope,
                center_label_rows=(
                    poisoned_rows
                    if scope.pseudo_center == poisoned_center
                    else _center_label_rows(scope.pseudo_center)
                ),
            )
            for scope in scopes
        ),
    )


def test_full_replay_bundle_is_closed_world_and_admission_is_root_bound() -> None:
    bundle = _bundle()
    result = evaluate_pseudo_admission(bundle)
    assert result.admitted is False
    assert result.failed_outer_centers == CENTERS
    assert all(
        "INSUFFICIENT_OPPORTUNITY_CASES" in row.reasons
        for row in result.outer_results
    )
    assert bundle.context_count == 1744
    assert bundle.action_evidence_count == 1744 * len(METHOD_IDS) * len(ACTION_IDS)
    assert bundle.policy_evidence_count == 1744 * len(METHOD_IDS)
    assert result.context_count == 1744
    assert result.policy_count == 1744 * len(METHOD_IDS)
    assert result.replay_bundle_hash == bundle.bundle_hash
    assert result.replay_input_root == bundle.input_root
    assert result.action_evidence_root == bundle.action_evidence_root
    assert result.policy_evidence_root == bundle.policy_evidence_root
    assert result.oracle_root == bundle.oracle_root
    assert result.method_menu_hash == method_menu_hash()


def test_replay_bundle_rejects_missing_context_and_admission_rejects_loose_rows() -> None:
    bundle = _bundle()
    outer = bundle.outer_bundles[-1]
    with pytest.raises(ProtocolError, match="universe is incomplete"):
        assemble_replay_evidence(
            outer.replay_inventory,
            outer.case_results[:-1],
        )
    with pytest.raises(ProtocolError, match="all-outer replay universe"):
        assemble_all_outer_replay_evidence(bundle.outer_bundles[:-1])
    with pytest.raises(ProtocolError, match="sealed all-outer replay evidence"):
        evaluate_pseudo_admission(outer)  # type: ignore[arg-type]


def test_all_outer_bundle_rejects_H_specific_pseudo_center_label_population() -> None:
    bundle = _bundle()
    poisoned = _outer_bundle_with_poisoned_center_population("0", "1")
    rows = tuple(
        poisoned if row.replay_inventory.outer_center == "0" else row
        for row in bundle.outer_bundles
    )
    with pytest.raises(ProtocolError, match="center-label population drifted"):
        assemble_all_outer_replay_evidence(rows)


def test_action_and_policy_evidence_cannot_be_caller_fabricated_or_replaced() -> None:
    scope = _scope("9", "0", _inventory().cases("0")[0])
    with pytest.raises(ProtocolError, match="not issued by replay"):
        PseudoRouteActionEvidence(
            scope=scope,
            method_id=P_PROTECTED,
            action_id=ACTION_IDS[0],
            opportunity=False,
            selected=False,
            crossing_indices=(),
            predicted_bacc_gain=0.0,
            realized_bacc_gain=0.0,
            realized_brier_loss_delta=0.0,
            realized_log_loss_delta=0.0,
            descriptor_hash=SHA,
            candidate_hash=SHA,
            replay_request_hash=SHA,
            terminal_label_hash=SHA,
            method_menu_hash=SHA,
            oracle_hash=SHA,
        )
    with pytest.raises(ProtocolError, match="not issued by replay"):
        PseudoRoutePolicyEvidence(
            scope=scope,
            method_id=P_PROTECTED,
            selected_action_ids=(),
            realized_bacc_gain=0.0,
            realized_brier_loss_delta=0.0,
            realized_log_loss_delta=0.0,
            oracle_bacc_gain=0.0,
            decision_hash=SHA,
            composition_hash=SHA,
            action_evidence_hashes=tuple(
                f"{index:064x}" for index in range(1, len(ACTION_IDS) + 1)
            ),
            replay_request_hash=SHA,
            terminal_label_hash=SHA,
            method_menu_hash=SHA,
            oracle_hash=SHA,
        )
    issued = _bundle().outer_bundles[0].action_evidence[0]
    with pytest.raises(ProtocolError, match="not issued by replay"):
        replace(issued, realized_bacc_gain=1.0)


def test_bundle_constructor_is_factory_sealed() -> None:
    bundle = _bundle()
    outer = bundle.outer_bundles[0]
    with pytest.raises(ProtocolError, match="not factory assembled"):
        PseudoReplayEvidenceBundle(
            replay_inventory=outer.replay_inventory,
            case_results=outer.case_results,
        )
    with pytest.raises(ProtocolError, match="not factory assembled"):
        AllOuterReplayEvidenceBundle(
            outer_bundles=bundle.outer_bundles,
        )
