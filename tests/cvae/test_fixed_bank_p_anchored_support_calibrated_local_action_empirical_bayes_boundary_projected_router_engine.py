from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.case_inventory import (
    DatasetCaseInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.donor_prior import (
    DonorPriorPrediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.engine import (
    CaseRouteRequest,
    RouteActionInput,
    build_case_route,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.contracts import (
    ActionDescriptor,
    ActionMetricVector,
    MetricStandardError,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.influence.descriptors import (
    build_action_descriptor,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.local_residual import (
    LocalResidualRecord,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.physical.endpoint_surface import (
    EndpointSurfaceReceipt,
    PhysicalCellSurface,
    _issue_physical_cell_surface,
    assemble_endpoint_surface,
    build_projection_from_endpoint,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.physical.library import (
    B_ACTION_ID,
    PhysicalCellIdentity,
    action_ids_for_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.replay_scope import (
    FinalDonorScope,
    PseudoReplayScope,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.replay import (
    PseudoCaseReplayRequest,
    TerminalCaseLabelInput,
    load_terminal_case_label_receipt,
    replay_pseudo_case,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.controls import (
    CYCLIC_ACTION_IDENTITY,
    METHOD_IDS,
    P_PROTECTED,
    SCALE_BP_PRIMARY,
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


SHA = "e" * 64


@lru_cache(maxsize=1)
def _inventory() -> DatasetCaseInventory:
    counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    return DatasetCaseInventory(
        SHA,
        SHA,
        SHA,
        tuple(
            (
                center,
                tuple(f"case-{center}-{index:03d}" for index in range(counts[center])),
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
                f"sample-{case}-{sample_index}",
            )
            for center in CENTERS
            for case in inventory.cases(center)
            for sample_index in range(2)
        ),
        case_inventory=inventory,
    )


def _physical_surfaces(
    *, target: str = "2", case_id: str = "case-2-000"
) -> tuple[PhysicalCellSurface, ...]:
    rows = []
    for action in action_ids_for_target(target):
        probabilities = (0.90, 0.80) if action == B_ACTION_ID else (0.60, 0.70)
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                rows.append(
                    _issue_physical_cell_surface(
                        identity=PhysicalCellIdentity(
                            target, action, training_seed, generation_seed
                        ),
                        case_id=case_id,
                        cache_content_hash=SHA,
                        row_order_hash=SHA,
                        probabilities=probabilities,
                        physical_bank_receipt_hash=SHA,
                        memmap_reference_hash=SHA,
                        memmap_slice_sha256=SHA,
                        memmap_row_index_hash=SHA,
                    )
                )
    return tuple(rows)


def _request(*, with_action: bool, pseudo: bool = False) -> CaseRouteRequest:
    case_id = "case-2-000"
    witness = RouteScopeWitness("2", case_id, _route_identity_inventory())
    route_hash = witness.witness_hash
    scope = (
        PseudoReplayScope("9", "2", case_id, witness, _inventory())
        if pseudo
        else FinalDonorScope("2", case_id, witness, _inventory())
    )
    support_bindings = witness.support_bindings
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
        for index, binding in enumerate(support_bindings)
    )
    plan = build_support_fold_plan(
        members,
        route_witness=witness,
    )
    portfolio = np.asarray([0.49, 0.80], dtype=np.float32)
    if not with_action:
        return CaseRouteRequest(
            case_id,
            scope,
            tuple(float(value) for value in portfolio),
            plan,
            (),
            (),
        )

    endpoint = assemble_endpoint_surface(
        _physical_surfaces(),
        target_center="2",
        case_id=case_id,
        family="B",
        direction="zero_to_one",
    )
    endpoint_projection = build_projection_from_endpoint(portfolio, endpoint)
    descriptor = build_action_descriptor(
        endpoint_projection.projection,
        case_id=case_id,
        posterior_eta=np.asarray([0.8, 0.2]),
        posterior_sd=np.asarray([0.05, 0.05]),
        seed_sd=np.asarray([0.02, 0.02]),
        positive_vote_fraction=np.asarray([2.0 / 3.0, 1.0 / 3.0]),
        support_positive_count=8.0,
        support_negative_count=8.0,
        support_row_count=16,
        bank_ess=4.5,
    )
    donor = DonorPriorPrediction(
        descriptor.descriptor_hash,
        ActionMetricVector(0.03, -0.01, -0.01),
        MetricStandardError.zeros(),
        SHA,
        scope.scope_hash,
        scope.fit_role,
    )
    records = []
    for index, member in enumerate(members):
        values = tuple(
            value + (index - 3.5) * 1.0e-4 for value in descriptor.values
        )
        support_descriptor = ActionDescriptor(
            member.case_id,
            descriptor.action_id,
            descriptor.family,
            descriptor.direction,
            descriptor.feature_names,
            values,
            1,
            2,
            SHA,
            SHA,
            SHA,
        )
        records.append(
            LocalResidualRecord(
                member.member_id,
                member.center_id,
                member.case_id,
                member.group_id,
                member.patient_id,
                member.slide_id,
                route_hash,
                member.member_hash,
                support_descriptor,
                ActionMetricVector.zeros(),
                ActionMetricVector.zeros(),
            )
        )
    return CaseRouteRequest(
        case_id,
        scope,
        tuple(float(value) for value in portfolio),
        plan,
        tuple(records),
        (RouteActionInput(endpoint_projection, descriptor, donor),),
    )


def _terminal_receipt(
    scope: PseudoReplayScope,
    *,
    reverse_labels: bool = False,
):
    labels = (0, 1) if reverse_labels else (1, 0)
    return load_terminal_case_label_receipt(
        scope,
        tuple(
            TerminalCaseLabelInput(
                case,
                tuple(
                    (scope.pseudo_center, case, f"sample-{case}-{sample_index}")
                    for sample_index in range(2)
                ),
                labels,
            )
            for case in scope.case_inventory.cases(scope.pseudo_center)
        ),
    )


def test_endpoint_assembly_binds_exact_physical_rectangle() -> None:
    surfaces = _physical_surfaces()
    endpoint = assemble_endpoint_surface(
        surfaces,
        target_center="2",
        case_id="case-2-000",
        family="B",
        direction="zero_to_one",
    )
    assert len(endpoint.physical_cell_identity_hashes) == 90
    assert endpoint.endpoint_array().dtype == np.float32
    with pytest.raises(ProtocolError, match="physical rectangle"):
        assemble_endpoint_surface(
            surfaces[:-1],
            target_center="2",
            case_id="case-2-000",
            family="B",
            direction="zero_to_one",
        )
    with pytest.raises(ProtocolError, match="physical rectangle"):
        EndpointSurfaceReceipt(
            "2",
            "case-2-000",
            "B",
            "zero_to_one",
            surfaces[:-1],
        )
    with pytest.raises(ProtocolError, match="memmap loader"):
        PhysicalCellSurface(
            identity=surfaces[0].identity,
            case_id=surfaces[0].case_id,
            cache_content_hash=SHA,
            row_order_hash=SHA,
            probabilities=surfaces[0].probabilities,
            physical_bank_receipt_hash=SHA,
            memmap_reference_hash=SHA,
            memmap_slice_sha256=SHA,
            memmap_row_index_hash=SHA,
        )


def test_case_route_engine_connects_crossfit_eb_selection_and_composition() -> None:
    result = build_case_route(_request(with_action=True))
    assert result.crossfit is not None
    assert result.final_local_model is not None
    assert result.selection_radius is not None
    assert result.calibrations[0].within_support is True
    assert result.decision.selected_action_ids == ("B::zero_to_one",)
    assert result.boundary_action.crossing_indices == (0,)
    assert result.full_endpoint_sensitivity.crossing_indices == (0,)
    assert result.boundary_action.composed_probability_hash != (
        result.full_endpoint_sensitivity.composed_probability_hash
    )


def test_case_route_engine_no_candidate_is_byte_exact_p() -> None:
    request = _request(with_action=False)
    result = build_case_route(request)
    assert result.decision.is_exact_p
    np.testing.assert_array_equal(
        result.boundary_action.as_array().view(np.uint32),
        np.asarray(request.portfolio_probabilities, dtype=np.float32).view(np.uint32),
    )


def test_pseudo_case_route_engine_preserves_h_j_d_scope_lineage() -> None:
    request = _request(with_action=True, pseudo=True)
    assert isinstance(request.route_scope, PseudoReplayScope)
    assert request.route_scope.route_scope_hash == request.support_plan.route_scope_hash
    result = build_case_route(request)
    assert result.decision.selected_action_ids == ("B::zero_to_one",)


def test_pseudo_case_replay_executes_primary_ablations_controls_and_oracle() -> None:
    request = _request(with_action=True, pseudo=True)
    assert isinstance(request.route_scope, PseudoReplayScope)
    receipt = _terminal_receipt(request.route_scope)
    replay_request = PseudoCaseReplayRequest(request, receipt)
    with pytest.raises(ProtocolError, match="may not be serialized"):
        pickle.dumps(replay_request)
    with pytest.raises(ProtocolError, match="may not be serialized"):
        pickle.dumps(receipt)
    assert "terminal_labels" not in replay_request.sealed_payload()
    assert "(1, 0)" not in repr(replay_request)
    result = replay_pseudo_case(replay_request)
    assert tuple(row.method_id for row in result.method_results) == METHOD_IDS
    by_method = {row.method_id: row for row in result.method_results}
    assert by_method[P_PROTECTED].selected_action_ids == ()
    assert by_method[SCALE_BP_PRIMARY].selected_action_ids == (
        "B::zero_to_one",
    )
    assert by_method[CYCLIC_ACTION_IDENTITY].selected_action_ids == ()
    assert result.oracle.selected_action_ids == ("B::zero_to_one",)
    assert len(result.action_evidence) == len(METHOD_IDS) * 6
    assert len(result.policy_evidence) == len(METHOD_IDS)
    assert request.action_inputs[0].donor_prediction.fit_role == "PSEUDO_H_J_D"


def test_terminal_labels_change_evidence_but_not_prelabel_route_decisions() -> None:
    request = _request(with_action=True, pseudo=True)
    assert isinstance(request.route_scope, PseudoReplayScope)
    first = replay_pseudo_case(
        PseudoCaseReplayRequest(request, _terminal_receipt(request.route_scope))
    )
    second = replay_pseudo_case(
        PseudoCaseReplayRequest(
            request,
            _terminal_receipt(request.route_scope, reverse_labels=True),
        )
    )
    assert tuple(
        (row.method_id, row.selected_action_ids, row.decision_hash)
        for row in first.method_results
    ) == tuple(
        (row.method_id, row.selected_action_ids, row.decision_hash)
        for row in second.method_results
    )
    assert first.terminal_label_hash != second.terminal_label_hash
    assert first.result_hash != second.result_hash


def test_terminal_label_loader_rejects_inexact_case_lineage() -> None:
    request = _request(with_action=True, pseudo=True)
    assert isinstance(request.route_scope, PseudoReplayScope)
    scope = request.route_scope
    case = scope.case_inventory.cases(scope.pseudo_center)[0]
    rows = tuple(
        TerminalCaseLabelInput(
            candidate,
            tuple(
                (scope.pseudo_center, candidate, f"sample-{candidate}-{index}")
                for index in range(2)
            ),
            (1, 0),
        )
        for candidate in scope.case_inventory.cases(scope.pseudo_center)
    )
    poisoned = replace(
        rows[0],
        ordered_sample_keys=(
            (scope.pseudo_center, case, "sample-not-in-manifest-0"),
            (scope.pseudo_center, case, "sample-not-in-manifest-1"),
        ),
    )
    with pytest.raises(ProtocolError, match="lineage drifted"):
        load_terminal_case_label_receipt(scope, (poisoned, *rows[1:]))


def test_pseudo_case_route_rejects_final_donor_fit_role() -> None:
    request = _request(with_action=True, pseudo=True)
    action = request.action_inputs[0]
    poisoned_prediction = replace(action.donor_prediction, fit_role="FINAL_H_C")
    poisoned_action = RouteActionInput(
        action.endpoint_projection,
        action.descriptor,
        poisoned_prediction,
    )
    with pytest.raises(ProtocolError, match="case-route request lineage"):
        replace(request, action_inputs=(poisoned_action,))
