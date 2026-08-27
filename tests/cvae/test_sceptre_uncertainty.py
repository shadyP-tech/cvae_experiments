from __future__ import annotations

from dataclasses import replace

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    build_three_role_partition,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.phase_order import (
    PhaseCapability,
    TerminalEvaluationCapability,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.uncertainty import (
    FIXED_ACCEPTANCE_PROBABILITY,
    SEED_CELL_GRID,
    DirichletBootstrapConfig,
    build_role_prediction_surface,
    paired_dirichlet_route_decision,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError


TARGET = "0"
ROLE = "SELECTION"
ROUTER_BUNDLE_HASH = canonical_hash({"router": "frozen-before-labels"})
G_PROPOSAL_HASH = canonical_hash({"proposal": "source-1"})


def _partition():
    identities = tuple(
        CaseIdentity(center, f"case-{center}-{index}", f"sample-{center}-{index}")
        for center in CENTERS
        for index in range(5)
    )
    return build_three_role_partition(identities, expected_total_case_count=45)


def _phase_capability(partition, fold, role: str = ROLE, *, nonce: str = "first"):
    return PhaseCapability(
        role={
            "SELECTION": "SELECTION_LABELS",
            "CALIBRATION": "CALIBRATION_LABELS",
        }[role],
        target_center=fold.target_center,
        fold_ordinal=fold.fold_ordinal,
        partition_hash=partition.partition_hash,
        router_bundle_hash=ROUTER_BUNDLE_HASH,
        g_proposal_hash=G_PROPOSAL_HASH,
        predecessor_decision_hash=canonical_hash(
            {"predecessor_decision": role}
        ),
        predecessor_seal_hash=canonical_hash({"predecessor_seal": role}),
        nonce_hash=canonical_hash({"nonce": nonce}),
    )


def _terminal_capability(partition):
    policy = canonical_hash({"policy": "complete"})
    route_policy = canonical_hash({"route_policy": "frozen"})
    attestation = canonical_hash({"attestation": "durable"})
    body = {
        "schema_version": "sceptre_terminal_evaluation_capability_v1",
        "partition_hash": partition.partition_hash,
        "router_bundle_hash": ROUTER_BUNDLE_HASH,
        "route_policy_hash": route_policy,
        "policy_seal_hash": policy,
        "durable_attestation_hash": attestation,
        "one_shot": True,
        "raw_labels_may_be_persisted": False,
    }
    return TerminalEvaluationCapability(
        partition_hash=partition.partition_hash,
        router_bundle_hash=ROUTER_BUNDLE_HASH,
        route_policy_hash=route_policy,
        policy_seal_hash=policy,
        durable_attestation_hash=attestation,
        capability_hash=canonical_hash(body),
    )


def _role_rows(fold, role: str = ROLE):
    role_cases = {
        "SELECTION": fold.selection_case_ids,
        "CALIBRATION": fold.calibration_case_ids,
        "EVALUATION": fold.evaluation_case_ids,
    }[role]
    observations: list[str] = []
    cases: list[str] = []
    labels: list[int] = []
    for case_id in role_cases:
        for label in (0, 1):
            observations.append(f"{case_id}::row-{label}")
            cases.append(case_id)
            labels.append(label)
    return tuple(observations), tuple(cases), tuple(labels)


def _prediction_mapping(
    labels: tuple[int, ...],
    *,
    strong_actions: frozenset[str] = frozenset({"1"}),
    identical_to_b: frozenset[str] = frozenset(),
    first_seed_only: frozenset[str] = frozenset(),
):
    result = {}
    for action in (*legal_routing_sources(TARGET), EXACT_B_CANDIDATE):
        cells = {}
        for seed_index, key in enumerate(SEED_CELL_GRID):
            strong = action in strong_actions and (
                action not in first_seed_only or seed_index == 0
            )
            if strong:
                values = tuple(0.9 if label == 1 else 0.1 for label in labels)
            elif action == EXACT_B_CANDIDATE or action in identical_to_b:
                values = tuple(0.6 for _label in labels)
            else:
                values = tuple(0.5 for _label in labels)
            cells[key] = values
        result[action] = cells
    return result


def _surface(
    *,
    role: str = ROLE,
    probabilities=None,
    capability: PhaseCapability | TerminalEvaluationCapability | None = None,
):
    partition = _partition()
    fold = partition.fold(TARGET, 0)
    observations, cases, labels = _role_rows(fold, role)
    mapping = _prediction_mapping(labels) if probabilities is None else probabilities
    if capability is None:
        capability = (
            _terminal_capability(partition)
            if role == "EVALUATION"
            else _phase_capability(partition, fold, role)
        )
    return build_role_prediction_surface(
        target_center=TARGET,
        fold=fold,
        partition_hash=partition.partition_hash,
        role=role,
        observation_ids=observations,
        case_ids=cases,
        labels=labels,
        probabilities_by_action_and_seed=mapping,
        candidate_menu_hash="candidate-menu-hash",
        exact_b_control_receipt_hash="exact-b-control-receipt",
        prediction_bundle_sha256=canonical_hash({"predictions": role}),
        phase_capability=capability,
    )


def test_exact_geometry_and_phase_identity_are_canonical() -> None:
    surface = _surface()

    assert surface.action_ids == (*legal_routing_sources(TARGET), EXACT_B_CANDIDATE)
    assert surface.tensor_shape == (9, 9, len(surface.observation_ids))
    assert set(surface.whole_case_ids) == set(surface.fold.selection_case_ids)
    assert all(
        tuple((cell.training_seed, cell.generation_seed) for cell in action.seed_cells)
        == SEED_CELL_GRID
        for action in surface.actions
    )

    observations, cases, labels = _role_rows(surface.fold)
    reversed_mapping = {
        action: dict(reversed(tuple(cells.items())))
        for action, cells in reversed(tuple(_prediction_mapping(labels).items()))
    }
    replay = build_role_prediction_surface(
        target_center=TARGET,
        fold=surface.fold,
        partition_hash=surface.partition_hash,
        role=ROLE,
        observation_ids=observations,
        case_ids=cases,
        labels=labels,
        probabilities_by_action_and_seed=reversed_mapping,
        candidate_menu_hash=surface.candidate_menu_hash,
        exact_b_control_receipt_hash=surface.exact_b_control_receipt_hash,
        prediction_bundle_sha256=surface.prediction_bundle_sha256,
        phase_capability=surface.phase_capability,
    )
    assert replay.surface_hash == surface.surface_hash

    changed_capability = replace(
        surface.phase_capability,
        nonce_hash=canonical_hash({"nonce": "second"}),
    )
    changed = replace(
        surface,
        phase_capability=changed_capability,
        phase_capability_identity_hash="",
        surface_hash="",
    )
    assert changed.phase_capability_identity_hash != surface.phase_capability_identity_hash
    assert changed.surface_hash != surface.surface_hash

    config = DirichletBootstrapConfig(draw_count=32, rng_seed=9)
    first_decision = paired_dirichlet_route_decision(
        surface, g_proposed_candidate="1", config=config
    )
    changed_decision = paired_dirichlet_route_decision(
        changed, g_proposed_candidate="1", config=config
    )
    assert first_decision.phase_capability_identity_hash != (
        changed_decision.phase_capability_identity_hash
    )
    assert first_decision.decision_hash != changed_decision.decision_hash


def test_seed_cells_are_pooled_then_meaned_as_nuisance_replications() -> None:
    partition = _partition()
    fold = partition.fold(TARGET, 0)
    observations, cases, labels = _role_rows(fold)
    probabilities = _prediction_mapping(
        labels,
        strong_actions=frozenset({"1"}),
        first_seed_only=frozenset({"1"}),
    )
    surface = build_role_prediction_surface(
        target_center=TARGET,
        fold=fold,
        partition_hash=partition.partition_hash,
        role=ROLE,
        observation_ids=observations,
        case_ids=cases,
        labels=labels,
        probabilities_by_action_and_seed=probabilities,
        candidate_menu_hash="menu",
        exact_b_control_receipt_hash="control",
        prediction_bundle_sha256=canonical_hash({"surface": 1}),
        phase_capability=_phase_capability(partition, fold),
    )
    decision = paired_dirichlet_route_decision(
        surface,
        g_proposed_candidate="1",
        config=DirichletBootstrapConfig(draw_count=64, rng_seed=7),
    )

    expected = (1.0 + 8.0 * 0.5) / 9.0
    assert decision.summaries_by_action["1"].point_bacc == pytest.approx(expected)
    assert decision.summaries_by_action["1"].bootstrap_expected_bacc == pytest.approx(
        expected
    )


def test_shared_paired_draws_and_fixed_gate_accept_the_g_proposal() -> None:
    partition = _partition()
    fold = partition.fold(TARGET, 0)
    observations, cases, labels = _role_rows(fold)
    probabilities = _prediction_mapping(
        labels,
        strong_actions=frozenset({"1"}),
        identical_to_b=frozenset({"2"}),
    )
    surface = build_role_prediction_surface(
        target_center=TARGET,
        fold=fold,
        partition_hash=partition.partition_hash,
        role=ROLE,
        observation_ids=observations,
        case_ids=cases,
        labels=labels,
        probabilities_by_action_and_seed=probabilities,
        candidate_menu_hash="menu",
        exact_b_control_receipt_hash="control",
        prediction_bundle_sha256=canonical_hash({"surface": 2}),
        phase_capability=_phase_capability(partition, fold),
    )
    config = DirichletBootstrapConfig(draw_count=128, rng_seed=19)
    first = paired_dirichlet_route_decision(
        surface, g_proposed_candidate="1", config=config
    )
    replay = paired_dirichlet_route_decision(
        surface, g_proposed_candidate="1", config=config
    )
    different_seed = paired_dirichlet_route_decision(
        surface,
        g_proposed_candidate="1",
        config=DirichletBootstrapConfig(draw_count=128, rng_seed=20),
    )

    assert first.decision_hash == replay.decision_hash
    assert first.shared_weight_draw_hash == replay.shared_weight_draw_hash
    assert first.bootstrap_config_hash != different_seed.bootstrap_config_hash
    assert first.shared_weight_draw_hash != different_seed.shared_weight_draw_hash
    assert first.g_proposed_candidate == "1"
    assert first.accepted is True
    assert first.route == "1"
    assert first.acceptance_probability == FIXED_ACCEPTANCE_PROBABILITY == 0.8
    assert first.summaries_by_action["1"].joint_acceptance_probability == 1.0

    candidate = first.summaries_by_action["2"]
    baseline = first.summaries_by_action[EXACT_B_CANDIDATE]
    assert candidate.bootstrap_expected_bacc == baseline.bootstrap_expected_bacc
    assert candidate.bootstrap_expected_brier == baseline.bootstrap_expected_brier
    assert candidate.bootstrap_expected_log_loss == baseline.bootstrap_expected_log_loss
    assert candidate.bacc_superiority_probability == 0.0
    assert candidate.brier_noninferiority_probability == 1.0
    assert candidate.log_loss_noninferiority_probability == 1.0
    assert candidate.joint_acceptance_probability == 0.0


def test_kernel_never_substitutes_a_better_nonproposed_action() -> None:
    surface = _surface()
    decision = paired_dirichlet_route_decision(
        surface,
        g_proposed_candidate="2",
        config=DirichletBootstrapConfig(draw_count=64, rng_seed=23),
    )

    assert decision.summaries_by_action["1"].joint_acceptance_probability == 1.0
    assert decision.summaries_by_action["2"].joint_acceptance_probability == 0.0
    assert decision.g_proposed_candidate == "2"
    assert decision.selected_candidate is None
    assert decision.route == EXACT_B_CANDIDATE
    assert (
        decision.reason
        == "FIXED_0_8_UPSTREAM_CANDIDATE_PAIRED_GATE_FALLBACK_TO_B"
    )


def test_descriptive_candidate_tie_does_not_reselect_or_force_fallback() -> None:
    partition = _partition()
    fold = partition.fold(TARGET, 0)
    _observations, _cases, labels = _role_rows(fold)
    probabilities = _prediction_mapping(
        labels,
        strong_actions=frozenset({"1", "2"}),
    )
    surface = _surface(probabilities=probabilities)
    decision = paired_dirichlet_route_decision(
        surface,
        g_proposed_candidate="2",
        config=DirichletBootstrapConfig(draw_count=64, rng_seed=29),
    )

    assert decision.summaries_by_action["1"].bootstrap_expected_bacc == (
        decision.summaries_by_action["2"].bootstrap_expected_bacc
    )
    assert decision.g_proposed_candidate == "2"
    assert decision.selected_candidate == "2"
    assert decision.route == "2"
    assert decision.accepted is True


def test_calibration_binds_support_decision_and_support_selected_candidate() -> None:
    surface = _surface(role="CALIBRATION")
    decision = paired_dirichlet_route_decision(
        surface,
        g_proposed_candidate="1",
        support_selected_candidate="1",
        config=DirichletBootstrapConfig(draw_count=32, rng_seed=31),
    )

    assert decision.support_decision_hash == surface.predecessor_decision_hash
    assert decision.support_selected_candidate == "1"
    assert decision.g_proposed_candidate == "1"
    assert decision.route == "1"

    with pytest.raises(ProtocolError, match="differs from support or G"):
        paired_dirichlet_route_decision(
            surface,
            g_proposed_candidate="1",
            support_selected_candidate="2",
            config=DirichletBootstrapConfig(draw_count=16, rng_seed=31),
        )
    with pytest.raises(ProtocolError, match="differs from support or G"):
        paired_dirichlet_route_decision(
            surface,
            g_proposed_candidate="1",
            config=DirichletBootstrapConfig(draw_count=16, rng_seed=31),
        )


def test_capability_scope_terminal_gate_and_illegal_proposals_fail_closed() -> None:
    partition = _partition()
    fold = partition.fold(TARGET, 0)
    valid_capability = _phase_capability(partition, fold)

    with pytest.raises(ProtocolError, match="typed phase capability"):
        _surface(capability=_terminal_capability(partition))

    wrong_role = replace(valid_capability, role="CALIBRATION_LABELS")
    with pytest.raises(ProtocolError, match="scope or partition"):
        _surface(capability=wrong_role)

    wrong_partition = replace(
        valid_capability,
        partition_hash=canonical_hash({"partition": "wrong"}),
    )
    with pytest.raises(ProtocolError, match="scope or partition"):
        _surface(capability=wrong_partition)

    evaluation = _surface(role="EVALUATION")
    assert isinstance(evaluation.phase_capability, TerminalEvaluationCapability)
    with pytest.raises(ProtocolError, match="evaluation labels cannot select"):
        paired_dirichlet_route_decision(
            evaluation,
            g_proposed_candidate="1",
            config=DirichletBootstrapConfig(draw_count=16, rng_seed=3),
        )

    bad_terminal = replace(
        _terminal_capability(partition),
        capability_hash=canonical_hash({"terminal": "forged"}),
    )
    with pytest.raises(ProtocolError, match="semantic replay"):
        _surface(role="EVALUATION", capability=bad_terminal)

    surface = _surface()
    for illegal in (TARGET, EXACT_B_CANDIDATE, "outside"):
        with pytest.raises(ProtocolError, match="outside exact C minus H"):
            paired_dirichlet_route_decision(
                surface,
                g_proposed_candidate=illegal,
                config=DirichletBootstrapConfig(draw_count=16, rng_seed=3),
            )


def test_geometry_lineage_and_nonfinite_inputs_fail_closed() -> None:
    surface = _surface()
    observations, cases, labels = _role_rows(surface.fold)
    valid = _prediction_mapping(labels)

    missing_action = dict(valid)
    missing_action.pop("1")
    with pytest.raises(ProtocolError, match="C minus H plus B"):
        build_role_prediction_surface(
            target_center=TARGET,
            fold=surface.fold,
            partition_hash=surface.partition_hash,
            role=ROLE,
            observation_ids=observations,
            case_ids=cases,
            labels=labels,
            probabilities_by_action_and_seed=missing_action,
            candidate_menu_hash="menu",
            exact_b_control_receipt_hash="control",
            prediction_bundle_sha256=canonical_hash({"bad": 1}),
            phase_capability=surface.phase_capability,
        )

    missing_seed = {action: dict(cells) for action, cells in valid.items()}
    missing_seed["1"].pop(SEED_CELL_GRID[-1])
    with pytest.raises(ProtocolError, match="exact seed grid"):
        build_role_prediction_surface(
            target_center=TARGET,
            fold=surface.fold,
            partition_hash=surface.partition_hash,
            role=ROLE,
            observation_ids=observations,
            case_ids=cases,
            labels=labels,
            probabilities_by_action_and_seed=missing_seed,
            candidate_menu_hash="menu",
            exact_b_control_receipt_hash="control",
            prediction_bundle_sha256=canonical_hash({"bad": 2}),
            phase_capability=surface.phase_capability,
        )

    nonfinite = {action: dict(cells) for action, cells in valid.items()}
    first_key = SEED_CELL_GRID[0]
    poisoned = list(nonfinite["1"][first_key])
    poisoned[0] = float("nan")
    nonfinite["1"][first_key] = tuple(poisoned)
    with pytest.raises(ProtocolError, match="finite probability"):
        build_role_prediction_surface(
            target_center=TARGET,
            fold=surface.fold,
            partition_hash=surface.partition_hash,
            role=ROLE,
            observation_ids=observations,
            case_ids=cases,
            labels=labels,
            probabilities_by_action_and_seed=nonfinite,
            candidate_menu_hash="menu",
            exact_b_control_receipt_hash="control",
            prediction_bundle_sha256=canonical_hash({"bad": 3}),
            phase_capability=surface.phase_capability,
        )

    wrong_cases = (*cases[:-1], "case-outside-fold")
    with pytest.raises(ProtocolError, match="typed fold role"):
        build_role_prediction_surface(
            target_center=TARGET,
            fold=surface.fold,
            partition_hash=surface.partition_hash,
            role=ROLE,
            observation_ids=observations,
            case_ids=wrong_cases,
            labels=labels,
            probabilities_by_action_and_seed=valid,
            candidate_menu_hash="menu",
            exact_b_control_receipt_hash="control",
            prediction_bundle_sha256=canonical_hash({"bad": 4}),
            phase_capability=surface.phase_capability,
        )


def test_acceptance_probability_is_immutable_and_decision_replays_proposal_only() -> None:
    config = DirichletBootstrapConfig(draw_count=32, rng_seed=11)
    assert config.acceptance_probability == 0.8
    with pytest.raises(TypeError):
        DirichletBootstrapConfig(  # type: ignore[call-arg]
            draw_count=32,
            rng_seed=11,
            acceptance_probability=0.7,
        )

    surface = _surface()
    with pytest.raises(ProtocolError, match="claim boundary"):
        replace(surface, descriptive_only=False)

    decision = paired_dirichlet_route_decision(
        surface, g_proposed_candidate="1", config=config
    )
    at_boundary = tuple(
        replace(row, joint_acceptance_probability=0.8, summary_hash="")
        if row.action_id == decision.g_proposed_candidate
        else row
        for row in decision.action_summaries
    )
    accepted = replace(decision, action_summaries=at_boundary, decision_hash="")
    assert accepted.accepted is True

    below_boundary = tuple(
        replace(row, joint_acceptance_probability=0.799999, summary_hash="")
        if row.action_id == decision.g_proposed_candidate
        else row
        for row in decision.action_summaries
    )
    rejected = replace(
        decision,
        action_summaries=below_boundary,
        selected_candidate=None,
        route=EXACT_B_CANDIDATE,
        accepted=False,
        reason="FIXED_0_8_UPSTREAM_CANDIDATE_PAIRED_GATE_FALLBACK_TO_B",
        decision_hash="",
    )
    assert rejected.accepted is False
    assert rejected.route == EXACT_B_CANDIDATE
