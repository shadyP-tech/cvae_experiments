from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.cvae.test_sceptre_model_freeze import frozen_fixture as _v3_fixture
from tests.cvae.test_sceptre_v4_lifecycle import (
    _manifest_and_frame,
    _prediction_store,
    _replay,
)

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.seals import (
    DurablePreterminalAttestation,
    EXPECTED_DECISION_KEYS,
    FreshProcessValidation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution.persistence import (
    PRETERMINAL_BUNDLE_MEMBER,
    _partition_payload,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution.validation_decisions import (
    reconstruct_partition,
    validate_decision_graph,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.execution.validation_terminal import (
    validate_terminal_lineage,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.label_broker import (
    RoleLabelBroker,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.outcome_builder import (
    build_role_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.phase_manager import (
    CandidateSetPhaseManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.phase_orchestrator import (
    run_routing_phases,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.route_policy import (
    FrozenRoutePolicy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.terminal_evaluation import (
    TerminalEvaluationResult,
    evaluate_terminal_surfaces,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import atomic_json


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("sceptre-v4-validation")
    fixture = _v3_fixture.__wrapped__()
    replay = _replay(fixture)
    manager = CandidateSetPhaseManager(fixture.partition, replay.context)
    manifest, manifest_hash, frame, row_specs = _manifest_and_frame(root, fixture)
    prediction_hash = "d" * 64
    lease_hash = "e" * 64
    broker = RoleLabelBroker(
        manager=manager,
        partition=fixture.partition,
        frame=frame,
        manifest_path=manifest,
        expected_manifest_sha256=manifest_hash,
        prediction_store_hash=prediction_hash,
        authorization_lease_hash=lease_hash,
    )
    candidates, baseline = _prediction_store(row_specs)
    phases = run_routing_phases(
        replay,
        partition=fixture.partition,
        manager=manager,
        broker=broker,
        candidate_probabilities=candidates,
        exact_b_probabilities=baseline,
        candidate_source_order=CENTERS,
        prediction_store_hash=prediction_hash,
    )
    partition_payload = _partition_payload(fixture.partition)
    bundle = {
        "partition": partition_payload,
        "routing_context": replay.context.to_payload(),
        "proposal_sets": [row.to_payload() for row in replay.proposal_sets],
        "support_decisions": [row.to_payload() for row in phases.support_decisions],
        "calibration_posteriors": [
            row.to_payload() for row in phases.calibration_posteriors
        ],
        "confirmation_decisions": [
            row.to_payload() for row in phases.confirmation_decisions
        ],
    }
    index = {
        "route_policy_hash": phases.route_policy.policy_artifact_hash,
        "policy_seal_hash": phases.route_policy.policy_seal_hash,
    }
    partition = reconstruct_partition(partition_payload, phases.route_policy)
    validate_decision_graph(
        index=index,
        bundle=bundle,
        development=replay.to_payload(),
        phases=phases.to_payload(),
        journal=phases.label_journal,
        policy=phases.route_policy,
        partition=partition,
    )
    return SimpleNamespace(
        root=root,
        fixture=fixture,
        replay=replay,
        manager=manager,
        broker=broker,
        candidates=candidates,
        baseline=baseline,
        phases=phases,
        partition=partition,
        partition_payload=partition_payload,
        bundle=bundle,
        index=index,
        prediction_hash=prediction_hash,
    )


def test_self_consistently_rehashed_policy_route_tamper_is_rejected(graph) -> None:
    policy = graph.phases.route_policy
    rows = list(policy.route_rows)
    first = list(rows[0])
    assert first[5] != EXACT_B_CANDIDATE
    first[5] = EXACT_B_CANDIDATE
    rows[0] = tuple(first)
    tampered_policy = FrozenRoutePolicy(
        partition_hash=policy.partition_hash,
        routing_context_hash=policy.routing_context_hash,
        proposal_set_seal_hash=policy.proposal_set_seal_hash,
        support_seal_hash=policy.support_seal_hash,
        policy_seal_hash=policy.policy_seal_hash,
        route_rows=tuple(rows),
    )
    tampered_phases = dict(graph.phases.to_payload())
    tampered_phases["route_policy_hash"] = tampered_policy.policy_artifact_hash
    tampered_phases["phase_hash"] = ""
    from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
        canonical_hash,
    )

    tampered_phases["phase_hash"] = canonical_hash(
        {key: value for key, value in tampered_phases.items() if key != "phase_hash"}
    )
    tampered_index = dict(graph.index)
    tampered_index["route_policy_hash"] = tampered_policy.policy_artifact_hash
    with pytest.raises(ProtocolError, match="decision seals drifted"):
        validate_decision_graph(
            index=tampered_index,
            bundle=graph.bundle,
            development=graph.replay.to_payload(),
            phases=tampered_phases,
            journal=graph.phases.label_journal,
            policy=tampered_policy,
            partition=graph.partition,
        )


def test_self_consistently_rehashed_terminal_metric_tamper_is_rejected(graph) -> None:
    source_hash = "a" * 64
    reconstruction_hash = "b" * 64
    attestation = DurablePreterminalAttestation(
        policy_seal_hash=graph.phases.policy_seal.seal_hash,
        validations=(
            FreshProcessValidation(
                101,
                graph.phases.policy_seal.seal_hash,
                source_hash,
                reconstruction_hash,
            ),
            FreshProcessValidation(
                202,
                graph.phases.policy_seal.seal_hash,
                source_hash,
                reconstruction_hash,
            ),
        ),
    )
    terminal = graph.manager.begin_terminal_evaluation(attestation)
    graph.broker.activate_terminal(terminal)
    evaluation_surfaces = []
    for target, fold_ordinal in EXPECTED_DECISION_KEYS:
        fold = graph.fixture.partition.fold(target, fold_ordinal)
        scoped = graph.broker.open_evaluation(target, fold_ordinal, terminal)
        model = graph.replay.context.model_for_target(target)
        evidence = build_role_evidence(
            scoped,
            fold=fold,
            partition_hash=graph.fixture.partition.partition_hash,
            candidate_probabilities=graph.candidates,
            exact_b_probabilities=graph.baseline,
            candidate_source_order=CENTERS,
            prediction_store_hash=graph.prediction_hash,
            candidate_menu_hash=model.candidate_menu_hash,
            exact_b_control_receipt_hash=model.exact_b_control_receipt_hash,
            phase_capability=terminal,
        )
        evaluation_surfaces.append(evidence.surface)
    result = evaluate_terminal_surfaces(
        graph.phases.route_policy,
        evaluation_surfaces,
        routing_context=graph.replay.context,
        prediction_store_hash=graph.prediction_hash,
        terminal_capability_hash=terminal.capability_hash,
    )
    preterminal_bundle = {
        "route_policy": graph.phases.route_policy.to_payload(),
        "partition": graph.partition_payload,
        "support_decisions": [
            row.to_payload() for row in graph.phases.support_decisions
        ],
        "label_journal_preterminal": dict(graph.phases.label_journal),
    }
    atomic_json(graph.root / PRETERMINAL_BUNDLE_MEMBER, preterminal_bundle)
    durable = {"attestation_hash": attestation.attestation_hash}
    journal = graph.broker.journal_payload()
    preterminal = {
        "route_policy_hash": graph.phases.route_policy.policy_artifact_hash,
        "prediction_store_hash": graph.prediction_hash,
    }
    validate_terminal_lineage(
        graph.root,
        result=result,
        durable=durable,
        journal=journal,
        preterminal=preterminal,
    )
    tampered_metric = replace(
        result.folds[0],
        case_count=result.folds[0].case_count + 1,
        fold_metric_hash="",
    )
    tampered = TerminalEvaluationResult(
        route_policy_hash=result.route_policy_hash,
        prediction_store_hash=result.prediction_store_hash,
        terminal_capability_hash=result.terminal_capability_hash,
        folds=(tampered_metric, *result.folds[1:]),
    )
    with pytest.raises(ProtocolError, match="terminal fold lineage drifted"):
        validate_terminal_lineage(
            graph.root,
            result=tampered,
            durable=durable,
            journal=journal,
            preterminal=preterminal,
        )
