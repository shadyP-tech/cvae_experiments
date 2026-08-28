from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.cvae.test_sceptre_model_freeze import frozen_fixture as _v3_fixture

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.evidence_builder import (
    build_target_prediction_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.seals import (
    DurablePreterminalAttestation,
    EXPECTED_DECISION_KEYS,
    FreshProcessValidation,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.development import (
    FrozenDevelopmentReplay,
    FrozenRoutingContext,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.label_broker import (
    RoleLabelBroker,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.outcome_builder import (
    CANDIDATE_EXCLUSION_SENTINEL,
    build_role_evidence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.phase_manager import (
    CandidateSetPhaseManager,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.phase_orchestrator import (
    run_routing_phases,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.proposal_set import (
    build_candidate_set_proposal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.route_policy import (
    FrozenRoutePolicy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.terminal_evaluation import (
    evaluate_terminal_surfaces,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    legal_routing_sources,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import sha256_file
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    evaluation_row_id,
)


@pytest.fixture(scope="module")
def fixture():
    return _v3_fixture.__wrapped__()


def _replay(fixture) -> FrozenDevelopmentReplay:
    old = fixture.full_router
    context = FrozenRoutingContext(
        models=fixture.models,
        partition_hash=fixture.partition.partition_hash,
        partition_identity_sha256=old.partition_identity_sha256,
        partition_fold_inventory_sha256=old.partition_fold_inventory_sha256,
        dirichlet_config=fixture.dirichlet_config,
    )
    proposals = []
    for model in fixture.models:
        evidence = build_target_prediction_evidence(
            fixture.raw,
            target_center=model.outer_target,
            raw_source_receipt_hash="a" * 64,
        )
        proposals.append(build_candidate_set_proposal(model, evidence))
    return FrozenDevelopmentReplay(
        fits=fixture.fits,
        context=context,
        proposal_sets=tuple(proposals),
    )


def _manifest_and_frame(tmp_path: Path, fixture):
    manifest = tmp_path / "manifest.csv"
    lines = ["case_id,center,label"]
    row_specs = []
    for identity in fixture.partition.identities:
        for label in (0, 1):
            lines.append(
                f"{identity.case_id},{identity.target_center},{label}"
            )
            row_specs.append((identity.case_id, identity.target_center, label))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_hash = sha256_file(manifest)
    rows = tuple(
        SimpleNamespace(
            row_ordinal=ordinal,
            manifest_row_index=ordinal,
            sample_id=evaluation_row_id(manifest_hash, ordinal),
            case_id=case_id,
            center=center,
        )
        for ordinal, (case_id, center, _label) in enumerate(row_specs)
    )
    return manifest, manifest_hash, SimpleNamespace(rows=rows), tuple(row_specs)


def _prediction_store(row_specs):
    row_count = len(row_specs)
    candidates = np.full((9, len(CENTERS), row_count), 0.65, dtype=np.float32)
    baseline = np.full((9, row_count), 0.65, dtype=np.float32)
    for row_ordinal, (_case_id, target, label) in enumerate(row_specs):
        target_ordinal = CENTERS.index(target)
        selected = legal_routing_sources(target)[0]
        selected_ordinal = CENTERS.index(selected)
        candidates[:, target_ordinal, row_ordinal] = CANDIDATE_EXCLUSION_SENTINEL
        candidates[:, selected_ordinal, row_ordinal] = 0.95 if label else 0.05
    candidates.setflags(write=False)
    baseline.setflags(write=False)
    return candidates, baseline


def test_complete_45_fold_lifecycle_rejects_forged_terminal_capability(
    tmp_path: Path,
    fixture,
) -> None:
    replay = _replay(fixture)
    manager = CandidateSetPhaseManager(fixture.partition, replay.context)
    manifest, manifest_hash, frame, row_specs = _manifest_and_frame(
        tmp_path, fixture
    )
    prediction_hash = canonical_hash({"prediction_store": "v4-lifecycle"})
    lease_hash = canonical_hash({"authorization_lease": "test-only"})
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
    assert len(phases.support_decisions) == 45
    assert len(phases.confirmation_decisions) == 45
    assert phases.proposal_set_seal.decision_count == 45
    assert phases.support_seal.decision_count == 45
    assert phases.policy_seal.decision_count == 45
    assert tuple(
        (row.target_center, row.fold_ordinal)
        for row in phases.support_decisions
    ) == EXPECTED_DECISION_KEYS
    assert all(row.selected_candidate is not None for row in phases.support_decisions)
    assert all(row.accepted for row in phases.confirmation_decisions)
    swapped = (
        phases.confirmation_decisions[1],
        phases.confirmation_decisions[0],
        *phases.confirmation_decisions[2:],
    )
    with pytest.raises(ProtocolError, match="inventory drifted"):
        replace(phases, confirmation_decisions=swapped, phase_hash="")
    replayed_policy = FrozenRoutePolicy.from_canonical_bytes(
        phases.route_policy.to_canonical_bytes()
    )
    assert replayed_policy == phases.route_policy

    lineage_manager = CandidateSetPhaseManager(fixture.partition, replay.context)
    for target, fold_ordinal in EXPECTED_DECISION_KEYS:
        lineage_manager.record_label_free_proposal_set(
            replay.proposal_for_target(target), fold_ordinal
        )
    lineage_manager.seal_all_proposal_sets()
    first_capability = lineage_manager.issue_selection_capability(*EXPECTED_DECISION_KEYS[0])
    forged_support = replace(
        phases.support_decisions[0],
        candidate_menu_hash="forged-menu",
        decision_hash="",
    )
    with pytest.raises(ProtocolError, match="lineage drifted"):
        lineage_manager.record_selection_decision(first_capability, forged_support)
    lineage_manager.record_selection_decision(
        first_capability, phases.support_decisions[0]
    )
    for key, support in zip(
        EXPECTED_DECISION_KEYS[1:], phases.support_decisions[1:], strict=True
    ):
        capability = lineage_manager.issue_selection_capability(*key)
        lineage_manager.record_selection_decision(capability, support)
    lineage_manager.seal_all_selection_decisions()
    calibration_capability = lineage_manager.issue_calibration_capability(
        *EXPECTED_DECISION_KEYS[0]
    )
    forged_confirmation = replace(
        phases.confirmation_decisions[0],
        fold_hash=canonical_hash({"forged": "fold"}),
        decision_hash="",
    )
    with pytest.raises(ProtocolError, match="lineage drifted"):
        lineage_manager.record_calibration_decision(
            calibration_capability, forged_confirmation
        )

    source_hash = canonical_hash({"source": "v4-lifecycle-test"})
    reconstruction_hash = canonical_hash(
        {
            "context": replay.context.context_hash,
            "policy": replayed_policy.policy_artifact_hash,
        }
    )
    attestation = DurablePreterminalAttestation(
        policy_seal_hash=phases.policy_seal.seal_hash,
        validations=(
            FreshProcessValidation(
                101,
                phases.policy_seal.seal_hash,
                source_hash,
                reconstruction_hash,
            ),
            FreshProcessValidation(
                202,
                phases.policy_seal.seal_hash,
                source_hash,
                reconstruction_hash,
            ),
        ),
    )
    terminal = manager.begin_terminal_evaluation(attestation)
    forged = replace(
        terminal,
        capability_hash=canonical_hash({"forged": terminal.capability_hash}),
    )
    with pytest.raises(ProtocolError, match="not minted by this manager"):
        broker.activate_terminal(forged)
    broker.activate_terminal(terminal)
    with pytest.raises(ProtocolError, match="invalid or reused"):
        broker.activate_terminal(terminal)

    evaluation_surfaces = []
    for target, fold_ordinal in EXPECTED_DECISION_KEYS:
        fold = fixture.partition.fold(target, fold_ordinal)
        scoped = broker.open_evaluation(target, fold_ordinal, terminal)
        model = replay.context.model_for_target(target)
        evidence = build_role_evidence(
            scoped,
            fold=fold,
            partition_hash=fixture.partition.partition_hash,
            candidate_probabilities=candidates,
            exact_b_probabilities=baseline,
            candidate_source_order=CENTERS,
            prediction_store_hash=prediction_hash,
            candidate_menu_hash=model.candidate_menu_hash,
            exact_b_control_receipt_hash=model.exact_b_control_receipt_hash,
            phase_capability=terminal,
        )
        evaluation_surfaces.append(evidence.surface)

    terminal_result = evaluate_terminal_surfaces(
        replayed_policy,
        evaluation_surfaces,
        routing_context=replay.context,
        prediction_store_hash=prediction_hash,
        terminal_capability_hash=terminal.capability_hash,
    )
    assert len(terminal_result.folds) == 45
    assert terminal_result.summary["fold_count"] == 45
    assert terminal_result.summary["evaluation_cases_exactly_once"] is True
    assert terminal_result.summary["expert_route_fold_count"] == 45
    assert phases.route_policy.policy_artifact_hash == replayed_policy.policy_artifact_hash
    assert broker.journal_payload()["raw_labels_persisted"] is False

    drifted_surfaces = list(evaluation_surfaces)
    drifted_surfaces[0] = replace(
        drifted_surfaces[0],
        candidate_menu_hash="forged-menu",
        surface_hash="",
    )
    with pytest.raises(ProtocolError, match="lineage drifted"):
        evaluate_terminal_surfaces(
            replayed_policy,
            drifted_surfaces,
            routing_context=replay.context,
            prediction_store_hash=prediction_hash,
            terminal_capability_hash=terminal.capability_hash,
        )


def test_label_broker_rejects_duplicate_row_identities(
    tmp_path: Path,
    fixture,
) -> None:
    replay = _replay(fixture)
    manager = CandidateSetPhaseManager(fixture.partition, replay.context)
    manifest, manifest_hash, frame, _row_specs = _manifest_and_frame(
        tmp_path, fixture
    )
    rows = list(frame.rows)
    duplicated = vars(rows[1]).copy()
    duplicated["row_ordinal"] = rows[0].row_ordinal
    rows[1] = SimpleNamespace(**duplicated)
    with pytest.raises(ProtocolError, match="row identities drifted"):
        RoleLabelBroker(
            manager=manager,
            partition=fixture.partition,
            frame=SimpleNamespace(rows=tuple(rows)),
            manifest_path=manifest,
            expected_manifest_sha256=manifest_hash,
            prediction_store_hash=canonical_hash({"prediction_store": "duplicate"}),
            authorization_lease_hash=canonical_hash({"lease": "duplicate"}),
        )
