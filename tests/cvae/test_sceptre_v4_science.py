from __future__ import annotations

from dataclasses import replace

import pytest

from tests.cvae.test_sceptre_model_freeze import frozen_fixture as _v3_fixture

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.hashing import (
    canonical_hash,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.outcome_surface import (
    EXACT_B_CANDIDATE,
    ConfusionCounts,
    FamilyOutcome,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.seals import (
    EXPECTED_DECISION_KEYS,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.uncertainty import (
    ActionUncertaintySummary,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.confirmation_gate import (
    apply_confirmation_gate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.development import (
    FrozenRoutingContext,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.posterior import (
    PairedCandidatePosterior,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.proposal_set import (
    build_candidate_set_proposal,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.route_policy import (
    FrozenRoutePolicy,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.support_posterior import (
    select_support_candidate,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router.evidence_builder import (
    build_target_prediction_evidence,
)


@pytest.fixture(scope="module")
def fixture():
    return _v3_fixture.__wrapped__()


def _context(fixture) -> FrozenRoutingContext:
    old = fixture.full_router
    return FrozenRoutingContext(
        models=fixture.models,
        partition_hash=fixture.partition.partition_hash,
        partition_identity_sha256=old.partition_identity_sha256,
        partition_fold_inventory_sha256=old.partition_fold_inventory_sha256,
        dirichlet_config=fixture.dirichlet_config,
    )


def _proposal(fixture, target: str):
    model = fixture.models[tuple(row.outer_target for row in fixture.models).index(target)]
    evidence = build_target_prediction_evidence(
        fixture.raw,
        target_center=target,
        raw_source_receipt_hash="a" * 64,
    )
    return build_candidate_set_proposal(model, evidence)


def _outcome(
    *,
    fixture,
    target: str,
    fold_ordinal: int,
    role: str,
    source: str,
    bacc: float,
    brier: float,
    log_loss: float,
) -> FamilyOutcome:
    fold = fixture.partition.fold(target, fold_ordinal)
    model = next(row for row in fixture.models if row.outer_target == target)
    correct = int(round(bacc * 100))
    confusion = ConfusionCounts(correct, 100 - correct, 100 - correct, correct)
    return FamilyOutcome(
        target_center=target,
        fold_ordinal=fold_ordinal,
        role=role,
        candidate_center=source,
        partition_hash=fixture.partition.partition_hash,
        case_set_hash=fold.case_set_hash(role),
        candidate_menu_hash=model.candidate_menu_hash,
        prediction_receipt_hash=canonical_hash(
            ["sceptre-v4-test-prediction", target, fold_ordinal, role, source]
        ),
        confusion=confusion,
        brier_sum=brier * confusion.row_count,
        log_loss_sum=log_loss * confusion.row_count,
        case_count=len(
            fold.selection_case_ids
            if role == "SELECTION"
            else fold.calibration_case_ids
        ),
        exact_b_control_receipt_hash=(
            model.exact_b_control_receipt_hash
            if source == EXACT_B_CANDIDATE
            else None
        ),
    )


def _support(fixture, *, tie: bool = False):
    target = "2"
    fold = fixture.partition.fold(target, 0)
    proposal = _proposal(fixture, target)
    baseline = _outcome(
        fixture=fixture,
        target=target,
        fold_ordinal=0,
        role="SELECTION",
        source=EXACT_B_CANDIDATE,
        bacc=0.50,
        brier=0.25,
        log_loss=0.70,
    )
    scores = {
        source: (0.80 if source in ({"1", "3"} if tie else {"1"}) else 0.55)
        for source in proposal.candidate_sources
    }
    candidates = tuple(
        _outcome(
            fixture=fixture,
            target=target,
            fold_ordinal=0,
            role="SELECTION",
            source=source,
            bacc=scores[source],
            brier=0.20,
            log_loss=0.60,
        )
        for source in proposal.candidate_sources
    )
    return proposal, select_support_candidate(
        candidates,
        exact_b=baseline,
        fold=fold,
        partition_hash=fixture.partition.partition_hash,
        proposal_set=proposal,
        routing_context=_context(fixture),
    )


def test_proposal_persists_all_eight_scores_without_inventing_b_advantage(
    fixture,
) -> None:
    proposal = _proposal(fixture, "2")
    payload = proposal.to_payload()

    assert len(proposal.ranked_sources) == 8
    assert set(proposal.ranked_sources) == set(proposal.candidate_sources)
    assert tuple(dict(proposal.predicted_utility_by_source)) == proposal.candidate_sources
    assert payload["exact_b_action"] == EXACT_B_CANDIDATE
    assert payload["exact_b_source_inner_score"] is None
    assert payload["exact_b_advantage_model_available"] is False
    assert payload["top_k_selected_from_consumed_results"] is False


def test_support_can_replace_wrong_G_top1_with_another_sealed_member(fixture) -> None:
    proposal, decision = _support(fixture)

    assert proposal.ranked_sources[0] != "1"
    assert decision.selected_candidate == "1"
    assert decision.route == "1"
    assert decision.fallback_required is False
    assert decision.reason == "UNIQUE_MAXIMUM_POSITIVE_SHRUNK_SUPPORT_GAIN"


def test_support_exact_tie_falls_back_to_b(fixture) -> None:
    _, decision = _support(fixture, tie=True)
    assert decision.selected_candidate is None
    assert decision.route == EXACT_B_CANDIDATE
    assert decision.reason == "SUPPORT_POSTERIOR_TIE_FALLBACK_TO_B"


def test_calibration_confirms_only_support_selected_member(fixture) -> None:
    proposal, support = _support(fixture)
    selected = support.selected_candidate
    assert selected is not None
    target = support.target_center
    actions = (*proposal.candidate_sources, EXACT_B_CANDIDATE)
    summaries = tuple(
        ActionUncertaintySummary(
            action_id=action,
            point_bacc=0.8 if action == selected else 0.5,
            point_brier=0.2 if action == selected else 0.25,
            point_log_loss=0.6 if action == selected else 0.7,
            bootstrap_expected_bacc=0.8 if action == selected else 0.5,
            bootstrap_expected_brier=0.2 if action == selected else 0.25,
            bootstrap_expected_log_loss=0.6 if action == selected else 0.7,
            bacc_superiority_probability=0.9 if action == selected else 0.0,
            brier_noninferiority_probability=0.9 if action == selected else 1.0,
            log_loss_noninferiority_probability=0.9 if action == selected else 1.0,
            joint_acceptance_probability=0.9 if action == selected else 0.0,
        )
        for action in actions
    )
    selected_summary = next(row for row in summaries if row.action_id == selected)
    posterior = PairedCandidatePosterior(
        target_center=target,
        fold_ordinal=0,
        fold_hash=support.fold_hash,
        partition_hash=support.partition_hash,
        calibration_case_set_hash=support.calibration_case_set_hash,
        routing_context_hash=support.routing_context_hash,
        proposal_set_hash=support.proposal_set_hash,
        support_decision_hash=support.decision_hash,
        candidate_center=selected,
        prediction_surface_hash="1" * 64,
        bootstrap_config_hash="2" * 64,
        shared_weight_draw_hash="3" * 64,
        action_summaries=summaries,
        candidate_summary_hash=selected_summary.summary_hash,
        joint_acceptance_probability=0.9,
    )
    candidate = _outcome(
        fixture=fixture,
        target=target,
        fold_ordinal=0,
        role="CALIBRATION",
        source=selected,
        bacc=0.80,
        brier=0.20,
        log_loss=0.60,
    )
    baseline = _outcome(
        fixture=fixture,
        target=target,
        fold_ordinal=0,
        role="CALIBRATION",
        source=EXACT_B_CANDIDATE,
        bacc=0.50,
        brier=0.25,
        log_loss=0.70,
    )
    decision = apply_confirmation_gate(
        support,
        posterior=posterior,
        candidate=candidate,
        exact_b=baseline,
        routing_context=_context(fixture),
    )

    assert decision.accepted is True
    assert decision.route == selected
    harmful = replace(
        candidate,
        brier_sum=0.30 * candidate.confusion.row_count,
        outcome_hash="",
    )
    rejected = apply_confirmation_gate(
        support,
        posterior=posterior,
        candidate=harmful,
        exact_b=baseline,
        routing_context=_context(fixture),
    )
    assert rejected.accepted is False
    assert rejected.route == EXACT_B_CANDIDATE


def test_policy_round_trip_allows_support_member_not_G_top1(fixture) -> None:
    proposal, support = _support(fixture)
    rows = tuple(
        (
            target,
            fold,
            proposal.proposal_set_hash,
            support.decision_hash,
            "1" if target != "1" else "0",
            "1" if target != "1" else "0",
            canonical_hash(["confirmation", target, fold]),
        )
        for target, fold in EXPECTED_DECISION_KEYS
    )
    policy = FrozenRoutePolicy(
        partition_hash=fixture.partition.partition_hash,
        routing_context_hash=_context(fixture).context_hash,
        proposal_set_seal_hash="4" * 64,
        support_seal_hash="5" * 64,
        policy_seal_hash="6" * 64,
        route_rows=rows,
    )
    replayed = FrozenRoutePolicy.from_canonical_bytes(policy.to_canonical_bytes())
    assert replayed == policy
    assert replayed.route_for("2", 0) == "1"
