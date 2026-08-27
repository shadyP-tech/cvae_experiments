"""Canonical evidence assembly and deterministic fail-closed selection."""

from __future__ import annotations

import math
from typing import Sequence

from ...protocol import ProtocolError
from .clustered_uncertainty import apply_calibrated_bound
from .contracts import (
    ActionQuery,
    ActionSelectionEvidence,
    BaccRankingPolicy,
    CandidatePoolReceipt,
    NormalizedUtility,
    OpportunityCaseReceipt,
    PairwiseRankerModel,
    P_ACTION_ID,
    SelectionDecision,
    UncertaintyCalibration,
)
from .pairwise_inference import predict_action_score, predict_pairwise_contrast


def assemble_action_selection_evidence(
    *,
    query: ActionQuery,
    equivalent_action_ids: Sequence[str],
    utility: NormalizedUtility,
    comparator_queries: Sequence[ActionQuery],
    candidate_pool: CandidatePoolReceipt,
    pairwise_model: PairwiseRankerModel,
    uncertainty_calibration: UncertaintyCalibration,
    opportunity_receipt: OpportunityCaseReceipt,
    ranking_policy: BaccRankingPolicy,
) -> ActionSelectionEvidence:
    """Build every score and bound from typed analytic inputs in one place."""

    if (
        not isinstance(query, ActionQuery)
        or not isinstance(utility, NormalizedUtility)
        or not isinstance(candidate_pool, CandidatePoolReceipt)
        or not isinstance(pairwise_model, PairwiseRankerModel)
        or not isinstance(uncertainty_calibration, UncertaintyCalibration)
        or not isinstance(opportunity_receipt, OpportunityCaseReceipt)
        or not isinstance(ranking_policy, BaccRankingPolicy)
    ):
        raise ProtocolError("Selection evidence assembly requires typed inputs.")
    comparators = tuple(comparator_queries)
    if (
        query.action_id == P_ACTION_ID
        or len({row.action_id for row in comparators}) != len(comparators)
        or query.action_id in {row.action_id for row in comparators}
        or P_ACTION_ID not in {row.action_id for row in comparators}
    ):
        raise ProtocolError("Selection evidence needs unique comparators including exact P.")
    score = predict_action_score(pairwise_model, query)
    pairwise_bounds = tuple(
        (
            comparator,
            apply_calibrated_bound(
                uncertainty_calibration,
                action_id=query.action_id,
                comparator_id=comparator.action_id,
                metric="pairwise",
                mean=predict_pairwise_contrast(
                    pairwise_model, query, comparator
                ).mean_contrast,
            ),
        )
        for comparator in comparators
    )
    return ActionSelectionEvidence(
        query=query,
        equivalent_action_ids=tuple(equivalent_action_ids),
        utility=utility,
        ranking_score=score,
        bacc=apply_calibrated_bound(
            uncertainty_calibration,
            action_id=query.action_id,
            comparator_id=P_ACTION_ID,
            metric="bacc",
            mean=utility.bacc_gain,
        ),
        brier=apply_calibrated_bound(
            uncertainty_calibration,
            action_id=query.action_id,
            comparator_id=P_ACTION_ID,
            metric="brier",
            mean=utility.brier_loss_delta,
        ),
        log=apply_calibrated_bound(
            uncertainty_calibration,
            action_id=query.action_id,
            comparator_id=P_ACTION_ID,
            metric="log",
            mean=utility.log_loss_delta,
        ),
        pairwise_bounds=pairwise_bounds,
        candidate_pool_receipt_hash=candidate_pool.receipt_hash,
        pairwise_model_hash=pairwise_model.model_hash,
        uncertainty_calibration_hash=uncertainty_calibration.calibration_hash,
        opportunity_case_receipt_hash=opportunity_receipt.receipt_hash,
        bacc_ranking_policy_hash=ranking_policy.policy_hash,
    )


def _fallback(
    *,
    raw_winner: str,
    reason: str,
    active_count: int,
    runner_up: str | None,
    candidate_pool: CandidatePoolReceipt,
    model: PairwiseRankerModel,
    calibration: UncertaintyCalibration,
    opportunity: OpportunityCaseReceipt,
    ranking_policy: BaccRankingPolicy,
) -> SelectionDecision:
    return SelectionDecision(
        selected_action_id=P_ACTION_ID,
        raw_winner_action_id=raw_winner,
        fallback_to_p=True,
        reason=reason,
        active_representative_count=active_count,
        runner_up_action_id=runner_up,
        selected_equivalent_action_ids=(P_ACTION_ID,),
        candidate_pool_receipt_hash=candidate_pool.receipt_hash,
        pairwise_model_hash=model.model_hash,
        uncertainty_calibration_hash=calibration.calibration_hash,
        opportunity_case_receipt_hash=opportunity.receipt_hash,
        bacc_ranking_policy_hash=ranking_policy.policy_hash,
        opportunity_active_representative_ids=opportunity.active_representative_ids,
    )


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)


def _validate_bound(calibration: UncertaintyCalibration, row: ActionSelectionEvidence, comparator: str, metric: str, bound) -> None:
    component = calibration.component(row.action_id, comparator, metric, bound.side)
    expected = bound.mean - component.offset if bound.side == "lower" else bound.mean + component.offset
    if component.component_hash != bound.component_hash or not _same(bound.bound, expected):
        raise ProtocolError("Selection bound drifted from uncertainty calibration arithmetic.")


def _validate_evidence(
    row: ActionSelectionEvidence,
    *,
    all_queries: dict[str, ActionQuery],
    candidate_pool: CandidatePoolReceipt,
    model: PairwiseRankerModel,
    calibration: UncertaintyCalibration,
    opportunity: OpportunityCaseReceipt,
    ranking_policy: BaccRankingPolicy,
) -> None:
    expected_comparators = {P_ACTION_ID, *(set(all_queries) - {row.action_id})}
    actual_comparators = {query.action_id for query, _ in row.pairwise_bounds}
    member = opportunity.opportunity.member(row.action_id)
    if (
        row.candidate_pool_receipt_hash != candidate_pool.receipt_hash
        or row.pairwise_model_hash != model.model_hash
        or row.uncertainty_calibration_hash != calibration.calibration_hash
        or row.opportunity_case_receipt_hash != opportunity.receipt_hash
        or row.bacc_ranking_policy_hash != ranking_policy.policy_hash
        or member.representative_action_id != row.action_id
        or (member.family, member.direction) != (row.family, row.direction)
        or row.utility.action_id != row.action_id
        or row.utility.baseline_probability_hash != opportunity.opportunity.baseline_hash
        or row.utility.candidate_probability_hash != member.probability_hash
        or row.equivalent_action_ids
        != opportunity.opportunity.equivalent_action_ids(row.action_id)
        or actual_comparators != expected_comparators
        or not _same(row.ranking_score, predict_action_score(model, row.query))
        or not _same(row.bacc.mean, row.utility.bacc_gain)
        or not _same(row.brier.mean, row.utility.brier_loss_delta)
        or not _same(row.log.mean, row.utility.log_loss_delta)
    ):
        raise ProtocolError("Selection evidence drifted from typed model, utility, or opportunity inputs.")
    for comparator_query, bound in row.pairwise_bounds:
        canonical_query = (
            ActionQuery.p_anchor(model.feature_names)
            if comparator_query.action_id == P_ACTION_ID
            else all_queries.get(comparator_query.action_id)
        )
        if canonical_query != comparator_query:
            raise ProtocolError("Selection pairwise comparator query drifted.")
        mean = predict_pairwise_contrast(model, row.query, comparator_query).mean_contrast
        if not _same(bound.mean, mean):
            raise ProtocolError("Selection pairwise mean drifted from the fitted model.")
        _validate_bound(calibration, row, comparator_query.action_id, "pairwise", bound)
    _validate_bound(calibration, row, P_ACTION_ID, "bacc", row.bacc)
    _validate_bound(calibration, row, P_ACTION_ID, "brier", row.brier)
    _validate_bound(calibration, row, P_ACTION_ID, "log", row.log)


def select_fail_closed_action(
    evidence: Sequence[ActionSelectionEvidence],
    *,
    candidate_pool: CandidatePoolReceipt,
    pairwise_model: PairwiseRankerModel,
    uncertainty_calibration: UncertaintyCalibration,
    opportunity_receipt: OpportunityCaseReceipt,
    ranking_policy: BaccRankingPolicy,
) -> SelectionDecision:
    """Select only a uniquely safe winner; every failed safety gate returns P."""

    rows = tuple(evidence)
    if (
        not isinstance(candidate_pool, CandidatePoolReceipt)
        or not isinstance(pairwise_model, PairwiseRankerModel)
        or not isinstance(uncertainty_calibration, UncertaintyCalibration)
        or not isinstance(opportunity_receipt, OpportunityCaseReceipt)
        or not isinstance(ranking_policy, BaccRankingPolicy)
    ):
        raise ProtocolError("Selection requires typed protocol inputs.")
    if (
        pairwise_model.candidate_pool_receipt_hash != candidate_pool.receipt_hash
        or pairwise_model.bacc_ranking_policy_hash != ranking_policy.policy_hash
        or pairwise_model.candidate_action_ids != opportunity_receipt.candidate_action_ids
        or uncertainty_calibration.outer_target_center != candidate_pool.outer_target_center
    ):
        raise ProtocolError("Selection model, pool, action inventory, or calibration drifted.")
    active_ids = opportunity_receipt.active_representative_ids
    if not active_ids:
        if rows:
            raise ProtocolError("Selection evidence exists for a zero-opportunity case.")
        return _fallback(
            raw_winner=P_ACTION_ID,
            reason="no_active_unique_action_opportunity",
            active_count=0,
            runner_up=None,
            candidate_pool=candidate_pool,
            model=pairwise_model,
            calibration=uncertainty_calibration,
            opportunity=opportunity_receipt,
            ranking_policy=ranking_policy,
        )
    if tuple(sorted(row.action_id for row in rows)) != active_ids:
        raise ProtocolError("Selection evidence does not exactly cover active representatives.")
    if len({row.action_id for row in rows}) != len(rows):
        raise ProtocolError("Selection evidence contains duplicate representatives.")
    all_queries = {row.action_id: row.query for row in rows}
    for row in rows:
        _validate_evidence(
            row,
            all_queries=all_queries,
            candidate_pool=candidate_pool,
            model=pairwise_model,
            calibration=uncertainty_calibration,
            opportunity=opportunity_receipt,
            ranking_policy=ranking_policy,
        )
    ranked = tuple(sorted(rows, key=lambda row: (-row.ranking_score, row.action_id)))
    winner = ranked[0]
    runner_up = ranked[1].action_id if len(ranked) > 1 else None

    failures = (
        (winner.bacc.bound <= 0.0, "winner_bacc_lcb_nonpositive"),
        (winner.brier.bound > 0.0, "winner_brier_ucb_harmful"),
        (winner.log.bound > 0.0, "winner_log_ucb_harmful"),
        (winner.pairwise_lower(P_ACTION_ID).bound <= 0.0, "winner_pairwise_lcb_vs_p_nonpositive"),
        (
            runner_up is not None and winner.pairwise_lower(runner_up).bound <= 0.0,
            "winner_pairwise_lcb_vs_runner_nonpositive",
        ),
    )
    for failed, reason in failures:
        if failed:
            return _fallback(
                raw_winner=winner.action_id,
                reason=reason,
                active_count=len(rows),
                runner_up=runner_up,
                candidate_pool=candidate_pool,
                model=pairwise_model,
                calibration=uncertainty_calibration,
                opportunity=opportunity_receipt,
                ranking_policy=ranking_policy,
            )
    return SelectionDecision(
        selected_action_id=winner.action_id,
        raw_winner_action_id=winner.action_id,
        fallback_to_p=False,
        reason="single_active_action_safe_vs_p" if runner_up is None else "winner_safe_vs_p_and_runner_up",
        active_representative_count=len(rows),
        runner_up_action_id=runner_up,
        selected_equivalent_action_ids=winner.equivalent_action_ids,
        candidate_pool_receipt_hash=candidate_pool.receipt_hash,
        pairwise_model_hash=pairwise_model.model_hash,
        uncertainty_calibration_hash=uncertainty_calibration.calibration_hash,
        opportunity_case_receipt_hash=opportunity_receipt.receipt_hash,
        bacc_ranking_policy_hash=ranking_policy.policy_hash,
        opportunity_active_representative_ids=active_ids,
    )


__all__ = ("assemble_action_selection_evidence", "select_fail_closed_action")
