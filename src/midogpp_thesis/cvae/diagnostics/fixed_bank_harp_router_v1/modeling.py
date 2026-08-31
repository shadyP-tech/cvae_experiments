"""Nested-center HARP fitting and conservative target policy selection."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
import struct

import numpy as np

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ...routing.harp_action_model import (
    LAMBDA_GRID,
    HarpActionScore,
    HarpActionModelBank,
    HarpTargetAction,
    HarpTrainingObservation,
    fit_harp_action_model_bank,
    model_bank_collection_payload,
    score_harp_actions,
)
from ...routing.harp_portfolio import (
    HarpPolicyConfig,
    HarpPortfolioDecision,
    select_harp_physical_portfolio,
    select_harp_portfolio,
)
from ...routing.harp_protocol import canonical_hash
from ...runtime.harp_probability_menu import (
    BASE_ACTION_ID,
    TARGET_SURFACE,
    HarpPredictionMenuSeal,
    HarpRouteDecision,
    HarpRoutedVectorSeal,
)
from ...runtime.harp_probability_menu.indexed import (
    HarpValidatedTargetMenuView,
    validated_target_menu_view,
)
from ...runtime.harp_probability_menu.routing import (
    _route_harp_probability_vector_from_validated_target_view,
)


def _fit_worker(
    payload: tuple[str, tuple[HarpTrainingObservation, ...], tuple[float, ...]]
) -> HarpActionModelBank:
    outer, rows, alphas = payload
    return fit_harp_action_model_bank(rows, outer_target_id=outer, alphas=alphas)


def fit_outer_model_banks(
    observations: Sequence[HarpTrainingObservation],
    *,
    alphas: Sequence[float],
    workers: int,
) -> tuple[HarpActionModelBank, ...]:
    rows = tuple(observations)
    if not rows:
        raise ProtocolError("HARP Stage-90 model surface is empty.")
    payloads = []
    for outer in CENTERS:
        scoped = tuple(row for row in rows if row.outer_target_id == outer)
        if not scoped or any(
            outer in {row.pseudo_query_id, row.candidate_source_id}
            or row.pseudo_query_id == row.candidate_source_id
            for row in scoped
        ):
            raise ProtocolError("HARP Stage-90 outer exclusion failed before fitting.")
        payloads.append((outer, scoped, tuple(float(value) for value in alphas)))
    if workers <= 1:
        banks = tuple(_fit_worker(payload) for payload in payloads)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            banks = tuple(pool.map(_fit_worker, payloads))
    if tuple(bank.outer_target_id for bank in banks) != CENTERS:
        raise ProtocolError("HARP Stage-90 model-bank coverage drifted.")
    for bank in banks:
        for outcome in bank.models:
            for audit in outcome.nested_lodo_audit:
                if audit.heldout_donor_id in audit.training_query_ids or audit.heldout_donor_id in audit.training_source_ids:
                    raise ProtocolError("HARP Stage-90 nested LODO leaked its donor.")
            for donor, model in outcome.delete_donor_models:
                if donor in model.training_query_ids or donor in model.training_source_ids:
                    raise ProtocolError("HARP Stage-90 delete-donor model leaked its donor.")
    return banks


def policy_hash(
    banks: Sequence[HarpActionModelBank],
    policy: HarpPolicyConfig,
    *,
    menu_seal_hash: str,
) -> str:
    return canonical_hash(
        {
            "schema_version": "midogpp_harp_stage90_fitted_policy_v2",
            "model_banks": model_bank_collection_payload(tuple(banks)),
            "policy": asdict(policy),
            "prediction_menu_seal_hash": menu_seal_hash,
            "target_outcomes_used": False,
        }
    )


def _route_scored_portfolio(
    view: HarpValidatedTargetMenuView,
    *,
    outer: str,
    scoped_actions: tuple[HarpTargetAction, ...],
    scores: tuple[HarpActionScore, ...],
    policy: HarpPolicyConfig,
    fitted_policy_hash: str,
    physical_lambda_one_only: bool,
) -> tuple[tuple[HarpPortfolioDecision, ...], HarpRoutedVectorSeal]:
    portfolio = (
        select_harp_physical_portfolio(
            tuple(
                score
                for score in scores
                if score.action.lambda_value == 1.0
            ),
            config=policy,
        )
        if physical_lambda_one_only
        else select_harp_portfolio(scores, config=policy)
    )
    by_sample = {row.sample_id: row for row in portfolio}
    action_lookup = {
        (row.sample_id, row.candidate_source_id, row.lambda_value): row
        for row in scoped_actions
    }
    baseline_action = view.action_for(
        surface_kind=TARGET_SURFACE,
        outer_target_id=outer,
        query_center_id=outer,
        selected_source_id=None,
        action_id=BASE_ACTION_ID,
    )
    row_ids, case_ids = view.identities_for(baseline_action)
    route_rows: list[HarpRouteDecision] = []
    ordered_portfolio: list[HarpPortfolioDecision] = []
    for sample_id, case_id in zip(row_ids, case_ids, strict=True):
        decision = by_sample.get(sample_id)
        if decision is None or decision.case_id != case_id:
            raise ProtocolError("HARP Stage-90 portfolio lacks a sealed target row.")
        if decision.routed:
            assert decision.selected_source_id is not None
            assert decision.selected_lambda is not None
            action = action_lookup[
                (sample_id, decision.selected_source_id, decision.selected_lambda)
            ]
            source = decision.selected_source_id
            lam = decision.selected_lambda
            direction = action.direction
        else:
            if decision.output_probability_bytes != decision.baseline_probability_bytes:
                raise ProtocolError("HARP Stage-90 exact-B fallback changed bytes.")
            source, lam, direction = None, 0.0, "NO_DISAGREEMENT"
        route_rows.append(
            HarpRouteDecision(
                surface_kind=TARGET_SURFACE,
                outer_target_id=outer,
                query_center_id=outer,
                row_id=sample_id,
                case_id=case_id,
                eligible=decision.routed,
                selected_source_id=source,
                lambda_value=lam,
                direction=direction,
                decision_reason=decision.reason,
                policy_hash=fitted_policy_hash,
                prediction_menu_seal_hash=view.seal_hash,
            )
        )
        ordered_portfolio.append(decision)
    vector = _route_harp_probability_vector_from_validated_target_view(
        view, route_rows
    )
    if physical_lambda_one_only and any(
        row.eligible and row.lambda_value != 1.0 for row in vector.decisions
    ):
        raise ProtocolError("HARP Stage-90 physical ablation escaped lambda=1.")
    for ordinal, decision in enumerate(ordered_portfolio):
        observed = struct.pack("<d", float(vector.routed_probabilities[ordinal]))
        if observed != decision.output_probability_bytes:
            raise ProtocolError("HARP Stage-90 portfolio/vector probability bytes drifted.")
    vector.assert_valid()
    return tuple(ordered_portfolio), vector


def _select_and_route_modes(
    menu: HarpPredictionMenuSeal,
    banks: Sequence[HarpActionModelBank],
    target_actions: Sequence[HarpTargetAction],
    *,
    policy: HarpPolicyConfig,
    fitted_policy_hash: str,
    modes: tuple[bool, ...],
) -> dict[
    bool, tuple[tuple[HarpPortfolioDecision, ...], tuple[HarpRoutedVectorSeal, ...]]
]:
    if not modes or len(set(modes)) != len(modes):
        raise ProtocolError("HARP Stage-90 routing modes must be unique and nonempty.")
    bank_by_outer = {bank.outer_target_id: bank for bank in banks}
    actions_by_outer: dict[str, list[HarpTargetAction]] = defaultdict(list)
    for action in target_actions:
        actions_by_outer[action.outer_target_id].append(action)
    scoped_by_outer: dict[str, tuple[HarpTargetAction, ...]] = {}
    # Preserve the original fail-fast ordering: reject an incomplete action
    # universe before probing or validating the physical probability menu.
    for outer in CENTERS:
        scoped_actions = tuple(actions_by_outer[outer])
        expected_sources = set(CENTERS) - {outer}
        actions_by_sample: dict[tuple[str, str], list[HarpTargetAction]] = defaultdict(list)
        for action in scoped_actions:
            actions_by_sample[(action.case_id, action.sample_id)].append(action)
        if not actions_by_sample or any(
            {action.candidate_source_id for action in sample_actions}
            != expected_sources
            or len(sample_actions) != len(expected_sources) * len(LAMBDA_GRID)
            for sample_actions in actions_by_sample.values()
        ):
            raise ProtocolError(
                "HARP Stage-90 target action menu lacks the complete legal candidate universe."
            )
        scoped_by_outer[outer] = scoped_actions

    view = validated_target_menu_view(menu)
    all_decisions: dict[bool, list[HarpPortfolioDecision]] = {
        mode: [] for mode in modes
    }
    vectors: dict[bool, list[HarpRoutedVectorSeal]] = {mode: [] for mode in modes}
    for outer in CENTERS:
        scoped_actions = scoped_by_outer[outer]
        scores = score_harp_actions(bank_by_outer[outer], scoped_actions)
        for mode in modes:
            scoped_decisions, vector = _route_scored_portfolio(
                view,
                outer=outer,
                scoped_actions=scoped_actions,
                scores=scores,
                policy=policy,
                fitted_policy_hash=fitted_policy_hash,
                physical_lambda_one_only=mode,
            )
            all_decisions[mode].extend(scoped_decisions)
            vectors[mode].append(vector)

    output: dict[
        bool,
        tuple[tuple[HarpPortfolioDecision, ...], tuple[HarpRoutedVectorSeal, ...]],
    ] = {}
    for mode in modes:
        decisions = tuple(sorted(all_decisions[mode], key=lambda row: row.row_key))
        if any(
            not row.routed
            and row.output_probability_bytes != row.baseline_probability_bytes
            for row in decisions
        ):
            raise ProtocolError("HARP Stage-90 fallback byte identity failed.")
        output[mode] = decisions, tuple(vectors[mode])

    # This is the second and final full validation for the whole phase.  The
    # predictive and physical projections share scores, but validator A and B
    # each call this function independently and build their own view/scores.
    view.assert_fully_valid()
    return output


def select_and_route(
    menu: HarpPredictionMenuSeal,
    banks: Sequence[HarpActionModelBank],
    target_actions: Sequence[HarpTargetAction],
    *,
    policy: HarpPolicyConfig,
    fitted_policy_hash: str,
    physical_lambda_one_only: bool = False,
) -> tuple[tuple[HarpPortfolioDecision, ...], tuple[HarpRoutedVectorSeal, ...]]:
    """Select one route role with a fresh validated target-menu view."""

    return _select_and_route_modes(
        menu,
        banks,
        target_actions,
        policy=policy,
        fitted_policy_hash=fitted_policy_hash,
        modes=(physical_lambda_one_only,),
    )[physical_lambda_one_only]


def select_and_route_pair(
    menu: HarpPredictionMenuSeal,
    banks: Sequence[HarpActionModelBank],
    target_actions: Sequence[HarpTargetAction],
    *,
    policy: HarpPolicyConfig,
    fitted_policy_hash: str,
) -> tuple[
    tuple[HarpPortfolioDecision, ...],
    tuple[HarpRoutedVectorSeal, ...],
    tuple[HarpPortfolioDecision, ...],
    tuple[HarpRoutedVectorSeal, ...],
]:
    """Score once, then derive predictive and physical policies independently."""

    output = _select_and_route_modes(
        menu,
        banks,
        target_actions,
        policy=policy,
        fitted_policy_hash=fitted_policy_hash,
        modes=(False, True),
    )
    decisions, vectors = output[False]
    physical_decisions, physical_vectors = output[True]
    return decisions, vectors, physical_decisions, physical_vectors


__all__ = (
    "fit_outer_model_banks",
    "policy_hash",
    "select_and_route",
    "select_and_route_pair",
)
