"""Complete deterministic primary, ablation, control, and oracle replay."""

from __future__ import annotations

import math

from ..controls import (
    CYCLIC_ACTION_IDENTITY,
    DONOR_ONLY,
    FULL_ENDPOINT_SENSITIVITY,
    LEGACY_SAME_RUN,
    LOCAL_ONLY,
    METHOD_IDS,
    P_PROTECTED,
    SCALE_BP_PRIMARY,
    SUPPORT_LABEL_PERMUTATION,
)
from ..engine import CaseRouteRequest, CaseRouteResult, build_case_route
from ..evidence.contracts import _issue_action_evidence, _issue_policy_evidence
from ..hashing import canonical_hash
from ..identity import ACTION_IDS
from ..influence.contracts import ActionMetricVector
from ..influence.metrics import realized_action_metrics
from ..protocol import ProtocolError
from .bundle import MethodReplayResult, PseudoCaseReplayResult, _issue_case_replay_result
from .composition import compose_replay_decision
from .contracts import PseudoCaseReplayRequest, method_menu_hash
from .methods import (
    ReplayActionScore,
    ablation_scores,
    protected_scores,
    relabel_scores,
    scores_from_route_result,
    select_replay_actions,
)
from .oracle import derive_action_oracle
from .permutation import build_support_label_permutation


def _metric(
    request: PseudoCaseReplayRequest,
    probabilities: object,
) -> ActionMetricVector:
    return realized_action_metrics(
        request.route_request.portfolio_probabilities,
        probabilities,
        request.terminal_labels,
        positive_denominator=request.positive_denominator,
        negative_denominator=request.negative_denominator,
        row_denominator=request.row_denominator,
    )


def _action_metrics(
    request: PseudoCaseReplayRequest,
    *,
    mode: str,
) -> dict[str, ActionMetricVector]:
    by_input = {row.action_id: row for row in request.route_request.action_inputs}
    output: dict[str, ActionMetricVector] = {}
    for action_id in ACTION_IDS:
        row = by_input.get(action_id)
        if row is None:
            output[action_id] = ActionMetricVector.zeros()
            continue
        projection = row.endpoint_projection.projection
        probabilities = (
            projection.projected_probabilities
            if mode == "boundary"
            else projection.full_endpoint_probabilities
        )
        output[action_id] = _metric(request, probabilities)
    return output


def _engine_method(
    *,
    method_id: str,
    route_request: CaseRouteRequest,
    scores: tuple[ReplayActionScore, ...],
    result: CaseRouteResult,
    mode: str,
) -> MethodReplayResult:
    predicted = select_replay_actions(
        route_request,
        scores,
        method_id=method_id,
    )
    if predicted.selected_action_ids != result.decision.selected_action_ids:
        raise ProtocolError("SCALE-BP engine/replay selection semantics diverged.")
    composition = (
        result.boundary_action
        if mode == "boundary"
        else result.full_endpoint_sensitivity
    )
    return MethodReplayResult(
        method_id=method_id,
        scores=scores,
        selected_action_ids=result.decision.selected_action_ids,
        decision_hash=result.decision.decision_hash,
        composition=composition,
    )


def replay_pseudo_case(request: PseudoCaseReplayRequest) -> PseudoCaseReplayResult:
    """Replay every frozen method and derive all evidence from terminal labels."""

    if not isinstance(request, PseudoCaseReplayRequest):
        raise ProtocolError("SCALE-BP pseudo replay request type drifted.")
    route = request.route_request
    primary_result = build_case_route(route)
    permuted_request, permutation_hash = build_support_label_permutation(route)
    permutation_result = build_case_route(permuted_request)

    primary_scores = scores_from_route_result(
        route, primary_result, method_id=SCALE_BP_PRIMARY
    )
    permutation_scores = scores_from_route_result(
        permuted_request,
        permutation_result,
        method_id=SUPPORT_LABEL_PERMUTATION,
    )
    score_menu = {
        P_PROTECTED: protected_scores(route, primary_scores),
        SCALE_BP_PRIMARY: primary_scores,
        DONOR_ONLY: ablation_scores(route, primary_result, method_id=DONOR_ONLY),
        LOCAL_ONLY: ablation_scores(route, primary_result, method_id=LOCAL_ONLY),
        LEGACY_SAME_RUN: ablation_scores(
            route, primary_result, method_id=LEGACY_SAME_RUN
        ),
        SUPPORT_LABEL_PERMUTATION: permutation_scores,
        CYCLIC_ACTION_IDENTITY: relabel_scores(
            primary_scores, method_id=CYCLIC_ACTION_IDENTITY
        ),
        FULL_ENDPOINT_SENSITIVITY: relabel_scores(
            primary_scores, method_id=FULL_ENDPOINT_SENSITIVITY
        ),
    }
    if tuple(score_menu) != METHOD_IDS:
        raise ProtocolError("SCALE-BP executor method menu drifted.")

    methods: dict[str, MethodReplayResult] = {}
    methods[SCALE_BP_PRIMARY] = _engine_method(
        method_id=SCALE_BP_PRIMARY,
        route_request=route,
        scores=primary_scores,
        result=primary_result,
        mode="boundary",
    )
    methods[SUPPORT_LABEL_PERMUTATION] = _engine_method(
        method_id=SUPPORT_LABEL_PERMUTATION,
        route_request=permuted_request,
        scores=permutation_scores,
        result=permutation_result,
        mode="boundary",
    )
    methods[FULL_ENDPOINT_SENSITIVITY] = MethodReplayResult(
        method_id=FULL_ENDPOINT_SENSITIVITY,
        scores=score_menu[FULL_ENDPOINT_SENSITIVITY],
        selected_action_ids=primary_result.decision.selected_action_ids,
        decision_hash=primary_result.decision.decision_hash,
        composition=primary_result.full_endpoint_sensitivity,
    )
    for method_id in (
        P_PROTECTED,
        DONOR_ONLY,
        LOCAL_ONLY,
        LEGACY_SAME_RUN,
        CYCLIC_ACTION_IDENTITY,
    ):
        scores = score_menu[method_id]
        decision = select_replay_actions(route, scores, method_id=method_id)
        methods[method_id] = MethodReplayResult(
            method_id=method_id,
            scores=scores,
            selected_action_ids=decision.selected_action_ids,
            decision_hash=decision.decision_hash,
            composition=compose_replay_decision(
                route, scores, decision, mode="boundary"
            ),
        )
    ordered_methods = tuple(methods[method_id] for method_id in METHOD_IDS)

    boundary_metrics = _action_metrics(request, mode="boundary")
    full_metrics = _action_metrics(request, mode="full_endpoint")
    oracle = derive_action_oracle(
        scope_hash=request.scope.scope_hash,
        geometry_scores=primary_scores,
        realized_metrics=boundary_metrics,
    )

    action_rows = []
    policy_rows = []
    by_input = {row.action_id: row for row in route.action_inputs}
    for method in ordered_methods:
        realized = (
            full_metrics
            if method.method_id == FULL_ENDPOINT_SENSITIVITY
            else boundary_metrics
        )
        issued_for_method = []
        for score in method.scores:
            action_input = by_input.get(score.action_id)
            descriptor_hash = (
                action_input.descriptor.descriptor_hash
                if action_input is not None
                else canonical_hash(
                    {
                        "schema_version": "scale_bp_absent_action_descriptor_v1",
                        "route_request_hash": route.request_hash,
                        "action_id": score.action_id,
                    }
                )
            )
            metric = realized[score.action_id]
            row = _issue_action_evidence(
                scope=request.scope,
                method_id=method.method_id,
                action_id=score.action_id,
                opportunity=score.opportunity,
                selected=score.action_id in method.selected_action_ids,
                crossing_indices=score.crossing_indices,
                predicted_bacc_gain=score.bacc_lower,
                realized_bacc_gain=metric.bacc_gain,
                realized_brier_loss_delta=metric.brier_loss_delta,
                realized_log_loss_delta=metric.log_loss_delta,
                descriptor_hash=descriptor_hash,
                candidate_hash=score.score_hash,
                replay_request_hash=request.request_hash,
                terminal_label_hash=request.terminal_label_hash,
                method_menu_hash=method_menu_hash(),
                oracle_hash=oracle.oracle_hash,
            )
            action_rows.append(row)
            issued_for_method.append(row)
        policy_metric = _metric(request, method.composition.composed_probabilities)
        selected_sum = ActionMetricVector.zeros()
        for action_id in method.selected_action_ids:
            selected_sum = selected_sum.plus(realized[action_id])
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(
                policy_metric.as_tuple(), selected_sum.as_tuple(), strict=True
            )
        ):
            raise ProtocolError("SCALE-BP replay policy metric is not additive.")
        policy_rows.append(
            _issue_policy_evidence(
                scope=request.scope,
                method_id=method.method_id,
                selected_action_ids=method.selected_action_ids,
                realized_bacc_gain=policy_metric.bacc_gain,
                realized_brier_loss_delta=policy_metric.brier_loss_delta,
                realized_log_loss_delta=policy_metric.log_loss_delta,
                oracle_bacc_gain=oracle.metrics.bacc_gain,
                decision_hash=method.decision_hash,
                composition_hash=method.composition.composition_hash,
                action_evidence_hashes=tuple(
                    sorted(row.evidence_hash for row in issued_for_method)
                ),
                replay_request_hash=request.request_hash,
                terminal_label_hash=request.terminal_label_hash,
                method_menu_hash=method_menu_hash(),
                oracle_hash=oracle.oracle_hash,
            )
        )
    return _issue_case_replay_result(
        scope=request.scope,
        replay_request_hash=request.request_hash,
        terminal_label_hash=request.terminal_label_hash,
        center_population_label_hash=(
            request.terminal_label_receipt.center_population_label_hash
        ),
        terminal_denominators=(
            request.positive_denominator,
            request.negative_denominator,
            request.row_denominator,
        ),
        method_results=ordered_methods,
        action_evidence=tuple(action_rows),
        policy_evidence=tuple(policy_rows),
        oracle=oracle,
        permutation_hash=permutation_hash,
    )


__all__ = ("replay_pseudo_case",)
