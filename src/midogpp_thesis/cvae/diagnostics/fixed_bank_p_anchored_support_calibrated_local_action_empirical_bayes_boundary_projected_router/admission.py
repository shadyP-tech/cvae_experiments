"""Pair-aware, policy-level pseudo-case admission for SCALE-BP.

Admission replays the complete frozen method menu for every ``(H,J,d)``
context.  Candidate ranking is measured inside each pseudo case and only then
aggregated case-to-center, preventing large cases or centers from dominating.
"""

from __future__ import annotations

from .admission_contracts import (
    AdmissionResult,
    AdmissionThresholds,
    AllOuterAdmissionResult,
    _issue_all_outer_admission_result,
)
from .admission_metrics import equal_center_mean, equal_center_metric_mean, spearman
from .admission_replay import (
    group_actions,
    group_policies,
    validate_context_rectangle,
)
from .controls import (
    LEGACY_SAME_RUN,
    METHOD_IDS,
    NEGATIVE_CONTROL_IDS,
    SCALE_BP_PRIMARY,
)
from .evidence.bundle import AllOuterReplayEvidenceBundle, PseudoReplayEvidenceBundle
from .pseudo_evidence import PseudoRouteActionEvidence, PseudoRoutePolicyEvidence
from .protocol import ProtocolError
from .replay.contracts import method_menu_hash
from .replay_scope import PseudoReplayScope


def evaluate_pseudo_admission(
    evidence_bundle: AllOuterReplayEvidenceBundle,
) -> AllOuterAdmissionResult:
    """Admit only when every exact outer-H replay passes independently."""

    if not isinstance(evidence_bundle, AllOuterReplayEvidenceBundle):
        raise ProtocolError(
            "SCALE-BP pseudo admission requires sealed all-outer replay evidence."
        )
    outer_results = tuple(
        _evaluate_outer_pseudo_admission(bundle)
        for bundle in evidence_bundle.outer_bundles
    )
    return _issue_all_outer_admission_result(
        replay_bundle_hash=evidence_bundle.bundle_hash,
        replay_input_root=evidence_bundle.input_root,
        action_evidence_root=evidence_bundle.action_evidence_root,
        policy_evidence_root=evidence_bundle.policy_evidence_root,
        oracle_root=evidence_bundle.oracle_root,
        method_menu_hash=method_menu_hash(),
        outer_results=outer_results,
    )


def _evaluate_outer_pseudo_admission(
    evidence_bundle: PseudoReplayEvidenceBundle,
) -> AdmissionResult:
    """Aggregate one factory-sealed exact all-method replay bundle for H."""

    if not isinstance(evidence_bundle, PseudoReplayEvidenceBundle):
        raise ProtocolError("SCALE-BP pseudo admission requires sealed replay evidence.")
    replay_inventory = evidence_bundle.replay_inventory
    actions = evidence_bundle.action_evidence
    policies = evidence_bundle.policy_evidence
    thresholds = AdmissionThresholds()
    if (
        not actions
        or not policies
    ):
        raise ProtocolError("SCALE-BP pseudo admission rectangle is empty.")
    action_groups = group_actions(actions)
    policy_groups = group_policies(policies)
    contexts = {key[:4] for key in action_groups} | {key[:4] for key in policy_groups}
    if {key[:4] for key in action_groups} != contexts or {
        key[:4] for key in policy_groups
    } != contexts:
        raise ProtocolError("SCALE-BP action/policy context rectangle is incomplete.")
    expected_contexts = {
        (replay_inventory.outer_center, center, case, scope_hash)
        for center, case, scope_hash in replay_inventory.scope_bindings
    }
    if contexts != expected_contexts or any(
        row.scope.case_inventory.inventory_hash
        != replay_inventory.case_inventory.inventory_hash
        for row in (*actions, *policies)
    ):
        raise ProtocolError("SCALE-BP expected pseudo replay universe is incomplete.")

    by_context: dict[
        tuple[str, str, str, str],
        tuple[
            dict[str, tuple[PseudoRouteActionEvidence, ...]],
            dict[str, PseudoRoutePolicyEvidence],
        ],
    ] = {}
    for context in sorted(contexts):
        action_methods = {
            key[4] for key in action_groups if key[:4] == context
        }
        policy_methods = {
            key[4] for key in policy_groups if key[:4] == context
        }
        if action_methods != set(METHOD_IDS) or policy_methods != set(METHOD_IDS):
            raise ProtocolError("SCALE-BP frozen method menu is incomplete.")
        method_actions = {
            method: action_groups[(*context, method)] for method in METHOD_IDS
        }
        method_policies = {
            method: policy_groups[(*context, method)] for method in METHOD_IDS
        }
        validate_context_rectangle(method_actions, method_policies)
        by_context[context] = method_actions, method_policies

    opportunity_contexts: set[tuple[str, str, str, str]] = set()
    represented_centers: set[str] = set()
    policy_metrics: dict[str, list[tuple[float, float, float]]] = {}
    case_spearman: dict[str, list[float]] = {}
    case_top1: dict[str, list[float]] = {}
    case_gap: dict[str, list[float]] = {}
    legacy_gap: dict[str, list[float]] = {}
    selected_case_count = 0
    harmful_count = 0

    for context, (method_actions, method_policies) in by_context.items():
        pseudo_center = context[1]
        primary_actions = tuple(
            row for row in method_actions[SCALE_BP_PRIMARY] if row.opportunity
        )
        primary_policy = method_policies[SCALE_BP_PRIMARY]
        legacy_policy = method_policies[LEGACY_SAME_RUN]
        policy_metrics.setdefault(pseudo_center, []).append(
            (
                primary_policy.realized_bacc_gain,
                primary_policy.realized_brier_loss_delta,
                primary_policy.realized_log_loss_delta,
            )
        )
        if primary_policy.selected_action_ids:
            selected_case_count += 1
            if (
                primary_policy.realized_bacc_gain < -thresholds.tie_tolerance
                or primary_policy.realized_brier_loss_delta > thresholds.tie_tolerance
                or primary_policy.realized_log_loss_delta > thresholds.tie_tolerance
            ):
                harmful_count += 1
        if not primary_actions:
            continue
        opportunity_contexts.add(context)
        represented_centers.add(pseudo_center)
        rank = spearman(
            (row.predicted_bacc_gain for row in primary_actions),
            (row.realized_bacc_gain for row in primary_actions),
        )
        if rank is not None:
            case_spearman.setdefault(pseudo_center, []).append(rank)
        predicted_best = max(
            primary_actions, key=lambda row: (row.predicted_bacc_gain, row.action_id)
        )
        realized_best = max(
            primary_actions, key=lambda row: (row.realized_bacc_gain, row.action_id)
        )
        case_top1.setdefault(pseudo_center, []).append(
            float(predicted_best.action_id == realized_best.action_id)
        )
        oracle = primary_policy.oracle_bacc_gain
        if oracle <= thresholds.tie_tolerance:
            continue
        case_gap.setdefault(pseudo_center, []).append(
            (oracle - primary_policy.realized_bacc_gain) / oracle
        )
        legacy_gap.setdefault(pseudo_center, []).append(
            (oracle - legacy_policy.realized_bacc_gain) / oracle
        )

    bacc, brier, log = equal_center_metric_mean(policy_metrics)
    rank_value = equal_center_mean(case_spearman) if case_spearman else None
    top1_value = equal_center_mean(case_top1) if case_top1 else None
    gap_value = equal_center_mean(case_gap) if case_gap else None
    legacy_gap_value = equal_center_mean(legacy_gap) if legacy_gap else None
    control_counts = tuple(
        sorted(
            (
                method,
                sum(
                    len(method_policies[method].selected_action_ids)
                    for _context, (_actions, method_policies) in by_context.items()
                ),
            )
            for method in NEGATIVE_CONTROL_IDS
        )
    )

    reasons: list[str] = []
    if len(opportunity_contexts) < thresholds.minimum_opportunity_cases:
        reasons.append("INSUFFICIENT_OPPORTUNITY_CASES")
    if len(represented_centers) < thresholds.minimum_represented_centers:
        reasons.append("INSUFFICIENT_REPRESENTED_CENTERS")
    if rank_value is None or rank_value <= thresholds.minimum_spearman:
        reasons.append("NONPOSITIVE_OR_UNDEFINED_WITHIN_CASE_SPEARMAN")
    if top1_value is None:
        reasons.append("UNDEFINED_TOP1_ACTION_AGREEMENT")
    if bacc <= thresholds.tie_tolerance:
        reasons.append("NONPOSITIVE_EQUAL_CENTER_BACC")
    if brier > thresholds.tie_tolerance:
        reasons.append("BRIER_SAFETY_FAILED")
    if log > thresholds.tie_tolerance:
        reasons.append("LOG_LOSS_SAFETY_FAILED")
    if gap_value is None or gap_value > thresholds.maximum_normalized_oracle_gap:
        reasons.append("ORACLE_GAP_FAILED_OR_UNDEFINED")
    if (
        legacy_gap_value is None
        or gap_value is None
        or gap_value > legacy_gap_value + thresholds.tie_tolerance
    ):
        reasons.append("LEGACY_NONINFERIORITY_FAILED_OR_UNDEFINED")
    if harmful_count > thresholds.maximum_harmful_selected_policy_count:
        reasons.append("HARM_BUDGET_EXCEEDED")
    if any(count for _method, count in control_counts):
        reasons.append("NEGATIVE_CONTROL_ROUTED")

    return AdmissionResult(
        outer_center=replay_inventory.outer_center,
        replay_inventory_hash=replay_inventory.receipt_hash,
        replay_bundle_hash=evidence_bundle.bundle_hash,
        replay_input_root=evidence_bundle.input_root,
        action_evidence_root=evidence_bundle.action_evidence_root,
        policy_evidence_root=evidence_bundle.policy_evidence_root,
        oracle_root=evidence_bundle.oracle_root,
        method_menu_hash=method_menu_hash(),
        admitted=not reasons,
        reasons=tuple(reasons),
        opportunity_case_count=len(opportunity_contexts),
        represented_center_count=len(represented_centers),
        selected_case_count=selected_case_count,
        equal_center_bacc_gain=bacc,
        equal_center_brier_loss_delta=brier,
        equal_center_log_loss_delta=log,
        opportunity_spearman=rank_value,
        spearman_case_count=sum(len(rows) for rows in case_spearman.values()),
        top1_action_agreement=top1_value,
        normalized_oracle_gap=gap_value,
        legacy_normalized_oracle_gap=legacy_gap_value,
        harmful_selected_policy_count=harmful_count,
        control_route_counts=control_counts,
        context_count=len(contexts),
        policy_count=len(policies),
    )


# Public compatibility names describe the stricter policy-level contracts.
PseudoActionEvidence = PseudoRouteActionEvidence
PseudoExclusionWitness = PseudoReplayScope


__all__ = (
    "AdmissionResult",
    "AdmissionThresholds",
    "AllOuterAdmissionResult",
    "PseudoActionEvidence",
    "PseudoExclusionWitness",
    "PseudoRouteActionEvidence",
    "PseudoRoutePolicyEvidence",
    "evaluate_pseudo_admission",
)
