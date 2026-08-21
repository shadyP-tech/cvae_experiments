"""Terminal-only descriptive diagnostics for sealed PCSI-PARC policies.

These diagnostics consume target truth only after the aggregate seal. They can
explain a frozen decision, but cannot change selection, authorization, or any
same-surface route.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    DIRECTION_IDS,
    PORTFOLIO_METHOD_ID,
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    UNPROJECTED_PARC_METHOD_ID,
)
from .contracts import BinaryLabel
from .engine import PreterminalResult
from .policy_regret import score_actual_center_policy
from .projection_lattice import THRESHOLD, as_binary32
from .reports import validate_transport_lineage_evidence


_PARC_POLICY_IDS = (PRIMARY_METHOD_ID, UNPROJECTED_PARC_METHOD_ID)
_ENDPOINT_NAMES = (
    "bacc_delta",
    "negative_brier_delta",
    "negative_log_loss_delta",
)


@dataclass(frozen=True)
class TerminalDiagnostics:
    projected_action_rows: tuple[Mapping[str, object], ...]
    policy_regret_rows: tuple[Mapping[str, object], ...]
    transport_rows: tuple[Mapping[str, object], ...]
    selected_case_rows: tuple[Mapping[str, object], ...]
    center_rows: tuple[Mapping[str, object], ...]
    action_frequency_rows: tuple[Mapping[str, object], ...]
    summary: Mapping[str, object]


def policy_regret_terminal_diagnostics(
    preterminal: PreterminalResult,
    terminal_labels: Sequence[BinaryLabel],
) -> TerminalDiagnostics:
    """Compute all six frozen information diagnostics descriptively."""

    labels = tuple(terminal_labels)
    label_map = {row.key: row.value for row in labels}
    if len(label_map) != len(labels):
        raise ProtocolError("PCSI-PARC terminal diagnostic labels are duplicated.")
    labels_by_center = {
        center: tuple(row for row in labels if row.center == center)
        for center in CENTERS
    }
    denominators = {
        center: (
            sum(row.value == 1 for row in labels_by_center[center]),
            sum(row.value == 0 for row in labels_by_center[center]),
        )
        for center in CENTERS
    }
    if any(min(values) <= 0 for values in denominators.values()):
        raise ProtocolError("PCSI-PARC terminal center lacks a binary class.")
    transport_lineage = validate_transport_lineage_evidence(preterminal)

    projected_rows = _projected_action_rows(
        preterminal, label_map=label_map, denominators=denominators
    )
    policy_rows = _policy_regret_rows(preterminal, labels_by_center)
    transport_rows = _transport_rows(
        preterminal, transport_lineage=transport_lineage
    )
    selected_case_rows = _selected_case_rows(
        preterminal, label_map=label_map, denominators=denominators
    )
    center_rows = _center_rows(projected_rows, policy_rows, selected_case_rows)
    action_frequency_rows = _action_frequency_rows(preterminal)
    summary = _summary(
        preterminal,
        projected_rows,
        policy_rows,
        transport_rows,
        selected_case_rows,
        action_frequency_rows,
        transport_lineage,
    )
    return TerminalDiagnostics(
        projected_rows,
        policy_rows,
        transport_rows,
        selected_case_rows,
        center_rows,
        action_frequency_rows,
        summary,
    )


def _projected_action_rows(
    preterminal: PreterminalResult,
    *,
    label_map: Mapping[tuple[str, str, str], int],
    denominators: Mapping[str, tuple[int, int]],
) -> tuple[Mapping[str, object], ...]:
    geometry = preterminal.donor_runtime.geometry_results[PROJECTION_GEOMETRY_ID]
    output: list[Mapping[str, object]] = []
    for center in CENTERS:
        action_by_hash = {
            row.action_hash: row for row in geometry.target_actions_by_center[center]
        }
        influences = {
            row.descriptor_hash: row
            for row in preterminal.policy_runtime.target_influences_by_policy_center[
                (PRIMARY_METHOD_ID, center)
            ]
        }
        selected = {
            decision.selected_action_hash
            for case in preterminal.policy_runtime.target_candidate_policies[
                (PRIMARY_METHOD_ID, center)
            ].cases
            for decision in case.decisions
            if decision.selected_action_hash is not None
        }
        for descriptor in geometry.target_descriptors_by_center[center]:
            action = action_by_hash[descriptor.action_hash]
            influence = influences[descriptor.descriptor_hash]
            endpoint = next(
                row
                for row in preterminal.predictions_by_center[center]
                if row.case_id == descriptor.case_id
            )
            actual = _case_bacc_contribution(
                center,
                endpoint.case_id,
                endpoint.sample_ids,
                endpoint.probabilities[PORTFOLIO_METHOD_ID],
                action.probabilities,
                label_map,
                denominators[center],
            )
            output.append(
                MappingProxyType(
                    {
                        "target_center": center,
                        "case_id": descriptor.case_id,
                        "geometry_id": descriptor.geometry_id,
                        "direction": descriptor.direction,
                        "representative": descriptor.representative,
                        "equivalence_members": list(descriptor.equivalence_members),
                        "equivalence_multiplicity": len(
                            descriptor.equivalence_members
                        ),
                        "crossing_count": descriptor.crossing_count,
                        "structural_zero": descriptor.crossing_count == 0,
                        "selected_by_primary": descriptor.action_hash in selected,
                        "target_influence_score": influence.target_score,
                        "realized_projected_action_bacc_contribution": actual,
                        "action_hash": descriptor.action_hash,
                        "descriptor_hash": descriptor.descriptor_hash,
                        "influence_hash": influence.influence_hash,
                        "raw_label_persisted": False,
                        "descriptive_only": True,
                    }
                )
            )
    return tuple(output)


def _policy_regret_rows(
    preterminal: PreterminalResult,
    labels_by_center: Mapping[str, Sequence[BinaryLabel]],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for policy_id in _PARC_POLICY_IDS:
        for center in CENTERS:
            key = policy_id, center
            candidate = preterminal.policy_runtime.target_candidate_policies[key]
            authorization = preterminal.policy_runtime.authorizations[key]
            replays = tuple(
                preterminal.policy_runtime.replays[
                    (authorization.geometry_id, center, pseudo)
                ]
                for pseudo in CENTERS
                if pseudo != center
            )
            replay_residuals = np.asarray(
                [row.residual_vector for row in replays], dtype=np.float64
            )
            expected_radius = tuple(
                float(max(0.0, np.max(replay_residuals[:, index])))
                for index in range(3)
            )
            if (
                len(replays) != len(CENTERS) - 1
                or tuple(row.replay_hash for row in replays)
                != authorization.replay_hashes
                or expected_radius != authorization.regret_radius_vector
                or not np.all(
                    np.isfinite(
                        np.asarray(
                            (
                                authorization.predicted_favorable_endpoint_vector,
                                authorization.regret_radius_vector,
                                authorization.diagnostic_lower_vector,
                            ),
                            dtype=np.float64,
                        )
                    )
                )
            ):
                raise ProtocolError(
                    "PCSI-PARC terminal policy Ldiag evidence drifted."
                )
            actual, evaluation_identity_hash = score_actual_center_policy(
                candidate,
                preterminal.predictions_by_center[center],
                labels_by_center[center],
            )
            predicted = authorization.predicted_favorable_endpoint_vector
            rows.append(
                MappingProxyType(
                    {
                        "target_center": center,
                        "policy_id": policy_id,
                        "geometry_id": authorization.geometry_id,
                        "endpoint_ids": list(_ENDPOINT_NAMES),
                        "predicted_favorable_endpoint_vector": list(predicted),
                        "regret_radius_vector": list(
                            authorization.regret_radius_vector
                        ),
                        "diagnostic_lower_vector": list(
                            authorization.diagnostic_lower_vector
                        ),
                        "diagnostic_lower_bacc": (
                            authorization.diagnostic_lower_vector[0]
                        ),
                        "regret_residual_count": len(replays),
                        "diagnostic_lower_uses_all_eight_observed_residuals": (
                            len(replays) == len(CENTERS) - 1
                        ),
                        "diagnostic_lower_finite_before_transport_authorization": True,
                        "diagnostic_lower_sentinel_used": False,
                        "actual_favorable_endpoint_vector": list(actual),
                        "target_residual_vector": [
                            float(predicted[index] - actual[index])
                            for index in range(3)
                        ],
                        "target_transport_passed": (
                            authorization.target_transport_passed
                        ),
                        "pseudo_transport_pass_count": (
                            authorization.effective_donor_count
                        ),
                        "all_eight_pseudo_transports_passed": (
                            authorization.effective_donor_count
                            == len(CENTERS) - 1
                        ),
                        "authorized_before_terminal_labels": (
                            authorization.authorized
                        ),
                        "candidate_changed_case_count": (
                            candidate.changed_case_count
                        ),
                        "terminal_evaluation_identity_hash": (
                            evaluation_identity_hash
                        ),
                        "target_policy_seal_hash": candidate.policy_seal_hash,
                        "authorization_hash": authorization.authorization_hash,
                        "terminal_labels_used_only_for_descriptive_actual_vector": True,
                        "may_change_same_surface_policy": False,
                        "formal_inference_claimed": False,
                    }
                )
            )
    if len(rows) != 2 * len(CENTERS):
        raise ProtocolError("PCSI-PARC terminal policy diagnostic topology drifted.")
    return tuple(rows)


def _transport_rows(
    preterminal: PreterminalResult,
    *,
    transport_lineage: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    rows = tuple(
        MappingProxyType(
            {
                "outer_target_center": outer,
                "screen_role": "target" if candidate is None else "pseudo_target",
                "pseudo_target_center": candidate,
                "candidate_distance": screen.candidate_distance,
                "threshold": screen.threshold,
                "normalized_distance_to_threshold": (
                    screen.candidate_distance / screen.threshold
                    if screen.threshold > 0.0
                    else (0.0 if screen.candidate_distance == 0.0 else None)
                ),
                "passed": screen.passed,
                "screen_hash": screen.screen_hash,
                "transport_semantics": transport_lineage[
                    "transport_semantics"
                ],
                "transport_label_free_claim": False,
                "transport_source_prior_labels_used_upstream": True,
                "transport_route_local_support_labels_used_upstream": True,
                "transport_held_case_evaluation_capability_used_directly": False,
                "transport_pseudo_evaluation_capability_used_directly": False,
                "transport_terminal_evaluation_capability_used_directly": False,
                "transport_authorization_valid": False,
                "transport_identity_feedback_detected": True,
                "transport_runtime_hash": transport_lineage[
                    "transport_runtime_hash"
                ],
                "transport_lineage_evidence_hash": transport_lineage[
                    "transport_lineage_evidence_hash"
                ],
                "descriptive_only": True,
            }
        )
        for outer in CENTERS
        for candidate in (
            None,
            *(center for center in CENTERS if center != outer),
        )
        for screen in (preterminal.policy_runtime.transport_screens[(outer, candidate)],)
    )
    if len(rows) != len(CENTERS) * len(CENTERS):
        raise ProtocolError("PCSI-PARC terminal transport diagnostic drifted.")
    return rows


def _selected_case_rows(
    preterminal: PreterminalResult,
    *,
    label_map: Mapping[tuple[str, str, str], int],
    denominators: Mapping[str, tuple[int, int]],
) -> tuple[Mapping[str, object], ...]:
    final_by_key = {
        (row.target_center, row.case_id): row
        for row in preterminal.policy_runtime.final_predictions_by_policy[
            PRIMARY_METHOD_ID
        ]
    }
    endpoint_by_key = {
        (center, row.case_id): row
        for center in CENTERS
        for row in preterminal.predictions_by_center[center]
    }
    rows: list[Mapping[str, object]] = []
    for center in CENTERS:
        target = preterminal.policy_runtime.target_candidate_policies[
            (PRIMARY_METHOD_ID, center)
        ]
        authorized = preterminal.policy_runtime.authorizations[
            (PRIMARY_METHOD_ID, center)
        ].authorized
        for case in target.cases:
            endpoint = endpoint_by_key[(center, case.case_id)]
            candidate_actual = _case_bacc_contribution(
                center,
                case.case_id,
                case.sample_ids,
                endpoint.probabilities[PORTFOLIO_METHOD_ID],
                case.probabilities,
                label_map,
                denominators[center],
            )
            final = final_by_key[(center, case.case_id)]
            final_actual = _case_bacc_contribution(
                center,
                case.case_id,
                case.sample_ids,
                endpoint.probabilities[PORTFOLIO_METHOD_ID],
                final.probabilities,
                label_map,
                denominators[center],
            )
            outcome = _classify_signed_contribution(candidate_actual)
            rows.append(
                MappingProxyType(
                    {
                        "target_center": center,
                        "case_id": case.case_id,
                        "candidate_changed": case.changed,
                        "center_policy_authorized": authorized,
                        "candidate_bacc_contribution_vs_P": candidate_actual,
                        "final_bacc_contribution_vs_P": final_actual,
                        "candidate_outcome": outcome,
                        "candidate_policy_hash": case.policy_hash,
                        "final_prediction_hash": final.prediction_hash,
                        "exact_center_normalized_contribution": True,
                        "raw_label_persisted": False,
                        "descriptive_only": True,
                    }
                )
            )
    return tuple(rows)


def _center_rows(
    projected_rows: Sequence[Mapping[str, object]],
    policy_rows: Sequence[Mapping[str, object]],
    case_rows: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    output: list[Mapping[str, object]] = []
    for center in CENTERS:
        actions = tuple(row for row in projected_rows if row["target_center"] == center)
        nonstructural_actions = tuple(
            row for row in actions if not row["structural_zero"]
        )
        policies = tuple(row for row in policy_rows if row["target_center"] == center)
        cases = tuple(row for row in case_rows if row["target_center"] == center)
        changed = tuple(row for row in cases if row["candidate_changed"])
        output.append(
            MappingProxyType(
                {
                    "target_center": center,
                    "target_influence_vs_realized_projected_action_bacc_midrank_spearman": (
                        _spearman(
                            [
                                float(row["target_influence_score"])
                                for row in nonstructural_actions
                            ],
                            [
                                float(row["realized_projected_action_bacc_contribution"])
                                for row in nonstructural_actions
                            ],
                        )
                    ),
                    "target_influence_rank_scope": (
                        "nonstructural_target_projected_equivalence_classes"
                    ),
                    "projected_equivalence_class_count": len(actions),
                    "projected_collapsed_alternative_count": sum(
                        int(row["equivalence_multiplicity"]) - 1 for row in actions
                    ),
                    "selected_candidate_case_count": len(changed),
                    "selected_helpful_case_count": sum(
                        row["candidate_outcome"] == "helpful" for row in changed
                    ),
                    "selected_harmful_case_count": sum(
                        row["candidate_outcome"] == "harmful" for row in changed
                    ),
                    "case_harm_rate": (
                        sum(row["candidate_outcome"] == "harmful" for row in changed)
                        / len(changed)
                        if changed
                        else 0.0
                    ),
                    "policy_vectors": {
                        str(row["policy_id"]): {
                            "predicted": list(
                                row["predicted_favorable_endpoint_vector"]
                            ),
                            "diagnostic_lower": list(
                                row["diagnostic_lower_vector"]
                            ),
                            "actual": list(row["actual_favorable_endpoint_vector"]),
                            "regret_residual_count": int(
                                row["regret_residual_count"]
                            ),
                            "authorized": bool(
                                row["authorized_before_terminal_labels"]
                            ),
                        }
                        for row in policies
                    },
                    "descriptive_only": True,
                }
            )
        )
    return tuple(output)


def _action_frequency_rows(
    preterminal: PreterminalResult,
) -> tuple[Mapping[str, object], ...]:
    counts: Counter[tuple[str, str, str, str, bool]] = Counter()
    for policy_id in COMPOSED_POLICY_IDS:
        for center in CENTERS:
            policy = preterminal.policy_runtime.target_candidate_policies[(policy_id, center)]
            for case in policy.cases:
                for decision in case.decisions:
                    counts[
                        (
                            policy_id,
                            policy.geometry_id,
                            decision.direction,
                            decision.selected_representative,
                            decision.selected_action_hash is not None,
                        )
                    ] += 1
    rows = tuple(
        MappingProxyType(
            {
                "policy_id": policy_id,
                "geometry_id": geometry_id,
                "direction": direction,
                "selected_representative": representative,
                "selected_non_P_action": selected,
                "case_decision_count": count,
                "terminal_labels_used": False,
                "descriptive_only": True,
            }
        )
        for (policy_id, geometry_id, direction, representative, selected), count
        in sorted(counts.items())
    )
    expected = len(COMPOSED_POLICY_IDS) * sum(
        len(preterminal.predictions_by_center[center]) for center in CENTERS
    ) * len(DIRECTION_IDS)
    if sum(int(row["case_decision_count"]) for row in rows) != expected:
        raise ProtocolError("PCSI-PARC action frequency surface is incomplete.")
    return rows


def _summary(
    preterminal: PreterminalResult,
    projected_rows: Sequence[Mapping[str, object]],
    policy_rows: Sequence[Mapping[str, object]],
    transport_rows: Sequence[Mapping[str, object]],
    selected_case_rows: Sequence[Mapping[str, object]],
    action_frequency_rows: Sequence[Mapping[str, object]],
    transport_lineage: Mapping[str, object],
) -> Mapping[str, object]:
    changed = tuple(row for row in selected_case_rows if row["candidate_changed"])
    harmful = tuple(row for row in changed if row["candidate_outcome"] == "harmful")
    helpful = tuple(row for row in changed if row["candidate_outcome"] == "helpful")
    worst = (
        min(
            changed,
            key=lambda row: (
                float(row["candidate_bacc_contribution_vs_P"]),
                str(row["target_center"]),
                str(row["case_id"]),
            ),
        )
        if changed
        else None
    )
    whole_rank: dict[str, dict[str, float]] = {}
    corrected_bacc_rank: dict[str, float] = {}
    for policy_id in _PARC_POLICY_IDS:
        rows = tuple(row for row in policy_rows if row["policy_id"] == policy_id)
        if (
            tuple(str(row["target_center"]) for row in rows) != CENTERS
            or any(int(row["regret_residual_count"]) != len(CENTERS) - 1 for row in rows)
            or any(
                not bool(
                    row[
                        "diagnostic_lower_finite_before_transport_authorization"
                    ]
                )
                or bool(row["diagnostic_lower_sentinel_used"])
                for row in rows
            )
        ):
            raise ProtocolError("PCSI-PARC policy-rho scope drifted.")
        corrected_bacc_rank[policy_id] = _spearman(
            [float(row["diagnostic_lower_bacc"]) for row in rows],
            [
                float(row["actual_favorable_endpoint_vector"][0])
                for row in rows
            ],
        )
        whole_rank[policy_id] = {
            endpoint: _spearman(
                [
                    float(row["predicted_favorable_endpoint_vector"][index])
                    for row in rows
                ],
                [
                    float(row["actual_favorable_endpoint_vector"][index])
                    for row in rows
                ],
            )
            for index, endpoint in enumerate(_ENDPOINT_NAMES)
        }
    multiplicities = Counter(
        int(row["equivalence_multiplicity"]) for row in projected_rows
    )
    nonstructural_projected = tuple(
        row for row in projected_rows if not row["structural_zero"]
    )
    payload = {
        "schema_version": "fixed_bank_pcsi_parc_terminal_policy_diagnostic_v1",
        "status": "NEEDS_EVIDENCE",
        "diagnostic_role": "BLOCKED_IDENTITY_FEEDBACK",
        "diagnostic_ids": [
            "target_influence_vs_realized_projected_action_BACC_midrank_Spearman",
            "predicted_whole_policy_gain_vs_realized_whole_policy_gain_midrank_Spearman",
            "transport_distance_by_center",
            "projected_equivalence_class_multiplicity",
            "selected_policy_helpful_vs_harmful_counts",
            "case_harm_rate",
        ],
        "target_influence_vs_realized_projected_action_bacc_midrank_spearman": (
            _spearman(
                [
                    float(row["target_influence_score"])
                    for row in nonstructural_projected
                ],
                [
                    float(row["realized_projected_action_bacc_contribution"])
                    for row in nonstructural_projected
                ],
            )
        ),
        "target_influence_rank_scope": (
            "nonstructural_target_projected_equivalence_classes"
        ),
        "target_influence_rank_row_count": len(nonstructural_projected),
        "policy_rho_diagnostic_lower_bacc_vs_realized_candidate_policy_bacc_midrank_spearman": (
            corrected_bacc_rank[PRIMARY_METHOD_ID]
        ),
        "policy_rho_by_parc_policy": corrected_bacc_rank,
        "policy_rho_primary_policy_id": PRIMARY_METHOD_ID,
        "policy_rho_center_count": len(CENTERS),
        "policy_rho_residual_count_per_center": len(CENTERS) - 1,
        "policy_rho_uses_all_eight_observed_residuals": True,
        "policy_rho_is_finite_before_transport_authorization": True,
        "policy_rho_sentinel_used": False,
        "additional_raw_ghat_vs_realized_whole_policy_gain_coordinatewise_midrank_spearman": whole_rank,
        "transport_diagnostic_row_count": len(transport_rows),
        "transport_pass_count": sum(bool(row["passed"]) for row in transport_rows),
        **dict(transport_lineage),
        "transport_protocol_status": "BLOCKED_IDENTITY_FEEDBACK",
        "completed_canonical_run_exists": False,
        "terminal_diagnostic_bundle_valid": False,
        "projected_equivalence_class_count": len(projected_rows),
        "projected_equivalence_multiplicity_counts": {
            str(key): value for key, value in sorted(multiplicities.items())
        },
        "projected_collapsed_alternative_count": sum(
            int(row["equivalence_multiplicity"]) - 1 for row in projected_rows
        ),
        "selected_candidate_case_count": len(changed),
        "selected_policy_helpful_case_count": len(helpful),
        "selected_policy_harmful_case_count": len(harmful),
        "selected_policy_neutral_case_count": len(changed) - len(helpful) - len(harmful),
        "case_harm_rate": len(harmful) / len(changed) if changed else 0.0,
        "worst_selected_exact_center_normalized_case_contribution": (
            float(worst["candidate_bacc_contribution_vs_P"])
            if worst is not None
            else None
        ),
        "worst_selected_case_identity": (
            {
                "target_center": str(worst["target_center"]),
                "case_id": str(worst["case_id"]),
            }
            if worst is not None
            else None
        ),
        "no_changed_case_sentinel_used": worst is None,
        "target_policy_diagnostic_count": len(policy_rows),
        "whole_policy_replay_count": len(preterminal.policy_runtime.replays),
        "authorized_target_policy_count_across_two_parc_surfaces": sum(
            bool(row["authorized_before_terminal_labels"]) for row in policy_rows
        ),
        "authorized_target_policy_count_by_policy": {
            policy_id: sum(
                bool(row["authorized_before_terminal_labels"])
                for row in policy_rows
                if row["policy_id"] == policy_id
            )
            for policy_id in _PARC_POLICY_IDS
        },
        "action_frequency_row_count": len(action_frequency_rows),
        "terminal_labels_opened_after_aggregate_seal": True,
        "terminal_information_used_by_selection": False,
        "terminal_information_used_by_authorization": False,
        "terminal_information_may_change_same_surface_routes": False,
        "rows_are_not_independent_inference_units": True,
        "success_gate_defined": False,
        "nominal_inference_claimed": False,
        "routing_success_claimed": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    return MappingProxyType(payload)


def _case_bacc_contribution(
    center: str,
    case_id: str,
    sample_ids: Sequence[str],
    portfolio_probabilities: Sequence[float],
    candidate_probabilities: Sequence[float],
    labels: Mapping[tuple[str, str, str], int],
    denominators: tuple[int, int],
) -> float:
    portfolio = as_binary32(portfolio_probabilities, name="terminal diagnostic P")
    candidate = as_binary32(candidate_probabilities, name="terminal diagnostic candidate")
    y = np.asarray(
        [labels[(center, case_id, sample)] for sample in sample_ids], dtype=np.int8
    )
    p_hard = portfolio >= THRESHOLD
    hard = candidate >= THRESHOLD
    positive = y == 1
    negative = ~positive
    n_positive, n_negative = denominators
    return float(
        0.5
        * (
            np.sum(
                hard[positive].astype(np.int8) - p_hard[positive].astype(np.int8),
                dtype=np.int64,
            )
            / n_positive
            + np.sum(
                (~hard[negative]).astype(np.int8)
                - (~p_hard[negative]).astype(np.int8),
                dtype=np.int64,
            )
            / n_negative
        )
    )


def _midranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _classify_signed_contribution(value: float) -> str:
    numeric = float(value)
    if numeric > 0.0:
        return "helpful"
    if numeric < 0.0:
        return "harmful"
    return "neutral"


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    x = _midranks(left)
    y = _midranks(right)
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


sample_influence_information_diagnostics = policy_regret_terminal_diagnostics
utility_information_diagnostics = policy_regret_terminal_diagnostics


__all__ = (
    "TerminalDiagnostics",
    "policy_regret_terminal_diagnostics",
    "sample_influence_information_diagnostics",
    "utility_information_diagnostics",
)
