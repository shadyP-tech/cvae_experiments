"""Terminal-only scoring and descriptive PCSI-PARC diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    CLAIM_ROLE,
    COMPOSED_POLICY_IDS,
    ENDPOINT_METHOD_IDS,
    PORTFOLIO_METHOD_ID,
    PRIMARY_METHOD_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .engine import PreterminalResult
from .hashing import canonical_hash
from .information_diagnostics import policy_regret_terminal_diagnostics
from .terminal_metrics import score_methods


@dataclass(frozen=True)
class TerminalEvaluation:
    method_metrics: tuple[Mapping[str, object], ...]
    center_contrasts: tuple[Mapping[str, object], ...]
    case_oracle_rows: tuple[Mapping[str, object], ...]
    projected_action_rows: tuple[Mapping[str, object], ...]
    policy_regret_rows: tuple[Mapping[str, object], ...]
    transport_diagnostic_rows: tuple[Mapping[str, object], ...]
    selected_case_rows: tuple[Mapping[str, object], ...]
    policy_regret_center_rows: tuple[Mapping[str, object], ...]
    action_frequency_rows: tuple[Mapping[str, object], ...]
    terminal_diagnostic: Mapping[str, object]
    selection_control: Mapping[str, object]
    diagnostic_summary: Mapping[str, object]
    terminal_seal: Mapping[str, object]
    capability_report: Mapping[str, object]
    evaluation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_hash",
            canonical_hash(self.to_payload(include_hash=False)),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "fixed_bank_pcsi_parc_terminal_evaluation_v1",
            "method_metrics": [dict(row) for row in self.method_metrics],
            "center_contrasts": [dict(row) for row in self.center_contrasts],
            "case_oracle_rows": [dict(row) for row in self.case_oracle_rows],
            "projected_action_rows": [
                dict(row) for row in self.projected_action_rows
            ],
            "policy_regret_rows": [dict(row) for row in self.policy_regret_rows],
            "transport_diagnostic_rows": [
                dict(row) for row in self.transport_diagnostic_rows
            ],
            "selected_case_rows": [dict(row) for row in self.selected_case_rows],
            "policy_regret_center_rows": [
                dict(row) for row in self.policy_regret_center_rows
            ],
            "action_frequency_rows": [dict(row) for row in self.action_frequency_rows],
            "terminal_diagnostic": dict(self.terminal_diagnostic),
            "selection_control": dict(self.selection_control),
            "diagnostic_summary": dict(self.diagnostic_summary),
            "terminal_seal": dict(self.terminal_seal),
            "capability_report": dict(self.capability_report),
        }
        return (
            {**payload, "evaluation_hash": self.evaluation_hash}
            if include_hash
            else payload
        )


def evaluate_terminal(preterminal: PreterminalResult) -> TerminalEvaluation:
    """Open labels after the aggregate seal and score the frozen method menu."""

    terminal_labels = preterminal.label_firewall.open_terminal_labels()
    label_map = {row.key: row.value for row in terminal_labels}
    expected = {
        (center, case_id, sample_id)
        for center in CENTERS
        for sample_id, case_id in zip(
            preterminal.surface.centers[center].sample_ids,
            preterminal.surface.centers[center].case_ids,
            strict=True,
        )
    }
    if set(label_map) != expected or len(label_map) != len(terminal_labels):
        raise ProtocolError(
            "PCSI-PARC terminal label capability does not match its surface."
        )

    probabilities, sample_ids = _probability_maps(preterminal)
    method_order = (*ENDPOINT_METHOD_IDS, *COMPOSED_POLICY_IDS)
    method_rows, center_rows, oracle_rows, center_metrics = score_methods(
        probabilities,
        sample_ids,
        label_map,
        method_order=method_order,
    )
    metrics_by_method = {str(row["method_id"]): row for row in method_rows}
    primary = metrics_by_method[PRIMARY_METHOD_ID]
    diagnostics = policy_regret_terminal_diagnostics(preterminal, terminal_labels)
    diagnostic = diagnostics.summary
    selection_control = MappingProxyType(
        _selection_aware_center_sign_flip(center_metrics)
    )
    proper_losses_nonincreasing = bool(
        float(primary["mean_center_brier_delta_vs_P"]) <= 0.0
        and float(primary["mean_center_log_loss_delta_vs_P"]) <= 0.0
    )
    summary = MappingProxyType(
        {
            "schema_version": "fixed_bank_pcsi_parc_diagnostic_summary_v1",
            "primary_method_id": PRIMARY_METHOD_ID,
            "primary_equal_center_bacc": primary["equal_center_bacc"],
            "primary_mean_center_bacc_delta_vs_P": primary[
                "mean_center_bacc_delta_vs_P"
            ],
            "primary_minimum_center_bacc_delta_vs_P": primary[
                "minimum_center_bacc_delta_vs_P"
            ],
            "primary_mean_center_brier_delta_vs_P": primary[
                "mean_center_brier_delta_vs_P"
            ],
            "primary_mean_center_log_loss_delta_vs_P": primary[
                "mean_center_log_loss_delta_vs_P"
            ],
            "primary_mean_proper_losses_nonincreasing_observed": (
                proper_losses_nonincreasing
            ),
            "primary_route_count": primary["route_count"],
            "primary_threshold_switch_count": primary["threshold_switch_count"],
            "primary_helpful_threshold_switch_count": primary[
                "helpful_threshold_switch_count"
            ],
            "primary_harmful_threshold_switch_count": primary[
                "harmful_threshold_switch_count"
            ],
            "terminal_diagnostic_status": diagnostic["status"],
            "authorized_target_policy_count_by_policy": diagnostic[
                "authorized_target_policy_count_by_policy"
            ],
            "primary_policy_rho_diagnostic_lower_bacc_vs_realized_candidate_policy_bacc_midrank_spearman": diagnostic[
                "policy_rho_diagnostic_lower_bacc_vs_realized_candidate_policy_bacc_midrank_spearman"
            ],
            "selection_aware_exact_upper_tail_fraction_descriptive": selection_control[
                "exact_upper_tail_fraction_descriptive"
            ],
            "claim_role": CLAIM_ROLE,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "terminal_descriptive_only": True,
            "transport_protocol_status": "BLOCKED_IDENTITY_FEEDBACK",
            "completed_canonical_run_exists": False,
            "terminal_diagnostic_bundle_valid": False,
            "success_gate_defined": False,
            "fresh_evidence": False,
            "routing_success_claimed": False,
            "promotion_eligible": False,
            "may_feed_another_experiment": False,
        }
    )
    terminal_label_hash = canonical_hash(
        [[*key, label_map[key]] for key in sorted(label_map)]
    )
    terminal_payload = {
        "schema_version": "fixed_bank_pcsi_parc_terminal_seal_v1",
        "aggregate_preterminal_seal_hash": preterminal.aggregate_seal[
            "aggregate_seal_hash"
        ],
        "terminal_label_identity_and_value_hash": terminal_label_hash,
        "method_metrics_hash": canonical_hash([dict(row) for row in method_rows]),
        "center_contrasts_hash": canonical_hash([dict(row) for row in center_rows]),
        "case_oracle_rows_hash": canonical_hash([dict(row) for row in oracle_rows]),
        "projected_action_rows_hash": canonical_hash(
            [dict(row) for row in diagnostics.projected_action_rows]
        ),
        "policy_regret_rows_hash": canonical_hash(
            [dict(row) for row in diagnostics.policy_regret_rows]
        ),
        "transport_diagnostic_rows_hash": canonical_hash(
            [dict(row) for row in diagnostics.transport_rows]
        ),
        "selected_case_rows_hash": canonical_hash(
            [dict(row) for row in diagnostics.selected_case_rows]
        ),
        "policy_regret_center_rows_hash": canonical_hash(
            [dict(row) for row in diagnostics.center_rows]
        ),
        "action_frequency_rows_hash": canonical_hash(
            [dict(row) for row in diagnostics.action_frequency_rows]
        ),
        "terminal_diagnostic_hash": canonical_hash(diagnostic),
        "selection_control_hash": canonical_hash(selection_control),
        "terminal_descriptive_only": True,
        "raw_labels_persisted": False,
        "fresh_evidence": False,
    }
    terminal_seal = MappingProxyType(
        {**terminal_payload, "terminal_seal_hash": canonical_hash(terminal_payload)}
    )
    return TerminalEvaluation(
        method_rows,
        center_rows,
        oracle_rows,
        diagnostics.projected_action_rows,
        diagnostics.policy_regret_rows,
        diagnostics.transport_rows,
        diagnostics.selected_case_rows,
        diagnostics.center_rows,
        diagnostics.action_frequency_rows,
        diagnostic,
        selection_control,
        summary,
        terminal_seal,
        MappingProxyType(preterminal.label_firewall.report_payload()),
    )


def _probability_maps(
    preterminal: PreterminalResult,
) -> tuple[
    Mapping[str, Mapping[str, Mapping[str, tuple[float, ...]]]],
    Mapping[str, Mapping[str, tuple[str, ...]]],
]:
    probabilities: dict[str, dict[str, dict[str, tuple[float, ...]]]] = {
        method: {center: {} for center in CENTERS}
        for method in (*ENDPOINT_METHOD_IDS, *COMPOSED_POLICY_IDS)
    }
    samples: dict[str, dict[str, tuple[str, ...]]] = {
        center: {} for center in CENTERS
    }
    for center in CENTERS:
        for endpoint in preterminal.predictions_by_center[center]:
            samples[center][endpoint.case_id] = endpoint.sample_ids
            for method in ENDPOINT_METHOD_IDS:
                probabilities[method][center][endpoint.case_id] = (
                    endpoint.probabilities[method]
                )
    for policy in COMPOSED_POLICY_IDS:
        for composed in preterminal.policy_runtime.final_predictions_by_policy[
            policy
        ]:
            probabilities[policy][composed.target_center][composed.case_id] = (
                composed.probabilities
            )
    expected_cases = {
        (center, case) for center in CENTERS for case in samples[center]
    }
    if any(
        {
            (center, case)
            for center in CENTERS
            for case in probabilities[method][center]
        }
        != expected_cases
        for method in probabilities
    ):
        raise ProtocolError("PCSI-PARC terminal probability maps are incomplete.")
    return (
        MappingProxyType(
            {
                method: MappingProxyType(
                    {
                        center: MappingProxyType(rows)
                        for center, rows in centers.items()
                    }
                )
                for method, centers in probabilities.items()
            }
        ),
        MappingProxyType(
            {center: MappingProxyType(rows) for center, rows in samples.items()}
        ),
    )


def _selection_aware_center_sign_flip(
    center_metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    p = center_metrics[PORTFOLIO_METHOD_ID]
    deltas = {
        method: np.asarray(
            [
                float(center_metrics[method][center]["center_bacc"])
                - float(p[center]["center_bacc"])
                for center in CENTERS
            ],
            dtype=np.float64,
        )
        for method in COMPOSED_POLICY_IDS
    }
    observed = float(np.mean(deltas[PRIMARY_METHOD_ID], dtype=np.float64))
    null_maxima = []
    for signs in product((-1.0, 1.0), repeat=len(CENTERS)):
        sign = np.asarray(signs, dtype=np.float64)
        null_maxima.append(
            max(
                float(np.mean(sign * deltas[method], dtype=np.float64))
                for method in COMPOSED_POLICY_IDS
            )
        )
    exceed = sum(value >= observed - 1.0e-15 for value in null_maxima)
    return {
        "schema_version": "fixed_bank_pcsi_parc_selection_aware_center_sign_flip_v1",
        "fixed_method_menu": list(COMPOSED_POLICY_IDS),
        "primary_method_id": PRIMARY_METHOD_ID,
        "observed_primary_mean_center_bacc_delta_vs_P": observed,
        "null_replicate_count": len(null_maxima),
        "exact_upper_tail_fraction_descriptive": exceed / len(null_maxima),
        "maximum_null_mean_delta": max(null_maxima),
        "method_identity_reselected_inside_each_null_replicate": True,
        "route_pipeline_refit_inside_null_replicate": False,
        "center_blocks_are_exchangeability_assumption": True,
        "descriptive_only": True,
        "formal_success_gate": False,
        "p_value_claimed": False,
        "nominal_significance_claimed": False,
    }


__all__ = ("TerminalEvaluation", "evaluate_terminal")
