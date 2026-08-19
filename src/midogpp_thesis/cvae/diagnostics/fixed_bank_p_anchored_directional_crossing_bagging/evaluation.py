"""Terminal-only PDCB utility, information, oracle, and selection diagnostics."""

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
    MODEL_BASED_METHOD_ID,
    PORTFOLIO_METHOD_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .engine import PreterminalResult
from .hashing import canonical_hash
from .information_diagnostics import crossing_information_diagnostics
from .terminal_metrics import score_methods


@dataclass(frozen=True)
class TerminalEvaluation:
    method_metrics: tuple[Mapping[str, object], ...]
    center_contrasts: tuple[Mapping[str, object], ...]
    case_oracle_rows: tuple[Mapping[str, object], ...]
    crossing_information_rows: tuple[Mapping[str, object], ...]
    crossing_information_center_rows: tuple[Mapping[str, object], ...]
    information_gate: Mapping[str, object]
    selection_control: Mapping[str, object]
    diagnostic_summary: Mapping[str, object]
    terminal_seal: Mapping[str, object]
    capability_report: Mapping[str, object]
    evaluation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_hash", canonical_hash(self.to_payload(include_hash=False)))

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = {
            "schema_version": "fixed_bank_pdcb_terminal_evaluation_v1",
            "method_metrics": [dict(row) for row in self.method_metrics],
            "center_contrasts": [dict(row) for row in self.center_contrasts],
            "case_oracle_rows": [dict(row) for row in self.case_oracle_rows],
            "crossing_information_rows": [dict(row) for row in self.crossing_information_rows],
            "crossing_information_center_rows": [
                dict(row) for row in self.crossing_information_center_rows
            ],
            "information_gate": dict(self.information_gate),
            "selection_control": dict(self.selection_control),
            "diagnostic_summary": dict(self.diagnostic_summary),
            "terminal_seal": dict(self.terminal_seal),
            "capability_report": dict(self.capability_report),
        }
        return {**payload, "evaluation_hash": self.evaluation_hash} if include_hash else payload


def evaluate_terminal(preterminal: PreterminalResult) -> TerminalEvaluation:
    """Open evaluation labels only after every composed probability is sealed."""

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
        raise ProtocolError("PDCB terminal label capability does not match its surface.")
    probabilities, sample_ids = _probability_maps(preterminal)
    method_order = (*ENDPOINT_METHOD_IDS, *COMPOSED_POLICY_IDS)
    method_rows, center_rows, oracle_rows, center_metrics = score_methods(
        probabilities,
        sample_ids,
        label_map,
        method_order=method_order,
    )
    metrics_by_method = {str(row["method_id"]): row for row in method_rows}
    primary = metrics_by_method[MODEL_BASED_METHOD_ID]
    info_rows, info_center_rows, information_gate = crossing_information_diagnostics(
        preterminal,
        label_map,
        primary_mean_center_bacc_delta_vs_p=float(
            primary["mean_center_bacc_delta_vs_P"]
        ),
        primary_mean_center_brier_delta_vs_p=float(
            primary["mean_center_brier_delta_vs_P"]
        ),
        primary_mean_center_log_loss_delta_vs_p=float(
            primary["mean_center_log_loss_delta_vs_P"]
        ),
        primary_helpful_switches=int(primary["helpful_threshold_switch_count"]),
        primary_harmful_switches=int(primary["harmful_threshold_switch_count"]),
    )
    selection_control = MappingProxyType(
        _selection_aware_center_sign_flip(center_metrics)
    )
    summary = MappingProxyType(
        {
            "schema_version": "fixed_bank_pdcb_diagnostic_summary_v1",
            "primary_method_id": MODEL_BASED_METHOD_ID,
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
            "primary_proper_loss_safety_pass": information_gate[
                "primary_proper_loss_safety_pass"
            ],
            "primary_route_count": primary["route_count"],
            "primary_threshold_switch_count": primary["threshold_switch_count"],
            "primary_helpful_threshold_switch_count": primary["helpful_threshold_switch_count"],
            "primary_harmful_threshold_switch_count": primary["harmful_threshold_switch_count"],
            "information_gate_status": information_gate["status"],
            "diagnosed_routing_bottleneck": information_gate["diagnosed_bottleneck"],
            "selection_aware_exact_p_value": selection_control["exact_p_value"],
            "claim_role": CLAIM_ROLE,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
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
        "schema_version": "fixed_bank_pdcb_terminal_seal_v1",
        "aggregate_preterminal_seal_hash": preterminal.aggregate_seal["aggregate_seal_hash"],
        "terminal_label_identity_and_value_hash": terminal_label_hash,
        "method_metrics_hash": canonical_hash([dict(row) for row in method_rows]),
        "center_contrasts_hash": canonical_hash([dict(row) for row in center_rows]),
        "case_oracle_rows_hash": canonical_hash([dict(row) for row in oracle_rows]),
        "crossing_information_rows_hash": canonical_hash([dict(row) for row in info_rows]),
        "information_gate_hash": canonical_hash(information_gate),
        "selection_control_hash": canonical_hash(selection_control),
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
        info_rows,
        info_center_rows,
        information_gate,
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
    samples: dict[str, dict[str, tuple[str, ...]]] = {center: {} for center in CENTERS}
    for center in CENTERS:
        for endpoint in preterminal.predictions_by_center[center]:
            samples[center][endpoint.case_id] = endpoint.sample_ids
            for method in ENDPOINT_METHOD_IDS:
                probabilities[method][center][endpoint.case_id] = endpoint.probabilities[method]
    for policy in COMPOSED_POLICY_IDS:
        for composed in preterminal.composed_predictions_by_policy[policy]:
            probabilities[policy][composed.target_center][composed.case_id] = composed.probabilities
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
        raise ProtocolError("PDCB terminal probability maps are incomplete.")
    return (
        MappingProxyType(
            {
                method: MappingProxyType(
                    {center: MappingProxyType(rows) for center, rows in centers.items()}
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
    observed = float(np.mean(deltas[MODEL_BASED_METHOD_ID], dtype=np.float64))
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
        "schema_version": "fixed_bank_pdcb_selection_aware_center_sign_flip_v1",
        "fixed_method_menu": list(COMPOSED_POLICY_IDS),
        "observed_primary_mean_center_bacc_delta_vs_P": observed,
        "null_replicate_count": len(null_maxima),
        "exact_p_value": exceed / len(null_maxima),
        "maximum_null_mean_delta": max(null_maxima),
        "method_identity_reselected_inside_each_null_replicate": True,
        "route_pipeline_refit_inside_null_replicate": False,
        "center_blocks_are_exchangeability_assumption": True,
        "nominal_significance_claimed": False,
    }


__all__ = ("TerminalEvaluation", "evaluate_terminal")
