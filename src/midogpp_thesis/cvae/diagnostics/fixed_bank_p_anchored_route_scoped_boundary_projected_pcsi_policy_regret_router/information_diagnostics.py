"""Terminal-only descriptions of frozen PCSI-RACR route decisions.

Nothing in this module is available until the complete preterminal route menu is
sealed. The rows explain an already frozen consumed-test diagnostic; they never
feed transport, calibration, selection, or another experiment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .case_regret import realized_case_favorable_vector
from .constants import (
    CENTERS,
    COMPOSED_POLICY_IDS,
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    RAW_OBSERVED_MAX_METHOD_ID,
)
from .contracts import BinaryLabel
from .engine import PreterminalResult
from .reports import validate_transport_lineage_evidence


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
    """Describe sealed per-case actions, replays, envelopes, and outcomes."""

    labels = tuple(terminal_labels)
    label_map = {row.key: row for row in labels}
    if len(label_map) != len(labels):
        raise ProtocolError("PCSI-RACR terminal diagnostic labels are duplicated.")
    by_case: dict[tuple[str, str], list[BinaryLabel]] = {}
    positive: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    for row in labels:
        by_case.setdefault((row.center, row.case_id), []).append(row)
        positive[row.center] += int(row.value == 1)
        negative[row.center] += int(row.value == 0)
    if any(positive[center] <= 0 or negative[center] <= 0 for center in CENTERS):
        raise ProtocolError("PCSI-RACR terminal center lacks a binary class.")

    lineage = validate_transport_lineage_evidence(preterminal)
    runtime = preterminal.policy_runtime
    endpoint_index = {
        (center, endpoint.case_id): endpoint
        for center in CENTERS
        for endpoint in preterminal.predictions_by_center[center]
    }

    action_rows: list[Mapping[str, object]] = []
    selected_rows: list[Mapping[str, object]] = []
    frequencies: Counter[tuple[str, str, str]] = Counter()
    for key in sorted(runtime.target_candidate_policies):
        policy_id, center, case_id = key
        candidate = runtime.target_candidate_policies[key]
        decision = runtime.decisions[key]
        if policy_id == PRIMARY_METHOD_ID:
            for directional in candidate.decisions:
                action_rows.append(
                    MappingProxyType(
                        {
                            "target_center": center,
                            "case_id": case_id,
                            "direction": directional.direction,
                            "selected_representative": directional.selected_representative,
                            "selected_action_hash": directional.selected_action_hash,
                            "target_influence": directional.target_influence,
                            "predicted_favorable_vector": list(
                                directional.predicted_favorable_endpoint_vector
                            ),
                            "candidate_policy_hash": candidate.policy_hash,
                            "route_decision_hash": decision.decision_hash,
                            "descriptive_only": True,
                        }
                    )
                )
                frequencies[
                    (policy_id, directional.direction, directional.selected_representative)
                ] += 1

        realized = (
            realized_case_favorable_vector(
                endpoint_index[(center, case_id)],
                candidate,
                by_case[(center, case_id)],
                center_n_positive=positive[center],
                center_n_negative=negative[center],
            )
            if candidate.changed
            else (0.0, 0.0, 0.0)
        )
        selected_rows.append(
            MappingProxyType(
                {
                    **decision.to_payload(),
                    "candidate_changed": candidate.changed,
                    "realized_candidate_favorable_vector": list(realized),
                    "emitted_change": decision.changed,
                    "terminal_labels_used_only_after_aggregate_seal": True,
                    "descriptive_only": True,
                }
            )
        )

    replay_rows = tuple(
        MappingProxyType(
            {**runtime.replays[key].to_payload(), "descriptive_only": True}
        )
        for key in sorted(runtime.replays)
    )
    transport_rows = tuple(
        MappingProxyType(
            {
                **runtime.transport_screens[key].to_payload(),
                "affects_route_decision": key.__class__.__name__ == "TargetRouteKey",
                "descriptive_only": True,
            }
        )
        for key in sorted(runtime.transport_screens, key=repr)
    )

    center_rows: list[Mapping[str, object]] = []
    for geometry_id, center in sorted(runtime.calibrations):
        calibration = runtime.calibrations[(geometry_id, center)]
        policy_id = (
            PRIMARY_METHOD_ID
            if geometry_id == PROJECTION_GEOMETRY_ID
            else RAW_OBSERVED_MAX_METHOD_ID
        )
        center_rows.append(
            MappingProxyType(
                {
                    **calibration.to_payload(),
                    "changed_route_count": sum(
                        runtime.decisions[(policy_id, center, endpoint.case_id)].changed
                        for endpoint in preterminal.predictions_by_center[center]
                    ),
                    "case_count": len(preterminal.predictions_by_center[center]),
                    "descriptive_only": True,
                }
            )
        )

    frequency_rows = tuple(
        MappingProxyType(
            {
                "policy_id": policy,
                "direction": direction,
                "representative": representative,
                "case_count": count,
                "descriptive_only": True,
            }
        )
        for (policy, direction, representative), count in sorted(frequencies.items())
    )
    change_counts = {
        policy: sum(
            runtime.decisions[(policy, center, endpoint.case_id)].changed
            for center in CENTERS
            for endpoint in preterminal.predictions_by_center[center]
        )
        for policy in COMPOSED_POLICY_IDS
    }
    summary = MappingProxyType(
        {
            "schema_version": "fixed_bank_pcsi_racr_terminal_diagnostic_summary_v1",
            "status": "PASS_TERMINAL_DESCRIPTIVE_ONLY",
            "authorized_target_policy_count_by_policy": change_counts,
            "policy_rho_diagnostic_lower_bacc_vs_realized_candidate_policy_bacc_midrank_spearman": None,
            "target_route_count": sum(
                len(preterminal.predictions_by_center[center]) for center in CENTERS
            ),
            "pseudo_case_replay_count": len(replay_rows),
            "transport_screen_count": len(transport_rows),
            "transport_lineage_hash": lineage["transport_hash"],
            "observed_donor_case_envelope": True,
            "conformal": False,
            "finite_sample_coverage": False,
            "success_gate_defined": False,
            "fresh_evidence": False,
            "may_feed_another_experiment": False,
        }
    )
    return TerminalDiagnostics(
        tuple(action_rows),
        replay_rows,
        transport_rows,
        tuple(selected_rows),
        tuple(center_rows),
        frequency_rows,
        summary,
    )


__all__ = ("TerminalDiagnostics", "policy_regret_terminal_diagnostics")
