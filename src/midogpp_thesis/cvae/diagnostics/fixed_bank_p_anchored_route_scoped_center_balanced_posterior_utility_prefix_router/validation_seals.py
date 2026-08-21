"""Candidate-to-terminal cryptographic seal-chain validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ...runtime.artifact_io import read_json
from .constants import CENTERS, PUBLICATION_STATUS, TERMINAL_DECISION
from .hashing import canonical_hash
from .terminal_diagnostics import GateFunnel
from .validation_candidates import CandidateTopology
from .validation_plans import PlanPosteriorTopology
from .validation_shared import Row, fail, index_rows, string_list, table_rows


def validate_seal_chain(
    root: Path,
    *,
    physical: Row,
    plan_topology: PlanPosteriorTopology,
    candidate_topology: CandidateTopology,
    capability: Row,
    method_metrics: Sequence[Row],
    center_metrics: Sequence[Row],
    oracle_rows: Sequence[Row],
    summary: Row,
    leakage: Row,
    terminal: Row,
    expected_gate_funnel: GateFunnel,
) -> str:
    plan_seal = read_json(root / "manifests/outer_plan_seal.json")
    decision_barrier = read_json(root / "manifests/decision_barrier.json")
    aggregate = read_json(root / "manifests/preterminal_aggregate_seal.json")

    candidate_seal = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_candidate_seal_v1",
            "plan_seal_hash": plan_seal["seal_hash"],
            "target": [
                [*key, candidate_topology.targets[key]["runtime_hash"]]
                for key in sorted(candidate_topology.targets)
            ],
            "pseudo": [
                [*key, candidate_topology.pseudos[key]["runtime_hash"]]
                for key in sorted(candidate_topology.pseudos)
            ],
            "terminal_labels_used": False,
        }
    )
    transport_lineage = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_structural_lineage_v1",
            "physical_surface_hash": physical["surface_hash"],
            "outer_plan_seal_hash": plan_seal["seal_hash"],
            "candidate_seal_hash": candidate_seal,
            "pseudo_posterior_reference_hashes": sorted(
                str(row["reference_hash"])
                for row in plan_topology.pseudo_references.values()
            ),
            "numeric_transport_is_authorization_gate": False,
        }
    )
    pre_evaluation = canonical_hash([candidate_seal, transport_lineage])

    replay_rows = table_rows(root, "pseudo_policy_replays")
    donor_rows = [
        row for row in replay_rows if row.get("record_type") == "donor_case_replay"
    ]
    policy_rows = [
        row for row in replay_rows if row.get("record_type") == "policy_replay"
    ]
    diagnostic_rows = [
        row
        for row in replay_rows
        if row.get("record_type") == "policy_replay_diagnostic"
    ]
    if len(diagnostic_rows) != len(CENTERS) or len(replay_rows) != (
        len(donor_rows) + len(policy_rows) + len(diagnostic_rows)
    ):
        fail("replay record topology")
    _validate_policy_replay_diagnostics(policy_rows, diagnostic_rows)
    calibrations = table_rows(root, "donor_bias_calibrations")
    replay_calibration_seal = canonical_hash(
        [
            pre_evaluation,
            canonical_hash(
                sorted(str(row["result_hash"]) for row in donor_rows)
            ),
            canonical_hash(
                sorted(str(row["calibration_hash"]) for row in calibrations)
            ),
            canonical_hash(
                sorted(str(row["runtime_hash"]) for row in policy_rows)
            ),
            {"policy_replay_bias_used": False},
        ]
    )
    decision_unhashed = {
        key: value
        for key, value in decision_barrier.items()
        if key != "decision_barrier_hash"
    }
    if (
        decision_unhashed
        != {
            "schema_version": "fixed_bank_cbpupr_decision_barrier_v1",
            "candidate_seal_hash": candidate_seal,
            "pre_evaluation_seal_hash": pre_evaluation,
            "replay_calibration_seal_hash": replay_calibration_seal,
            "pseudo_evaluation_opened_after_candidate_seal": True,
            "target_evaluation_opened": False,
        }
        or decision_barrier.get("decision_barrier_hash")
        != canonical_hash(decision_unhashed)
        or aggregate.get("schema_version")
        != "fixed_bank_cbpupr_preterminal_aggregate_seal_v1"
        or aggregate.get("target_evaluation_opened") is not False
        or summary.get("preterminal_hash") != aggregate.get("preterminal_hash")
    ):
        fail("durable preterminal seal chain")

    gate_rows = table_rows(root, "gate_funnel")
    if len(gate_rows) != 1:
        fail("gate funnel topology")
    gate = GateFunnel.from_payload(gate_rows[0])
    if gate != expected_gate_funnel:
        fail("gate funnel semantic lineage")
    expected_preterminal_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_preterminal_result_v1",
            "candidate_seal_hash": candidate_seal,
            "pre_evaluation_seal_hash": pre_evaluation,
            "replay_calibration_seal_hash": replay_calibration_seal,
            "aggregate_seal_hash": aggregate.get("aggregate_seal_hash"),
            "gate_funnel_hash": gate.funnel_hash,
            "target_label_used": False,
        }
    )
    events = capability.get("events")
    if not isinstance(events, list):
        fail("terminal capability event list")
    terminal_event = next(
        (
            row
            for row in events
            if isinstance(row, dict)
            and row.get("role") == "target_terminal_after_aggregate_seal"
        ),
        None,
    )
    if terminal_event is None:
        fail("terminal capability event")
    expected_terminal_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_terminal_seal_v1",
            "aggregate_seal_hash": aggregate.get("aggregate_seal_hash"),
            "label_identity_hash": terminal_event["identity_hash"],
            "method_rows": list(method_metrics),
            "selection_aware_center_sign_flip": summary.get(
                "selection_aware_center_sign_flip"
            ),
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
        }
    )
    expected_result_hash = canonical_hash(
        {
            "schema_version": "fixed_bank_cbpupr_terminal_result_v1",
            "method_rows": list(method_metrics),
            "center_rows": list(center_metrics),
            "oracle_rows": list(oracle_rows),
            "terminal_seal_hash": expected_terminal_hash,
            "diagnostic_summary": dict(summary),
            "raw_labels_persisted": False,
        }
    )
    if (
        aggregate.get("preterminal_hash") != expected_preterminal_hash
        or terminal.get("aggregate_seal_hash")
        != aggregate.get("aggregate_seal_hash")
        or terminal.get("terminal_seal_hash") != expected_terminal_hash
        or terminal.get("terminal_result_hash") != expected_result_hash
        or leakage.get("outer_plan_seal_hash") != plan_seal.get("seal_hash")
        or leakage.get("aggregate_preterminal_seal_hash")
        != aggregate.get("aggregate_seal_hash")
        or leakage.get("capability_report_hash") != canonical_hash(capability)
    ):
        fail("terminal/preterminal lineage")
    return expected_preterminal_hash


def _validate_policy_replay_diagnostics(
    policy_rows: Sequence[Row], diagnostic_rows: Sequence[Row]
) -> None:
    indexed = index_rows(
        diagnostic_rows, ("outer_center",), "policy replay diagnostics"
    )
    if set(key[0] for key in indexed) != set(CENTERS):
        fail("policy replay diagnostic center rectangle")
    for (outer,), row in indexed.items():
        observed = [
            candidate
            for candidate in policy_rows
            if candidate.get("replay", {}).get("outer_center") == outer  # type: ignore[union-attr]
        ]
        replay_hashes = sorted(
            str(candidate["replay"]["replay_hash"])  # type: ignore[index]
            for candidate in observed
        )
        donor_centers = sorted(
            {
                str(candidate["replay"]["donor_center"])  # type: ignore[index]
                for candidate in observed
            }
        )
        payload = {
            "schema_version": "fixed_bank_cbpupr_policy_replay_diagnostic_v1",
            "outer_center": outer,
            "donor_centers": donor_centers,
            "replay_hashes": replay_hashes,
            "policy_replay_bias_used": False,
            "authorization_gate": False,
        }
        if (
            row.get("record_type") != "policy_replay_diagnostic"
            or string_list(row, "donor_centers", allow_empty=True)
            != tuple(donor_centers)
            or string_list(row, "replay_hashes", allow_empty=True)
            != tuple(replay_hashes)
            or row.get("policy_replay_bias_used") is not False
            or row.get("authorization_gate") is not False
            or row.get("diagnostic_hash") != canonical_hash(payload)
        ):
            fail("policy replay diagnostic lineage")


__all__ = ("validate_seal_chain",)
