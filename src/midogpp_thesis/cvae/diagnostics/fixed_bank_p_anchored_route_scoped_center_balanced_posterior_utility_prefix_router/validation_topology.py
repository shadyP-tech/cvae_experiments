"""Thin coordinator for exact persisted CBPUPR topology validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .validation_arrays import validate_composed_probability_stores
from .validation_candidates import validate_candidate_topology
from .validation_capabilities import validate_capability_topology
from .validation_decisions import validate_route_decisions
from .validation_diagnostics import validate_preterminal_diagnostics
from .validation_endpoint_evidence import validate_endpoint_evidence
from .validation_fingerprints import validate_fingerprint_topology
from .validation_label_truth import validate_label_derived_truth
from .validation_origin import PhysicalOriginTopology
from .validation_plans import validate_plan_posterior_topology
from .validation_posterior_models import validate_posterior_model_predictions
from .validation_replays import validate_persisted_replays_and_calibrations
from .validation_seals import validate_seal_chain
from .validation_shared import Row, fail, mapping_field, string_list, table_rows
from .validation_transport import validate_transport_diagnostics


def validate_exact_topology_and_lineage(
    root: Path,
    *,
    config: object,
    origin: PhysicalOriginTopology,
    physical: Row,
    fingerprints: Sequence[Row],
    plans: Sequence[Row],
    support: Sequence[Row],
    models: Sequence[Row],
    posteriors: Sequence[Row],
    pseudo_references: Sequence[Row],
    target_candidates: Sequence[Row],
    pseudo_candidates: Sequence[Row],
    decisions: Sequence[Row],
    composed: Sequence[Row],
    capability: Row,
    method_metrics: Sequence[Row],
    center_metrics: Sequence[Row],
    oracle_rows: Sequence[Row],
    summary: Row,
    leakage: Row,
    terminal: Row,
) -> None:
    plan_topology = validate_plan_posterior_topology(
        root,
        physical=physical,
        plan_rows=plans,
        support_rows=support,
        model_rows=models,
        posterior_rows=posteriors,
        pseudo_reference_rows=pseudo_references,
    )
    validate_capability_topology(
        root, topology=plan_topology, capability=capability
    )
    validate_fingerprint_topology(
        physical=physical,
        rows=fingerprints,
        topology=plan_topology,
    )
    posterior_probabilities = validate_posterior_model_predictions(
        root,
        origin=origin,
        topology=plan_topology,
    )
    endpoint_topology = validate_endpoint_evidence(
        root,
        config=config,
        origin=origin,
        topology=plan_topology,
        capability=capability,
    )
    candidate_topology = validate_candidate_topology(
        root,
        topology=plan_topology,
        posterior_probabilities=posterior_probabilities,
        endpoint_topology=endpoint_topology,
        target_rows=target_candidates,
        pseudo_rows=pseudo_candidates,
    )
    transport = validate_transport_diagnostics(
        root,
        rows=table_rows(root, "transport_diagnostics"),
        topology=plan_topology,
        candidate_topology=candidate_topology,
        origin=origin,
    )

    replay_rows = table_rows(root, "pseudo_policy_replays")
    donor_replay_rows = [
        row
        for row in replay_rows
        if row.get("record_type") == "donor_case_replay"
    ]
    replay_validation = validate_persisted_replays_and_calibrations(
        pseudo_candidate_rows=pseudo_candidates,
        donor_case_replay_rows=donor_replay_rows,
        policy_replay_rows=[
            row for row in replay_rows if row.get("record_type") == "policy_replay"
        ],
        donor_bias_calibration_rows=table_rows(root, "donor_bias_calibrations"),
        case_sample_count_by_center_case={
            key: len(string_list(row, "evaluation_sample_ids"))
            for key, row in plan_topology.plans.items()
        },
        selected_candidate_utilities={
            key: record.action.estimate.utility
            for key, record in candidate_topology.selected_action_by_runtime.items()
            if key[0] != key[1] and record is not None
        },
    )
    decision_index = validate_route_decisions(
        root,
        plan_topology=plan_topology,
        candidate_topology=candidate_topology,
        rows=decisions,
        calibrations=replay_validation["calibrations_by_outer_control"],
    )
    for (center, _method), row in decision_index.items():
        if dict(mapping_field(row, "structural_transport")) != (
            transport.structural_by_center[center].to_payload()
        ):
            fail("route decision/transport diagnostic cross-table lineage")
    validate_composed_probability_stores(
        root,
        plan_topology=plan_topology,
        candidate_topology=candidate_topology,
        decisions=decision_index,
        rows=composed,
        donor_case_replay_rows=donor_replay_rows,
    )
    gate_funnel = validate_preterminal_diagnostics(
        root,
        candidate_topology=candidate_topology,
        decisions=decision_index,
    )
    preterminal_hash = validate_seal_chain(
        root,
        physical=physical,
        plan_topology=plan_topology,
        candidate_topology=candidate_topology,
        capability=capability,
        method_metrics=method_metrics,
        center_metrics=center_metrics,
        oracle_rows=oracle_rows,
        summary=summary,
        leakage=leakage,
        terminal=terminal,
        expected_gate_funnel=gate_funnel,
    )
    validate_label_derived_truth(
        root,
        config=config,
        origin=origin,
        plan_topology=plan_topology,
        candidate_topology=candidate_topology,
        endpoint_topology=endpoint_topology,
        capability=capability,
        method_metrics=method_metrics,
        center_metrics=center_metrics,
        oracle_rows=oracle_rows,
        summary=summary,
        terminal=terminal,
        gate_funnel=gate_funnel,
        preterminal_hash=preterminal_hash,
        policy_replay_count=int(replay_validation["policy_replay_count"]),
    )


__all__ = ("validate_exact_topology_and_lineage",)
