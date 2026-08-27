"""Center-probability assembly and sealed worker artifact emission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from types import MappingProxyType

import numpy as np

from ..artifacts.chunks import write_center_chunk
from ..execution.dtos import OuterCenterResult, OuterCenterTask
from ..hashing import canonical_hash
from ..identity import GovernanceError
from ..label_capabilities import DelegatedWorkerLabelJournal
from ..physical_memmaps import MappedPhysicalStore
from ..routing.admission import AdmissionMetrics
from ..routing.selection import RouteDecision
from ..terminal.scoring import sealed_probability_hash
from ..utility.actions import ActionRectangle
from .contracts import (
    DonorPhaseOutput,
    FinalRouteOutput,
    METHOD_IDS,
    ParsedTaskPayload,
    ROUTE_CHUNK_SCHEMA,
    WORKER_CAPABILITY_CHUNK_SCHEMA,
)


def emit_outer_center_result(
    task: OuterCenterTask,
    parsed: ParsedTaskPayload,
    store: MappedPhysicalStore,
    journal: DelegatedWorkerLabelJournal,
    donor_phase: DonorPhaseOutput,
    routes: Sequence[FinalRouteOutput],
) -> OuterCenterResult:
    """Seal one outer center without returning labels or fitted objects."""

    route_rows = tuple(routes)
    route_hashes = tuple(row.route_hash for row in route_rows)
    method_probabilities = assemble_center_probabilities(store, task, route_rows)
    method_probability_hashes = {
        method_id: sealed_probability_hash(method_probabilities[method_id])
        for method_id in METHOD_IDS
    }
    admission = admission_payload(donor_phase.admission)
    decision_fragment_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_outer_decision_fragment_v1",
            "target_center": task.target_center,
            "task_hash": task.task_hash,
            "route_hashes": route_hashes,
            "method_probability_hashes": method_probability_hashes,
            "admission_metrics_hash": donor_phase.admission.metrics_hash,
            "method_ids": METHOD_IDS,
            "route_count": len(route_rows),
        }
    )
    audit = journal.complete(decision_fragment_hash=decision_fragment_hash)

    route_payload = {
        "schema_version": ROUTE_CHUNK_SCHEMA,
        "target_center": task.target_center,
        "case_ids": list(task.case_ids),
        "row_sample_ids": list(store.rows_by_center[task.target_center]),
        "route_records": [dict(row.record) for row in route_rows],
        "method_probabilities": {
            method_id: method_probabilities[method_id].tolist()
            for method_id in METHOD_IDS
        },
        "method_probability_hashes": method_probability_hashes,
        "decision_fragment_hash": decision_fragment_hash,
        "admission": admission,
        "scientific_contracts_hash": parsed.settings.contracts_hash,
        "raw_labels_persisted": False,
        "row_level_label_values_persisted": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "terminal_diagnostic_only": True,
    }
    route_chunk = write_center_chunk(
        parsed.artifact_root,
        target_center=task.target_center,
        phase_id="route_decisions",
        payload=route_payload,
        record_count=len(route_rows),
        bindings={
            "protocol_hash": task.protocol_hash,
            "task_hash": task.task_hash,
            "delegation_hash": parsed.delegation.delegation_hash,
            "physical_index_hash": parsed.physical_index_hash,
            "label_identity_hash": parsed.label_identity_hash,
            "scientific_contracts_hash": parsed.settings.contracts_hash,
            "decision_fragment_hash": decision_fragment_hash,
        },
    )
    capability_payload = {
        "schema_version": WORKER_CAPABILITY_CHUNK_SCHEMA,
        "target_center": task.target_center,
        "worker_capability_audit": audit.to_payload(),
        "worker_event_log": journal.event_log_payload(),
        "donor_phase_hash": donor_phase.donor_phase_hash,
        "pseudo_replay_hash": donor_phase.pseudo_replay_hash,
        "pseudo_record_count": donor_phase.pseudo_record_count,
        "pseudo_model_manifest_hash": donor_phase.pseudo_model_manifest_hash,
        "decision_fragment_hash": decision_fragment_hash,
        "scientific_contracts_hash": parsed.settings.contracts_hash,
        "raw_labels_persisted": False,
        "row_level_label_values_persisted": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
    }
    capability_chunk = write_center_chunk(
        parsed.artifact_root,
        target_center=task.target_center,
        phase_id="worker_capability",
        payload=capability_payload,
        record_count=audit.event_count,
        bindings={
            "task_hash": task.task_hash,
            "delegation_hash": parsed.delegation.delegation_hash,
            "worker_audit_hash": audit.audit_hash,
            "decision_fragment_hash": decision_fragment_hash,
        },
    )
    chunks = tuple(
        sorted(
            (route_chunk, capability_chunk),
            key=lambda row: (row.phase_id, row.member),
        )
    )
    no_refit_reconstruction_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_outer_no_refit_reconstruction_v1",
            "target_center": task.target_center,
            "task_hash": task.task_hash,
            "physical_index_hash": parsed.physical_index_hash,
            "label_identity_hash": parsed.label_identity_hash,
            "scientific_contracts_hash": parsed.settings.contracts_hash,
            "route_chunk_hash": route_chunk.chunk_hash,
            "worker_capability_chunk_hash": capability_chunk.chunk_hash,
            "route_hashes": route_hashes,
            "method_probability_hashes": method_probability_hashes,
            "decision_fragment_hash": decision_fragment_hash,
            "scientific_refit_required_for_reconstruction": False,
        }
    )
    summary = {
        "schema_version": "scale_bp_v2_outer_worker_summary_v1",
        "target_center": task.target_center,
        "worker_capability_audit": audit.to_payload(),
        "admission": admission,
        "method_ids": list(METHOD_IDS),
        "method_probability_hashes": method_probability_hashes,
        "decision_fragment_hash": decision_fragment_hash,
        "scientific_contracts_hash": parsed.settings.contracts_hash,
        "pseudo_replay_hash": donor_phase.pseudo_replay_hash,
        "pseudo_record_count": donor_phase.pseudo_record_count,
        "raw_labels_returned": False,
        "raw_labels_persisted": False,
        "fresh_evidence": False,
        "promotion_eligible": False,
    }
    return OuterCenterResult(
        target_center=task.target_center,
        task_hash=task.task_hash,
        completed_support_fold_ids=task.support_fold_ids,
        route_hashes=route_hashes,
        chunks=chunks,
        no_refit_reconstruction_hash=no_refit_reconstruction_hash,
        summary_json=json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    )


def assemble_center_probabilities(
    store: MappedPhysicalStore,
    task: OuterCenterTask,
    routes: Sequence[FinalRouteOutput],
) -> Mapping[str, np.ndarray]:
    """Restore case-local outputs to the immutable physical row order."""

    row_count = len(store.rows_by_center[task.target_center])
    output = {
        method_id: np.empty(row_count, dtype=np.float64) for method_id in METHOD_IDS
    }
    covered = np.zeros(row_count, dtype=bool)
    by_case = {row.case_id: row for row in routes}
    if tuple(by_case) != task.case_ids or len(by_case) != len(tuple(routes)):
        raise GovernanceError("SCALE-BP v2 final route inventory drifted.")
    for case_id in task.case_ids:
        route = by_case[case_id]
        indices = store.case_indices(task.target_center, case_id)
        expected_samples = tuple(
            store.rows_by_center[task.target_center][index] for index in indices
        )
        if route.sample_ids != expected_samples or np.any(covered[list(indices)]):
            raise GovernanceError("SCALE-BP v2 final probability row binding drifted.")
        for method_id in METHOD_IDS:
            values = np.ascontiguousarray(
                route.method_probabilities[method_id], dtype=np.float64
            )
            if values.shape != (len(indices),):
                raise GovernanceError("SCALE-BP v2 case probability width drifted.")
            output[method_id][list(indices)] = values
        covered[list(indices)] = True
    if not np.all(covered) or any(
        not np.isfinite(values).all()
        or np.any((values < 0.0) | (values > 1.0))
        for values in output.values()
    ):
        raise GovernanceError("SCALE-BP v2 final center probability coverage drifted.")
    for values in output.values():
        values.setflags(write=False)
    return MappingProxyType(output)


def decision_payload(
    decision: RouteDecision, rectangle: ActionRectangle
) -> dict[str, object]:
    if decision.rectangle_hash != rectangle.rectangle_hash:
        raise GovernanceError("SCALE-BP v2 decision/audit rectangle drifted.")
    selected = next(
        (
            assessment
            for assessment in decision.assessments
            if assessment.action_id == decision.selected_action_id
        ),
        None,
    )
    selected_evidence = (
        None
        if selected is None
        else {
            "mean": selected.estimate.mean.to_payload(),
            "lower": selected.bounds.lower.to_payload(),
            "upper": selected.bounds.upper.to_payload(),
            "shrinkage_weights": list(selected.estimate.shrinkage_weights),
            "within_support": selected.estimate.within_support,
            "bank_viable": selected.estimate.bank_viable,
        }
    )
    return {
        "decision_hash": decision.decision_hash,
        "method_id": decision.method_id,
        "selected_action_id": decision.selected_action_id,
        "reason": decision.reason,
        "selected_action_hash": decision.selected_action_hash,
        "selected_evidence": selected_evidence,
        "assessments": [
            {
                "action_id": row.action_id,
                "eligible": row.eligible,
                "reasons": list(row.reasons),
                "assessment_hash": row.assessment_hash,
                "estimate_hash": row.estimate.estimate_hash,
                "bounds_hash": row.bounds.bounds_hash,
            }
            for row in decision.assessments
        ],
        "action_switch_audit": [
            {
                "action_id": cell.action_id,
                "evidence_packet_hash": cell.evidence.packet_hash,
                "threshold_switch_count": cell.evidence.threshold_switch_count,
                "harmful_switch_count": cell.evidence.harmful_switch_count,
                "harmful_switch_count_status": (
                    cell.evidence.harmful_switch_count_status
                ),
            }
            for cell in rectangle.cells
        ],
        "p_candidate": {
            "representation": "IMPLICIT_ZERO_UTILITY_EXACT_FALLBACK",
            "expected_utility_anchor": 0.0,
            "candidate_assessment_emitted": False,
            "wins_without_unique_robust_safe_positive_action": True,
        },
        "exact_p_fallback": decision.is_exact_p,
    }


def admission_payload(metrics: AdmissionMetrics) -> dict[str, object]:
    return {
        "schema_version": "scale_bp_v2_admission_aggregate_v1",
        "case_count": metrics.case_count,
        "represented_center_count": metrics.represented_center_count,
        "top1_oracle_agreement": metrics.top1_oracle_agreement,
        "mean_within_case_spearman": metrics.mean_within_case_spearman,
        "mean_normalized_oracle_gap": metrics.mean_normalized_oracle_gap,
        "selected_count": metrics.selected_count,
        "harmful_selected_count": metrics.harmful_selected_count,
        "proper_safe_selected_count": metrics.proper_safe_selected_count,
        "passed": metrics.passed,
        "failed_gates": list(metrics.failed_gates),
        "metrics_hash": metrics.metrics_hash,
        "may_authorize_controls": False,
    }


__all__ = (
    "admission_payload",
    "assemble_center_probabilities",
    "decision_payload",
    "emit_outer_center_result",
)
