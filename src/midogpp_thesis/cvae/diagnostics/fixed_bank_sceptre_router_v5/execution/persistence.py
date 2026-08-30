"""Closed-world durable artifacts for SCEPTRE v5 production execution."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json, sha256_file
from ...fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ...fixed_bank_sceptre_router.partitions import ThreeRolePartition
from ...fixed_bank_sceptre_router.seals import DurablePreterminalAttestation
from ..development import FrozenDevelopmentReplay
from ..identity import EXPERIMENT_ID, PUBLICATION_STATUS, TERMINAL_DECISION
from ..phase_orchestrator import SealedRoutingPhases
from ..source_seal import build_source_snapshot_payload
from ..terminal_evaluation import TerminalEvaluationResult


PRETERMINAL_BUNDLE_MEMBER = "manifests/preterminal_bundle.json"
PRETERMINAL_INDEX_MEMBER = "manifests/preterminal_content_index.json"
DURABLE_ATTESTATION_MEMBER = "reports/preterminal_fresh_process_attestation.json"
TERMINAL_RESULT_MEMBER = "tables/terminal_result.json"
FINAL_SUMMARY_MEMBER = "reports/summary.json"
FINAL_INDEX_MEMBER = "manifests/final_content_index.json"
FINAL_VALIDATION_MEMBER = "reports/final_fresh_process_attestation.json"
VALIDATION_REPORT_MEMBER = "reports/validation_report.json"
FAILURE_REPORT_MEMBER = "reports/failure_report.json"


def persist_preterminal_bundle(
    root: str | Path,
    *,
    config_hash: str,
    admission: Mapping[str, object],
    input_binding: Mapping[str, object],
    authorization_lease: Mapping[str, object],
    runtime: Mapping[str, object],
    source_store: Mapping[str, object],
    prediction_store: Mapping[str, object],
    prediction_member_hashes: Mapping[str, str],
    partition: ThreeRolePartition,
    development: FrozenDevelopmentReplay,
    phases: SealedRoutingPhases,
) -> Mapping[str, object]:
    """Persist everything needed to reconstruct the policy without labels."""

    destination = Path(root)
    if not isinstance(partition, ThreeRolePartition):
        raise ProtocolError("SCEPTRE v5 preterminal partition is untyped.")
    if not isinstance(development, FrozenDevelopmentReplay):
        raise ProtocolError("SCEPTRE v5 preterminal development is untyped.")
    if not isinstance(phases, SealedRoutingPhases):
        raise ProtocolError("SCEPTRE v5 preterminal phases are untyped.")
    members = {
        str(name): require_sha256(digest, f"prediction member {name}")
        for name, digest in sorted(prediction_member_hashes.items())
    }
    bundle_body = {
        "schema_version": "sceptre_v5_preterminal_bundle_v1",
        "config_hash": require_sha256(config_hash, "preterminal config"),
        "admission": dict(admission),
        "input_binding": dict(input_binding),
        "authorization_lease": dict(authorization_lease),
        "runtime": dict(runtime),
        "source_snapshot": dict(build_source_snapshot_payload()),
        "source_store": dict(source_store),
        "partition": _partition_payload(partition),
        "development": development.to_payload(),
        "routing_context": development.context.to_payload(),
        "proposal_sets": [row.to_payload() for row in development.proposal_sets],
        "routing_phases": phases.to_payload(),
        "support_decisions": [
            row.to_payload() for row in phases.support_decisions
        ],
        "calibration_posteriors": [
            row.to_payload() for row in phases.calibration_posteriors
        ],
        "confirmation_decisions": [
            row.to_payload() for row in phases.confirmation_decisions
        ],
        "route_policy": phases.route_policy.to_payload(),
        "label_journal_preterminal": dict(phases.label_journal),
        "prediction_store": dict(prediction_store),
        "prediction_member_hashes": members,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "evaluation_labels_opened": False,
        "raw_labels_persisted": False,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }
    bundle = {**bundle_body, "bundle_hash": canonical_hash(bundle_body)}
    _write_new_json(destination / PRETERMINAL_BUNDLE_MEMBER, bundle)
    index_body = {
        "schema_version": "sceptre_v5_preterminal_content_index_v1",
        "bundle_member": PRETERMINAL_BUNDLE_MEMBER,
        "bundle_file_sha256": sha256_file(destination / PRETERMINAL_BUNDLE_MEMBER),
        "bundle_hash": bundle["bundle_hash"],
        "prediction_member_hashes": members,
        "partition_hash": partition.partition_hash,
        "routing_context_hash": development.context.context_hash,
        "phase_hash": phases.phase_hash,
        "route_policy_hash": phases.route_policy.policy_artifact_hash,
        "policy_seal_hash": phases.policy_seal.seal_hash,
        "source_tree_sha256": bundle["source_snapshot"][
            "source_snapshot_tree_sha256"
        ],
        "source_store_hash": source_store["source_store_hash"],
        "evaluation_labels_opened": False,
        "raw_labels_persisted": False,
    }
    index = {**index_body, "content_index_hash": canonical_hash(index_body)}
    _write_new_json(destination / PRETERMINAL_INDEX_MEMBER, index)
    return index


def persist_durable_attestation(
    root: str | Path,
    attestation: DurablePreterminalAttestation,
    *,
    preterminal_content_index_hash: str,
) -> Mapping[str, object]:
    if not isinstance(attestation, DurablePreterminalAttestation):
        raise ProtocolError("SCEPTRE v5 durable attestation is untyped.")
    body = {
        "schema_version": "sceptre_v5_durable_preterminal_attestation_v1",
        "preterminal_content_index_hash": require_sha256(
            preterminal_content_index_hash, "preterminal content index"
        ),
        "policy_seal_hash": attestation.policy_seal_hash,
        "validation_receipts": [asdict(row) for row in attestation.validations],
        "attestation_hash": attestation.attestation_hash,
        "independent_process_ids": [
            row.process_id for row in attestation.validations
        ],
        "evaluation_labels_opened": False,
        "raw_labels_persisted": False,
    }
    _write_new_json(Path(root) / DURABLE_ATTESTATION_MEMBER, body)
    return body


def persist_terminal_bundle(
    root: str | Path,
    result: TerminalEvaluationResult,
    *,
    final_label_journal: Mapping[str, object],
) -> Mapping[str, object]:
    if not isinstance(result, TerminalEvaluationResult):
        raise ProtocolError("SCEPTRE v5 terminal result is untyped.")
    destination = Path(root)
    result_payload = result.to_payload()
    _write_new_json(destination / TERMINAL_RESULT_MEMBER, result_payload)
    summary_body = {
        "schema_version": "sceptre_v5_terminal_summary_v1",
        "result_hash": result.result_hash,
        "route_policy_hash": result.route_policy_hash,
        "summary": dict(result.summary),
        "label_journal": dict(final_label_journal),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "nelbo_compatibility_claimed": False,
        "raw_labels_persisted": False,
        "may_feed_another_experiment": False,
    }
    summary = {**summary_body, "summary_hash": canonical_hash(summary_body)}
    _write_new_json(destination / FINAL_SUMMARY_MEMBER, summary)
    preterminal = read_json(destination / PRETERMINAL_INDEX_MEMBER)
    durable = read_json(destination / DURABLE_ATTESTATION_MEMBER)
    final_body = {
        "schema_version": "sceptre_v5_final_content_index_v1",
        "preterminal_content_index_hash": preterminal.get("content_index_hash"),
        "durable_attestation_hash": durable.get("attestation_hash"),
        "terminal_result_hash": result.result_hash,
        "terminal_result_file_sha256": sha256_file(
            destination / TERMINAL_RESULT_MEMBER
        ),
        "summary_hash": summary["summary_hash"],
        "summary_file_sha256": sha256_file(destination / FINAL_SUMMARY_MEMBER),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "raw_labels_persisted": False,
    }
    final = {**final_body, "content_index_hash": canonical_hash(final_body)}
    _write_new_json(destination / FINAL_INDEX_MEMBER, final)
    return final


def persist_final_validation(
    root: str | Path,
    *,
    validations: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    rows = [dict(row) for row in validations]
    if len(rows) != 2 or len({row.get("process_id") for row in rows}) != 2:
        raise ProtocolError("SCEPTRE v5 requires two final validator processes.")
    body = {
        "schema_version": "sceptre_v5_final_fresh_process_attestation_v1",
        "validations": rows,
        "byte_identical_reconstruction": (
            len({row.get("reconstruction_hash") for row in rows}) == 1
        ),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
    }
    if body["byte_identical_reconstruction"] is not True:
        raise ProtocolError("SCEPTRE v5 final validators disagree.")
    payload = {**body, "attestation_hash": canonical_hash(body)}
    destination = Path(root)
    _write_new_json(destination / FINAL_VALIDATION_MEMBER, payload)
    report_body = {
        "schema_version": "sceptre_v5_validation_report_v1",
        "status": "PASS",
        "final_attestation_hash": payload["attestation_hash"],
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "nelbo_compatibility_claimed": False,
    }
    report = {**report_body, "report_hash": canonical_hash(report_body)}
    _write_new_json(destination / VALIDATION_REPORT_MEMBER, report)
    return report


def persist_failure_report(
    root: str | Path,
    *,
    config_hash: str,
    authorization_lease: object,
    phase: str,
    bound_hashes: Mapping[str, str],
    error: BaseException,
    scratch_root: Path | None,
) -> Mapping[str, object]:
    """Preserve a compact forensic record before owned scratch is deleted."""

    hashes = {
        str(role): require_sha256(value, f"failure {role}")
        for role, value in sorted(bound_hashes.items())
    }
    body = {
        "schema_version": "sceptre_v5_failure_report_v1",
        "experiment_id": EXPERIMENT_ID,
        "config_hash": require_sha256(config_hash, "failure config"),
        "authorization_lease_hash": require_sha256(
            getattr(authorization_lease, "lease_hash", None), "failure lease"
        ),
        "authorization_lease_status_at_failure": str(
            getattr(authorization_lease, "status", "")
        ),
        "phase": str(phase),
        "bound_hashes": hashes,
        "error_class": error.__class__.__name__,
        "error": str(error)[:2000],
        "scratch_root_cleaned_after_report": scratch_root is not None,
        "scratch_root": None if scratch_root is None else str(scratch_root),
        "authorization_exhausted": True,
        "cross_run_recovery_allowed": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "raw_labels_persisted": False,
    }
    payload = {**body, "failure_report_hash": canonical_hash(body)}
    _write_new_json(Path(root) / FAILURE_REPORT_MEMBER, payload)
    return payload


def prediction_store_payload(
    artifact_root: str | Path, prediction: object
) -> tuple[Mapping[str, object], Mapping[str, str]]:
    destination = Path(artifact_root)
    paths = (
        Path(getattr(prediction, "candidate_array_path")),
        Path(getattr(prediction, "exact_b_array_path")),
        Path(getattr(prediction, "index_path")),
        Path(getattr(prediction, "receipt_path")),
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ProtocolError("SCEPTRE v5 prediction store is incomplete.")
    try:
        names = tuple(path.relative_to(destination).as_posix() for path in paths)
    except ValueError as exc:
        raise ProtocolError("SCEPTRE v5 prediction store escaped artifact root.") from exc
    hashes = {name: sha256_file(path) for name, path in zip(names, paths, strict=True)}
    receipt = dict(getattr(prediction, "receipt"))
    payload = {
        "schema_version": "sceptre_v5_prediction_store_binding_v1",
        "receipt_hash": require_sha256(
            getattr(prediction, "receipt_hash"), "prediction receipt"
        ),
        "member_hashes": hashes,
        "candidate_shape": list(getattr(prediction, "geometry").candidate_shape),
        "exact_b_shape": list(getattr(prediction, "geometry").exact_b_shape),
        "candidate_source_order": list(getattr(prediction, "geometry").centers),
        "receipt_schema": receipt.get("schema_version"),
        "read_only_memmap": True,
        "labels_opened": False,
    }
    return payload, hashes


def source_store_payload(source: object) -> Mapping[str, object]:
    """Reduce the ephemeral generated stream store to a durable sealed receipt."""

    paths = (
        Path(getattr(source, "array_path")),
        Path(getattr(source, "index_path")),
        Path(getattr(source, "receipt_path")),
    )
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ProtocolError("SCEPTRE v5 source stream store is incomplete.")
    receipt = dict(getattr(source, "receipt"))
    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    receipt_hash = require_sha256(
        getattr(source, "receipt_hash"), "source stream receipt"
    )
    if receipt.get("receipt_sha256") != canonical_hash(receipt_body):
        raise ProtocolError("SCEPTRE v5 source stream receipt hash drifted.")
    body = {
        "schema_version": "sceptre_v5_source_store_binding_v1",
        "attempt_id": str(getattr(source, "attempt_id")),
        "receipt_hash": receipt_hash,
        "source_array_file_sha256": sha256_file(paths[0]),
        "source_index_file_sha256": sha256_file(paths[1]),
        "source_receipt_file_sha256": sha256_file(paths[2]),
        "source_receipt": receipt,
        "geometry": getattr(source, "geometry").to_payload(),
        "record_hashes": [
            canonical_hash(record.to_payload())
            for record in getattr(source, "records")
        ],
        "stream_count": len(getattr(source, "records")),
        "source_store_retained_after_terminal_evaluation": False,
        "labels_opened": False,
    }
    return {**body, "source_store_hash": canonical_hash(body)}


def _partition_payload(partition: ThreeRolePartition) -> dict[str, object]:
    return {
        "schema_version": "sceptre_v5_partition_persistence_v1",
        "partition_seed": partition.partition_seed,
        "partition_hash": partition.partition_hash,
        "case_count": len(
            {(row.target_center, row.case_id) for row in partition.identities}
        ),
        "identities": [
            {
                "target_center": row.target_center,
                "case_id": row.case_id,
                "sample_id": row.sample_id,
            }
            for row in partition.identities
        ],
        "folds": [
            {
                "target_center": fold.target_center,
                "fold_ordinal": fold.fold_ordinal,
                "selection_case_ids": list(fold.selection_case_ids),
                "calibration_case_ids": list(fold.calibration_case_ids),
                "evaluation_case_ids": list(fold.evaluation_case_ids),
                "fold_hash": fold.fold_hash,
            }
            for fold in partition.folds
        ],
        "whole_case_roles_disjoint": True,
        "evaluation_cases_exactly_once": True,
        "labels_persisted": False,
    }


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise ProtocolError(f"SCEPTRE v5 refuses to overwrite {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, dict(payload))


__all__ = (
    "DURABLE_ATTESTATION_MEMBER",
    "FINAL_INDEX_MEMBER",
    "FINAL_SUMMARY_MEMBER",
    "FINAL_VALIDATION_MEMBER",
    "FAILURE_REPORT_MEMBER",
    "PRETERMINAL_BUNDLE_MEMBER",
    "PRETERMINAL_INDEX_MEMBER",
    "TERMINAL_RESULT_MEMBER",
    "VALIDATION_REPORT_MEMBER",
    "persist_durable_attestation",
    "persist_final_validation",
    "persist_failure_report",
    "persist_preterminal_bundle",
    "persist_terminal_bundle",
    "prediction_store_payload",
    "source_store_payload",
)
