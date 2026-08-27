"""Closed-world persistence for SCEPTRE v3 preterminal and terminal bundles."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping, Sequence

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    read_json,
    sha256_file,
)

from ..fixed_bank_sceptre_router.hashing import canonical_hash, require_sha256
from ..fixed_bank_sceptre_router.partitions import ThreeRolePartition
from ..fixed_bank_sceptre_router.seals import DurablePreterminalAttestation
from .development_orchestrator import FrozenDevelopmentReplay
from .phase_orchestrator import SealedRoutingPhases
from .terminal_evaluation import TerminalEvaluationResult


PRETERMINAL_JSON_MEMBERS = (
    "provenance/input_binding.json",
    "provenance/source_snapshot.json",
    "provenance/authorization_consumption_lease.json",
    "manifests/prediction_store.json",
    "models/frozen_prelabel_router.json",
    "models/g_proposals.json",
    "tables/development_replay.json",
    "manifests/partition.json",
    "tables/support_decisions.json",
    "tables/uncertainty_decisions.json",
    "tables/calibration_decisions.json",
    "manifests/g_seal.json",
    "manifests/selection_seal.json",
    "manifests/policy_seal.json",
    "manifests/frozen_route_policy.json",
    "reports/label_capability_journal.json",
)
PRETERMINAL_INDEX_MEMBER = "manifests/preterminal_content_index.json"
PRETERMINAL_LAUNCH_MEMBERS = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
)
DURABLE_ATTESTATION_MEMBER = "reports/preterminal_fresh_process_attestation.json"
TERMINAL_RESULT_MEMBER = "tables/terminal_result.json"
FINAL_INDEX_MEMBER = "manifests/content_index.json"
FINAL_FRESH_ATTESTATION_MEMBER = "reports/final_fresh_process_attestation.json"
VALIDATION_REPORT_MEMBER = "reports/validation_report.json"
VALIDATION_INDEX_MEMBER = "manifests/validation_index.json"


def persist_preterminal_bundle(
    root: str | Path,
    *,
    config_hash: str,
    input_binding: Mapping[str, object],
    source_snapshot: Mapping[str, object],
    authorization_lease: Mapping[str, object],
    prediction_store: Mapping[str, object],
    prediction_member_hashes: Mapping[str, str],
    partition: ThreeRolePartition,
    development: FrozenDevelopmentReplay,
    phases: SealedRoutingPhases,
) -> Mapping[str, object]:
    """Persist the complete policy graph before any evaluation label opens."""

    destination = Path(root).resolve()
    if not destination.is_dir() or destination.is_symlink():
        raise ProtocolError("SCEPTRE v3 output root is absent or unsafe.")
    if (
        development.router.partition_hash != partition.partition_hash
        or phases.route_policy.partition_hash != partition.partition_hash
        or phases.prediction_store_hash != prediction_store.get("store_hash")
    ):
        raise ProtocolError("SCEPTRE v3 preterminal lineage drifted.")
    payloads: dict[str, Mapping[str, object]] = {
        "provenance/input_binding.json": dict(input_binding),
        "provenance/source_snapshot.json": dict(source_snapshot),
        "provenance/authorization_consumption_lease.json": dict(
            authorization_lease
        ),
        "manifests/prediction_store.json": dict(prediction_store),
        "models/frozen_prelabel_router.json": development.router.to_payload(),
        "models/g_proposals.json": {
            "schema_version": "sceptre_v3_g_proposal_inventory_v1",
            "proposals": [row.to_payload() for row in development.proposals],
            "proposal_count": len(development.proposals),
            "target_global": True,
            "labels_consumed": False,
        },
        "tables/development_replay.json": {
            **development.receipt_payload(),
            "replay_hash": development.replay_hash,
        },
        "manifests/partition.json": partition_payload(partition),
        "tables/support_decisions.json": _decision_inventory(
            "SUPPORT", phases.support_decisions
        ),
        "tables/uncertainty_decisions.json": _decision_inventory(
            "UNCERTAINTY", phases.uncertainty_decisions
        ),
        "tables/calibration_decisions.json": _decision_inventory(
            "CALIBRATION", phases.calibration_decisions
        ),
        "manifests/g_seal.json": asdict(phases.g_seal),
        "manifests/selection_seal.json": asdict(phases.selection_seal),
        "manifests/policy_seal.json": asdict(phases.policy_seal),
        "manifests/frozen_route_policy.json": phases.route_policy.to_payload(),
        "reports/label_capability_journal.json": dict(phases.label_journal),
    }
    if tuple(payloads) != PRETERMINAL_JSON_MEMBERS:
        raise ProtocolError("SCEPTRE v3 preterminal member order drifted.")
    for member, payload in payloads.items():
        _persist_or_match(destination / member, payload)
    member_hashes = {
        member: sha256_file(destination / member)
        for member in PRETERMINAL_JSON_MEMBERS
    }
    for member in PRETERMINAL_LAUNCH_MEMBERS:
        path = destination / member
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("SCEPTRE v3 launch member is absent or unsafe.")
        member_hashes[member] = sha256_file(path)
    for member, expected in sorted(prediction_member_hashes.items()):
        safe = _safe_relative_member(member)
        digest = require_sha256(expected, "prediction member")
        path = destination / safe
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ProtocolError("SCEPTRE v3 prediction member bytes drifted.")
        member_hashes[safe] = digest
    unhashed = {
        "schema_version": "sceptre_v3_preterminal_content_index_v1",
        "config_hash": require_sha256(config_hash, "config"),
        "partition_hash": partition.partition_hash,
        "router_hash": development.router.full_router_sha256,
        "development_replay_hash": development.replay_hash,
        "prediction_store_hash": phases.prediction_store_hash,
        "phase_hash": phases.phase_hash,
        "route_policy_hash": phases.route_policy.policy_artifact_hash,
        "policy_seal_hash": phases.policy_seal.seal_hash,
        "members": member_hashes,
        "target_labels_opened_for_evaluation": False,
        "raw_labels_persisted": False,
        "closed_world": True,
    }
    index = {**unhashed, "content_index_hash": canonical_hash(unhashed)}
    _persist_or_match(destination / PRETERMINAL_INDEX_MEMBER, index)
    return index


def persist_durable_attestation(
    root: str | Path,
    attestation: DurablePreterminalAttestation,
    *,
    preterminal_content_index_hash: str,
) -> Mapping[str, object]:
    if not isinstance(attestation, DurablePreterminalAttestation):
        raise ProtocolError("SCEPTRE v3 durable attestation is untyped.")
    body = {
        "schema_version": "sceptre_v3_durable_preterminal_attestation_v1",
        "preterminal_content_index_hash": require_sha256(
            preterminal_content_index_hash, "preterminal index"
        ),
        "policy_seal_hash": attestation.policy_seal_hash,
        "validator_process_ids": [
            row.process_id for row in attestation.validations
        ],
        "validator_receipt_hashes": [
            row.receipt_hash for row in attestation.validations
        ],
        "source_tree_sha256": attestation.validations[0].source_tree_sha256,
        "reconstruction_hash": attestation.validations[0].reconstruction_hash,
        "attestation_hash": attestation.attestation_hash,
        "fresh_process_count": 2,
        "byte_identical_reconstruction": True,
        "target_labels_opened_for_evaluation": False,
    }
    _persist_or_match(Path(root).resolve() / DURABLE_ATTESTATION_MEMBER, body)
    return body


def persist_terminal_bundle(
    root: str | Path,
    result: TerminalEvaluationResult,
    *,
    reports: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    if not isinstance(result, TerminalEvaluationResult):
        raise ProtocolError("SCEPTRE v3 terminal result is untyped.")
    destination = Path(root).resolve()
    _persist_or_match(destination / TERMINAL_RESULT_MEMBER, result.to_payload())
    report_hashes: dict[str, str] = {}
    for member, payload in sorted(reports.items()):
        safe = _safe_relative_member(member)
        if not safe.startswith("reports/"):
            raise ProtocolError("SCEPTRE v3 final report escapes reports/.")
        _persist_or_match(destination / safe, payload)
        report_hashes[safe] = sha256_file(destination / safe)
    preterminal = read_json(destination / PRETERMINAL_INDEX_MEMBER)
    attestation = read_json(destination / DURABLE_ATTESTATION_MEMBER)
    unhashed = {
        "schema_version": "sceptre_v3_final_content_index_v1",
        "preterminal_content_index_hash": preterminal.get("content_index_hash"),
        "durable_attestation_hash": attestation.get("attestation_hash"),
        "terminal_result_hash": result.result_hash,
        "terminal_result_file_sha256": sha256_file(
            destination / TERMINAL_RESULT_MEMBER
        ),
        "report_member_hashes": report_hashes,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
        "raw_labels_persisted": False,
        "closed_world": True,
    }
    index = {**unhashed, "content_index_hash": canonical_hash(unhashed)}
    _persist_or_match(destination / FINAL_INDEX_MEMBER, index)
    return index


def persist_validation_index(
    root: str | Path,
    *,
    final_attestation_member: str,
    validation_report_member: str,
) -> Mapping[str, object]:
    """Persist a self-hashed index for subsequent publication-layer validation."""

    destination = Path(root).resolve()
    final_index = read_json(destination / FINAL_INDEX_MEMBER)
    attestation = read_json(destination / _safe_relative_member(final_attestation_member))
    report = read_json(destination / _safe_relative_member(validation_report_member))
    if (
        attestation.get("schema_version")
        != "sceptre_v3_final_fresh_process_attestation_v1"
        or attestation.get("status") != "PASS"
        or report.get("schema_version")
        != "sceptre_v3_final_validation_report_v1"
        or report.get("status") != "PASS"
        or report.get("final_fresh_process_attestation_hash")
        != attestation.get("attestation_hash")
    ):
        raise ProtocolError("SCEPTRE v3 post-validation payload drifted.")
    body = {
        "schema_version": "sceptre_v3_postvalidation_index_v1",
        "status": "PASS",
        "final_content_index_hash": final_index.get("content_index_hash"),
        "final_content_index_file_sha256": sha256_file(
            destination / FINAL_INDEX_MEMBER
        ),
        "final_fresh_process_attestation_hash": attestation.get("attestation_hash"),
        "validation_report_hash": report.get("report_hash"),
        "members": {
            final_attestation_member: sha256_file(
                destination / final_attestation_member
            ),
            validation_report_member: sha256_file(
                destination / validation_report_member
            ),
        },
        "fresh_process_count": 2,
        "self_validation_claimed": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
        "raw_labels_persisted": False,
    }
    payload = {**body, "validation_index_hash": canonical_hash(body)}
    _persist_or_match(destination / VALIDATION_INDEX_MEMBER, payload)
    return payload


def partition_payload(partition: ThreeRolePartition) -> dict[str, object]:
    if not isinstance(partition, ThreeRolePartition):
        raise ProtocolError("SCEPTRE v3 partition persistence is untyped.")
    return {
        "schema_version": "sceptre_v3_partition_persistence_v1",
        "partition_seed": partition.partition_seed,
        "partition_hash": partition.partition_hash,
        "identities": [asdict(row) for row in partition.identities],
        "folds": [asdict(row) for row in partition.folds],
        "labels_consumed": False,
    }


def _decision_inventory(role: str, rows: Sequence[object]) -> dict[str, object]:
    payload = [asdict(row) for row in rows]
    return {
        "schema_version": "sceptre_v3_decision_inventory_v1",
        "role": role,
        "rows": payload,
        "row_count": len(payload),
        "raw_labels_persisted": False,
        "inventory_hash": canonical_hash(payload),
    }


def _persist_or_match(path: Path, payload: Mapping[str, object]) -> None:
    normalized = _json_value(payload)
    if not isinstance(normalized, dict):
        raise ProtocolError("SCEPTRE v3 persistence payload is not a JSON object.")
    if path.is_symlink():
        raise ProtocolError("SCEPTRE v3 persistence target is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != normalized:
            raise ProtocolError(
                "Existing SCEPTRE v3 member differs; refusing repair or overwrite."
            )
        return
    atomic_json(path, normalized)


def _json_value(value: object) -> object:
    """Normalize tuples and other JSON containers before byte persistence."""

    return json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _safe_relative_member(member: str) -> str:
    path = Path(str(member))
    if path.is_absolute() or ".." in path.parts or str(path) != str(member):
        raise ProtocolError("SCEPTRE v3 member path is unsafe.")
    return path.as_posix()


__all__ = (
    "DURABLE_ATTESTATION_MEMBER",
    "FINAL_FRESH_ATTESTATION_MEMBER",
    "FINAL_INDEX_MEMBER",
    "PRETERMINAL_INDEX_MEMBER",
    "PRETERMINAL_JSON_MEMBERS",
    "PRETERMINAL_LAUNCH_MEMBERS",
    "TERMINAL_RESULT_MEMBER",
    "VALIDATION_REPORT_MEMBER",
    "VALIDATION_INDEX_MEMBER",
    "partition_payload",
    "persist_durable_attestation",
    "persist_preterminal_bundle",
    "persist_terminal_bundle",
    "persist_validation_index",
)
