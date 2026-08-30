"""Public fresh-process validation façade for SCEPTRE v5 artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....runtime.artifact_io import sha256_file
from ...fixed_bank_sceptre_router.hashing import canonical_hash
from ..config import load_config
from ..identity import PUBLICATION_STATUS, TERMINAL_DECISION
from ..route_policy import FrozenRoutePolicy
from ..source_seal import build_source_snapshot_payload
from ..terminal_evaluation import TerminalEvaluationResult
from .persistence import (
    DURABLE_ATTESTATION_MEMBER,
    FINAL_INDEX_MEMBER,
    FINAL_SUMMARY_MEMBER,
    FINAL_VALIDATION_MEMBER,
    PRETERMINAL_BUNDLE_MEMBER,
    PRETERMINAL_INDEX_MEMBER,
    TERMINAL_RESULT_MEMBER,
    VALIDATION_REPORT_MEMBER,
)
from .validation_authority import validate_preterminal_authority
from .validation_decisions import reconstruct_partition, validate_decision_graph
from .validation_io import read_validation_object
from .validation_journal import validate_label_journal
from .validation_physical import validate_physical_graph
from .validation_terminal import (
    reconstruct_durable_attestation,
    validate_terminal_lineage,
    validate_terminal_result_hash,
)


def validate_preterminal_bundle(root: str | Path) -> Mapping[str, object]:
    """Reconstruct the complete sealed graph before evaluation-label access."""

    destination = Path(root).resolve()
    index = read_validation_object(destination / PRETERMINAL_INDEX_MEMBER)
    bundle = read_validation_object(destination / PRETERMINAL_BUNDLE_MEMBER)
    index_body = {
        key: value for key, value in index.items() if key != "content_index_hash"
    }
    bundle_body = {key: value for key, value in bundle.items() if key != "bundle_hash"}
    prediction_hashes = index.get("prediction_member_hashes")
    if not isinstance(prediction_hashes, Mapping):
        raise ProtocolError("SCEPTRE v5 prediction member index is malformed.")
    if (
        index.get("schema_version")
        != "sceptre_v5_preterminal_content_index_v1"
        or bundle.get("schema_version") != "sceptre_v5_preterminal_bundle_v1"
        or index.get("content_index_hash") != canonical_hash(index_body)
        or bundle.get("bundle_hash") != canonical_hash(bundle_body)
        or index.get("bundle_hash") != bundle.get("bundle_hash")
        or index.get("bundle_file_sha256")
        != sha256_file(destination / PRETERMINAL_BUNDLE_MEMBER)
        or index.get("evaluation_labels_opened") is not False
        or bundle.get("evaluation_labels_opened") is not False
        or index.get("raw_labels_persisted") is not False
        or bundle.get("raw_labels_persisted") is not False
    ):
        raise ProtocolError("SCEPTRE v5 preterminal index or bundle drifted.")

    source = bundle.get("source_snapshot")
    if not isinstance(source, Mapping) or dict(source) != dict(
        build_source_snapshot_payload()
    ):
        raise ProtocolError("SCEPTRE v5 source snapshot reconstruction drifted.")
    config = load_config(destination / "config.resolved.yaml")
    if config.config_hash != bundle.get("config_hash"):
        raise ProtocolError("SCEPTRE v5 config reconstruction drifted.")
    policy_payload = bundle.get("route_policy")
    if not isinstance(policy_payload, Mapping):
        raise ProtocolError("SCEPTRE v5 route policy is absent.")
    policy = FrozenRoutePolicy.from_payload(policy_payload)
    input_binding = bundle.get("input_binding")
    source_store = bundle.get("source_store")
    prediction_store = bundle.get("prediction_store")
    partition_payload = bundle.get("partition")
    development = bundle.get("development")
    phases = bundle.get("routing_phases")
    journal = bundle.get("label_journal_preterminal")
    if not all(
        isinstance(value, Mapping)
        for value in (
            input_binding,
            source_store,
            prediction_store,
            partition_payload,
            development,
            phases,
            journal,
        )
    ):
        raise ProtocolError("SCEPTRE v5 preterminal graph is malformed.")

    validate_preterminal_authority(destination, bundle, config)
    partition = reconstruct_partition(partition_payload, policy)
    validate_decision_graph(
        index=index,
        bundle=bundle,
        development=development,
        phases=phases,
        journal=journal,
        policy=policy,
        partition=partition,
    )
    validate_physical_graph(
        destination,
        index=index,
        bundle=bundle,
        input_binding=input_binding,
        source_store=source_store,
        prediction_store=prediction_store,
        phases=phases,
        prediction_hashes=prediction_hashes,
        partition=partition,
    )
    reconstruction_body = {
        "schema_version": "sceptre_v5_preterminal_reconstruction_v1",
        "content_index_hash": index["content_index_hash"],
        "bundle_hash": bundle["bundle_hash"],
        "config_hash": config.config_hash,
        "partition_hash": partition_payload.get("partition_hash"),
        "routing_context_hash": policy.routing_context_hash,
        "route_policy_hash": policy.policy_artifact_hash,
        "policy_seal_hash": policy.policy_seal_hash,
        "prediction_store_hash": phases.get("prediction_store_hash"),
        "source_store_hash": source_store.get("source_store_hash"),
        "source_tree_sha256": source.get("source_snapshot_tree_sha256"),
        "raw_labels_persisted": False,
    }
    return {
        **reconstruction_body,
        "reconstruction_hash": canonical_hash(reconstruction_body),
    }


def validate_final_bundle(root: str | Path) -> Mapping[str, object]:
    """Reconstruct the terminal metrics and bind them to the frozen policy."""

    destination = Path(root).resolve()
    preterminal = validate_preterminal_bundle(destination)
    durable = read_validation_object(destination / DURABLE_ATTESTATION_MEMBER)
    final = read_validation_object(destination / FINAL_INDEX_MEMBER)
    result = read_validation_object(destination / TERMINAL_RESULT_MEMBER)
    summary = read_validation_object(destination / FINAL_SUMMARY_MEMBER)
    final_body = {
        key: value for key, value in final.items() if key != "content_index_hash"
    }
    typed_result = TerminalEvaluationResult.from_payload(result)
    result_hash = validate_terminal_result_hash(result)
    if result_hash != typed_result.result_hash:
        raise ProtocolError("SCEPTRE v5 terminal result reconstruction disagreed.")
    summary_body = {
        key: value for key, value in summary.items() if key != "summary_hash"
    }
    attestation = reconstruct_durable_attestation(durable)
    if (
        durable.get("schema_version")
        != "sceptre_v5_durable_preterminal_attestation_v1"
        or durable.get("preterminal_content_index_hash")
        != preterminal["content_index_hash"]
        or durable.get("attestation_hash") != attestation.attestation_hash
        or durable.get("policy_seal_hash") != preterminal["policy_seal_hash"]
        or final.get("schema_version") != "sceptre_v5_final_content_index_v1"
        or final.get("content_index_hash") != canonical_hash(final_body)
        or final.get("preterminal_content_index_hash")
        != preterminal["content_index_hash"]
        or final.get("durable_attestation_hash") != durable.get("attestation_hash")
        or final.get("terminal_result_hash") != result_hash
        or final.get("terminal_result_file_sha256")
        != sha256_file(destination / TERMINAL_RESULT_MEMBER)
        or final.get("summary_hash") != summary.get("summary_hash")
        or final.get("summary_file_sha256")
        != sha256_file(destination / FINAL_SUMMARY_MEMBER)
        or summary.get("summary_hash") != canonical_hash(summary_body)
        or summary.get("result_hash") != result_hash
        or result.get("publication_status") != PUBLICATION_STATUS
        or result.get("terminal_decision") != TERMINAL_DECISION
        or result.get("raw_labels_persisted") is not False
        or len(result.get("folds", ())) != 45
    ):
        raise ProtocolError("SCEPTRE v5 final artifact graph drifted.")
    journal = summary.get("label_journal")
    events = journal.get("events") if isinstance(journal, Mapping) else None
    evaluation_events = (
        [
            row
            for row in events
            if isinstance(row, Mapping)
            and str(row.get("event", "")).startswith("EVALUATION_LABELS")
        ]
        if isinstance(events, list)
        else []
    )
    if (
        not isinstance(journal, Mapping)
        or not isinstance(events, list)
        or not all(isinstance(row, Mapping) for row in events)
        or journal.get("raw_labels_persisted") is not False
        or len(evaluation_events) != 45
        or any(row.get("raw_labels_persisted") is not False for row in evaluation_events)
    ):
        raise ProtocolError("SCEPTRE v5 terminal label journal drifted.")
    validate_label_journal(journal)
    validate_terminal_lineage(
        destination,
        result=typed_result,
        durable=durable,
        journal=journal,
        preterminal=preterminal,
    )
    body = {
        "schema_version": "sceptre_v5_final_reconstruction_v1",
        "preterminal_reconstruction_hash": preterminal["reconstruction_hash"],
        "durable_attestation_hash": durable["attestation_hash"],
        "final_content_index_hash": final["content_index_hash"],
        "terminal_result_hash": result_hash,
        "summary_hash": summary["summary_hash"],
        "raw_labels_persisted": False,
    }
    return {**body, "reconstruction_hash": canonical_hash(body)}


def validate_complete_bundle(root: str | Path) -> Mapping[str, object]:
    """Authenticate the final graph plus the two-validator attestation files."""

    destination = Path(root).resolve()
    final = dict(validate_final_bundle(destination))
    attestation = read_validation_object(destination / FINAL_VALIDATION_MEMBER)
    report = read_validation_object(destination / VALIDATION_REPORT_MEMBER)
    attestation_body = {
        key: value for key, value in attestation.items() if key != "attestation_hash"
    }
    report_body = {
        key: value for key, value in report.items() if key != "report_hash"
    }
    validations = attestation.get("validations")
    if not isinstance(validations, list):
        raise ProtocolError("SCEPTRE v5 final validations are malformed.")
    process_ids = {
        row.get("process_id") for row in validations if isinstance(row, Mapping)
    }
    reconstruction_hashes = {
        row.get("reconstruction_hash")
        for row in validations
        if isinstance(row, Mapping)
    }
    if (
        attestation.get("schema_version")
        != "sceptre_v5_final_fresh_process_attestation_v1"
        or len(validations) != 2
        or len(process_ids) != 2
        or reconstruction_hashes != {final["reconstruction_hash"]}
        or attestation.get("byte_identical_reconstruction") is not True
        or attestation.get("attestation_hash") != canonical_hash(attestation_body)
        or report.get("schema_version") != "sceptre_v5_validation_report_v1"
        or report.get("status") != "PASS"
        or report.get("final_attestation_hash") != attestation.get("attestation_hash")
        or report.get("report_hash") != canonical_hash(report_body)
        or report.get("fresh_evidence") is not False
        or report.get("routing_success_claimed") is not False
        or report.get("nelbo_compatibility_claimed") is not False
    ):
        raise ProtocolError("SCEPTRE v5 complete validation graph drifted.")
    body = {
        "schema_version": "sceptre_v5_complete_reconstruction_v1",
        "final_reconstruction_hash": final["reconstruction_hash"],
        "final_attestation_hash": attestation["attestation_hash"],
        "validation_report_hash": report["report_hash"],
        "raw_labels_persisted": False,
    }
    return {**body, "reconstruction_hash": canonical_hash(body)}


__all__ = (
    "validate_complete_bundle",
    "validate_final_bundle",
    "validate_preterminal_bundle",
)
