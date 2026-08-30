"""Terminal capability, metric, and durable-attestation reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ...fixed_bank_sceptre_router.hashing import canonical_hash
from ...fixed_bank_sceptre_router.seals import (
    DurablePreterminalAttestation,
    EXPECTED_DECISION_KEYS,
    FreshProcessValidation,
)
from ..identity import PUBLICATION_STATUS, TERMINAL_DECISION
from ..route_policy import FrozenRoutePolicy
from ..support_posterior import SupportPosteriorDecision
from ..terminal_evaluation import TerminalEvaluationResult
from .persistence import PRETERMINAL_BUNDLE_MEMBER
from .validation_decisions import reconstruct_partition
from .validation_io import read_validation_object


def validate_terminal_lineage(
    destination: Path,
    *,
    result: TerminalEvaluationResult,
    durable: Mapping[str, object],
    journal: Mapping[str, object],
    preterminal: Mapping[str, object],
) -> None:
    """Replay capability issuance, journal prefix, and all 45 fold lineages."""

    bundle = read_validation_object(destination / PRETERMINAL_BUNDLE_MEMBER)
    policy_payload = bundle.get("route_policy")
    partition_payload = bundle.get("partition")
    support_payloads = bundle.get("support_decisions")
    preterminal_journal = bundle.get("label_journal_preterminal")
    if (
        not isinstance(policy_payload, Mapping)
        or not isinstance(partition_payload, Mapping)
        or not isinstance(support_payloads, list)
        or not isinstance(preterminal_journal, Mapping)
    ):
        raise ProtocolError("SCEPTRE v5 terminal lineage inputs are malformed.")
    policy = FrozenRoutePolicy.from_payload(policy_payload)
    partition = reconstruct_partition(partition_payload, policy)
    support_rows = tuple(
        SupportPosteriorDecision.from_payload(row)
        for row in support_payloads
        if isinstance(row, Mapping)
    )
    support_by_key = {
        (row.target_center, row.fold_ordinal): row for row in support_rows
    }
    if tuple(support_by_key) != EXPECTED_DECISION_KEYS:
        raise ProtocolError("SCEPTRE v5 terminal support coverage drifted.")
    capability_body = {
        "schema_version": "sceptre_terminal_evaluation_capability_v1",
        "partition_hash": partition.partition_hash,
        "router_bundle_hash": policy.routing_context_hash,
        "route_policy_hash": policy.policy_artifact_hash,
        "policy_seal_hash": policy.policy_seal_hash,
        "durable_attestation_hash": durable.get("attestation_hash"),
        "one_shot": True,
        "raw_labels_may_be_persisted": False,
    }
    terminal_capability_hash = canonical_hash(capability_body)
    pre_events = preterminal_journal.get("events")
    events = journal.get("events")
    if not isinstance(pre_events, list) or not isinstance(events, list):
        raise ProtocolError("SCEPTRE v5 final journal event inventory is malformed.")
    expected_event_count = len(pre_events) + 1 + len(EXPECTED_DECISION_KEYS)
    activation = events[len(pre_events)] if len(events) > len(pre_events) else None
    evaluation = events[len(pre_events) + 1 :]
    if (
        events[: len(pre_events)] != pre_events
        or len(events) != expected_event_count
        or not isinstance(activation, Mapping)
        or activation.get("event") != "TERMINAL_CAPABILITY_ACTIVATED"
        or activation.get("terminal_capability_hash") != terminal_capability_hash
        or activation.get("row_count") != 0
        or activation.get("manifest_rows_decoded") != 0
        or journal.get("partition_hash") != preterminal_journal.get("partition_hash")
        or journal.get("prediction_store_hash")
        != preterminal_journal.get("prediction_store_hash")
        or journal.get("authorization_lease_hash")
        != preterminal_journal.get("authorization_lease_hash")
        or journal.get("manifest_sha256")
        != preterminal_journal.get("manifest_sha256")
        or result.route_policy_hash != policy.policy_artifact_hash
        or result.route_policy_hash != preterminal.get("route_policy_hash")
        or result.prediction_store_hash != preterminal.get("prediction_store_hash")
        or result.terminal_capability_hash != terminal_capability_hash
    ):
        raise ProtocolError("SCEPTRE v5 terminal capability or journal prefix drifted.")
    for key, event, metric in zip(
        EXPECTED_DECISION_KEYS,
        evaluation,
        result.folds,
        strict=True,
    ):
        fold = partition.fold(*key)
        support = support_by_key[key]
        if (
            not isinstance(event, Mapping)
            or event.get("event") != "EVALUATION_LABELS_DECODED"
            or (event.get("target_center"), event.get("fold_ordinal")) != key
            or event.get("case_set_hash") != fold.case_set_hash("EVALUATION")
            or event.get("case_set_hash") != support.evaluation_case_set_hash
            or not isinstance(event.get("row_count"), int)
            or int(event["row_count"]) <= 0
            or event.get("manifest_rows_decoded") != event.get("row_count")
            or metric.fold_hash != fold.fold_hash
            or metric.evaluation_case_set_hash != fold.case_set_hash("EVALUATION")
            or metric.route != policy.route_for(*key)
            or metric.case_count != len(fold.evaluation_case_ids)
            or metric.route_aggregate.observation_count != event.get("row_count")
            or metric.exact_b_aggregate.observation_count != event.get("row_count")
        ):
            raise ProtocolError("SCEPTRE v5 terminal fold lineage drifted.")


def reconstruct_durable_attestation(
    payload: Mapping[str, object],
) -> DurablePreterminalAttestation:
    """Rehydrate the two independent preterminal validator receipts."""

    raw = payload.get("validation_receipts")
    if not isinstance(raw, list) or len(raw) != 2:
        raise ProtocolError("SCEPTRE v5 durable validation receipts drifted.")
    rows = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ProtocolError("SCEPTRE v5 durable validation receipt is malformed.")
        rows.append(
            FreshProcessValidation(
                process_id=int(value.get("process_id", 0)),
                policy_seal_hash=str(value.get("policy_seal_hash", "")),
                source_tree_sha256=str(value.get("source_tree_sha256", "")),
                reconstruction_hash=str(value.get("reconstruction_hash", "")),
                receipt_hash=str(value.get("receipt_hash", "")),
            )
        )
    return DurablePreterminalAttestation(
        policy_seal_hash=str(payload.get("policy_seal_hash", "")),
        validations=(rows[0], rows[1]),
        attestation_hash=str(payload.get("attestation_hash", "")),
    )


def validate_terminal_result_hash(payload: Mapping[str, object]) -> str:
    """Replay the terminal result hash independently of DTO construction."""

    folds = payload.get("folds")
    if not isinstance(folds, list):
        raise ProtocolError("SCEPTRE v5 terminal folds are malformed.")
    fold_hashes = []
    for fold in folds:
        if not isinstance(fold, Mapping):
            raise ProtocolError("SCEPTRE v5 terminal fold is malformed.")
        body = {key: value for key, value in fold.items() if key != "fold_metric_hash"}
        if fold.get("fold_metric_hash") != canonical_hash(body):
            raise ProtocolError("SCEPTRE v5 terminal fold hash drifted.")
        fold_hashes.append(fold["fold_metric_hash"])
    body = {
        "schema_version": "sceptre_v5_terminal_evaluation_result_v1",
        "route_policy_hash": payload.get("route_policy_hash"),
        "prediction_store_hash": payload.get("prediction_store_hash"),
        "terminal_capability_hash": payload.get("terminal_capability_hash"),
        "fold_metric_hashes": fold_hashes,
        "summary": payload.get("summary"),
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "nelbo_compatibility_claimed": False,
        "raw_labels_persisted": False,
    }
    expected = canonical_hash(body)
    if payload.get("result_hash") != expected:
        raise ProtocolError("SCEPTRE v5 terminal result hash drifted.")
    return expected


__all__ = (
    "reconstruct_durable_attestation",
    "validate_terminal_lineage",
    "validate_terminal_result_hash",
)
