"""Pure contract for the externally issued OE-PPUR v3 amendment.

This module can construct and validate canonical amendment *content*.  It does
not create directories, publish bytes, consume the single-use lease, launch
workers, or open target labels.  Publication remains an explicit preparation
step outside the scientific runner.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from ...protocol import ProtocolError
from .hashing import require_sha256
from .identity import (
    AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    DIRECT_INPUT_ARTIFACT_IDS,
    EXPERIMENT_ID,
    EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)


# Input #6 is a v3 resolution alias, but the amendment is a child of the
# immutable original ledger itself.  Keeping these identities distinct avoids
# accidentally making the consumer alias appear to be a new authority parent.
IMMUTABLE_PARENT_LEDGER_ARTIFACT_ID = (
    "midogpp_uniform_b_test_consumption_ledger_v1"
)
PARENT_LEDGER_MEMBER = "reports/test_consumption_ledger.json"
AMENDMENT_SCHEMA = "oe_ppur_v3_single_use_authorization_amendment_v1"
AMENDMENT_STATUS = "AUTHORIZED_SINGLE_USE_NOT_CONSUMED"

AMENDMENT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "amendment_artifact_id",
        "parent_artifact_id",
        "parent_member",
        "parent_sha256",
        "direct_original_parent_only",
        "consumer_experiment_id",
        "consumer_output_artifact_id",
        "authorized_consumer_experiment_ids",
        "consumer_count",
        "authorized_run_count",
        "authorization_scope",
        "authorization_basis",
        "execution_authorized",
        "consumed_test_reuse_authorized",
        "single_use_execution_identity",
        "authorization_exhausted",
        "source_contract_hash",
        "protocol_hash",
        "lifecycle_source_seal_sha256",
        "direct_input_artifact_ids",
        "test_manifest_sha256",
        "test_cache_content_sha256",
        "test_cache_row_order_sha256",
        "target_labels_open_only_after_durable_preterminal_attestation",
        "parsed_probability_matrix_science_receipt_required",
        "previous_stage90_outputs_used",
        "previous_stage90_amendments_used",
        "previous_stage90_run_state_or_scratch_used",
        "cross_run_recovery_allowed",
        "publication_status",
        "terminal_decision",
        "fresh_evidence",
        "may_feed_another_experiment",
    }
)


def build_authorization_amendment_payload(
    *,
    source_contract_hash: str,
    protocol_hash: str,
    lifecycle_source_seal_sha256: str,
) -> dict[str, object]:
    """Build the sole valid one-consumer, one-run amendment payload."""

    source = require_sha256(source_contract_hash, "source contract hash")
    protocol = require_sha256(protocol_hash, "protocol hash")
    lifecycle = require_sha256(
        lifecycle_source_seal_sha256,
        "lifecycle source seal",
    )
    if source == "0" * 64 or protocol == "0" * 64 or lifecycle == "0" * 64:
        raise ProtocolError(
            "OE-PPUR v3 authorization amendment hashes cannot be placeholders."
        )
    return {
        "schema_version": AMENDMENT_SCHEMA,
        "status": AMENDMENT_STATUS,
        "amendment_artifact_id": AUTHORIZATION_AMENDMENT_ARTIFACT_ID,
        "parent_artifact_id": IMMUTABLE_PARENT_LEDGER_ARTIFACT_ID,
        "parent_member": PARENT_LEDGER_MEMBER,
        "parent_sha256": EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256,
        "direct_original_parent_only": True,
        "consumer_experiment_id": EXPERIMENT_ID,
        "consumer_output_artifact_id": OUTPUT_ARTIFACT_ID,
        "authorized_consumer_experiment_ids": [EXPERIMENT_ID],
        "consumer_count": 1,
        "authorized_run_count": 1,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_basis": AUTHORIZATION_BASIS,
        "execution_authorized": True,
        "consumed_test_reuse_authorized": True,
        "single_use_execution_identity": True,
        "authorization_exhausted": False,
        "source_contract_hash": source,
        "protocol_hash": protocol,
        "lifecycle_source_seal_sha256": lifecycle,
        "direct_input_artifact_ids": list(DIRECT_INPUT_ARTIFACT_IDS),
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "test_cache_content_sha256": EXPECTED_TEST_CACHE_CONTENT_HASH,
        "test_cache_row_order_sha256": EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        "target_labels_open_only_after_durable_preterminal_attestation": True,
        "parsed_probability_matrix_science_receipt_required": True,
        "previous_stage90_outputs_used": False,
        "previous_stage90_amendments_used": False,
        "previous_stage90_run_state_or_scratch_used": False,
        "cross_run_recovery_allowed": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "may_feed_another_experiment": False,
    }


def validate_authorization_amendment_payload(
    payload: Mapping[str, object],
    *,
    source_contract_hash: str,
    protocol_hash: str,
    lifecycle_source_seal_sha256: str,
) -> dict[str, object]:
    """Exact-match externally supplied content against the canonical payload."""

    if not isinstance(payload, Mapping) or set(payload) != AMENDMENT_FIELDS:
        raise ProtocolError("OE-PPUR v3 authorization amendment schema drifted.")
    normalized = dict(payload)
    expected = build_authorization_amendment_payload(
        source_contract_hash=source_contract_hash,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=lifecycle_source_seal_sha256,
    )
    # bool is an int subclass; exact equality alone would admit True for a
    # declared count.  Preserve the explicit integer type boundary.
    if (
        type(normalized.get("consumer_count")) is not int
        or type(normalized.get("authorized_run_count")) is not int
        or normalized != expected
    ):
        raise ProtocolError("OE-PPUR v3 authorization amendment drifted.")
    return normalized


def authorization_amendment_bytes(
    *,
    source_contract_hash: str,
    protocol_hash: str,
    lifecycle_source_seal_sha256: str,
) -> bytes:
    """Serialize deterministic bytes for a separate no-overwrite publisher."""

    payload = build_authorization_amendment_payload(
        source_contract_hash=source_contract_hash,
        protocol_hash=protocol_hash,
        lifecycle_source_seal_sha256=lifecycle_source_seal_sha256,
    )
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_authorization_amendment_sha256(
    *,
    source_contract_hash: str,
    protocol_hash: str,
    lifecycle_source_seal_sha256: str,
) -> str:
    return hashlib.sha256(
        authorization_amendment_bytes(
            source_contract_hash=source_contract_hash,
            protocol_hash=protocol_hash,
            lifecycle_source_seal_sha256=lifecycle_source_seal_sha256,
        )
    ).hexdigest()


# Descriptive compatibility alias for early implementation callers.  New
# preparation code should use the concise public name requested above.
canonical_authorization_amendment_bytes = authorization_amendment_bytes


__all__ = (
    "AMENDMENT_FIELDS",
    "AMENDMENT_SCHEMA",
    "AMENDMENT_STATUS",
    "IMMUTABLE_PARENT_LEDGER_ARTIFACT_ID",
    "PARENT_LEDGER_MEMBER",
    "authorization_amendment_bytes",
    "build_authorization_amendment_payload",
    "canonical_authorization_amendment_bytes",
    "canonical_authorization_amendment_sha256",
    "validate_authorization_amendment_payload",
)
