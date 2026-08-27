"""Parser-only contract for a future OE-PPUR v2 authorization amendment.

The package deliberately contains no amendment factory and no authorized JSON
artifact.  This module can only validate bytes that were separately issued and
hash-pinned after explicit authorization.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Protocol

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
    ORIGINAL_PARENT_LEDGER_ARTIFACT_ID,
    OUTPUT_ARTIFACT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)


AMENDMENT_SCHEMA = "oe_ppur_v2_single_use_authorization_amendment_v1"
AMENDMENT_STATUS = "AUTHORIZED_SINGLE_USE_NOT_CONSUMED"
AMENDMENT_REQUIRED_FIELDS = frozenset(
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


class AuthorizationConfig(Protocol):
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: tuple[str, ...]
    source_contract_hash: str | None
    expected_authorization_amendment_sha256: str | None

    @property
    def execution_authorized(self) -> bool: ...

    @property
    def protocol(self) -> Mapping[str, object]: ...


def validate_authorization_amendment(
    payload: Mapping[str, object], *, config: AuthorizationConfig
) -> dict[str, object]:
    """Validate a v2-only, one-run, one-consumer direct-parent amendment."""

    if not isinstance(payload, Mapping) or set(payload) != AMENDMENT_REQUIRED_FIELDS:
        raise ProtocolError("OE-PPUR v2 authorization amendment schema drifted.")
    normalized = dict(payload)
    protocol_hash = require_sha256(
        config.protocol.get("protocol_hash"), "protocol hash"
    )
    source_hash = require_sha256(
        config.source_contract_hash, "source contract hash"
    )
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or tuple(config.input_artifact_ids) != DIRECT_INPUT_ARTIFACT_IDS
        or config.execution_authorized is not True
        or normalized.get("schema_version") != AMENDMENT_SCHEMA
        or normalized.get("status") != AMENDMENT_STATUS
        or normalized.get("amendment_artifact_id")
        != AUTHORIZATION_AMENDMENT_ARTIFACT_ID
        or normalized.get("parent_artifact_id")
        != ORIGINAL_PARENT_LEDGER_ARTIFACT_ID
        or normalized.get("parent_member")
        != "reports/test_consumption_ledger.json"
        or normalized.get("parent_sha256")
        != EXPECTED_ORIGINAL_PARENT_LEDGER_SHA256
        or normalized.get("direct_original_parent_only") is not True
        or normalized.get("consumer_experiment_id") != EXPERIMENT_ID
        or normalized.get("consumer_output_artifact_id") != OUTPUT_ARTIFACT_ID
        or normalized.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or normalized.get("consumer_count") != 1
        or isinstance(normalized.get("consumer_count"), bool)
        or normalized.get("authorized_run_count") != 1
        or isinstance(normalized.get("authorized_run_count"), bool)
        or normalized.get("authorization_scope") != AUTHORIZATION_SCOPE
        or normalized.get("authorization_basis") != AUTHORIZATION_BASIS
        or normalized.get("execution_authorized") is not True
        or normalized.get("consumed_test_reuse_authorized") is not True
        or normalized.get("single_use_execution_identity") is not True
        or normalized.get("authorization_exhausted") is not False
        or normalized.get("source_contract_hash") != source_hash
        or normalized.get("protocol_hash") != protocol_hash
        or normalized.get("direct_input_artifact_ids")
        != list(DIRECT_INPUT_ARTIFACT_IDS)
        or normalized.get("test_manifest_sha256")
        != EXPECTED_TEST_MANIFEST_SHA256
        or normalized.get("test_cache_content_sha256")
        != EXPECTED_TEST_CACHE_CONTENT_HASH
        or normalized.get("test_cache_row_order_sha256")
        != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or normalized.get(
            "target_labels_open_only_after_durable_preterminal_attestation"
        )
        is not True
        or normalized.get("parsed_probability_matrix_science_receipt_required")
        is not True
        or normalized.get("previous_stage90_outputs_used") is not False
        or normalized.get("previous_stage90_amendments_used") is not False
        or normalized.get("previous_stage90_run_state_or_scratch_used")
        is not False
        or normalized.get("cross_run_recovery_allowed") is not False
        or normalized.get("publication_status") != PUBLICATION_STATUS
        or normalized.get("terminal_decision") != TERMINAL_DECISION
        or normalized.get("fresh_evidence") is not False
        or normalized.get("may_feed_another_experiment") is not False
    ):
        raise ProtocolError("OE-PPUR v2 authorization amendment drifted.")
    return normalized


def load_and_validate_authorization_amendment(
    path: str | Path, *, config: AuthorizationConfig
) -> tuple[dict[str, object], str]:
    """Read immutable amendment bytes once and exact-match the config pin."""

    expected = require_sha256(
        config.expected_authorization_amendment_sha256,
        "expected authorization amendment hash",
    )
    raw = _read_regular_file_bytes(Path(path), maximum_bytes=1024 * 1024)
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise ProtocolError("OE-PPUR v2 authorization amendment bytes drifted.")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError("Cannot parse OE-PPUR v2 authorization amendment.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("OE-PPUR v2 authorization amendment is not an object.")
    return validate_authorization_amendment(payload, config=config), observed


def _unique_object(rows: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in rows:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_regular_file_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProtocolError(
            "OE-PPUR v2 authorization amendment is absent or unsafe."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ProtocolError(
                "OE-PPUR v2 authorization amendment is not a bounded regular file."
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise ProtocolError(
                    "OE-PPUR v2 authorization amendment is oversized."
                )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ProtocolError(
                "OE-PPUR v2 authorization amendment changed during admission."
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


__all__ = (
    "AMENDMENT_REQUIRED_FIELDS",
    "AMENDMENT_SCHEMA",
    "AMENDMENT_STATUS",
    "load_and_validate_authorization_amendment",
    "validate_authorization_amendment",
)
