"""Compact whole-artifact seal composer for OE-PPUR v4."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .artifact.contracts import (
    CompleteArtifactSealReceipt,
    COMPLETE_ARTIFACT_INDEX_MEMBER,
)
from .artifact.schema import (
    build_complete_index_payload,
    issue_complete_artifact_seal as _issue_complete_artifact_seal,
    validate_complete_index_schema as _validate_complete_index_schema,
)
from .artifact_io import (
    artifact_inventory_hash as _artifact_inventory_hash,
    catalog_member_hashes as _catalog_member_hashes,
    internal_member_hashes as _internal_member_hashes,
    safe_artifact_root as _safe_artifact_root,
    validate_exact_artifact_inventory as _validate_exact_artifact_inventory,
)
from .artifact.semantics import (
    _PreparedStateBinding,
    _SemanticReopenResult,
    _require_prepared_complete_state_type,
    _semantic_reopen_complete_artifact,
    _validate_committed_complete_state,
    _validate_prepared_complete_state_for_build,
)
from .output_persistence import (
    _fsync_directory,
    _read_json_object,
    _sha256_file,
    _write_json_exclusive,
)
from .output_validation import _issue_final_aggregate_bundle


def build_complete_artifact_seal(
    root: str | Path,
    *,
    expected_complete_state: object,
) -> CompleteArtifactSealReceipt:
    """Publish the complete index while durable state is COMPLETION_PENDING."""

    destination = _safe_artifact_root(root)
    _require_prepared_complete_state_type(expected_complete_state)
    _validate_exact_artifact_inventory(destination, index_present=False)
    prepared, final_bundle = _validate_prepared_complete_state_for_build(
        destination,
        expected_complete_state,
    )
    member_hashes = _catalog_member_hashes(
        destination,
        expected_complete_file_sha256=prepared.complete_file_sha256,
    )
    internal_hashes = _internal_member_hashes(destination)
    semantic = _semantic_reopen_complete_artifact(
        destination,
        complete_state_payload=prepared.complete_payload,
        final_bundle=final_bundle,
    )
    inventory_hash = _artifact_inventory_hash(member_hashes, internal_hashes)
    payload = build_complete_index_payload(
        prepared_state_hash=prepared.state_hash,
        prepared_state_receipt_hash=prepared.receipt_hash,
        final_bundle_receipt_hash=prepared.final_bundle_receipt_hash,
        source_seal_hash=semantic.source_seal_hash,
        semantic_validation_hash=semantic.semantic_validation_hash,
        catalog_member_sha256=member_hashes,
        internal_member_sha256=internal_hashes,
        artifact_inventory_hash=inventory_hash,
    )
    _write_json_exclusive(destination / COMPLETE_ARTIFACT_INDEX_MEMBER, payload)
    _fsync_directory(destination)
    _validate_exact_artifact_inventory(destination, index_present=True)

    if (
        _catalog_member_hashes(
            destination,
            expected_complete_file_sha256=prepared.complete_file_sha256,
        )
        != member_hashes
        or _internal_member_hashes(destination) != internal_hashes
        or _read_json_object(destination / COMPLETE_ARTIFACT_INDEX_MEMBER)
        != payload
    ):
        raise ProtocolError("OE-PPUR v4 complete artifact changed while sealing.")
    return _issue_complete_artifact_seal(
        destination,
        payload=payload,
        index_file_sha256=_sha256_file(
            destination / COMPLETE_ARTIFACT_INDEX_MEMBER
        ),
    )


def validate_complete_artifact_seal(
    root: str | Path,
    *,
    expected: CompleteArtifactSealReceipt,
    expected_complete_state: object | None = None,
) -> CompleteArtifactSealReceipt:
    """Reopen the pending virtual-COMPLETE or committed COMPLETE artifact."""

    if type(expected) is not CompleteArtifactSealReceipt:
        raise ProtocolError("OE-PPUR v4 complete artifact validation is untyped.")
    destination = _safe_artifact_root(root)
    _validate_exact_artifact_inventory(destination, index_present=True)
    if expected_complete_state is None:
        complete_payload = _validate_committed_complete_state(
            destination,
            expected_complete_state=None,
        )
        complete_file_sha256 = None
        final_bundle = _issue_final_aggregate_bundle(destination)
        prepared = None
    else:
        prepared, final_bundle = _validate_prepared_complete_state_for_build(
            destination,
            expected_complete_state,
        )
        complete_payload = prepared.complete_payload
        complete_file_sha256 = prepared.complete_file_sha256
    payload = _validate_complete_index_schema(
        _read_json_object(destination / COMPLETE_ARTIFACT_INDEX_MEMBER)
    )
    member_hashes = _catalog_member_hashes(
        destination,
        expected_complete_file_sha256=complete_file_sha256,
    )
    internal_hashes = _internal_member_hashes(destination)
    inventory_hash = _artifact_inventory_hash(member_hashes, internal_hashes)
    if (
        payload["prepared_state_hash"] != complete_payload["state_hash"]
        or (
            prepared is not None
            and (
                payload["prepared_state_receipt_hash"] != prepared.receipt_hash
                or payload["final_bundle_receipt_hash"]
                != prepared.final_bundle_receipt_hash
            )
        )
        or payload["catalog_member_sha256"] != member_hashes
        or payload["internal_member_sha256"] != internal_hashes
        or payload["artifact_inventory_hash"] != inventory_hash
    ):
        raise ProtocolError("OE-PPUR v4 complete artifact content hashes drifted.")

    semantic = _semantic_reopen_complete_artifact(
        destination,
        complete_state_payload=complete_payload,
        final_bundle=final_bundle,
    )
    if (
        payload["final_bundle_receipt_hash"] != final_bundle.receipt_hash
        or payload["source_seal_hash"] != semantic.source_seal_hash
        or payload["semantic_validation_hash"] != semantic.semantic_validation_hash
    ):
        raise ProtocolError("OE-PPUR v4 complete artifact semantic seal drifted.")
    observed = _issue_complete_artifact_seal(
        destination,
        payload=payload,
        index_file_sha256=_sha256_file(
            destination / COMPLETE_ARTIFACT_INDEX_MEMBER
        ),
    )
    if observed != expected:
        raise ProtocolError("OE-PPUR v4 complete artifact seal receipt drifted.")
    return observed


__all__ = (
    "CompleteArtifactSealReceipt",
    "build_complete_artifact_seal",
    "validate_complete_artifact_seal",
)
