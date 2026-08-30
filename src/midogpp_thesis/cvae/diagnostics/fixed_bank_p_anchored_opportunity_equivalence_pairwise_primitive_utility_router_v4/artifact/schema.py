"""Pure complete-index schema validation for OE-PPUR v4.

No persistence writer is imported here.  This keeps the index grammar and its
typed receipt below the lifecycle composition layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ....protocol import ProtocolError
from ..hashing import canonical_hash, require_sha256
from ..identity import OUTPUT_ARTIFACT_ID
from .contracts import (
    COMPLETE_ARTIFACT_INDEX_MEMBER,
    COMPLETE_CATALOG_MEMBERS,
    COMPLETE_INTERNAL_MEMBERS,
    CompleteArtifactSealReceipt,
    _issue_complete_artifact_seal,
)


def build_complete_index_payload(
    *,
    prepared_state_hash: str,
    prepared_state_receipt_hash: str,
    final_bundle_receipt_hash: str,
    source_seal_hash: str,
    semantic_validation_hash: str,
    catalog_member_sha256: Mapping[str, str],
    internal_member_sha256: Mapping[str, str],
    artifact_inventory_hash: str,
) -> dict[str, object]:
    body = {
        "schema_version": "oe_ppur_v4_complete_artifact_index_v1",
        "output_artifact_id": OUTPUT_ARTIFACT_ID,
        "prepared_state_hash": prepared_state_hash,
        "prepared_state_receipt_hash": prepared_state_receipt_hash,
        "final_bundle_receipt_hash": final_bundle_receipt_hash,
        "source_seal_hash": source_seal_hash,
        "semantic_validation_hash": semantic_validation_hash,
        "catalog_members": list(COMPLETE_CATALOG_MEMBERS),
        "catalog_member_sha256": dict(catalog_member_sha256),
        "self_index_member": COMPLETE_ARTIFACT_INDEX_MEMBER,
        "self_index_excluded_from_member_hashes": True,
        "internal_member_sha256": dict(internal_member_sha256),
        "artifact_inventory_hash": artifact_inventory_hash,
        "run_state_hash_uses_prepared_complete_bytes": True,
        "run_state_was_completion_pending_at_seal": True,
        "post_commit_revalidation_required": True,
        "raw_labels_persisted": False,
        "cross_run_recovery_allowed": False,
    }
    return {**body, "complete_artifact_index_hash": canonical_hash(body)}


def validate_complete_index_schema(
    payload: Mapping[str, object],
) -> dict[str, object]:
    expected_keys = {
        "schema_version",
        "output_artifact_id",
        "prepared_state_hash",
        "prepared_state_receipt_hash",
        "final_bundle_receipt_hash",
        "source_seal_hash",
        "semantic_validation_hash",
        "catalog_members",
        "catalog_member_sha256",
        "self_index_member",
        "self_index_excluded_from_member_hashes",
        "internal_member_sha256",
        "artifact_inventory_hash",
        "run_state_hash_uses_prepared_complete_bytes",
        "run_state_was_completion_pending_at_seal",
        "post_commit_revalidation_required",
        "raw_labels_persisted",
        "cross_run_recovery_allowed",
        "complete_artifact_index_hash",
    }
    body = {
        key: value
        for key, value in payload.items()
        if key != "complete_artifact_index_hash"
    }
    member_hashes = payload.get("catalog_member_sha256")
    internal_hashes = payload.get("internal_member_sha256")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version")
        != "oe_ppur_v4_complete_artifact_index_v1"
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("catalog_members") != list(COMPLETE_CATALOG_MEMBERS)
        or not isinstance(member_hashes, Mapping)
        or set(member_hashes)
        != {
            member
            for member in COMPLETE_CATALOG_MEMBERS
            if member != COMPLETE_ARTIFACT_INDEX_MEMBER
        }
        or not isinstance(internal_hashes, Mapping)
        or set(internal_hashes) != set(COMPLETE_INTERNAL_MEMBERS)
        or payload.get("self_index_member") != COMPLETE_ARTIFACT_INDEX_MEMBER
        or payload.get("self_index_excluded_from_member_hashes") is not True
        or payload.get("run_state_hash_uses_prepared_complete_bytes") is not True
        or payload.get("run_state_was_completion_pending_at_seal") is not True
        or payload.get("post_commit_revalidation_required") is not True
        or payload.get("raw_labels_persisted") is not False
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("complete_artifact_index_hash") != canonical_hash(body)
    ):
        raise ProtocolError("OE-PPUR v4 complete artifact index schema drifted.")
    for role in (
        "prepared_state_hash",
        "prepared_state_receipt_hash",
        "final_bundle_receipt_hash",
        "source_seal_hash",
        "semantic_validation_hash",
        "artifact_inventory_hash",
        "complete_artifact_index_hash",
    ):
        require_sha256(payload.get(role), role.replace("_", " "))
    for member, digest in (*member_hashes.items(), *internal_hashes.items()):
        require_sha256(digest, f"{member} content hash")
    return dict(payload)


def issue_complete_artifact_seal(
    root: Path,
    *,
    payload: Mapping[str, object],
    index_file_sha256: str,
) -> CompleteArtifactSealReceipt:
    index = validate_complete_index_schema(payload)
    return _issue_complete_artifact_seal(
        artifact_root=root,
        prepared_state_hash=str(index["prepared_state_hash"]),
        prepared_state_receipt_hash=str(index["prepared_state_receipt_hash"]),
        final_bundle_receipt_hash=str(index["final_bundle_receipt_hash"]),
        artifact_inventory_hash=str(index["artifact_inventory_hash"]),
        complete_artifact_index_hash=str(index["complete_artifact_index_hash"]),
        complete_artifact_index_file_sha256=index_file_sha256,
        semantic_validation_hash=str(index["semantic_validation_hash"]),
        source_seal_hash=str(index["source_seal_hash"]),
    )


__all__ = (
    "build_complete_index_payload",
    "issue_complete_artifact_seal",
    "validate_complete_index_schema",
)
