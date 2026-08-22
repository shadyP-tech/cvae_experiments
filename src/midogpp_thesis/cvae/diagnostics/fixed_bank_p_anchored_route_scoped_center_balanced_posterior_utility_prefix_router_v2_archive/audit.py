"""Read-only orchestration for the exact failed CBPUPR v2 preterminal run."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ...protocol import ProtocolError
from . import contracts as archive_contracts
from .contracts import (
    AUDIT_SCHEMA,
    CLAIM_ROLE,
    CLAIM_SCOPE,
    CONFIG_CONTRACT_HASH,
    ELIGIBLE_NEXT_ACTION,
    EXPERIMENT_ID,
    FAILED_ERROR,
    FAILED_ERROR_CLASS,
    FAILED_PHASE,
    OUTPUT_ARTIFACT_ID,
    PROTOCOL_CONTRACT_HASH,
    PUBLICATION_STATUS,
    REPAIR_SOURCE_MANIFEST_SHA256,
    REPAIR_SOURCE_MEMBER_COUNT,
    REPAIR_SOURCE_TREE_SHA256,
    TERMINAL_DECISION,
    TERMINAL_JOURNAL_MEMBERS,
    V2_PRETERMINAL_ARTIFACT_DIRECTORIES,
    V2_PRETERMINAL_SCRATCH_DIRECTORIES,
)
from .hashing import canonical_hash
from .identity import (
    FORBIDDEN_CLAIM_FLAGS,
    validate_config,
    validate_failed_state,
    validate_preflight,
    validate_protocol_manifest,
    validate_provenance,
)
from .physical import validate_label_free_physical_graph
from .tree import (
    audit_exact_tree,
    exclusive_existing_run_lock,
    logical_path,
    validate_scratch_duplicates,
)


# Shared with the small synthetic fixture; production still uses the frozen
# contract and full claim boundary.
_FORBIDDEN_CLAIM_FLAGS = FORBIDDEN_CLAIM_FLAGS


def audit_failed_v2_preterminal_for_archive(
    root: Path,
    *,
    scratch_root: Path | None,
) -> Mapping[str, object]:
    """Audit a failure root; ``None`` is the only no-scratch API state."""

    logical_root = logical_path(root, role="artifact")
    if not logical_root.is_dir():
        raise ProtocolError("CBPUPR v2 preterminal archive root is absent or unsafe.")
    logical_scratch = (
        None
        if scratch_root is None
        else logical_path(scratch_root, role="scratch")
    )
    if logical_scratch is not None and not logical_scratch.is_dir():
        raise ProtocolError(
            "CBPUPR v2 preterminal supplied scratch is absent or unsafe."
        )
    with exclusive_existing_run_lock(logical_root):
        return audit_locked_failed_v2_preterminal(
            logical_root=logical_root,
            observed_root=logical_root,
            logical_scratch=logical_scratch,
            observed_scratch=logical_scratch,
        )


def audit_locked_failed_v2_preterminal(
    *,
    logical_root: Path,
    observed_root: Path,
    logical_scratch: Path | None,
    observed_scratch: Path | None,
) -> Mapping[str, object]:
    contract = archive_contracts.CANONICAL_ARCHIVE_CONTRACT
    artifact_rows = audit_exact_tree(
        observed_root,
        role="artifact",
        expected_directories=V2_PRETERMINAL_ARTIFACT_DIRECTORIES,
        expected_members=contract.artifact_members,
    )
    if TERMINAL_JOURNAL_MEMBERS & {str(row["path"]) for row in artifact_rows}:
        raise ProtocolError("CBPUPR v2 preterminal terminal journal is present.")

    state = validate_failed_state(observed_root)
    validate_config(
        observed_root,
        logical_root=logical_root,
        logical_scratch=logical_scratch,
    )
    provenance, provenance_hashes = validate_provenance(observed_root)
    protocol = validate_protocol_manifest(
        observed_root,
        provenance_hashes=provenance_hashes,
        expected_input_hashes=dict(contract.input_artifact_hashes),
    )
    graph = validate_label_free_physical_graph(observed_root)
    validate_preflight(observed_root)

    if observed_scratch is None:
        if logical_scratch is not None:
            raise ProtocolError(
                "CBPUPR v2 preterminal supplied scratch disappeared during audit."
            )
        scratch_state = "EXPLICIT_NO_SCRATCH_API_STATE"
        scratch_rows: list[dict[str, object]] = []
        scratch_directories: list[str] = []
        scratch_verified = False
    else:
        if logical_scratch is None:
            raise ProtocolError("CBPUPR v2 preterminal scratch binding drifted.")
        scratch_rows = audit_exact_tree(
            observed_scratch,
            role="scratch",
            expected_directories=V2_PRETERMINAL_SCRATCH_DIRECTORIES,
            expected_members=contract.scratch_members,
        )
        validate_scratch_duplicates(
            artifact_rows=artifact_rows,
            scratch_rows=scratch_rows,
        )
        scratch_state = "EXACT_OBSERVED_FULL_SCRATCH"
        scratch_directories = sorted(V2_PRETERMINAL_SCRATCH_DIRECTORIES)
        scratch_verified = True

    payload: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA,
        "status": "PASS",
        "archive_contract_hash": contract.contract_hash,
        "source_root": str(logical_root),
        "source_scratch_root": (
            str(logical_scratch) if logical_scratch is not None else None
        ),
        "source_experiment_id": EXPERIMENT_ID,
        "source_output_artifact_id": OUTPUT_ARTIFACT_ID,
        "source_run_status": "FAILED",
        "source_run_phase": FAILED_PHASE,
        "source_error": FAILED_ERROR,
        "source_error_class": FAILED_ERROR_CLASS,
        "source_run_updated_at_utc": state["updated_at_utc"],
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "claim_scope": CLAIM_SCOPE,
        "claim_role": CLAIM_ROLE,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "config_contract_hash": CONFIG_CONTRACT_HASH,
        "protocol_contract_hash": PROTOCOL_CONTRACT_HASH,
        "protocol_manifest_hash": protocol["protocol_manifest_hash"],
        "input_artifact_hashes": dict(contract.input_artifact_hashes),
        "provenance_input_count": len(provenance),
        "repair_source_manifest_sha256": REPAIR_SOURCE_MANIFEST_SHA256,
        "repair_source_tree_sha256": REPAIR_SOURCE_TREE_SHA256,
        "repair_source_member_count": REPAIR_SOURCE_MEMBER_COUNT,
        "physical_surface_hash": graph["physical_surface_hash"],
        "physical_surface_seal_hash": graph["physical_surface_seal_hash"],
        "global_prediction_seal_hash": graph["global_prediction_seal_hash"],
        "source_stream_lock_hash": graph["source_stream_lock_hash"],
        "physical_probability_surface_complete": True,
        "route_endpoint_products_persisted": False,
        "durable_preterminal_barrier_persisted": False,
        "terminal_access_journal_status": "ABSENT_NOT_OPENED",
        "terminal_labels_opened": False,
        "terminal_products_persisted": False,
        "artifact_directories": sorted(V2_PRETERMINAL_ARTIFACT_DIRECTORIES),
        "artifact_members": artifact_rows,
        "scratch_state": scratch_state,
        "scratch_verified": scratch_verified,
        "scratch_directories": scratch_directories,
        "scratch_members": scratch_rows,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "quarantined_bytes_may_feed_rerun": False,
        "quarantined_bytes_may_feed_successor": False,
        "v2_rerun_authorized": False,
        "v3_input_reuse_authorized": False,
        "routing_success_claim_authorized": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
        "eligible_next_action": ELIGIBLE_NEXT_ACTION,
    }
    return {**payload, "archive_audit_hash": canonical_hash(payload)}


__all__ = (
    "audit_failed_v2_preterminal_for_archive",
    "audit_locked_failed_v2_preterminal",
)
