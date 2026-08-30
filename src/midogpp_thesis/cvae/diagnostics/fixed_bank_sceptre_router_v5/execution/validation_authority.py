"""Authorization and workspace-provenance reconstruction for SCEPTRE v5."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ...fixed_bank_sceptre_router.hashing import canonical_hash
from ..experiment_contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    INPUT_ARTIFACT_IDS,
)
from .authorization_lease import validate_authorization_lease_payload
from .validation_io import read_validation_object
from .worker_runtime import validate_worker_runtime_smoke


def validate_preterminal_authority(
    destination: Path,
    bundle: Mapping[str, object],
    config: object,
) -> None:
    """Replay the exact-eight input, admission, lease, and runtime bindings."""

    admission = bundle.get("admission")
    input_binding = bundle.get("input_binding")
    lease = bundle.get("authorization_lease")
    runtime = bundle.get("runtime")
    if not all(
        isinstance(value, Mapping)
        for value in (admission, input_binding, lease, runtime)
    ):
        raise ProtocolError("SCEPTRE v5 preterminal authority graph is malformed.")
    validate_authorization_lease_payload(lease, expected_status="CLAIMED_IN_PROGRESS")
    admission_body = {
        key: value for key, value in admission.items() if key != "admission_hash"
    }
    expected_input_keys = {
        "schema_version",
        "config_hash",
        "bank_lock_hash",
        "cache_binding_hash",
        "test_cache_content_hash",
        "test_cache_row_order_hash",
        "generation_lock_hash",
        "source_inner_amendment_sha256",
        "execution_amendment_sha256",
        "parent_ledger_sha256",
        "manifest_sha256",
        "input_artifact_ids",
        "target_labels_opened",
    }
    preflight = runtime.get("workstation_preflight")
    smoke = runtime.get("worker_runtime_smoke")
    if not isinstance(preflight, Mapping) or not isinstance(smoke, Mapping):
        raise ProtocolError("SCEPTRE v5 persisted runtime admission is malformed.")
    validate_worker_runtime_smoke(smoke)
    provenance = read_validation_object(
        destination / "provenance/input_artifacts.json"
    )
    raw_rows = provenance.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("SCEPTRE v5 persisted workspace provenance is malformed.")
    rows_by_id = {str(row.get("artifact_id")): row for row in raw_rows}
    workspace_provenance = {
        artifact_id: rows_by_id[artifact_id]
        for artifact_id in INPUT_ARTIFACT_IDS
        if artifact_id in rows_by_id
    }
    workspace_binding = {
        "status": "PASS",
        "experiment_id": getattr(config, "experiment_id"),
        "output_artifact_id": getattr(config, "output_artifact_id"),
        "input_artifact_count": 8,
        "source_snapshot_bound": True,
    }
    if (
        set(input_binding) != expected_input_keys
        or input_binding.get("schema_version")
        != "sceptre_v5_admitted_input_binding_v1"
        or admission.get("admission_hash") != canonical_hash(admission_body)
        or admission.get("input_binding_hash") != canonical_hash(input_binding)
        or lease.get("admission_hash") != admission.get("admission_hash")
        or bundle.get("config_hash") != input_binding.get("config_hash")
        or bundle.get("config_hash") != getattr(config, "config_hash")
        or admission.get("artifact_root") != str(destination)
        or admission.get("workspace_binding_hash")
        != canonical_hash(workspace_binding)
        or len(workspace_provenance) != 8
        or admission.get("workspace_provenance_hash")
        != canonical_hash(workspace_provenance)
        or admission.get("workstation_preflight_hash")
        != canonical_hash(preflight)
        or admission.get("worker_runtime_smoke_hash")
        != smoke.get("worker_runtime_smoke_hash")
        or input_binding.get("input_artifact_ids") != list(INPUT_ARTIFACT_IDS)
        or input_binding.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
        or input_binding.get("generation_lock_hash")
        != EXPECTED_GENERATION_LOCK_HASH
        or input_binding.get("test_cache_content_hash")
        != EXPECTED_TEST_CACHE_CONTENT_HASH
        or input_binding.get("test_cache_row_order_hash")
        != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or input_binding.get("source_inner_amendment_sha256")
        != EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
        or input_binding.get("parent_ledger_sha256")
        != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or input_binding.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or admission.get("execution_amendment_sha256")
        != input_binding.get("execution_amendment_sha256")
        or input_binding.get("execution_amendment_sha256")
        != getattr(config, "expected_execution_amendment_sha256")
        or admission.get("source_snapshot_manifest_sha256")
        != getattr(config, "expected_source_snapshot_manifest_sha256")
        or admission.get("source_snapshot_tree_sha256")
        != getattr(config, "expected_source_snapshot_tree_sha256")
        or admission.get("target_labels_opened") is not False
        or admission.get("filesystem_mutations") != 0
        or input_binding.get("target_labels_opened") is not False
    ):
        raise ProtocolError("SCEPTRE v5 preterminal authority lineage drifted.")


__all__ = ("validate_preterminal_authority",)
