"""Read-only admission checks and lease-bound scratch allocation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json, sha256_file
from ..config import HarpStage90V4Config


def validate_preflight(value: Mapping[str, object]) -> None:
    """Require the exact workstation and numerical preflight contract."""

    expected = {
        "status": "PASS",
        "persistent_gpu_workers": 2,
        "gpu_devices": ["cuda:0", "cuda:1"],
        "probability_transport_dtype": "float32",
        "scientific_reduction_dtype": "float64",
        "physical_expert_weight": 1.0,
        "tf32_enabled": False,
        "amp_enabled": False,
        "parent_cuda_context_created": False,
        "shared_validated_menu_index": True,
        "labels_consumed": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ProtocolError("HARP v4 workstation preflight contract drifted.")


def exact_output_root(config: HarpStage90V4Config, value: str | Path) -> Path:
    """Resolve only the workspace-bound v4 output root."""

    text = str(value)
    if "://" in text:
        raise ProtocolError("HARP v4 runner requires a workspace-resolved output path.")
    root = Path(text).resolve()
    if "://" not in config.artifact_root and Path(config.artifact_root).resolve() != root:
        raise ProtocolError("HARP v4 CLI/config output roots differ.")
    return root


def assert_pristine_output(root: Path) -> None:
    """Reject predecessor or prior scientific state before lease claim."""

    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v4 prepared output root is absent or unsafe.")
    allowed = {"config.resolved.yaml", "provenance/input_artifacts.json"}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("HARP v4 prepared output contains a symlink.")
        if path.is_file() and path.relative_to(root).as_posix() not in allowed:
            raise ProtocolError("HARP v4 output contains prior scientific state.")


def validate_parent_ledger(config: HarpStage90V4Config) -> str:
    """Validate the immutable predecessor evidence ledger by full-file SHA-256."""

    expected = config.expected_hashes["parent_ledger_sha256"]
    if type(expected) is not str:
        raise ProtocolError("HARP v4 parent ledger hash is absent.")
    path = config.resolved_path("parent_ledger_path")
    if sha256_file(path) != expected:
        raise ProtocolError("HARP v4 parent ledger bytes drifted.")
    read_json(path)
    return expected


def dedicated_scratch(
    config: HarpStage90V4Config,
    *,
    admission_hash: str,
    authorization_lease_hash: str,
    root: Path,
) -> Path:
    """Allocate or reopen only the scratch directory bound to this live lease."""

    configured = Path(str(config.runtime["scratch_root"]))
    if not configured.is_absolute() or configured == root or configured.is_symlink():
        raise ProtocolError("HARP v4 dedicated scratch root is unsafe.")
    scratch = configured / admission_hash[:20]
    receipt = scratch / "scratch_binding.json"
    binding = {
        "schema_version": "midogpp_harp_v4_scratch_binding_v1",
        "admission_hash": admission_hash,
        "authorization_lease_hash": authorization_lease_hash,
        "process_id": os.getpid(),
        "config_hash": config.config_hash,
        "label_free_resumption_only": True,
        "development_or_evaluation_labels_stored": False,
    }
    if scratch.exists() or scratch.is_symlink():
        if (
            scratch.is_symlink()
            or not scratch.is_dir()
            or not receipt.is_file()
            or receipt.is_symlink()
            or read_json(receipt) != binding
        ):
            raise ProtocolError(
                "HARP v4 pre-existing scratch is not bound to this live lease."
            )
    else:
        configured.mkdir(parents=True, exist_ok=True)
        if configured.is_symlink() or not configured.is_dir():
            raise ProtocolError("HARP v4 dedicated scratch parent is unsafe.")
        scratch.mkdir(mode=0o700, parents=False, exist_ok=False)
        atomic_json(receipt, binding)
    return scratch


def authorization_provenance(value: object) -> dict[str, object]:
    """Project typed activation provenance into admission and dry-run reports."""

    required = (
        "amendment_sha256",
        "amendment_hash",
        "input_binding_hash",
        "scientific_contract_hash",
        "workspace_registration_execution_contract_hash",
        "source_snapshot_schema",
        "source_snapshot_manifest_sha256",
        "source_snapshot_tree_sha256",
        "source_snapshot_member_count",
    )
    if any(not hasattr(value, name) for name in required):
        raise ProtocolError("HARP v4 authorization provenance is untyped.")
    return {
        "execution_amendment_sha256": value.amendment_sha256,
        "execution_amendment_hash": value.amendment_hash,
        "authorized_input_binding_hash": value.input_binding_hash,
        "scientific_contract_hash": value.scientific_contract_hash,
        "workspace_registration_execution_contract_hash": (
            value.workspace_registration_execution_contract_hash
        ),
        "source_snapshot_schema": value.source_snapshot_schema,
        "source_snapshot_manifest_sha256": value.source_snapshot_manifest_sha256,
        "source_snapshot_tree_sha256": value.source_snapshot_tree_sha256,
        "source_snapshot_member_count": value.source_snapshot_member_count,
    }


__all__ = (
    "assert_pristine_output",
    "authorization_provenance",
    "dedicated_scratch",
    "exact_output_root",
    "validate_parent_ledger",
    "validate_preflight",
)
