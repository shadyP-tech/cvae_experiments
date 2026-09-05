"""Read-only admission checks and lease-bound scratch allocation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....protocol import ProtocolError
from ....runtime.artifact_io import atomic_json, read_json, sha256_file
from ....runtime.harp_v18_execution.action_capacity import (
    validate_action_capacity_certificate,
)
from ..config import HarpStage90V18Config
from ..source_train_label_access_fence import source_train_label_access_has_begun


def validate_preflight(value: Mapping[str, object]) -> None:
    """Require the exact workstation and numerical preflight contract."""

    expected = {
        "schema_version": "midogpp_harp_v18_workstation_preflight_v1",
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
        "source_train_target_classifier_task_count": 81,
        "total_classifier_fit_count": 810,
        "H_q_r_seven_expert_folds_used": False,
        "labels_consumed": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ProtocolError("HARP v18 workstation preflight contract drifted.")
    certificate = value.get("action_capacity_certificate")
    validate_action_capacity_certificate(certificate)
    receipt = value.get("physical_input_receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("labels_consumed") is not False
        or receipt.get("bank_independence_attestation_hash")
        != value.get("bank_independence_attestation_hash")
    ):
        raise ProtocolError("HARP v18 physical input preflight binding drifted.")


def exact_output_root(config: HarpStage90V18Config, value: str | Path) -> Path:
    """Resolve only the workspace-bound v18 output root."""

    text = str(value)
    if "://" in text:
        raise ProtocolError("HARP v18 runner requires a workspace-resolved output path.")
    root = Path(text).resolve()
    if "://" not in config.artifact_root and Path(config.artifact_root).resolve() != root:
        raise ProtocolError("HARP v18 CLI/config output roots differ.")
    return root


def validate_pristine_or_label_free_recovery(
    root: Path,
    *,
    admission_hash: str | None = None,
) -> str:
    """Admit a pristine snapshot or exact pre-label state for the same lease."""

    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v18 prepared output root is absent or unsafe.")
    if source_train_label_access_has_begun(root):
        raise ProtocolError(
            "HARP v18 label-free recovery is closed after the source-label fence."
        )
    administrative = {"config.resolved.yaml", "provenance/input_artifacts.json"}
    scientific: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ProtocolError("HARP v18 prepared output contains a symlink.")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in administrative:
                scientific.append(relative)
    if not scientific:
        return "PRISTINE"
    if admission_hash is None:
        raise ProtocolError("HARP v18 output contains prior scientific state.")
    allowed_files = {
        "manifests/admission.json",
        "manifests/protocol_manifest.json",
        "manifests/action_capacity_certificate.json",
        "manifests/label_free_progress_journal.json",
        "manifests/center_menu_root_binding.json",
        "manifests/source_train_menu_seals.json",
        "manifests/target_evaluation_menu_seals.json",
        "manifests/bank_independence_attestations.json",
    }
    allowed_prefixes = (
        "stores/physical_menu/",
        "stores/label_free_compatibility/",
        "manifests/source_target_role_seals/",
    )
    if any(
        relative not in allowed_files
        and not any(relative.startswith(prefix) for prefix in allowed_prefixes)
        for relative in scientific
    ):
        raise ProtocolError(
            "HARP v18 recovery is closed after a label capability or route state exists."
        )
    admission = read_json(root / "manifests/admission.json")
    if admission.get("admission_hash") != admission_hash:
        raise ProtocolError("HARP v18 recovery admission identity drifted.")
    capacity_path = root / "manifests/action_capacity_certificate.json"
    if capacity_path.exists():
        capacity = validate_action_capacity_certificate(read_json(capacity_path))
        if capacity.get("capacity_certificate_hash") != admission.get(
            "action_capacity_certificate_hash"
        ):
            raise ProtocolError(
                "HARP v18 recovery capacity certificate binding drifted."
            )
    journal_path = root / "manifests/label_free_progress_journal.json"
    if not journal_path.exists():
        if set(scientific).issubset(
            {
                "manifests/admission.json",
                "manifests/action_capacity_certificate.json",
                "manifests/protocol_manifest.json",
            }
        ):
            return "ACTIVE_LEASE_PREJOURNAL_RECOVERY"
        raise ProtocolError("HARP v18 recovery journal is absent.")
    if not capacity_path.exists():
        raise ProtocolError("HARP v18 recovery capacity certificate is absent.")
    journal = read_json(journal_path)
    if (
        journal.get("admission_hash") != admission_hash
        or journal.get("labels_available") is not False
    ):
        raise ProtocolError("HARP v18 recovery journal is not label-free.")
    return "LABEL_FREE_RECOVERY"


def assert_pristine_output(root: Path) -> None:
    """Backward-compatible strict inspection used by mutation-free dry runs."""

    validate_pristine_or_label_free_recovery(root)


def validate_parent_ledger(config: HarpStage90V18Config) -> str:
    """Validate the immutable predecessor evidence ledger by full-file SHA-256."""

    expected = config.expected_hashes["parent_ledger_sha256"]
    if type(expected) is not str:
        raise ProtocolError("HARP v18 parent ledger hash is absent.")
    path = config.resolved_path("parent_ledger_path")
    if sha256_file(path) != expected:
        raise ProtocolError("HARP v18 parent ledger bytes drifted.")
    read_json(path)
    return expected


def dedicated_scratch(
    config: HarpStage90V18Config,
    *,
    admission_hash: str,
    authorization_lease_hash: str,
    root: Path,
) -> Path:
    """Allocate or reopen only the scratch directory bound to this live lease."""

    configured = Path(str(config.runtime["scratch_root"]))
    if not configured.is_absolute() or configured == root or configured.is_symlink():
        raise ProtocolError("HARP v18 dedicated scratch root is unsafe.")
    scratch = configured / admission_hash[:20]
    receipt = scratch / "scratch_binding.json"
    binding = {
        "schema_version": "midogpp_harp_v18_scratch_binding_v1",
        "experiment_id": config.experiment_id,
        "execution_revision": config.execution_revision,
        "admission_hash": admission_hash,
        "authorization_lease_hash": authorization_lease_hash,
        "config_hash": config.config_hash,
        "process_independent_identity": True,
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
                "HARP v18 pre-existing scratch is not bound to this live lease."
            )
    else:
        configured.mkdir(parents=True, exist_ok=True)
        if configured.is_symlink() or not configured.is_dir():
            raise ProtocolError("HARP v18 dedicated scratch parent is unsafe.")
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
        raise ProtocolError("HARP v18 authorization provenance is untyped.")
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
    "validate_pristine_or_label_free_recovery",
    "validate_parent_ledger",
    "validate_preflight",
)
