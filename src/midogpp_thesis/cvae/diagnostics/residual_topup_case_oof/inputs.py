"""Narrow adapters for validated locks, label-free rows, and pre-GPU admission."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...expert_bank.uniform_b_v2_promotion import (
    load_promotion_config,
    validate_promoted_bank,
)
from ...protocol import ProtocolError
from ....workspace.runtime import MidogppWorkspace
from ..residual_topup_router.partitions import (
    LabelFreeValidationFrame,
    PartitionSurface,
    SUPPORT_PARTITION_COLUMNS,
    ValidatedLocks,
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_workspace_provenance,
)
from .artifact_io import read_json, sha256_file
from .config import EXPECTED_MANIFEST_SHA256


def validate_pre_gpu_firewall(
    config: object,
    frame: LabelFreeValidationFrame,
) -> Mapping[str, object]:
    """Bind source-only bank evidence and the val-only cache before CUDA work."""

    bank_root = Path(getattr(config, "expert_bank_root"))
    promotion_config = load_promotion_config(bank_root / "config.resolved.yaml")
    bank_checks = validate_promoted_bank(
        bank_root, config=promotion_config, allow_pending=False
    )
    bank_index = read_json(bank_root / "manifests/expert_bank_index.json")
    leakage = read_json(bank_root / "reports/leakage_report.json")
    source_evidence = read_json(
        bank_root / "manifests/source_evidence_lock.json"
    )
    records = bank_index.get("records")
    binding = dict(getattr(frame, "cache_binding", {}))
    manifest_path = Path(getattr(config, "validation_manifest_path"))
    if (
        bank_checks.get("status") != "PASS"
        or bank_checks.get("all_experts_source_only") is not True
        or not isinstance(records, list)
        or len(records) != 27
        or any(
            not isinstance(row, Mapping)
            or row.get("fresh_source_only_training") is not True
            or row.get("parent_checkpoint_used") is not False
            for row in records
        )
        or leakage.get("status") != "PASS"
        or leakage.get("fresh_source_only_training") is not True
        or int(leakage.get("identity_overlap_failures", -1)) != 0
        or int(source_evidence.get("identity_overlap_failures", -1)) != 0
        or binding.get("validation_split") != "val"
        or binding.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or binding.get("labels_persisted") is not False
        or binding.get("manifest_opened") is not False
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256
    ):
        raise ProtocolError("Case-OOF pre-GPU data firewall failed.")
    return {
        "status": "PASS",
        "bank_validation_status": "PASS",
        "bank_lock_hash": str(bank_index.get("bank_lock_hash")),
        "expert_count": len(records),
        "fresh_source_only_training": True,
        "parent_checkpoint_used": False,
        "bank_identity_overlap_failures": 0,
        "validation_cache_split": "val",
        "validation_manifest_split": "val",
        "validation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "validation_cache_label_fields_absent": True,
        "direct_source_train_vs_validation_identity_rows_available_in_promoted_bank": False,
        "source_vs_validation_overlap_basis": (
            "inherited_split_contract_attestation_source_only_training_"
            "versus_hash_bound_validation_split_val"
        ),
        "attestation_scope": (
            "bank_internal_zero_identity_overlap_plus_distinct_source_train_"
            "and_validation_val_split_contract_not_a_new_cross_artifact_row_audit"
        ),
        "gpu_work_authorized": True,
    }


def validate_active_diagnostic_workspace_binding(config: object) -> Mapping[str, object]:
    """Fail closed unless the canonical registry authorizes this diagnostic."""

    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(str(getattr(config, "experiment_id")))
        output = workspace.artifacts[str(getattr(config, "output_artifact_id"))]
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("Case-OOF canonical workspace binding failed.") from exc
    if (
        experiment.status != "diagnostic"
        or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only"
        or experiment.output_artifact_id != getattr(config, "output_artifact_id")
        or experiment.input_artifact_ids != getattr(config, "input_artifact_ids")
        or output.stage != "90_oracles_and_diagnostics"
        or output.claim_scope != "diagnostic_only"
    ):
        raise ProtocolError("Case-OOF experiment is not the active diagnostic binding.")
    return {
        "status": "PASS",
        "registry_status": experiment.status,
        "experiment_id": experiment.experiment_id,
        "output_artifact_id": experiment.output_artifact_id,
        "stage": experiment.stage,
        "claim_scope": experiment.claim_scope,
    }


__all__ = (
    "LabelFreeValidationFrame",
    "PartitionSurface",
    "SUPPORT_PARTITION_COLUMNS",
    "ValidatedLocks",
    "build_partition_surface",
    "load_label_free_validation_frame",
    "load_validated_locks",
    "validate_pre_gpu_firewall",
    "validate_active_diagnostic_workspace_binding",
    "validate_workspace_provenance",
)
