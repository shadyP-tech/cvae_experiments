"""Source-inner artifact writer and filesystem-level validator."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ...real_features.classifier_reference.protocol import ProtocolError
from ..reporting import prepare_artifact_dirs, write_csv_rows, write_json
from .prior_recovery_artifact_shared import (
    _assert_columns,
    _read_csv,
    _read_json,
    _require_files,
    _validate_workspace_provenance,
)
from .prior_recovery_common import selection_evidence_hash
from .prior_recovery_config import PriorRecoveryConfig
from .prior_recovery_provenance import validate_provenance_indices
from .prior_recovery_runtime_cache import validate_feature_frame_index
from .prior_recovery_timing import validate_runtime_reports, write_run_state
from .prior_recovery_schema import (
    SOURCE_CHECKPOINT_AUDIT_COLUMNS,
    SOURCE_INNER_METRIC_COLUMNS,
)
from .prior_recovery_source_validation import (
    derive_source_checkpoint_audits,
    validate_source_inner_evidence_view,
    validate_source_protocol,
)
from .source_inner_selection import (
    RecipeLock,
    load_recipe_lock,
    write_recipe_lock,
)


def write_source_inner_bundle(
    root: Path,
    *,
    metric_rows: Sequence[Mapping[str, object]],
    nested_reference_rows: Sequence[Mapping[str, object]],
    nested_tuning_rows: Sequence[Mapping[str, object]],
    sampler_rows: Sequence[Mapping[str, object]],
    identity_audit_rows: Sequence[Mapping[str, object]],
    locks: Sequence[RecipeLock],
    protocol_manifest: Mapping[str, object],
    selection_bundle_hash: str,
) -> Path:
    root = prepare_artifact_dirs(root)
    checkpoint_index = _read_json(root / "manifests/checkpoint_index.json")
    checkpoint_audit_rows = derive_source_checkpoint_audits(
        metric_rows,
        checkpoint_index=checkpoint_index,
    )
    write_csv_rows(
        root / "tables/source_inner_metrics.csv",
        metric_rows,
        SOURCE_INNER_METRIC_COLUMNS,
    )
    write_csv_rows(root / "tables/nested_real_references.csv", nested_reference_rows)
    write_csv_rows(root / "tables/nested_classifier_tuning.csv", nested_tuning_rows)
    write_csv_rows(root / "tables/sampler_realizations.csv", sampler_rows)
    write_csv_rows(root / "tables/identity_overlap_audit.csv", identity_audit_rows)
    write_csv_rows(
        root / "tables/checkpoint_reuse_audit.csv",
        checkpoint_audit_rows,
        SOURCE_CHECKPOINT_AUDIT_COLUMNS,
    )
    write_json(root / "manifests/protocol_manifest.json", dict(protocol_manifest))
    write_json(
        root / "manifests/selection_evidence_manifest.json",
        {
            "schema_version": "midogpp_prior_recovery_selection_evidence_v1",
            "selection_bundle_hash": selection_bundle_hash,
        },
    )
    for lock in locks:
        write_recipe_lock(
            root / f"manifests/recipe_locks/{lock.outer_target_center}.json",
            lock,
        )
    valid = [lock for lock in locks if lock.status == "VALID"]
    conditional = [lock for lock in valid if lock.primary_arm in {"C", "D"}]
    factorial_triggered = (
        len(valid) == len(locks)
        and len(conditional) == len(locks)
        and bool(locks)
    )
    gate = {
        "status": (
            "FACTORIAL_TRIGGERED"
            if factorial_triggered
            else (
                "NEGATIVE_GATE_COMPLETE"
                if len(valid) == len(locks)
                else "INVALID_LOCKS_PRESENT"
            )
        ),
        "n_locks": len(locks),
        "n_valid_locks": len(valid),
        "n_conditional_locks": len(conditional),
        "factorial_triggered": factorial_triggered,
        "invalid_centers": [
            lock.outer_target_center for lock in locks if lock.status != "VALID"
        ],
        "outer_scoring_used": False,
        "selection_bundle_hash": selection_bundle_hash,
    }
    identity_pass = all(row.get("status") == "PASS" for row in identity_audit_rows)
    leakage = {
        "status": "PASS" if len(locks) == len(valid) and identity_pass else "FAIL",
        "outer_target_rows_passed_to_training_or_selection": False,
        "outer_target_labels_used_for_selection": False,
        "target_eval_labels_used_for_selection": False,
        "center_4_excluded": True,
        "identity_overlap_status": "PASS" if identity_pass else "FAIL",
        "routing_performed": False,
        "composition_performed": False,
        "selection_bundle_hash": selection_bundle_hash,
    }
    write_json(root / "reports/gate_decision.json", gate)
    write_json(root / "reports/leakage_report.json", leakage)
    write_run_state(
        root,
        protocol_hash=str(protocol_manifest["protocol_hash"]),
        mode="source_inner",
        status="COMPLETE",
    )
    try:
        validate_source_inner_bundle(root)
    except Exception:
        write_run_state(
            root,
            protocol_hash=str(protocol_manifest["protocol_hash"]),
            mode="source_inner",
            status="FAILED",
        )
        raise
    return root


def validate_source_inner_bundle(
    root: Path,
    *,
    expected_config: PriorRecoveryConfig | None = None,
    require_factorial: bool = False,
) -> dict[str, RecipeLock]:
    root = Path(root)
    required = (
        "tables/source_inner_metrics.csv",
        "tables/nested_real_references.csv",
        "tables/nested_classifier_tuning.csv",
        "tables/sampler_realizations.csv",
        "tables/identity_overlap_audit.csv",
        "tables/checkpoint_reuse_audit.csv",
        "manifests/protocol_manifest.json",
        "manifests/selection_evidence_manifest.json",
        "manifests/checkpoint_index.json",
        "manifests/task_fisher_index.json",
        "manifests/feature_frame_index.json",
        "reports/gate_decision.json",
        "reports/leakage_report.json",
        "tables/runtime_timings.csv",
        "reports/runtime_summary.json",
        "reports/run_state.json",
    )
    _require_files(root, required)
    protocol = _read_json(root / "manifests/protocol_manifest.json")
    _validate_workspace_provenance(root, protocol=protocol, mode="source_inner")
    evidence = _read_json(root / "manifests/selection_evidence_manifest.json")
    gate = _read_json(root / "reports/gate_decision.json")
    leakage = _read_json(root / "reports/leakage_report.json")
    checkpoint_index, fisher_index = validate_provenance_indices(root)
    rows = _read_csv(root / "tables/source_inner_metrics.csv")
    nested_rows = _read_csv(root / "tables/nested_real_references.csv")
    nested_tuning_rows = _read_csv(root / "tables/nested_classifier_tuning.csv")
    sampler_rows = _read_csv(root / "tables/sampler_realizations.csv")
    identity_rows = _read_csv(root / "tables/identity_overlap_audit.csv")
    checkpoint_audits = _read_csv(root / "tables/checkpoint_reuse_audit.csv")
    frame_index = validate_feature_frame_index(
        root,
        expected_frame_hashes={str(row.get("frame_hash", "")) for row in rows},
    )
    _assert_columns(rows, SOURCE_INNER_METRIC_COLUMNS, "source_inner_metrics.csv")
    _assert_columns(
        checkpoint_audits,
        SOURCE_CHECKPOINT_AUDIT_COLUMNS,
        "checkpoint_reuse_audit.csv",
    )
    validate_source_protocol(protocol, expected_config=expected_config)
    validate_runtime_reports(
        root,
        protocol_hash=str(protocol["protocol_hash"]),
        mode="source_inner",
        checkpoint_index=checkpoint_index,
        frame_index=frame_index,
    )
    heldouts = tuple(str(value) for value in protocol["heldout_centers"])
    bundle_hash = selection_evidence_hash(
        metric_rows=rows,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        protocol_manifest=protocol,
        checkpoint_index=checkpoint_index,
        task_fisher_index=fisher_index,
        feature_frame_index=frame_index,
    )
    if evidence.get("selection_bundle_hash") != bundle_hash:
        raise ProtocolError("Source-inner selection evidence bundle hash mismatch.")
    if evidence.get("schema_version") != (
        "midogpp_prior_recovery_selection_evidence_v1"
    ):
        raise ProtocolError("Unexpected source-inner selection evidence schema.")
    if any(row.get("selection_bundle_hash") != bundle_hash for row in rows):
        raise ProtocolError(
            "Source-inner metric row is not bound to the selection evidence bundle."
        )
    observed_locks: dict[str, RecipeLock] = {}
    for outer in heldouts:
        path = root / f"manifests/recipe_locks/{outer}.json"
        if not path.is_file():
            raise ProtocolError(f"Missing RecipeLock for center {outer}.")
        observed_locks[outer] = load_recipe_lock(path)
    locks = validate_source_inner_evidence_view(
        metric_rows=rows,
        nested_reference_rows=nested_rows,
        nested_tuning_rows=nested_tuning_rows,
        sampler_rows=sampler_rows,
        identity_rows=identity_rows,
        checkpoint_audit_rows=checkpoint_audits,
        checkpoint_index=checkpoint_index,
        task_fisher_index=fisher_index,
        protocol=protocol,
        selection_bundle_hash=bundle_hash,
        observed_locks=observed_locks,
    )
    expected_gate = _expected_gate(locks, bundle_hash=bundle_hash)
    if gate != expected_gate:
        raise ProtocolError("Source-inner gate report does not match recomputed locks.")
    if leakage.get("selection_bundle_hash") != bundle_hash:
        raise ProtocolError("Source-inner leakage report bundle hash mismatch.")
    expected_leakage_status = (
        "PASS" if all(lock.status == "VALID" for lock in locks.values()) else "FAIL"
    )
    if (
        leakage.get("status") != expected_leakage_status
        or leakage.get("identity_overlap_status") != "PASS"
    ):
        raise ProtocolError(
            "Source-inner leakage report status is inconsistent with evidence."
        )
    for field, expected in {
        "outer_target_rows_passed_to_training_or_selection": False,
        "outer_target_labels_used_for_selection": False,
        "target_eval_labels_used_for_selection": False,
        "center_4_excluded": True,
        "routing_performed": False,
        "composition_performed": False,
    }.items():
        if leakage.get(field) is not expected:
            raise ProtocolError(f"Source-inner leakage field {field} mismatch.")
    if require_factorial and (
        gate.get("factorial_triggered") is not True
        or leakage.get("status") != "PASS"
    ):
        raise ProtocolError(
            "Source-inner artifact is not eligible for the outer factorial run."
        )
    return locks


def _expected_gate(
    locks: Mapping[str, RecipeLock],
    *,
    bundle_hash: str,
) -> dict[str, object]:
    values = list(locks.values())
    valid = [lock for lock in values if lock.status == "VALID"]
    conditional = [lock for lock in valid if lock.primary_arm in {"C", "D"}]
    triggered = (
        len(valid) == len(values)
        and len(conditional) == len(values)
        and bool(values)
    )
    return {
        "status": (
            "FACTORIAL_TRIGGERED"
            if triggered
            else (
                "NEGATIVE_GATE_COMPLETE"
                if len(valid) == len(values)
                else "INVALID_LOCKS_PRESENT"
            )
        ),
        "n_locks": len(values),
        "n_valid_locks": len(valid),
        "n_conditional_locks": len(conditional),
        "factorial_triggered": triggered,
        "invalid_centers": [
            lock.outer_target_center for lock in values if lock.status != "VALID"
        ],
        "outer_scoring_used": False,
        "selection_bundle_hash": bundle_hash,
    }
