"""Failed-state, config, provenance, protocol, and preflight identity checks."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import yaml

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from . import contracts as archive_contracts
from .contracts import (
    CLAIM_ROLE,
    CLAIM_SCOPE,
    CONFIG_CONTRACT_HASH,
    EXPECTED_AMENDMENT_SHA256,
    EXPECTED_LEDGER_SHA256,
    EXPECTED_MANIFEST_SHA256,
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
    STAGE_ID,
    TERMINAL_DECISION,
)
from .hashing import canonical_hash, without


_RUN_STATE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "phase",
        "error",
        "error_class",
        "updated_at_utc",
        "cross_run_recovery_allowed",
        "terminal_recovery_allowed",
    }
)
_PROTOCOL_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "output_artifact_id",
        "config_contract_hash",
        "protocol_contract_hash",
        "stage",
        "claim_scope",
        "claim_role",
        "input_artifact_hashes",
        "cache_binding_hash",
        "pre_gpu_firewall",
        "exact_six_original_inputs",
        "previous_stage90_output_or_checkpoint_used",
        "test_split_previously_consumed",
        "fresh_evidence",
        "publication_status",
        "protocol_manifest_hash",
    }
)
FORBIDDEN_CLAIM_FLAGS = (
    "fresh_evidence",
    "quarantined_v1_output_used",
    "quarantined_v1_scratch_or_checkpoint_used",
    "quarantined_v1_terminal_outputs_used",
    "prior_v1_label_capability_history_used",
    "prior_v1_amendment_used",
    "routing_success_claimed",
    "routing_quality_claimed",
    "downstream_utility_claimed",
    "nelbo_compatibility_claimed",
    "expert_selection_claimed",
    "deployment_claimed",
    "nominal_coverage_claimed",
    "nominal_significance_claimed",
    "source_expert_updated",
    "target_expert_used",
    "shared_model_updated_with_target_labels",
    "promotion_eligible",
    "may_feed_stage50",
    "may_feed_stage60",
    "may_feed_stage70",
    "may_feed_another_stage90",
    "may_feed_another_experiment",
    "previous_stage90_outputs_used",
    "previous_stage90_amendments_used",
    "previous_probability_surface_used",
    "previous_stage90_scratch_or_checkpoint_used",
)


def validate_failed_state(root: Path) -> Mapping[str, object]:
    state = read_json(root / "reports/run_state.json")
    try:
        timestamp = datetime.fromisoformat(str(state.get("updated_at_utc")))
    except ValueError as exc:
        raise ProtocolError("CBPUPR v2 preterminal failure timestamp drifted.") from exc
    if (
        set(state) != _RUN_STATE_KEYS
        or state.get("schema_version") != "fixed_bank_cbpupr_run_state_v1"
        or state.get("status") != "FAILED"
        or state.get("phase") != FAILED_PHASE
        or state.get("error") != FAILED_ERROR
        or state.get("error_class") != FAILED_ERROR_CLASS
        or timestamp.tzinfo is None
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("CBPUPR v2 preterminal failed state drifted.")
    return state


def validate_config(
    root: Path,
    *,
    logical_root: Path,
    logical_scratch: Path | None,
) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load((root / "config.resolved.yaml").read_text("utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolError("CBPUPR v2 preterminal config is unreadable.") from exc
    if not isinstance(raw, Mapping):
        raise ProtocolError("CBPUPR v2 preterminal config is not an object.")
    experiment = _mapping(raw, "experiment")
    inputs = _mapping(raw, "inputs")
    protocol = _mapping(raw, "protocol")
    runtime = _mapping(raw, "runtime")
    boundary = _mapping(raw, "claim_boundary")
    if (
        experiment.get("id") != EXPERIMENT_ID
        or experiment.get("claim_scope") != CLAIM_SCOPE
        or experiment.get("status") != PUBLICATION_STATUS
        or _resolved_text(experiment.get("artifact_root")) != logical_root
        or protocol.get("schema_version") != "fixed_bank_cbpupr_protocol_v2"
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("execution_revision")
        != "v2_canonical_row_order_mechanical_repair"
        or protocol.get("publication_status") != PUBLICATION_STATUS
        or protocol.get("terminal_decision") != TERMINAL_DECISION
        or protocol.get("claim_scope") != CLAIM_SCOPE
        or protocol.get("claim_role") != CLAIM_ROLE
        or protocol.get("split") != "test"
        or protocol.get("split_previously_consumed") is not True
        or protocol.get("fresh_evidence") is not False
        or protocol.get("repair_source_manifest_sha256")
        != REPAIR_SOURCE_MANIFEST_SHA256
        or protocol.get("repair_source_tree_sha256") != REPAIR_SOURCE_TREE_SHA256
        or protocol.get("repair_source_member_count") != REPAIR_SOURCE_MEMBER_COUNT
        or protocol.get("previous_stage90_outputs_used") is not False
        or protocol.get("previous_stage90_scratch_or_checkpoints_used") is not False
        or protocol.get("routing_success_claimed") is not False
        or protocol.get("promotion_eligible") is not False
        or protocol.get("may_feed_another_experiment") is not False
        or runtime.get("schema_version")
        != "fixed_bank_cbpupr_workstation_runtime_v2"
        or runtime.get("cross_run_recovery_allowed") is not False
        or runtime.get("terminal_recovery_allowed") is not False
        or runtime.get("owned_task_checkpoint_replay_allowed") is not False
        or runtime.get("foreign_checkpoint_reuse_forbidden") is not True
        or runtime.get("previous_stage90_scratch_reuse_forbidden") is not True
        or runtime.get("repair_source_manifest_sha256")
        != REPAIR_SOURCE_MANIFEST_SHA256
        or runtime.get("repair_source_tree_sha256") != REPAIR_SOURCE_TREE_SHA256
        or runtime.get("repair_source_member_count") != REPAIR_SOURCE_MEMBER_COUNT
        or boundary.get("schema_version") != "fixed_bank_cbpupr_claim_boundary_v2"
        or boundary.get("publication_status") != PUBLICATION_STATUS
        or boundary.get("terminal_decision") != TERMINAL_DECISION
        or boundary.get("claim_role") != CLAIM_ROLE
        or boundary.get("consumed_test_data") is not True
        or boundary.get("terminal_stage90_diagnostic") is not True
        or any(boundary.get(flag) is not False for flag in FORBIDDEN_CLAIM_FLAGS)
    ):
        raise ProtocolError("CBPUPR v2 preterminal config identity drifted.")
    scratch_preference = runtime.get("scratch_preference")
    if (
        not isinstance(scratch_preference, list)
        or len(scratch_preference) != 2
        or scratch_preference[1] != "artifact_parent"
    ):
        raise ProtocolError("CBPUPR v2 preterminal scratch binding drifted.")
    configured_scratch = _resolved_text(scratch_preference[0])
    if logical_scratch is None:
        if configured_scratch.exists() or configured_scratch.is_symlink():
            raise ProtocolError(
                "CBPUPR v2 preterminal explicit no-scratch state drifted."
            )
    elif configured_scratch != logical_scratch:
        raise ProtocolError("CBPUPR v2 preterminal scratch binding drifted.")
    expected_inputs = {
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "expected_test_consumption_ledger_sha256": EXPECTED_LEDGER_SHA256,
        "expected_ledger_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "expected_ledger_amendment_parent_sha256": EXPECTED_LEDGER_SHA256,
        "ledger_amendment_authorized_experiment_id": EXPERIMENT_ID,
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        raise ProtocolError("CBPUPR v2 preterminal input identity drifted.")
    if read_json(root / "manifests/action_library.json") != dict(
        _mapping(raw, "action_library")
    ) or read_json(root / "manifests/policy_menu.json") != dict(
        _mapping(raw, "policy_menu")
    ):
        raise ProtocolError("CBPUPR v2 preterminal action/policy identity drifted.")
    return raw


def validate_provenance(
    root: Path,
) -> tuple[tuple[Mapping[str, object], ...], dict[str, str]]:
    payload = read_json(root / "provenance/input_artifacts.json")
    rows = payload.get("input_artifacts")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("stage") != STAGE_ID
        or payload.get("claim_scope") != CLAIM_SCOPE
        or payload.get("selection_used_target_eval_artifacts") is not False
        or not isinstance(rows, list)
        or len(rows) != 6
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise ProtocolError("CBPUPR v2 preterminal provenance drifted.")
    typed_rows = tuple(row for row in rows if isinstance(row, Mapping))
    expected_ids = tuple(
        sorted(
            dict(
                archive_contracts.CANONICAL_ARCHIVE_CONTRACT.input_artifact_hashes
            )
        )
    )
    observed_ids = tuple(str(row.get("artifact_id")) for row in typed_rows)
    if (
        observed_ids != expected_ids
        or len(set(observed_ids)) != 6
        or any(
            row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
            for row in typed_rows
        )
    ):
        raise ProtocolError("CBPUPR v2 preterminal provenance coverage drifted.")
    return typed_rows, {
        str(row["artifact_id"]): canonical_hash(dict(row)) for row in typed_rows
    }


def validate_protocol_manifest(
    root: Path,
    *,
    provenance_hashes: Mapping[str, str],
    expected_input_hashes: Mapping[str, str],
) -> Mapping[str, object]:
    protocol = read_json(root / "manifests/protocol_manifest.json")
    firewall = protocol.get("pre_gpu_firewall")
    if (
        set(protocol) != _PROTOCOL_KEYS
        or protocol.get("schema_version")
        != "fixed_bank_cbpupr_protocol_manifest_v1"
        or protocol.get("experiment_id") != EXPERIMENT_ID
        or protocol.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or protocol.get("config_contract_hash") != CONFIG_CONTRACT_HASH
        or protocol.get("protocol_contract_hash") != PROTOCOL_CONTRACT_HASH
        or protocol.get("stage") != STAGE_ID
        or protocol.get("claim_scope") != CLAIM_SCOPE
        or protocol.get("claim_role") != CLAIM_ROLE
        or protocol.get("input_artifact_hashes") != dict(provenance_hashes)
        or protocol.get("input_artifact_hashes") != dict(expected_input_hashes)
        or not _sha256(protocol.get("cache_binding_hash"))
        or not isinstance(firewall, Mapping)
        or firewall.get("status") != "PASS"
        or firewall.get("repair_source_manifest_validated") is not True
        or firewall.get("repair_source_manifest_sha256")
        != REPAIR_SOURCE_MANIFEST_SHA256
        or firewall.get("repair_source_tree_sha256") != REPAIR_SOURCE_TREE_SHA256
        or firewall.get("repair_source_member_count") != REPAIR_SOURCE_MEMBER_COUNT
        or firewall.get("target_labels_opened") is not False
        or firewall.get("target_expert_used") is not False
        or firewall.get(
            "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed"
        )
        is not False
        or protocol.get("exact_six_original_inputs") is not True
        or protocol.get("previous_stage90_output_or_checkpoint_used") is not False
        or protocol.get("test_split_previously_consumed") is not True
        or protocol.get("fresh_evidence") is not False
        or protocol.get("publication_status") != PUBLICATION_STATUS
        or protocol.get("protocol_manifest_hash")
        != canonical_hash(without(protocol, "protocol_manifest_hash"))
    ):
        raise ProtocolError("CBPUPR v2 preterminal protocol identity drifted.")
    return protocol


def validate_preflight(root: Path) -> None:
    payload = read_json(root / "reports/workstation_preflight.json")
    if (
        payload.get("schema_version") != "fixed_bank_cbpupr_workstation_preflight_v1"
        or payload.get("status") != "PASS"
        or payload.get("outer_route_count") != 218
        or payload.get("target_probability_cell_count") != 810
        or payload.get("expected_target_posterior_model_fit_count") != 436
        or payload.get("scratch_absent_at_launch") is not True
        or payload.get("owned_task_checkpoint_replay_allowed") is not False
        or payload.get("foreign_checkpoint_reuse_forbidden") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("CBPUPR v2 preterminal preflight drifted.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"CBPUPR v2 preterminal config section absent: {key}.")
    return value


def _resolved_text(value: object) -> Path:
    text = str(value)
    if text.startswith(("artifact://", "output://")):
        raise ProtocolError("CBPUPR v2 preterminal archive requires resolved paths.")
    return Path(text).resolve()


def _sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


__all__ = (
    "FORBIDDEN_CLAIM_FLAGS",
    "validate_config",
    "validate_failed_state",
    "validate_preflight",
    "validate_protocol_manifest",
    "validate_provenance",
)
