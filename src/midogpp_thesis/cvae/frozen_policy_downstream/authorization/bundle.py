"""Closed-world artifact schemas and deterministic Stage-70 payload builders."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ..contracts import AUTHORIZED_CONSUMER_EXPERIMENT_ID, POLICY_ARMS
from .config import FinalAuthorizationConfig, ReservationConfig
from .contracts import (
    AuthorizationValidationInputs,
    CacheBinding,
    CLAIM_SCOPE,
    EXPECTED_EVALUATION_PLAN_ROWS,
    EXPECTED_SPLIT,
    EXPECTED_TEST_ROWS,
    FINAL_AUTHORIZATION_EXPERIMENT_ID,
    FINAL_AUTHORIZATION_PHASE,
    FINAL_DESCRIPTIVE_STATUS,
    FRESH_CONFIRMATORY_STATUS,
    PURPOSE,
    RESERVATION_DESCRIPTIVE_STATUS,
    RESERVATION_EXPERIMENT_ID,
    RESERVATION_PHASE,
    RUN_COMPLETE,
    FinalAuthorizationToken,
    make_final_authorization_token,
)


TARGET_IDENTITY_COLUMNS = (
    "evaluation_row_id",
    "contract_row_index",
    "target_center",
    "split",
)
RESERVATION_REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/input_binding.json",
    "manifests/protocol_manifest.json",
    "manifests/identity_lock.json",
    "manifests/evaluation_plan.json",
    "manifests/content_index.json",
    "reports/authorization_decision.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/target_identity.csv",
)
FINAL_REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/input_binding.json",
    "manifests/protocol_manifest.json",
    "manifests/identity_lock.json",
    "manifests/evaluation_plan.json",
    "manifests/authorization_token.json",
    "manifests/content_index.json",
    "reports/authorization_decision.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)


def projected_reservation_payload(value: object) -> dict[str, object]:
    """Normalize the public data-projector result without importing labels."""

    manifest_sha256 = str(_field(value, "manifest_sha256"))
    reservation_id = str(_field(value, "reservation_id"))
    protocol_hash = str(_field(value, "protocol_hash"))
    rows_raw = _field(value, "rows")
    expected_raw = _field_any(value, "expected_rows_by_center", "rows_by_center")
    if isinstance(rows_raw, (str, bytes)) or not isinstance(rows_raw, Sequence):
        raise ProtocolError("Stage-70 projected reservation rows are malformed.")
    if not isinstance(expected_raw, Mapping):
        raise ProtocolError("Stage-70 projected reservation center counts are malformed.")
    rows = target_identity_rows(rows_raw)
    counts = {str(key): int(item) for key, item in expected_raw.items()}
    if set(counts) != set(CENTERS) or sum(counts.values()) != EXPECTED_TEST_ROWS:
        raise ProtocolError("Stage-70 projected reservation center coverage drifted.")
    observed_counts = {
        center: sum(row["target_center"] == center for row in rows)
        for center in CENTERS
    }
    if observed_counts != counts:
        raise ProtocolError("Stage-70 projected reservation row counts drifted.")
    return {
        "manifest_sha256": manifest_sha256,
        "reservation_id": reservation_id,
        "protocol_hash": protocol_hash,
        "expected_rows_by_center": counts,
        "row_count": len(rows),
        "target_identity_table_hash": stable_hash(rows),
        "rows": rows,
    }


def target_identity_rows(rows: Sequence[object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for raw in rows:
        output.append(
            {
                "evaluation_row_id": str(_field(raw, "evaluation_row_id")),
                "contract_row_index": int(_field(raw, "contract_row_index")),
                "target_center": str(_field(raw, "center")),
                "split": str(_field(raw, "split")),
            }
        )
    if len(output) != EXPECTED_TEST_ROWS:
        raise ProtocolError("Stage-70 target-identity row count drifted.")
    if (
        len({str(row["evaluation_row_id"]) for row in output}) != len(output)
        or len({int(row["contract_row_index"]) for row in output}) != len(output)
        or any(not row["evaluation_row_id"] for row in output)
        or any(row["target_center"] not in CENTERS for row in output)
        or any(row["split"] != EXPECTED_SPLIT for row in output)
    ):
        raise ProtocolError("Stage-70 target-identity rows violate the reservation contract.")
    return output


def reservation_identity_lock(
    config: ReservationConfig,
    projected: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_target_identity_lock_v1",
        "experiment_id": RESERVATION_EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "purpose": PURPOSE,
        "fresh_evidence": False,
        "scoring_manifest_sha256": projected["manifest_sha256"],
        "target_evaluation_reservation_id": projected["reservation_id"],
        "target_evaluation_reservation_protocol_hash": projected["protocol_hash"],
        "target_identity_table_hash": projected["target_identity_table_hash"],
        "row_count": projected["row_count"],
        "rows_by_center": dict(_mapping(projected, "expected_rows_by_center")),
        "split": EXPECTED_SPLIT,
        "opaque_evaluation_row_ids_only": True,
        "sample_ids_persisted": False,
        "image_paths_persisted": False,
        "target_label_values_persisted": False,
        "cache_artifact_id": config.cache_artifact_id,
        "cache_extractor_protocol_hash": (
            config.expected_cache_extractor_protocol_hash
        ),
    }
    payload["identity_lock_hash"] = stable_hash(payload)
    return payload


def reservation_evaluation_plan(
    config: ReservationConfig,
    identity: Mapping[str, object],
) -> dict[str, object]:
    rows_by_center = _mapping(identity, "rows_by_center")
    records = [
        {
            "target_center": center,
            "split": EXPECTED_SPLIT,
            "reserved_row_count": int(rows_by_center[center]),
            "authorized_action": "label_blind_cache_extraction_only",
            "prospective_cache_artifact_id": config.cache_artifact_id,
            "target_expert_exclusion_applies_at_later_prediction": True,
            "prediction_allowed": False,
            "label_access_allowed": False,
            "metric_scoring_allowed": False,
        }
        for center in CENTERS
    ]
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_cache_extraction_reservation_plan_v1",
        "experiment_id": RESERVATION_EXPERIMENT_ID,
        "target_evaluation_reservation_id": identity[
            "target_evaluation_reservation_id"
        ],
        "target_evaluation_reservation_protocol_hash": identity[
            "target_evaluation_reservation_protocol_hash"
        ],
        "identity_lock_hash": identity["identity_lock_hash"],
        "cache_experiment_id": config.cache_experiment_id,
        "cache_artifact_id": config.cache_artifact_id,
        "cache_extractor_protocol_hash": (
            config.expected_cache_extractor_protocol_hash
        ),
        "prospective_cache_root": str(config.prospective_cache_root),
        "records": records,
        "prediction_count": 0,
        "metric_count": 0,
    }
    payload["evaluation_plan_hash"] = stable_hash(payload)
    return payload


def reservation_protocol_manifest(
    config: ReservationConfig,
    inputs: AuthorizationValidationInputs,
    identity: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_target_evaluation_reservation_protocol_v1",
        "phase": RESERVATION_PHASE,
        "experiment_id": RESERVATION_EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "purpose": PURPOSE,
        "fresh_evidence": False,
        "config_contract_hash": config.contract_hash,
        "authorized_consumer_experiment_id": AUTHORIZED_CONSUMER_EXPERIMENT_ID,
        "scoring_manifest_sha256": identity["scoring_manifest_sha256"],
        "test_consumption_status": inputs.consumption_ledger["status"],
        "descriptive_locked_model_scoring_allowed": True,
        "identity_lock_hash": identity["identity_lock_hash"],
        "evaluation_plan_hash": plan["evaluation_plan_hash"],
        "validated_upstreams": inputs.bindings_payload(),
        "generation_lock_hash": inputs.generation_lock_hash,
        "policy_lock_hashes": {
            item.policy_id: item.policy_lock_hash for item in inputs.policies
        },
        "cache_extractor_protocol_hash": (
            config.expected_cache_extractor_protocol_hash
        ),
        "target_labels_opened": False,
        "generation_performed": False,
        "classifier_fit_performed": False,
        "prediction_performed": False,
        "metric_scoring_performed": False,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def final_identity_lock(
    config: FinalAuthorizationConfig,
    reservation_identity: Mapping[str, object],
    reservation_content_hash: str,
    cache: CacheBinding,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_final_target_identity_lock_v1",
        "experiment_id": FINAL_AUTHORIZATION_EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "purpose": PURPOSE,
        "fresh_evidence": False,
        "reservation_artifact_id": config.reservation_artifact_id,
        "reservation_content_hash": reservation_content_hash,
        "reservation_identity_lock_hash": reservation_identity["identity_lock_hash"],
        "target_evaluation_reservation_id": reservation_identity[
            "target_evaluation_reservation_id"
        ],
        "target_evaluation_reservation_protocol_hash": reservation_identity[
            "target_evaluation_reservation_protocol_hash"
        ],
        "scoring_manifest_sha256": cache.manifest_sha256,
        "target_identity_table_hash": reservation_identity[
            "target_identity_table_hash"
        ],
        "target_cache_artifact_id": cache.artifact_id,
        "target_cache_content_hash": cache.content_hash,
        "target_cache_row_order_hash": cache.row_order_hash,
        "target_cache_shard_sha256_by_center": dict(cache.shard_sha256_by_center),
        "row_count": cache.row_count,
        "rows_by_center": dict(cache.rows_by_center),
        "sample_ids_persisted": False,
        "image_paths_persisted": False,
        "target_label_values_persisted": False,
    }
    payload["identity_lock_hash"] = stable_hash(payload)
    return payload


def final_evaluation_plan(
    config: FinalAuthorizationConfig,
    inputs: AuthorizationValidationInputs,
    identity: Mapping[str, object],
    cache: CacheBinding,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for replicate in inputs.policy_replicates:
        assignments = []
        for item in tuple(getattr(replicate, "assignments", ())):
            assignments.append(
                {
                    "assignment_id": str(getattr(item, "assignment_id")),
                    "source_center": str(getattr(item, "source_center")),
                    "source_stream_id": str(getattr(item, "source_stream_id")),
                    "source_ordinal": int(getattr(item, "source_ordinal")),
                    "source_budget_per_class": int(
                        getattr(item, "source_budget_per_class")
                    ),
                    "target_expert": False,
                }
            )
        target = str(getattr(replicate, "target_center"))
        records.append(
            {
                "policy_id": str(getattr(replicate, "policy_id")),
                "policy_lock_hash": str(getattr(replicate, "policy_lock_hash")),
                "policy_plan_hash": str(getattr(replicate, "policy_plan_hash")),
                "assignment_table_hash": str(
                    getattr(replicate, "assignment_table_hash")
                ),
                "replicate_id": str(getattr(replicate, "replicate_id")),
                "target_center": target,
                "training_seed": int(getattr(replicate, "training_seed")),
                "generation_seed": int(getattr(replicate, "generation_seed")),
                "assignments": assignments,
                "assignment_payload_hash": stable_hash(assignments),
                "class_shuffle_seed_by_label": dict(
                    getattr(replicate, "class_shuffle_seed_by_label")
                ),
                "synthetic_rows_per_class": 1024,
                "classifier_config_hash": inputs.classifier_spec["config_hash"],
                "target_cache_shard_sha256": cache.shard_sha256_by_center[target],
                "target_cache_row_order_hash": cache.row_order_hash,
                "target_expert_excluded": True,
                "prediction_allowed": True,
                "label_access_allowed": False,
                "metric_scoring_allowed": False,
            }
        )
    if len(records) != EXPECTED_EVALUATION_PLAN_ROWS:
        raise ProtocolError("Stage-70 final evaluation-plan coverage drifted.")
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_final_prediction_plan_v1",
        "experiment_id": FINAL_AUTHORIZATION_EXPERIMENT_ID,
        "authorized_consumer_experiment_id": config.consumer_experiment_id,
        "identity_lock_hash": identity["identity_lock_hash"],
        "generation_lock_hash": inputs.generation_lock_hash,
        "classifier": dict(inputs.classifier_spec),
        "policy_arms": list(POLICY_ARMS),
        "factorial": {
            "heldout_target_centers": list(CENTERS),
            "training_seed_count": 3,
            "generation_seed_count": 3,
            "policy_count": 3,
            "replicates_per_policy": 81,
            "total_prediction_cells": 243,
            "synthetic_rows_per_class": 1024,
        },
        "records": records,
        "target_labels_opened": False,
        "prediction_count": 0,
        "metric_count": 0,
    }
    payload["evaluation_plan_hash"] = stable_hash(payload)
    return payload


def final_protocol_manifest(
    config: FinalAuthorizationConfig,
    inputs: AuthorizationValidationInputs,
    identity: Mapping[str, object],
    plan: Mapping[str, object],
    cache: CacheBinding,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_final_prediction_authorization_protocol_v1",
        "phase": FINAL_AUTHORIZATION_PHASE,
        "experiment_id": FINAL_AUTHORIZATION_EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "purpose": PURPOSE,
        "fresh_evidence": False,
        "config_contract_hash": config.contract_hash,
        "authorized_consumer_experiment_id": config.consumer_experiment_id,
        "fresh_confirmatory_status": FRESH_CONFIRMATORY_STATUS,
        "descriptive_status": FINAL_DESCRIPTIVE_STATUS,
        "identity_lock_hash": identity["identity_lock_hash"],
        "evaluation_plan_hash": plan["evaluation_plan_hash"],
        "scoring_manifest_sha256": identity["scoring_manifest_sha256"],
        "cache_extractor_protocol_hash": cache.cache_extractor_protocol_hash,
        "validated_cache": cache.to_payload(),
        "validated_upstreams": inputs.bindings_payload(),
        "generation_lock_hash": inputs.generation_lock_hash,
        "classifier_config_hash": inputs.classifier_spec["config_hash"],
        "policy_lock_hashes": {
            item.policy_id: item.policy_lock_hash for item in inputs.policies
        },
        "all_policy_assignments_frozen": True,
        "target_expert_excluded_in_every_replicate": True,
        "full_9_by_3_by_3_factorial_per_policy": True,
        "prediction_allowed": True,
        "target_labels_opened": False,
        "metric_scoring_allowed": False,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def final_authorization_token(
    config: FinalAuthorizationConfig,
    inputs: AuthorizationValidationInputs,
    identity: Mapping[str, object],
    plan: Mapping[str, object],
    protocol: Mapping[str, object],
    cache: CacheBinding,
) -> FinalAuthorizationToken:
    return make_final_authorization_token(
        {
            "schema_version": (
                "midogpp_stage70_final_prediction_authorization_token_v1"
            ),
            "phase": FINAL_AUTHORIZATION_PHASE,
            "status": RUN_COMPLETE,
            "experiment_id": FINAL_AUTHORIZATION_EXPERIMENT_ID,
            "claim_scope": CLAIM_SCOPE,
            "purpose": PURPOSE,
            "fresh_evidence": False,
            "fresh_confirmatory_status": FRESH_CONFIRMATORY_STATUS,
            "descriptive_status": FINAL_DESCRIPTIVE_STATUS,
            "authorized_consumer_experiment_id": config.consumer_experiment_id,
            "authorization_protocol_hash": protocol["protocol_hash"],
            "identity_lock_hash": identity["identity_lock_hash"],
            "evaluation_plan_hash": plan["evaluation_plan_hash"],
            "reservation_content_hash": identity["reservation_content_hash"],
            "target_cache_content_hash": cache.content_hash,
            "target_cache_row_order_hash": cache.row_order_hash,
            "scoring_manifest_sha256": cache.manifest_sha256,
            "generation_lock_hash": inputs.generation_lock_hash,
            "classifier_config_hash": inputs.classifier_spec["config_hash"],
            "policy_bindings": [item.to_payload() for item in inputs.policies],
            "prediction_allowed": True,
            "label_access_allowed": False,
            "metric_scoring_allowed": False,
            "policy_or_seed_selection_allowed": False,
        }
    )


def authorization_decision(*, final: bool) -> dict[str, object]:
    phase = FINAL_AUTHORIZATION_PHASE if final else RESERVATION_PHASE
    experiment_id = (
        FINAL_AUTHORIZATION_EXPERIMENT_ID if final else RESERVATION_EXPERIMENT_ID
    )
    descriptive = FINAL_DESCRIPTIVE_STATUS if final else RESERVATION_DESCRIPTIVE_STATUS
    payload: dict[str, object] = {
        "schema_version": (
            "midogpp_stage70_final_authorization_decision_v1"
            if final
            else "midogpp_stage70_reservation_decision_v1"
        ),
        "phase": phase,
        "status": RUN_COMPLETE,
        "experiment_id": experiment_id,
        "claim_scope": CLAIM_SCOPE,
        "purpose": PURPOSE,
        "fresh_evidence": False,
        "fresh_confirmatory_status": FRESH_CONFIRMATORY_STATUS,
        "descriptive_status": descriptive,
        "authorized_consumer_experiment_id": AUTHORIZED_CONSUMER_EXPERIMENT_ID,
        "cache_extraction_allowed": not final,
        "prediction_allowed": final,
        "label_access_allowed": False,
        "metric_scoring_allowed": False,
        "generation_or_policy_refit_allowed": False,
    }
    payload["decision_hash"] = stable_hash(payload)
    return payload


def leakage_report(*, final: bool) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_stage70_final_authorization_leakage_v1"
            if final
            else "midogpp_stage70_reservation_leakage_v1"
        ),
        "status": "PASS",
        "purpose": PURPOSE,
        "fresh_evidence": False,
        "previously_consumed_test_rows": True,
        "unconsumed_eligible_split_available": False,
        "scoring_manifest_access": (
            "sha256_bytes_only"
            if final
            else "identity_projection_only_no_label_or_path_field_access"
        ),
        "target_label_values_opened": False,
        "target_label_values_persisted": False,
        "sample_ids_persisted": False,
        "image_paths_persisted": False,
        "target_identity_used_as_predictive_feature": False,
        "target_identity_role": "fold_membership_and_target_expert_exclusion_only",
        "target_expert_excluded": True,
        "policy_or_seed_selection_performed": False,
        "generation_performed": False,
        "classifier_fit_performed": False,
        "prediction_performed": False,
        "metric_scoring_performed": False,
    }


def run_state(*, final: bool, status: str) -> dict[str, object]:
    return {
        "schema_version": (
            "midogpp_stage70_final_authorization_run_state_v1"
            if final
            else "midogpp_stage70_reservation_run_state_v1"
        ),
        "phase": FINAL_AUTHORIZATION_PHASE if final else RESERVATION_PHASE,
        "status": status,
        "claim_scope": CLAIM_SCOPE,
        "purpose": PURPOSE,
        "fresh_evidence": False,
    }


def write_resolved_config(
    path: Path,
    config: ReservationConfig | FinalAuthorizationConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config.to_payload(), sort_keys=False),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read Stage-70 authorization JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Stage-70 authorization JSON must be an object: {path}.")
    return payload


def write_target_identity(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TARGET_IDENTITY_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in TARGET_IDENTITY_COLUMNS})


def read_target_identity(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != TARGET_IDENTITY_COLUMNS:
                raise ProtocolError("Stage-70 target-identity table schema drifted.")
            rows = [
                {
                    "evaluation_row_id": str(row["evaluation_row_id"]),
                    "contract_row_index": int(row["contract_row_index"]),
                    "target_center": str(row["target_center"]),
                    "split": str(row["split"]),
                }
                for row in reader
            ]
    except (OSError, ValueError) as exc:
        raise ProtocolError("Cannot read Stage-70 target-identity table.") from exc
    return rows


def write_content_index(root: Path, required_files: Sequence[str]) -> None:
    excluded = {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
    records = []
    for relative in required_files:
        if relative in excluded:
            continue
        member = root / relative
        if not member.is_file() or member.is_symlink():
            raise ProtocolError(f"Stage-70 content member is missing: {relative}.")
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_authorization_content_index_v1",
        "records": records,
    }
    payload["content_hash"] = stable_hash(payload)
    write_json(root / "manifests/content_index.json", payload)


def input_provenance(
    *,
    config: ReservationConfig | FinalAuthorizationConfig,
    inputs: AuthorizationValidationInputs,
    scoring_manifest_sha256: str,
    reservation_content_hash: str | None = None,
    cache: CacheBinding | None = None,
) -> dict[str, object]:
    paths = {
        "canonical_reference_root": str(config.canonical_reference_root),
        "bank_root": str(config.bank_root),
        "generation_lock_root": str(config.generation_lock_root),
        "equal_union_policy_root": str(config.equal_union_policy_root),
        "metadata_policy_root": str(config.metadata_policy_root),
        "utility_policy_root": str(config.utility_policy_root),
    }
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_authorization_input_provenance_v1",
        "config_contract_hash": config.contract_hash,
        "input_paths": paths,
        "validated_bindings": inputs.bindings_payload(),
        "scoring_manifest": {
            "path": str(config.scoring_manifest_path),
            "sha256": scoring_manifest_sha256,
            "authorization_access": "binary_hash_only_no_csv_or_label_parse",
        },
    }
    if isinstance(config, ReservationConfig):
        payload["prospective_cache"] = {
            "path": str(config.prospective_cache_root),
            "artifact_id": config.cache_artifact_id,
            "extractor_protocol_hash": config.expected_cache_extractor_protocol_hash,
        }
    else:
        payload["reservation"] = {
            "path": str(config.reservation_root),
            "content_hash": str(reservation_content_hash or ""),
        }
        payload["cache"] = {
            "path": str(config.cache_root),
            **({} if cache is None else cache.to_payload()),
        }
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash Stage-70 input: {path}.") from exc
    return digest.hexdigest()


def assert_embedded_hash(payload: Mapping[str, object], field: str) -> None:
    observed = payload.get(field)
    unhashed = {key: value for key, value in payload.items() if key != field}
    if observed != stable_hash(unhashed):
        raise ProtocolError(f"Stage-70 embedded hash drifted: {field}.")


def assert_closed_world(root: Path, required_files: Sequence[str]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ProtocolError("Stage-70 authorization root is absent or a symlink.")
    symlinks = sorted(
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_symlink()
    )
    if symlinks:
        raise ProtocolError(f"Stage-70 authorization contains symlinks: {symlinks}.")
    actual = {
        member.relative_to(root).as_posix()
        for member in root.rglob("*")
        if member.is_file()
    }
    expected = set(required_files)
    if actual != expected:
        raise ProtocolError(
            "Stage-70 authorization closed-world coverage drifted: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ProtocolError(f"Stage-70 projector payload lacks {name!r}.")
        return value[name]
    if not hasattr(value, name):
        raise ProtocolError(f"Stage-70 projector object lacks {name!r}.")
    return getattr(value, name)


def _field_any(value: object, *names: str) -> object:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if not isinstance(value, Mapping) and hasattr(value, name):
            return getattr(value, name)
    raise ProtocolError(f"Stage-70 projector payload lacks all fields {names!r}.")


def _mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"Stage-70 authorization payload lacks mapping {key!r}.")
    return value


__all__ = (
    "FINAL_REQUIRED_FILES",
    "RESERVATION_REQUIRED_FILES",
    "TARGET_IDENTITY_COLUMNS",
    "assert_closed_world",
    "assert_embedded_hash",
    "authorization_decision",
    "final_evaluation_plan",
    "final_authorization_token",
    "final_identity_lock",
    "final_protocol_manifest",
    "input_provenance",
    "leakage_report",
    "projected_reservation_payload",
    "read_json",
    "read_target_identity",
    "reservation_evaluation_plan",
    "reservation_identity_lock",
    "reservation_protocol_manifest",
    "run_state",
    "sha256_file",
    "target_identity_rows",
    "write_content_index",
    "write_json",
    "write_resolved_config",
    "write_target_identity",
)
