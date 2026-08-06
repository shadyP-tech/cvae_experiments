"""Independent fail-closed validation of the Stage-70 descriptive bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np

from ...data.contract.stage70_target_evaluation.contracts import EXPECTED_TEST_ROWS
from ...data.features.stage70_test_cache import CACHE_ARTIFACT_ID
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..generation.contracts import (
    COMMON_OUTPUT_DIM,
    SOURCE_BUDGET_PER_CLASS,
    SOURCE_STREAM_NAMESPACE,
    TOTAL_PER_CLASS,
)
from ..protocol import ProtocolError
from ..reporting import write_json
from .authorization.contracts import FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID
from .bundle import REQUIRED_FILES
from .contracts import (
    CLAIM_SCOPE,
    CONTROL_ARM,
    EXPERIMENT_ID,
    METADATA_ARM,
    POLICY_ARMS,
    UTILITY_ARM,
)
from .prediction import PersistedPredictionPass
from .prediction_seal import VerifiedPredictionArtifact, verify_persisted_prediction_artifact
from .source_blocks import SOURCE_BLOCK_CACHE_SCHEMA


GridKey = tuple[str, str, int, int]
SourceKey = tuple[str, int, int]

_EXPECTED_GRID = {
    (arm, center, training_seed, generation_seed)
    for arm in POLICY_ARMS
    for center in CENTERS
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
}
_EXPECTED_SOURCE_GRID = {
    (center, training_seed, generation_seed)
    for center in CENTERS
    for training_seed in TRAINING_SEEDS
    for generation_seed in GENERATION_SEEDS
}
_EVALUATION_SPLIT = "test_previously_consumed_for_representation_adoption"
_FRESH_STATUS = "BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT"

_METRIC_COLUMNS = (
    "schema_version",
    "claim_scope",
    "claim_role",
    "row_role",
    "policy_id",
    "target_center",
    "training_seed",
    "generation_seed",
    "replicate_id",
    "n_eval",
    "n_cases",
    "bacc",
    "macro_f1",
    "macro_f1_role",
    "prediction_sha256",
    "probability_sha256",
    "prediction_cell_hash",
    "target_identity_hash",
    "composition_manifest_hash",
    "train_content_sha256",
    "classifier_config_hash",
    "scaler_state_hash",
    "target_row_order_hash",
    "label_manifest_sha256",
    "reused_from_policy_id",
    "authorization_binding_hash",
    "final_authorization_hash",
    "authorization_protocol_hash",
    "identity_lock_hash",
    "evaluation_plan_hash",
    "reservation_content_hash",
    "target_evaluation_reservation_id",
    "target_evaluation_reservation_protocol_hash",
    "target_cache_artifact_id",
    "target_cache_content_hash",
    "target_cache_row_order_hash",
    "target_cache_shard_sha256",
    "phase_01_sha256",
    "prediction_index_sha256",
    "prediction_arrays_sha256",
    "prediction_seal_sha256",
    "phase_02_sha256",
    "target_labels_used_for_scoring_only",
    "fresh_confirmatory_evidence",
    "policy_or_seed_selection_performed",
)
_CONFUSION_COLUMNS = (
    "schema_version",
    "policy_id",
    "target_center",
    "training_seed",
    "generation_seed",
    "case_id",
    "tn",
    "fp",
    "fn",
    "tp",
    "replicate_id",
    "prediction_sha256",
    "target_identity_hash",
    "label_manifest_sha256",
    "authorization_binding_hash",
    "prediction_index_sha256",
    "prediction_arrays_sha256",
    "target_labels_used_for_scoring_only",
)
_SUMMARY_COLUMNS = (
    "schema_version",
    "policy_id",
    "equal_center_equal_seed_mean_bacc",
    "equal_center_equal_seed_mean_macro_f1",
    "minimum_cell_bacc",
    "maximum_cell_bacc",
    "cell_count",
    "fresh_confirmatory_evidence",
)
_DELTA_COLUMNS = (
    "schema_version",
    "comparison_id",
    "policy_id",
    "control_policy_id",
    "target_center",
    "training_seed",
    "generation_seed",
    "bacc_delta",
    "macro_f1_delta",
    "role",
    "paired",
    "fresh_confirmatory_evidence",
)
_BOOTSTRAP_COLUMNS = (
    "schema_version",
    "comparison_id",
    "observed_mean_bacc_delta",
    "bootstrap_mean_bacc_delta",
    "percentile_2_5",
    "percentile_97_5",
    "seed",
    "valid_replicates",
    "attempted_replicates",
    "rejected_replicates",
    "centers_resampled",
    "cases_resampled_within_center",
    "full_crossed_seed_grid_retained",
    "flattened_seed_pairs_resampled_as_iid",
    "invalid_class_denominator_draws_rejected",
    "interval_role",
    "fresh_confirmatory_inference",
)


def validate_frozen_policy_downstream_bundle(
    root: str | Path,
    *,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Validate every Stage-70 artifact relationship without trusting PASS text."""

    path = Path(root)
    _reject_symlinks(path)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Stage-70 descriptive artifact is incomplete: {missing}.")

    _validate_run_state(path)
    declared_source_members = _declared_source_member_paths(path)
    content_hashes = _validate_content_index(
        path,
        dynamic_members=declared_source_members,
    )
    protocol = _validate_protocol_manifest(path)
    authorization = _validate_authorization_phase(path, protocol)
    evaluation = _validate_evaluation_plan(path, protocol, authorization)
    if authorization.get("classifier_config_hash") != _evaluation_classifier_hash(evaluation):
        raise ProtocolError("Stage-70 authorization/evaluation classifier binding drifted.")
    source_count = _validate_source_block_index(path, protocol, content_hashes=content_hashes)
    composition = _validate_composition_index(path, evaluation)

    # This is deliberately the current public, disk-only verifier used by the
    # label-opening boundary.  Bundle validation must never duplicate a weaker
    # approximation of that transaction check.
    verified = verify_persisted_prediction_artifact(_persisted_capability_from_disk(path))
    if dict(verified.phase_01_binding) != authorization:
        raise ProtocolError("Stage-70 verified authorization binding changed during validation.")
    predictions = _validate_prediction_joins(
        verified.records,
        evaluation=evaluation,
        composition=composition,
    )
    label_manifest_sha256 = _validate_scoring_phases(
        path,
        protocol=protocol,
        verified=verified,
    )

    metrics = _read_csv(path / "tables/target_metrics.csv", _METRIC_COLUMNS)
    confusions = _read_csv(path / "tables/case_confusions.csv", _CONFUSION_COLUMNS)
    summaries = _read_csv(path / "tables/arm_summaries.csv", _SUMMARY_COLUMNS)
    deltas = _read_csv(path / "tables/paired_deltas.csv", _DELTA_COLUMNS)
    bootstrap = _read_csv(path / "tables/bootstrap_summary.csv", _BOOTSTRAP_COLUMNS)
    metric_by_key = _validate_metrics(
        metrics,
        predictions=predictions,
        evaluation=evaluation,
        label_manifest_sha256=label_manifest_sha256,
        verified=verified,
    )
    _validate_case_confusions(confusions, metrics=metric_by_key, predictions=predictions)
    _validate_arm_summaries(summaries, metrics=metric_by_key)
    delta_by_id = _validate_paired_deltas(deltas, metrics=metric_by_key)
    _validate_bootstrap(bootstrap, deltas=delta_by_id)
    _validate_derived_reports(
        path,
        protocol=protocol,
        authorization=authorization,
        prediction_index_sha256=verified.prediction_index_sha256,
        label_manifest_sha256=label_manifest_sha256,
        metrics=metric_by_key,
        deltas=delta_by_id,
    )

    checks: dict[str, object] = {
        "status": "PASS",
        "decision": "DESCRIPTIVE_COMPARISON_COMPLETE",
        "claim_scope": CLAIM_SCOPE,
        "fresh_confirmatory_status": _FRESH_STATUS,
        "prediction_cell_count": len(predictions),
        "metric_row_count": len(metrics),
        "case_confusion_row_count": len(confusions),
        "paired_delta_count": len(deltas),
        "bootstrap_summary_count": len(bootstrap),
        "utility_control_exact_equivalence": True,
        "target_labels_used_for_scoring_only": True,
        "routing_policy_promoted": False,
    }
    if source_count != 81:
        raise ProtocolError("Stage-70 source-block count drifted after validation.")
    if not allow_pending:
        observed = _json(path / "reports/validation_report.json")
        expected = {
            "schema_version": "midogpp_stage70_validation_report_v1",
            **checks,
        }
        if observed != expected:
            raise ProtocolError("Stage-70 validation report drifted.")
    return checks


def write_validation_report(root: str | Path, checks: Mapping[str, object]) -> None:
    write_json(
        Path(root) / "reports/validation_report.json",
        {
            "schema_version": "midogpp_stage70_validation_report_v1",
            **dict(checks),
        },
    )


def _reject_symlinks(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ProtocolError("Stage-70 artifact root must be a real directory.")
    symlinks = sorted(
        member.relative_to(path).as_posix()
        for member in path.rglob("*")
        if member.is_symlink()
    )
    if symlinks:
        raise ProtocolError(f"Stage-70 artifact contains symlinks: {symlinks}.")


def _validate_run_state(path: Path) -> None:
    observed = _json(path / "reports/run_state.json")
    expected = {
        "schema_version": "midogpp_stage70_run_state_v1",
        "status": "COMPLETE",
        "phase": "SCORING_COMPLETE",
    }
    if observed != expected:
        raise ProtocolError("Stage-70 run state is not COMPLETE/SCORING_COMPLETE.")


def _declared_source_member_paths(path: Path) -> set[str]:
    payload = _json(path / "manifests/source_block_index.json")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ProtocolError("Stage-70 source-block index lacks records.")
    members: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Stage-70 source-block index contains a non-object row.")
        member = _safe_relative_path(
            raw.get("member_path"),
            "source-block member",
        )
        if not member.startswith("arrays/source_blocks/") or not member.endswith(".npz"):
            raise ProtocolError("Stage-70 source-block member path escapes its namespace.")
        if member in members:
            raise ProtocolError("Stage-70 source-block index duplicates a member path.")
        members.add(member)
    if len(members) != 81:
        raise ProtocolError("Stage-70 source-block index record geometry drifted.")
    return members


def _validate_content_index(
    path: Path,
    *,
    dynamic_members: set[str],
) -> dict[str, str]:
    payload = _json(path / "manifests/content_index.json")
    _exact_keys(payload, {"schema_version", "files"}, "content index")
    if payload["schema_version"] != "midogpp_stage70_content_index_v1":
        raise ProtocolError("Stage-70 content-index schema drifted.")
    raw_records = payload["files"]
    if not isinstance(raw_records, list):
        raise ProtocolError("Stage-70 content index lacks file records.")
    indexed: dict[str, str] = {}
    ordered_paths: list[str] = []
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Stage-70 content index contains a non-object row.")
        _exact_keys(raw, {"path", "sha256"}, "content-index row")
        relative = _safe_relative_path(raw["path"], "content-index member")
        if relative in {"manifests/content_index.json", "reports/validation_report.json"}:
            raise ProtocolError("Stage-70 content index contains a self/excluded member.")
        if relative in indexed:
            raise ProtocolError("Stage-70 content index duplicates a file.")
        indexed[relative] = _hash_string(raw["sha256"], "content-index SHA-256", 64)
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths):
        raise ProtocolError("Stage-70 content-index rows are not canonical.")

    excluded = {"manifests/content_index.json", "reports/validation_report.json"}
    actual_paths = {
        member.relative_to(path).as_posix()
        for member in path.rglob("*")
        if member.is_file() and member.relative_to(path).as_posix() not in excluded
    }
    static_members = set(REQUIRED_FILES).difference(excluded)
    allowed_members = static_members | dynamic_members
    if actual_paths != allowed_members:
        missing_members = allowed_members - actual_paths
        if any(member.startswith("arrays/source_blocks/") for member in missing_members):
            raise ProtocolError(
                "Stage-70 source-block member is missing: "
                f"{sorted(missing_members)}."
            )
        raise ProtocolError(
            "Stage-70 artifact membership is not closed-world: "
            f"missing={sorted(allowed_members - actual_paths)}, "
            f"unexpected={sorted(actual_paths - allowed_members)}."
        )
    if set(indexed) != actual_paths:
        raise ProtocolError("Stage-70 content index is not exact closed-world membership.")
    for relative, expected_sha in indexed.items():
        if _sha256_file(path / relative) != expected_sha:
            raise ProtocolError(f"Stage-70 content member hash drifted: {relative}.")
    return indexed


def _validate_protocol_manifest(path: Path) -> dict[str, object]:
    payload = _json(path / "manifests/protocol_manifest.json")
    expected_keys = {
        "schema_version",
        "experiment_id",
        "claim_scope",
        "config_contract_hash",
        "final_authorization_hash",
        "authorization_protocol_hash",
        "target_cache_content_hash",
        "target_cache_row_order_hash",
        "dataset_contract_hash",
        "scoring_manifest_sha256",
        "representation_id",
        "backbone_identity_hash",
        "policy_arms",
        "evaluation_split",
        "fresh_confirmatory_evidence",
        "fresh_confirmatory_status",
        "routing_policy_promotion_allowed",
        "deployment_claim_allowed",
        "target_support_used",
        "policy_or_seed_selection_performed",
        "predictions_persisted_before_labels_opened",
        "labels_used_for_scoring_only",
        "protocol_hash",
    }
    _exact_keys(payload, expected_keys, "protocol manifest")
    fixed = {
        "schema_version": "midogpp_stage70_descriptive_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "policy_arms": list(POLICY_ARMS),
        "evaluation_split": _EVALUATION_SPLIT,
        "fresh_confirmatory_evidence": False,
        "fresh_confirmatory_status": _FRESH_STATUS,
        "routing_policy_promotion_allowed": False,
        "deployment_claim_allowed": False,
        "target_support_used": False,
        "policy_or_seed_selection_performed": False,
        "predictions_persisted_before_labels_opened": True,
        "labels_used_for_scoring_only": True,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise ProtocolError("Stage-70 protocol claim/firewall contract drifted.")
    for field in (
        "config_contract_hash",
        "final_authorization_hash",
        "authorization_protocol_hash",
        "target_cache_content_hash",
        "target_cache_row_order_hash",
        "dataset_contract_hash",
        "backbone_identity_hash",
    ):
        _hash_string(payload[field], f"protocol {field}", (16, 64))
    scoring_sha = _hash_string(
        payload["scoring_manifest_sha256"], "protocol scoring manifest", 64
    )
    if payload["dataset_contract_hash"] != scoring_sha:
        raise ProtocolError("Stage-70 dataset/scoring-manifest contract drifted.")
    if not isinstance(payload["representation_id"], str) or not payload["representation_id"]:
        raise ProtocolError("Stage-70 representation identity is empty.")
    observed_hash = _hash_string(payload["protocol_hash"], "protocol hash", 16)
    unhashed = dict(payload)
    unhashed.pop("protocol_hash")
    if observed_hash != _stable_hash(unhashed):
        raise ProtocolError("Stage-70 protocol semantic hash drifted.")
    return payload


def _validate_authorization_phase(
    path: Path,
    protocol: Mapping[str, object],
) -> dict[str, object]:
    payload = _json(path / "reports/phase_01_authorization_complete.json")
    fixed = {
        "schema_version": "midogpp_stage70_phase_01_authorization_binding_v2",
        "phase": "AUTHORIZATION_COMPLETE",
        "final_authorization_artifact_id": FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
        "final_authorization_hash": protocol["final_authorization_hash"],
        "authorization_protocol_hash": protocol["authorization_protocol_hash"],
        "target_cache_artifact_id": CACHE_ARTIFACT_ID,
        "target_cache_content_hash": protocol["target_cache_content_hash"],
        "target_cache_row_order_hash": protocol["target_cache_row_order_hash"],
        "target_cache_row_count": EXPECTED_TEST_ROWS,
        "scoring_manifest_sha256": protocol["scoring_manifest_sha256"],
        "target_labels_opened": False,
    }
    if any(payload.get(field) != value for field, value in fixed.items()):
        raise ProtocolError("Stage-70 authorization phase binding drifted.")
    for field in (
        "final_authorization_content_hash",
        "identity_lock_hash",
        "evaluation_plan_hash",
        "reservation_content_hash",
        "reservation_identity_lock_hash",
        "target_evaluation_reservation_protocol_hash",
        "target_identity_table_hash",
        "cache_extractor_protocol_hash",
        "classifier_config_hash",
        "authorized_cell_hash",
        "global_target_identity_hash",
        "authorization_binding_hash",
    ):
        _hash_string(payload.get(field), f"authorization {field}", (16, 64))
    return payload


def _persisted_capability_from_disk(path: Path) -> PersistedPredictionPass:
    phase_01 = _json(path / "reports/phase_01_authorization_complete.json")
    return PersistedPredictionPass(
        artifact_root=path,
        authorization_binding_hash=_hash_string(
            phase_01.get("authorization_binding_hash"),
            "authorization binding hash",
            16,
        ),
        phase_01_sha256=_sha256_file(
            path / "reports/phase_01_authorization_complete.json"
        ),
        prediction_index_sha256=_sha256_file(path / "manifests/prediction_index.json"),
        prediction_arrays_sha256=_sha256_file(path / "arrays/target_predictions.npz"),
        prediction_seal_sha256=_sha256_file(path / "manifests/prediction_seal.json"),
        phase_02_sha256=_sha256_file(path / "reports/phase_02_predictions_persisted.json"),
    )


def _validate_evaluation_plan(
    path: Path,
    protocol: Mapping[str, object],
    authorization: Mapping[str, object],
) -> dict[GridKey, Mapping[str, object]]:
    payload = _json(path / "manifests/evaluation_plan.json")
    _exact_keys(
        payload,
        {
            "schema_version",
            "final_authorization_hash",
            "classifier_config_hash",
            "policy_arms",
            "records",
            "prediction_cells",
            "training_seeds_retained",
            "generation_seeds_retained",
            "seed_selection",
            "target_labels_opened",
            "evaluation_plan_hash",
        },
        "evaluation plan",
    )
    fixed = {
        "schema_version": "midogpp_stage70_descriptive_evaluation_plan_v1",
        "final_authorization_hash": authorization["final_authorization_hash"],
        "policy_arms": list(POLICY_ARMS),
        "prediction_cells": 243,
        "training_seeds_retained": list(TRAINING_SEEDS),
        "generation_seeds_retained": list(GENERATION_SEEDS),
        "seed_selection": False,
        "target_labels_opened": False,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise ProtocolError("Stage-70 evaluation-plan protocol drifted.")
    _hash_string(payload["classifier_config_hash"], "classifier config hash", (16, 64))
    observed_hash = _hash_string(payload["evaluation_plan_hash"], "evaluation-plan hash", 16)
    unhashed = dict(payload)
    unhashed.pop("evaluation_plan_hash")
    if observed_hash != _stable_hash(unhashed):
        raise ProtocolError("Stage-70 evaluation-plan semantic hash drifted.")
    records = _mapping_records(payload.get("records"), 243, "evaluation plan")
    by_key = _grid_records(records, "evaluation plan")
    expected_row_keys = {
        "policy_id",
        "target_center",
        "training_seed",
        "generation_seed",
        "replicate_id",
        "policy_lock_hash",
        "policy_plan_hash",
        "assignment_table_hash",
        "assignment_count",
        "synthetic_rows_per_class",
        "target_expert_excluded",
    }
    for row in records:
        _exact_keys(row, expected_row_keys, "evaluation-plan row")
        for field in (
            "replicate_id",
            "policy_lock_hash",
            "policy_plan_hash",
            "assignment_table_hash",
        ):
            _hash_string(row[field], f"evaluation {field}", (16, 64))
        assignment_count = _json_int(row.get("assignment_count"), "assignment count")
        if (
            assignment_count <= 0
            or assignment_count > 8
            or row.get("synthetic_rows_per_class") != TOTAL_PER_CLASS
            or row.get("target_expert_excluded") is not True
        ):
            raise ProtocolError("Stage-70 evaluation-plan row semantics drifted.")
    if protocol["policy_arms"] != payload["policy_arms"]:
        raise ProtocolError("Stage-70 protocol/evaluation arm binding drifted.")
    classifier_hash = str(payload["classifier_config_hash"])
    return {
        key: {**dict(row), "__plan_classifier_config_hash": classifier_hash}
        for key, row in by_key.items()
    }


def _validate_source_block_index(
    path: Path,
    protocol: Mapping[str, object],
    *,
    content_hashes: Mapping[str, str],
) -> int:
    payload = _json(path / "manifests/source_block_index.json")
    _exact_keys(
        payload,
        {"schema_version", "source_block_count", "records", "target_labels_opened"},
        "source-block index",
    )
    records = _mapping_records(payload.get("records"), 81, "source-block index")
    if (
        payload.get("schema_version") != "midogpp_stage70_source_block_index_v1"
        or payload.get("source_block_count") != 81
        or payload.get("target_labels_opened") is not False
    ):
        raise ProtocolError("Stage-70 source-block index header drifted.")
    expected_keys = {
        "schema_version",
        "cache_key",
        "cache_status",
        "source_center",
        "training_seed",
        "generation_seed",
        "source_stream_id",
        "expert_lock_hash",
        "checkpoint_hash",
        "bank_lock_hash",
        "generation_lock_hash",
        "dataset_contract_hash",
        "evaluation_split",
        "representation_id",
        "backbone_identity_hash",
        "budget_per_class",
        "output_sha256",
        "path",
        "persistent_path",
        "member_path",
        "member_sha256",
        "artifact_member",
    }
    observed_grid: set[SourceKey] = set()
    stream_ids: set[str] = set()
    member_paths: set[str] = set()
    for row in records:
        _exact_keys(row, expected_keys, "source-block row")
        source_key = (
            _json_str(row.get("source_center"), "source center"),
            _json_int(row.get("training_seed"), "source training seed"),
            _json_int(row.get("generation_seed"), "source generation seed"),
        )
        if source_key in observed_grid:
            raise ProtocolError("Stage-70 source-block grid is duplicated.")
        observed_grid.add(source_key)
        stream_id = _hash_string(row.get("source_stream_id"), "source stream id", 16)
        cache_key = _hash_string(row.get("cache_key"), "source cache key", 16)
        if stream_id in stream_ids:
            raise ProtocolError("Stage-70 source stream identity is reused.")
        stream_ids.add(stream_id)
        if row.get("schema_version") != SOURCE_BLOCK_CACHE_SCHEMA:
            raise ProtocolError("Stage-70 source-block record schema drifted.")
        if row.get("cache_status") not in {"GENERATED", "REUSED_VALIDATED"}:
            raise ProtocolError("Stage-70 source-block cache status drifted.")
        for field in (
            "expert_lock_hash",
            "checkpoint_hash",
            "bank_lock_hash",
            "generation_lock_hash",
            "dataset_contract_hash",
            "backbone_identity_hash",
        ):
            _hash_string(row.get(field), f"source {field}", (16, 64))
        if (
            row.get("dataset_contract_hash") != protocol["dataset_contract_hash"]
            or row.get("evaluation_split") != _EVALUATION_SPLIT
            or row.get("representation_id") != protocol["representation_id"]
            or row.get("backbone_identity_hash") != protocol["backbone_identity_hash"]
            or row.get("budget_per_class") != TOTAL_PER_CLASS
        ):
            raise ProtocolError("Stage-70 source-block provenance drifted.")
        output_sha = _hash_string(row.get("output_sha256"), "source output hash", 64)
        filename = _safe_filename(row.get("path"), "source-block filename")
        if filename != f"{cache_key}.npz":
            raise ProtocolError("Stage-70 source-block filename/cache identity drifted.")
        persistent = _safe_filename(row.get("persistent_path"), "persistent source-block path")
        relative = _safe_relative_path(row.get("member_path"), "source-block member")
        artifact_member = _safe_relative_path(
            row.get("artifact_member"), "source-block artifact member"
        )
        member_sha = _hash_string(row.get("member_sha256"), "source member SHA-256", 64)
        if (
            persistent != filename
            or relative != f"arrays/source_blocks/{filename}"
            or artifact_member != relative
            or relative in member_paths
        ):
            raise ProtocolError("Stage-70 source-block member path drifted or duplicated.")
        member_paths.add(relative)
        member = path / relative
        if not member.is_file() or member.is_symlink():
            raise ProtocolError("Stage-70 source-block member is missing or unsafe.")
        if content_hashes.get(relative) != member_sha or _sha256_file(member) != member_sha:
            raise ProtocolError("Stage-70 source-block member SHA-256 drifted.")
        expected_internal_key = _expected_source_internal_key(row)
        expected_cache_key = _stable_hash(
            {
                "schema_version": SOURCE_BLOCK_CACHE_SCHEMA,
                "protocol_version": SOURCE_BLOCK_CACHE_SCHEMA,
                "source_generation_key": expected_internal_key,
                "checkpoint_hash": row["checkpoint_hash"],
                "bank_lock_hash": row["bank_lock_hash"],
                "generation_lock_hash": row["generation_lock_hash"],
                "dataset_contract_hash": row["dataset_contract_hash"],
                "evaluation_split": row["evaluation_split"],
                "representation_id": row["representation_id"],
                "backbone_identity_hash": row["backbone_identity_hash"],
                "budget_per_class": TOTAL_PER_CLASS,
            }
        )
        if cache_key != expected_cache_key:
            raise ProtocolError("Stage-70 source-block cache key is not reproducible.")
        _validate_source_block_member(
            member,
            row=row,
            expected_internal_key=expected_internal_key,
            expected_output_sha256=output_sha,
        )
    if observed_grid != _EXPECTED_SOURCE_GRID:
        raise ProtocolError("Stage-70 source-block grid is incomplete or out of scope.")
    return len(records)


def _expected_source_internal_key(row: Mapping[str, object]) -> dict[str, object]:
    center = str(row["source_center"])
    training_seed = int(row["training_seed"])
    generation_seed = int(row["generation_seed"])
    expert_lock_hash = str(row["expert_lock_hash"])
    bank_lock_hash = str(row["bank_lock_hash"])
    class_seed_by_label = {
        str(label): _derived_generation_seed(
            bank_lock_hash=bank_lock_hash,
            expert_lock_hash=expert_lock_hash,
            generation_seed=generation_seed,
            class_label=label,
        )
        for label in (0, 1)
    }
    stream_id = _stable_hash(
        {
            "namespace": SOURCE_STREAM_NAMESPACE,
            "bank_lock_hash": bank_lock_hash,
            "expert_lock_hash": expert_lock_hash,
            "source_center": center,
            "training_seed": training_seed,
            "generation_seed": generation_seed,
        }
    )
    if stream_id != row["source_stream_id"]:
        raise ProtocolError("Stage-70 source stream identity is not reproducible.")
    return {
        "schema_version": "midogpp_uniform_b_v2_source_generation_key_v1",
        "source_center": center,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "expert_lock_hash": expert_lock_hash,
        "stream_id": stream_id,
        "class_seed_by_label": class_seed_by_label,
        "max_samples_per_class": TOTAL_PER_CLASS,
        "equal_union_prefix_per_class": SOURCE_BUDGET_PER_CLASS,
    }


def _validate_source_block_member(
    member: Path,
    *,
    row: Mapping[str, object],
    expected_internal_key: Mapping[str, object],
    expected_output_sha256: str,
) -> None:
    try:
        with np.load(member, allow_pickle=False) as archive:
            if set(archive.files) != {"embeddings", "labels", "metadata_json"}:
                raise ProtocolError("Stage-70 source-block archive is not closed-world.")
            embeddings = np.asarray(archive["embeddings"])
            labels = np.asarray(archive["labels"])
            metadata_raw = np.asarray(archive["metadata_json"])
            if metadata_raw.shape != ():
                raise ProtocolError("Stage-70 source-block metadata is not scalar.")
            metadata = json.loads(str(metadata_raw.item()))
    except ProtocolError:
        raise
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read Stage-70 source-block member: {member}.") from exc
    if not isinstance(metadata, Mapping):
        raise ProtocolError("Stage-70 source-block metadata is not an object.")
    _exact_keys(
        metadata,
        {
            "schema_version",
            "protocol_version",
            "cache_key",
            "source_generation_key",
            "checkpoint_hash",
            "bank_lock_hash",
            "generation_lock_hash",
            "dataset_contract_hash",
            "evaluation_split",
            "representation_id",
            "backbone_identity_hash",
            "budget_per_class",
            "output_sha256",
        },
        "source metadata",
    )
    expected_metadata = {
        "schema_version": SOURCE_BLOCK_CACHE_SCHEMA,
        "protocol_version": SOURCE_BLOCK_CACHE_SCHEMA,
        "cache_key": row["cache_key"],
        "source_generation_key": dict(expected_internal_key),
        "checkpoint_hash": row["checkpoint_hash"],
        "bank_lock_hash": row["bank_lock_hash"],
        "generation_lock_hash": row["generation_lock_hash"],
        "dataset_contract_hash": row["dataset_contract_hash"],
        "evaluation_split": row["evaluation_split"],
        "representation_id": row["representation_id"],
        "backbone_identity_hash": row["backbone_identity_hash"],
        "budget_per_class": TOTAL_PER_CLASS,
        "output_sha256": expected_output_sha256,
    }
    if dict(metadata) != expected_metadata:
        raise ProtocolError("Stage-70 source-block embedded metadata drifted.")
    if (
        embeddings.dtype != np.dtype(np.float32)
        or labels.dtype != np.dtype(np.int64)
        or embeddings.shape != (2 * TOTAL_PER_CLASS, COMMON_OUTPUT_DIM)
        or labels.shape != (2 * TOTAL_PER_CLASS,)
        or not np.isfinite(embeddings).all()
        or int(np.sum(labels == 0)) != TOTAL_PER_CLASS
        or int(np.sum(labels == 1)) != TOTAL_PER_CLASS
        or set(int(value) for value in np.unique(labels)) != {0, 1}
        or _array_bundle_sha256(embeddings, labels) != expected_output_sha256
    ):
        raise ProtocolError("Stage-70 source-block content/geometry drifted.")


def _validate_composition_index(
    path: Path,
    evaluation: Mapping[GridKey, Mapping[str, object]],
) -> dict[GridKey, Mapping[str, object]]:
    payload = _json(path / "manifests/composition_index.json")
    _exact_keys(
        payload,
        {"schema_version", "composition_count", "records", "target_labels_opened"},
        "composition index",
    )
    records = _mapping_records(payload.get("records"), 243, "composition index")
    if (
        payload.get("schema_version") != "midogpp_stage70_composition_index_v1"
        or payload.get("composition_count") != 243
        or payload.get("target_labels_opened") is not False
    ):
        raise ProtocolError("Stage-70 composition-index header drifted.")
    by_key = _grid_records(records, "composition index")
    expected_keys = {
        "policy_id",
        "target_center",
        "training_seed",
        "generation_seed",
        "replicate_id",
        "policy_lock_hash",
        "assignment_table_hash",
        "composition_manifest_hash",
        "train_content_sha256",
        "pre_shuffle_sha256_by_label",
        "post_shuffle_sha256_by_label",
    }
    for key, row in by_key.items():
        _exact_keys(row, expected_keys, "composition row")
        plan = evaluation[key]
        for field in ("replicate_id", "policy_lock_hash", "assignment_table_hash"):
            if row.get(field) != plan.get(field):
                raise ProtocolError("Stage-70 evaluation/composition provenance drifted.")
        _hash_string(row.get("composition_manifest_hash"), "composition manifest hash", 16)
        _hash_string(row.get("train_content_sha256"), "composition content hash", 64)
        for field in ("pre_shuffle_sha256_by_label", "post_shuffle_sha256_by_label"):
            hashes = row.get(field)
            if not isinstance(hashes, Mapping) or set(hashes) != {"0", "1"}:
                raise ProtocolError("Stage-70 composition class hashes drifted.")
            for value in hashes.values():
                _hash_string(value, f"composition {field}", 64)
    for center in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                control = by_key[(CONTROL_ARM, center, training_seed, generation_seed)]
                utility = by_key[(UTILITY_ARM, center, training_seed, generation_seed)]
                for field in (
                    "train_content_sha256",
                    "pre_shuffle_sha256_by_label",
                    "post_shuffle_sha256_by_label",
                ):
                    if utility[field] != control[field]:
                        raise ProtocolError("Stage-70 utility/control composition is not exact.")
    return by_key


def _validate_prediction_joins(
    records: Sequence[Mapping[str, object]],
    *,
    evaluation: Mapping[GridKey, Mapping[str, object]],
    composition: Mapping[GridKey, Mapping[str, object]],
) -> dict[GridKey, Mapping[str, object]]:
    by_key = _grid_records(records, "verified prediction index")
    for key, row in by_key.items():
        plan = evaluation[key]
        composed = composition[key]
        expected = {
            "replicate_id": plan["replicate_id"],
            "composition_manifest_hash": composed["composition_manifest_hash"],
            "train_content_sha256": composed["train_content_sha256"],
        }
        if any(row.get(field) != value for field, value in expected.items()):
            raise ProtocolError("Stage-70 plan/composition/prediction join drifted.")
        if row.get("classifier_config_hash") != _evaluation_classifier_hash(evaluation):
            raise ProtocolError("Stage-70 prediction classifier provenance drifted.")
        for field in (
            "prediction_sha256",
            "probability_sha256",
            "prediction_cell_hash",
            "target_identity_hash",
            "composition_manifest_hash",
            "train_content_sha256",
            "classifier_config_hash",
            "scaler_state_hash",
            "target_row_order_hash",
        ):
            _hash_string(row.get(field), f"prediction {field}", (16, 64))
        expected_reuse = CONTROL_ARM if key[0] == UTILITY_ARM else ""
        if row.get("reused_from_policy_id") != expected_reuse:
            raise ProtocolError("Stage-70 prediction reuse provenance drifted.")
    return by_key


def _validate_scoring_phases(
    path: Path,
    *,
    protocol: Mapping[str, object],
    verified: VerifiedPredictionArtifact,
) -> str:
    labels_opened = _json(path / "reports/phase_03_labels_opened.json")
    complete = _json(path / "reports/phase_04_scoring_complete.json")
    label_sha = _hash_string(
        labels_opened.get("label_manifest_sha256"), "label manifest SHA-256", 64
    )
    expected_open = {
        "schema_version": "midogpp_stage70_phase_marker_v1",
        "phase": "LABELS_OPENED_AFTER_PREDICTIONS_PERSISTED",
        "authorization_binding_hash": verified.authorization_binding_hash,
        "final_authorization_hash": verified.phase_01_binding["final_authorization_hash"],
        "target_cache_content_hash": verified.phase_01_binding["target_cache_content_hash"],
        "phase_01_sha256": verified.phase_01_sha256,
        "prediction_index_sha256": verified.prediction_index_sha256,
        "prediction_arrays_sha256": verified.prediction_arrays_sha256,
        "prediction_seal_sha256": verified.prediction_seal_sha256,
        "phase_02_sha256": verified.phase_02_sha256,
        "label_manifest_sha256": label_sha,
        "labels_used_for_scoring_only": True,
    }
    expected_complete = {
        "schema_version": "midogpp_stage70_phase_marker_v1",
        "phase": "SCORING_COMPLETE",
        "metric_row_count": 243,
        "authorization_binding_hash": verified.authorization_binding_hash,
        "final_authorization_hash": verified.phase_01_binding["final_authorization_hash"],
        "target_cache_content_hash": verified.phase_01_binding["target_cache_content_hash"],
        "phase_01_sha256": verified.phase_01_sha256,
        "prediction_index_sha256": verified.prediction_index_sha256,
        "prediction_arrays_sha256": verified.prediction_arrays_sha256,
        "prediction_seal_sha256": verified.prediction_seal_sha256,
        "phase_02_sha256": verified.phase_02_sha256,
        "label_manifest_sha256": label_sha,
    }
    if labels_opened != expected_open or complete != expected_complete:
        raise ProtocolError("Stage-70 scoring phase/hash binding drifted.")
    if label_sha != protocol["scoring_manifest_sha256"]:
        raise ProtocolError("Stage-70 scoring labels differ from the authorized manifest.")
    return label_sha


def _validate_metrics(
    rows: list[dict[str, str]],
    *,
    predictions: Mapping[GridKey, Mapping[str, object]],
    evaluation: Mapping[GridKey, Mapping[str, object]],
    label_manifest_sha256: str,
    verified: VerifiedPredictionArtifact,
) -> dict[GridKey, dict[str, str]]:
    if len(rows) != 243:
        raise ProtocolError("Stage-70 metric row count drifted.")
    by_key: dict[GridKey, dict[str, str]] = {}
    classifier_hash = _evaluation_classifier_hash(evaluation)
    phase = verified.phase_01_binding
    shard_hashes = phase.get("target_cache_shard_sha256_by_center")
    if not isinstance(shard_hashes, Mapping):
        raise ProtocolError("Stage-70 phase-01 shard identities are malformed.")
    for row in rows:
        key = _csv_grid_key(row, "metric")
        if key in by_key:
            raise ProtocolError("Stage-70 metric grid contains a duplicate cell.")
        by_key[key] = row
        prediction = predictions[key]
        fixed = {
            "schema_version": "midogpp_stage70_target_metric_v1",
            "claim_scope": CLAIM_SCOPE,
            "claim_role": "descriptive_locked_policy_comparison",
            "row_role": "target_evaluation_metric",
            "macro_f1_role": "secondary_descriptive_only",
            "target_labels_used_for_scoring_only": "True",
            "fresh_confirmatory_evidence": "False",
            "policy_or_seed_selection_performed": "False",
            "label_manifest_sha256": label_manifest_sha256,
            "classifier_config_hash": classifier_hash,
            "authorization_binding_hash": verified.authorization_binding_hash,
            "final_authorization_hash": str(phase["final_authorization_hash"]),
            "authorization_protocol_hash": str(phase["authorization_protocol_hash"]),
            "identity_lock_hash": str(phase["identity_lock_hash"]),
            "evaluation_plan_hash": str(phase["evaluation_plan_hash"]),
            "reservation_content_hash": str(phase["reservation_content_hash"]),
            "target_evaluation_reservation_id": str(
                phase["target_evaluation_reservation_id"]
            ),
            "target_evaluation_reservation_protocol_hash": str(
                phase["target_evaluation_reservation_protocol_hash"]
            ),
            "target_cache_artifact_id": str(phase["target_cache_artifact_id"]),
            "target_cache_content_hash": str(phase["target_cache_content_hash"]),
            "target_cache_row_order_hash": str(phase["target_cache_row_order_hash"]),
            "target_cache_shard_sha256": str(shard_hashes[key[1]]),
            "phase_01_sha256": verified.phase_01_sha256,
            "prediction_index_sha256": verified.prediction_index_sha256,
            "prediction_arrays_sha256": verified.prediction_arrays_sha256,
            "prediction_seal_sha256": verified.prediction_seal_sha256,
            "phase_02_sha256": verified.phase_02_sha256,
        }
        if any(row.get(field) != value for field, value in fixed.items()):
            raise ProtocolError("Stage-70 metric claim/protocol fields drifted.")
        prediction_join = {
            "replicate_id": prediction.get("replicate_id"),
            "prediction_sha256": prediction.get("prediction_sha256"),
            "probability_sha256": prediction.get("probability_sha256"),
            "prediction_cell_hash": prediction.get("prediction_cell_hash"),
            "target_identity_hash": prediction.get("target_identity_hash"),
            "composition_manifest_hash": prediction.get("composition_manifest_hash"),
            "train_content_sha256": prediction.get("train_content_sha256"),
            "classifier_config_hash": prediction.get("classifier_config_hash"),
            "scaler_state_hash": prediction.get("scaler_state_hash"),
            "target_row_order_hash": prediction.get("target_row_order_hash"),
            "reused_from_policy_id": prediction.get("reused_from_policy_id"),
        }
        if any(row.get(field) != str(value) for field, value in prediction_join.items()):
            raise ProtocolError("Stage-70 metric/prediction provenance join drifted.")
        n_eval = _csv_int(row, "n_eval", "metric")
        n_cases = _csv_int(row, "n_cases", "metric")
        prediction_cases = prediction.get("case_ids")
        prediction_ids = prediction.get("evaluation_row_ids")
        if (
            not isinstance(prediction_ids, Sequence)
            or isinstance(prediction_ids, (str, bytes))
            or n_eval != len(prediction_ids)
            or not isinstance(prediction_cases, Sequence)
            or isinstance(prediction_cases, (str, bytes))
            or n_cases != len(set(str(value) for value in prediction_cases))
            or n_cases <= 0
        ):
            raise ProtocolError("Stage-70 metric evaluation geometry drifted.")
        for field in ("bacc", "macro_f1"):
            value = _csv_float(row, field, "metric")
            if value < 0.0 or value > 1.0:
                raise ProtocolError("Stage-70 metric value is outside [0, 1].")
    if set(by_key) != _EXPECTED_GRID:
        raise ProtocolError("Stage-70 metric grid is incomplete or out of scope.")
    for center in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                control = by_key[(CONTROL_ARM, center, training_seed, generation_seed)]
                utility = by_key[(UTILITY_ARM, center, training_seed, generation_seed)]
                for field in (
                    "bacc",
                    "macro_f1",
                    "prediction_sha256",
                    "probability_sha256",
                    "train_content_sha256",
                    "classifier_config_hash",
                    "scaler_state_hash",
                    "target_row_order_hash",
                ):
                    if utility[field] != control[field]:
                        raise ProtocolError("Stage-70 utility/control metric provenance drifted.")
    return by_key


def _validate_case_confusions(
    rows: list[dict[str, str]],
    *,
    metrics: Mapping[GridKey, Mapping[str, str]],
    predictions: Mapping[GridKey, Mapping[str, object]],
) -> None:
    if not rows:
        raise ProtocolError("Stage-70 case-confusion table is empty.")
    grouped: dict[GridKey, list[dict[str, str]]] = {}
    observed_keys: set[tuple[GridKey, str]] = set()
    truth_geometry: dict[tuple[str, str], tuple[int, int]] = {}
    for row in rows:
        key = _csv_grid_key(row, "case confusion")
        case_id = row.get("case_id", "")
        identity = (key, case_id)
        if not case_id or identity in observed_keys:
            raise ProtocolError("Stage-70 case-confusion identity is empty or duplicated.")
        observed_keys.add(identity)
        if (
            row.get("schema_version") != "midogpp_stage70_case_confusion_v1"
            or row.get("target_labels_used_for_scoring_only") != "True"
        ):
            raise ProtocolError("Stage-70 case-confusion schema/firewall drifted.")
        metric = metrics[key]
        for field in (
            "replicate_id",
            "prediction_sha256",
            "target_identity_hash",
            "label_manifest_sha256",
            "authorization_binding_hash",
            "prediction_index_sha256",
            "prediction_arrays_sha256",
        ):
            if row.get(field) != metric.get(field):
                raise ProtocolError("Stage-70 case-confusion provenance join drifted.")
        counts = tuple(_csv_int(row, field, "case confusion") for field in ("tn", "fp", "fn", "tp"))
        if any(value < 0 for value in counts) or sum(counts) <= 0:
            raise ProtocolError("Stage-70 case-confusion counts are invalid.")
        geometry = (counts[0] + counts[1], counts[2] + counts[3])
        previous = truth_geometry.setdefault((key[1], case_id), geometry)
        if previous != geometry:
            raise ProtocolError("Stage-70 truth geometry changes across policy/seed cells.")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != _EXPECTED_GRID:
        raise ProtocolError("Stage-70 case-confusion grid is incomplete.")
    for key, group in grouped.items():
        prediction_cases = predictions[key].get("case_ids")
        if not isinstance(prediction_cases, Sequence) or isinstance(
            prediction_cases, (str, bytes)
        ):
            raise ProtocolError("Stage-70 prediction case identities are malformed.")
        expected_cases = {str(value) for value in prediction_cases}
        if {row["case_id"] for row in group} != expected_cases:
            raise ProtocolError("Stage-70 confusion/prediction case coverage drifted.")
        totals = {
            field: sum(_csv_int(row, field, "case confusion") for row in group)
            for field in ("tn", "fp", "fn", "tp")
        }
        negative = totals["tn"] + totals["fp"]
        positive = totals["tp"] + totals["fn"]
        if negative <= 0 or positive <= 0:
            raise ProtocolError("Stage-70 metric cell lacks a class denominator.")
        bacc = 0.5 * (totals["tn"] / negative + totals["tp"] / positive)
        f1_negative = _safe_f1(2 * totals["tn"], 2 * totals["tn"] + totals["fp"] + totals["fn"])
        f1_positive = _safe_f1(2 * totals["tp"], 2 * totals["tp"] + totals["fp"] + totals["fn"])
        macro_f1 = 0.5 * (f1_negative + f1_positive)
        metric = metrics[key]
        if (
            sum(totals.values()) != _csv_int(metric, "n_eval", "metric")
            or len(group) != _csv_int(metric, "n_cases", "metric")
            or not _close(bacc, _csv_float(metric, "bacc", "metric"))
            or not _close(macro_f1, _csv_float(metric, "macro_f1", "metric"))
        ):
            raise ProtocolError("Stage-70 metric/confusion reconstruction drifted.")


def _validate_arm_summaries(
    rows: list[dict[str, str]],
    *,
    metrics: Mapping[GridKey, Mapping[str, str]],
) -> None:
    if len(rows) != 3 or {row.get("policy_id") for row in rows} != set(POLICY_ARMS):
        raise ProtocolError("Stage-70 arm-summary key geometry drifted.")
    for row in rows:
        arm = str(row["policy_id"])
        arm_rows = [metric for key, metric in metrics.items() if key[0] == arm]
        expected = {
            "equal_center_equal_seed_mean_bacc": _equal_center_mean(arm_rows, "bacc"),
            "equal_center_equal_seed_mean_macro_f1": _equal_center_mean(arm_rows, "macro_f1"),
            "minimum_cell_bacc": min(_csv_float(item, "bacc", "metric") for item in arm_rows),
            "maximum_cell_bacc": max(_csv_float(item, "bacc", "metric") for item in arm_rows),
        }
        if (
            row.get("schema_version") != "midogpp_stage70_arm_summary_v1"
            or row.get("fresh_confirmatory_evidence") != "False"
            or _csv_int(row, "cell_count", "arm summary") != 81
            or any(not _close(_csv_float(row, field, "arm summary"), value) for field, value in expected.items())
        ):
            raise ProtocolError("Stage-70 arm summary is not derived from metric rows.")


def _validate_paired_deltas(
    rows: list[dict[str, str]],
    *,
    metrics: Mapping[GridKey, Mapping[str, str]],
) -> dict[str, list[dict[str, str]]]:
    definitions = {
        "metadata_max_tie_union_minus_equal_union": (
            METADATA_ARM,
            "sole_predeclared_descriptive_policy_contrast",
        ),
        "utility_regret_minus_equal_union": (
            UTILITY_ARM,
            "deterministic_fallback_equivalence_audit",
        ),
    }
    grouped: dict[str, list[dict[str, str]]] = {key: [] for key in definitions}
    observed: set[tuple[str, str, int, int]] = set()
    for row in rows:
        comparison = row.get("comparison_id", "")
        if comparison not in definitions:
            raise ProtocolError("Stage-70 paired-delta comparison is undeclared.")
        arm, role = definitions[comparison]
        key = _csv_grid_key({**row, "policy_id": arm}, "paired delta")
        delta_key = (comparison, key[1], key[2], key[3])
        if delta_key in observed:
            raise ProtocolError("Stage-70 paired-delta grid contains a duplicate.")
        observed.add(delta_key)
        if (
            row.get("schema_version") != "midogpp_stage70_paired_delta_v1"
            or row.get("policy_id") != arm
            or row.get("control_policy_id") != CONTROL_ARM
            or row.get("role") != role
            or row.get("paired") != "True"
            or row.get("fresh_confirmatory_evidence") != "False"
        ):
            raise ProtocolError("Stage-70 paired-delta contract drifted.")
        control = metrics[(CONTROL_ARM, key[1], key[2], key[3])]
        policy = metrics[(arm, key[1], key[2], key[3])]
        expected_bacc = _csv_float(policy, "bacc", "metric") - _csv_float(control, "bacc", "metric")
        expected_f1 = _csv_float(policy, "macro_f1", "metric") - _csv_float(control, "macro_f1", "metric")
        if (
            not _close(_csv_float(row, "bacc_delta", "paired delta"), expected_bacc)
            or not _close(_csv_float(row, "macro_f1_delta", "paired delta"), expected_f1)
        ):
            raise ProtocolError("Stage-70 paired delta is not derived from metric rows.")
        grouped[comparison].append(row)
    if len(rows) != 162 or any(len(group) != 81 for group in grouped.values()):
        raise ProtocolError("Stage-70 paired-delta key geometry drifted.")
    return grouped


def _validate_bootstrap(
    rows: list[dict[str, str]],
    *,
    deltas: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    expected_ids = {
        "metadata_max_tie_union_minus_equal_union": "metadata_max_tie_union_minus_equal_union",
        "utility_regret_minus_equal_union_equivalence": "utility_regret_minus_equal_union",
    }
    if len(rows) != 2 or {row.get("comparison_id") for row in rows} != set(expected_ids):
        raise ProtocolError("Stage-70 bootstrap comparison geometry drifted.")
    accounting: tuple[int, int, int, int] | None = None
    for row in rows:
        comparison = row["comparison_id"]
        valid = _csv_int(row, "valid_replicates", "bootstrap")
        attempted = _csv_int(row, "attempted_replicates", "bootstrap")
        rejected = _csv_int(row, "rejected_replicates", "bootstrap")
        seed = _csv_int(row, "seed", "bootstrap")
        current = (seed, valid, attempted, rejected)
        if accounting is None:
            accounting = current
        if (
            accounting != current
            or valid <= 0
            or rejected < 0
            or attempted != valid + rejected
            or row.get("schema_version") != "midogpp_stage70_descriptive_bootstrap_v1"
            or row.get("centers_resampled") != "True"
            or row.get("cases_resampled_within_center") != "True"
            or row.get("full_crossed_seed_grid_retained") != "True"
            or row.get("flattened_seed_pairs_resampled_as_iid") != "False"
            or row.get("invalid_class_denominator_draws_rejected") != "True"
            or row.get("interval_role") != "descriptive_resampling_uncertainty_only"
            or row.get("fresh_confirmatory_inference") != "False"
        ):
            raise ProtocolError("Stage-70 bootstrap audit accounting drifted.")
        values = [
            _csv_float(item, "bacc_delta", "paired delta")
            for item in deltas[expected_ids[comparison]]
        ]
        observed = _csv_float(row, "observed_mean_bacc_delta", "bootstrap")
        lower = _csv_float(row, "percentile_2_5", "bootstrap")
        upper = _csv_float(row, "percentile_97_5", "bootstrap")
        _csv_float(row, "bootstrap_mean_bacc_delta", "bootstrap")
        if not _close(observed, sum(values) / len(values)) or lower > upper:
            raise ProtocolError("Stage-70 bootstrap summary is not bound to paired deltas.")
        if comparison.startswith("utility_") and any(
            not _close(_csv_float(row, field, "bootstrap"), 0.0)
            for field in (
                "observed_mean_bacc_delta",
                "bootstrap_mean_bacc_delta",
                "percentile_2_5",
                "percentile_97_5",
            )
        ):
            raise ProtocolError("Stage-70 utility/control bootstrap is not exact equivalence.")


def _validate_derived_reports(
    path: Path,
    *,
    protocol: Mapping[str, object],
    authorization: Mapping[str, object],
    prediction_index_sha256: str,
    label_manifest_sha256: str,
    metrics: Mapping[GridKey, Mapping[str, str]],
    deltas: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    leakage = _json(path / "reports/leakage_report.json")
    identity = _json(path / "reports/identity_overlap_report.json")
    equivalence = _json(path / "reports/utility_control_equivalence.json")
    decision = _json(path / "reports/publication_decision.json")
    expected_leakage = {
        "schema_version": "midogpp_stage70_leakage_report_v1",
        "status": "PASS",
        "final_authorization_hash": authorization["final_authorization_hash"],
        "target_labels_opened_after_prediction_seal": True,
        "target_labels_used_for_fit_selection_or_prediction": False,
        "target_labels_used_for_scoring_only": True,
        "target_support_used": False,
        "routing_recomputed": False,
        "stage50_or_stage90_input_used": False,
        "fresh_confirmatory_evidence": False,
    }
    expected_identity = {
        "schema_version": "midogpp_stage70_identity_overlap_v1",
        "status": "PASS",
        "target_expert_assignments": 0,
        "center_4_rows": 0,
        "legacy_label_encoded_identifiers_persisted": 0,
    }
    utility_rows = deltas["utility_regret_minus_equal_union"]
    exact_metric = len(utility_rows) == 81 and all(
        _close(_csv_float(row, "bacc_delta", "paired delta"), 0.0)
        and _close(_csv_float(row, "macro_f1_delta", "paired delta"), 0.0)
        for row in utility_rows
    )
    exact_hashes = all(
        metrics[(UTILITY_ARM, center, training_seed, generation_seed)]["prediction_sha256"]
        == metrics[(CONTROL_ARM, center, training_seed, generation_seed)]["prediction_sha256"]
        and metrics[(UTILITY_ARM, center, training_seed, generation_seed)]["probability_sha256"]
        == metrics[(CONTROL_ARM, center, training_seed, generation_seed)]["probability_sha256"]
        for center in CENTERS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    )
    expected_equivalence = {
        "schema_version": "midogpp_stage70_utility_control_equivalence_v1",
        "status": "PASS",
        "cell_count": len(utility_rows),
        "exact_metric_equivalence": exact_metric,
        "exact_prediction_and_probability_hash_equivalence": exact_hashes,
        "independent_policy_hypothesis_test": False,
    }
    expected_decision = {
        "schema_version": "midogpp_stage70_publication_decision_v1",
        "status": "PASS",
        "decision": "DESCRIPTIVE_COMPARISON_COMPLETE",
        "claim_scope": CLAIM_SCOPE,
        "fresh_confirmatory_status": _FRESH_STATUS,
        "routing_policy_promoted": False,
        "deployment_utility_claimed": False,
        "new_center_generalization_claimed": False,
        "external_generalization_claimed": False,
    }
    if leakage != expected_leakage:
        raise ProtocolError("Stage-70 leakage report is inconsistent with validated phases.")
    if identity != expected_identity:
        raise ProtocolError("Stage-70 identity report is inconsistent with validated grids.")
    if not exact_metric or not exact_hashes or equivalence != expected_equivalence:
        raise ProtocolError("Stage-70 utility-control report is not derived from results.")
    if decision != expected_decision:
        raise ProtocolError("Stage-70 publication decision exceeds the validated claim boundary.")
    if (
        protocol["scoring_manifest_sha256"] != label_manifest_sha256
        or _sha256_file(path / "manifests/prediction_index.json") != prediction_index_sha256
    ):
        raise ProtocolError("Stage-70 derived report phase binding drifted.")


def _evaluation_classifier_hash(
    evaluation: Mapping[GridKey, Mapping[str, object]],
) -> str:
    # The plan-level hash is not repeated in each plan row.  It is loaded from
    # the canonical manifest to avoid inferring classifier identity from a
    # result table.
    values = {
        str(row.get("classifier_config_hash"))
        for row in evaluation.values()
        if "classifier_config_hash" in row
    }
    if values:
        if len(values) != 1:
            raise ProtocolError("Stage-70 evaluation classifier identity drifted.")
        return next(iter(values))
    # Rows produced by v1 do not repeat it; caller replaces this sentinel below.
    first = next(iter(evaluation.values()))
    marker = first.get("__plan_classifier_config_hash")
    if not isinstance(marker, str):
        raise ProtocolError("Stage-70 evaluation plan lacks classifier identity.")
    return marker


def _mapping_records(value: object, count: int, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or len(value) != count or not all(isinstance(row, Mapping) for row in value):
        raise ProtocolError(f"Stage-70 {name} record geometry drifted.")
    return list(value)  # type: ignore[return-value]


def _grid_records(
    records: Sequence[Mapping[str, object]],
    name: str,
) -> dict[GridKey, Mapping[str, object]]:
    by_key: dict[GridKey, Mapping[str, object]] = {}
    for row in records:
        key = (
            _json_str(row.get("policy_id"), f"{name} policy"),
            _json_str(row.get("target_center"), f"{name} target"),
            _json_int(row.get("training_seed"), f"{name} training seed"),
            _json_int(row.get("generation_seed"), f"{name} generation seed"),
        )
        if key in by_key:
            raise ProtocolError(f"Stage-70 {name} contains a duplicate grid cell.")
        by_key[key] = row
    if set(by_key) != _EXPECTED_GRID:
        raise ProtocolError(f"Stage-70 {name} grid is incomplete or out of scope.")
    return by_key


def _csv_grid_key(row: Mapping[str, str], name: str) -> GridKey:
    try:
        return (
            row["policy_id"],
            row["target_center"],
            int(row["training_seed"]),
            int(row["generation_seed"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Stage-70 {name} grid identity is malformed.") from exc


def _equal_center_mean(rows: Sequence[Mapping[str, str]], field: str) -> float:
    means: list[float] = []
    for center in CENTERS:
        center_rows = [row for row in rows if row["target_center"] == center]
        if len(center_rows) != 9:
            raise ProtocolError("Stage-70 summary lost the crossed seed grid.")
        means.append(sum(_csv_float(row, field, "metric") for row in center_rows) / 9.0)
    return sum(means) / len(means)


def _read_csv(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(expected_columns):
                raise ProtocolError(f"Stage-70 CSV schema drifted: {path}.")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise ProtocolError(f"Cannot read Stage-70 CSV: {path}.") from exc


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read Stage-70 JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Stage-70 JSON must be an object: {path}.")
    return payload


def _exact_keys(payload: Mapping[object, object], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ProtocolError(f"Stage-70 {name} schema keys drifted.")


def _safe_relative_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"Stage-70 {name} is empty or non-string.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts) or pure.as_posix() != value:
        raise ProtocolError(f"Stage-70 {name} is not a canonical relative path.")
    return value


def _safe_filename(value: object, name: str) -> str:
    relative = _safe_relative_path(value, name)
    if len(PurePosixPath(relative).parts) != 1:
        raise ProtocolError(f"Stage-70 {name} is not a filename.")
    return relative


def _json_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"Stage-70 {name} is malformed.")
    return value


def _json_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"Stage-70 {name} is not an integer.")
    return value


def _hash_string(value: object, name: str, length: int | Sequence[int]) -> str:
    lengths = {length} if isinstance(length, int) else set(length)
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"Stage-70 {name} is not a canonical hash.")
    return value


def _csv_int(row: Mapping[str, str], field: str, name: str) -> int:
    try:
        value = row[field]
        if not value or value.strip() != value:
            raise ValueError
        return int(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Stage-70 {name} field {field!r} is not an integer.") from exc


def _csv_float(row: Mapping[str, str], field: str, name: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Stage-70 {name} field {field!r} is not numeric.") from exc
    if not math.isfinite(value):
        raise ProtocolError(f"Stage-70 {name} field {field!r} is non-finite.")
    return value


def _safe_f1(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-12)


def _derived_generation_seed(
    *,
    bank_lock_hash: str,
    expert_lock_hash: str,
    generation_seed: int,
    class_label: int,
) -> int:
    payload = {
        "namespace": SOURCE_STREAM_NAMESPACE,
        "bank_lock_hash": bank_lock_hash,
        "expert_lock_hash": expert_lock_hash,
        "generation_seed": generation_seed,
        "class_label": class_label,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big", signed=False)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash Stage-70 member: {path}.") from exc
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _array_bundle_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


__all__ = (
    "validate_frozen_policy_downstream_bundle",
    "write_validation_report",
)
