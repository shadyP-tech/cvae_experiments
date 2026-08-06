from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.common.hashing import stable_hash
from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from midogpp_thesis.data.features.stage70_test_cache import CACHE_ARTIFACT_ID
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from midogpp_thesis.cvae.frozen_policy_downstream import validation as validation_module
from midogpp_thesis.cvae.frozen_policy_downstream.authorization.contracts import (
    FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
)
from midogpp_thesis.cvae.frozen_policy_downstream.bundle import write_content_index
from midogpp_thesis.cvae.frozen_policy_downstream.contracts import (
    CLAIM_SCOPE,
    CONTROL_ARM,
    EXPERIMENT_ID,
    METADATA_ARM,
    POLICY_ARMS,
    UTILITY_ARM,
    array_sha256,
)
from midogpp_thesis.cvae.frozen_policy_downstream.source_blocks import (
    SOURCE_BLOCK_CACHE_SCHEMA,
    source_block_cache_key,
)
from midogpp_thesis.cvae.frozen_policy_downstream.validation import (
    validate_frozen_policy_downstream_bundle,
    write_validation_report,
)
from midogpp_thesis.cvae.generation.contracts import SourceGenerationKey
from midogpp_thesis.cvae.generation.generation import derived_generation_seed
from midogpp_thesis.cvae.protocol import ProtocolError


AUTHORIZATION_HASH = "a" * 16
AUTHORIZATION_PROTOCOL_HASH = "b" * 16
CLASSIFIER_HASH = "c" * 16
SCORING_MANIFEST_SHA256 = "d" * 64
BACKBONE_HASH = "e" * 16
BANK_LOCK_HASH = "b" * 16
GENERATION_LOCK_HASH = "d" * 16
CHECKPOINT_HASH = "f" * 64
REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
TARGET_CACHE_CONTENT_HASH = "2" * 16
TARGET_CACHE_ROW_ORDER_HASH = "3" * 16


@pytest.fixture
def valid_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Full production source blocks are 81 x (2048, 3840).  The isolated source
    # member parser has its own mutation test below; bundle geometry tests patch
    # only that expensive byte-level leaf while retaining every index/path/hash
    # and closed-world check around it.
    monkeypatch.setattr(validation_module, "_validate_source_block_member", lambda *args, **kwargs: None)
    _write_valid_bundle(tmp_path)
    return tmp_path


def test_valid_stage70_bundle_passes_independent_validation(valid_bundle: Path) -> None:
    observed = validate_frozen_policy_downstream_bundle(valid_bundle)

    assert observed["status"] == "PASS"
    assert observed["prediction_cell_count"] == 243
    assert observed["utility_control_exact_equivalence"] is True


def test_failed_run_state_is_rejected_before_static_pass_reports(valid_bundle: Path) -> None:
    state = _read_json(valid_bundle / "reports/run_state.json")
    state.update(status="FAILED", phase="FAILED_BEFORE_PUBLICATION")
    _write_json(valid_bundle / "reports/run_state.json", state)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="run state"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_protocol_claim_tamper_is_rejected_even_with_recomputed_hash(valid_bundle: Path) -> None:
    manifest_path = valid_bundle / "manifests/protocol_manifest.json"
    payload = _read_json(manifest_path)
    payload["deployment_claim_allowed"] = True
    payload["protocol_hash"] = _semantic_hash_without(payload, "protocol_hash")
    _write_json(manifest_path, payload)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="claim/firewall"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_evaluation_plan_duplicate_cell_is_rejected_with_valid_hash(valid_bundle: Path) -> None:
    plan_path = valid_bundle / "manifests/evaluation_plan.json"
    payload = _read_json(plan_path)
    payload["records"][1] = dict(payload["records"][0])  # type: ignore[index]
    payload["evaluation_plan_hash"] = _semantic_hash_without(payload, "evaluation_plan_hash")
    _write_json(plan_path, payload)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="duplicate grid cell"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_composition_grid_tamper_is_rejected(valid_bundle: Path) -> None:
    index_path = valid_bundle / "manifests/composition_index.json"
    payload = _read_json(index_path)
    payload["records"][1]["training_seed"] = payload["records"][0]["training_seed"]  # type: ignore[index]
    payload["records"][1]["generation_seed"] = payload["records"][0]["generation_seed"]  # type: ignore[index]
    payload["records"][1]["target_center"] = payload["records"][0]["target_center"]  # type: ignore[index]
    _write_json(index_path, payload)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="duplicate grid cell"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_missing_source_block_record_is_rejected(valid_bundle: Path) -> None:
    index_path = valid_bundle / "manifests/source_block_index.json"
    payload = _read_json(index_path)
    payload["records"] = payload["records"][:-1]  # type: ignore[index]
    payload["source_block_count"] = 80
    _write_json(index_path, payload)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="source-block index record geometry"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_missing_source_block_member_is_rejected(valid_bundle: Path) -> None:
    payload = _read_json(valid_bundle / "manifests/source_block_index.json")
    relative = str(payload["records"][0]["artifact_member"])  # type: ignore[index]
    (valid_bundle / relative).unlink()
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="member is missing"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_misdescribed_source_block_member_is_rejected(valid_bundle: Path) -> None:
    index_path = valid_bundle / "manifests/source_block_index.json"
    payload = _read_json(index_path)
    payload["records"][0]["artifact_member"] = payload["records"][1]["artifact_member"]  # type: ignore[index]
    _write_json(index_path, payload)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="member path"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_duplicate_metric_cell_is_rejected(valid_bundle: Path) -> None:
    path = valid_bundle / "tables/target_metrics.csv"
    rows = _read_csv(path)
    for field in ("policy_id", "target_center", "training_seed", "generation_seed"):
        rows[1][field] = rows[0][field]
    _write_csv(path, rows)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="duplicate cell"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_metric_prediction_provenance_mismatch_is_rejected(valid_bundle: Path) -> None:
    path = valid_bundle / "tables/target_metrics.csv"
    rows = _read_csv(path)
    rows[0]["prediction_sha256"] = "f" * 64
    _write_csv(path, rows)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="provenance join"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_static_equivalence_pass_report_cannot_override_derived_mismatch(valid_bundle: Path) -> None:
    path = valid_bundle / "reports/utility_control_equivalence.json"
    payload = _read_json(path)
    payload["cell_count"] = 80
    _write_json(path, payload)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="not derived"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


@pytest.mark.parametrize(
    ("relative", "field", "replacement", "message"),
    (
        (
            "tables/arm_summaries.csv",
            "policy_id",
            CONTROL_ARM,
            "arm-summary key geometry",
        ),
        (
            "tables/paired_deltas.csv",
            "comparison_id",
            "undeclared_comparison",
            "comparison is undeclared",
        ),
        (
            "tables/bootstrap_summary.csv",
            "comparison_id",
            "undeclared_comparison",
            "bootstrap comparison geometry",
        ),
    ),
)
def test_derived_table_key_geometry_is_exact(
    valid_bundle: Path,
    relative: str,
    field: str,
    replacement: str,
    message: str,
) -> None:
    path = valid_bundle / relative
    rows = _read_csv(path)
    rows[1][field] = replacement
    _write_csv(path, rows)
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match=message):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_unindexed_extra_member_is_rejected_as_not_closed_world(valid_bundle: Path) -> None:
    (valid_bundle / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ProtocolError, match="closed-world"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_reindexed_extra_member_is_rejected_as_not_closed_world(
    valid_bundle: Path,
) -> None:
    (valid_bundle / "unexpected.txt").write_text("not an artifact member\n")
    _reindex(valid_bundle)

    with pytest.raises(ProtocolError, match="membership is not closed-world"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_symlink_member_is_rejected(valid_bundle: Path) -> None:
    (valid_bundle / "linked-state.json").symlink_to(
        valid_bundle / "reports/run_state.json"
    )

    with pytest.raises(ProtocolError, match="symlink"):
        validate_frozen_policy_downstream_bundle(valid_bundle)


def test_source_member_rejects_embedded_metadata_mismatch(tmp_path: Path) -> None:
    member = tmp_path / "source.npz"
    row, internal_key = _source_record("0", 17, 17, member_name="source.npz")
    np.savez_compressed(
        member,
        embeddings=np.zeros((1, 1), dtype=np.float32),
        labels=np.zeros((1,), dtype=np.int64),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema_version": SOURCE_BLOCK_CACHE_SCHEMA,
                    "protocol_version": SOURCE_BLOCK_CACHE_SCHEMA,
                    "cache_key": "0" * 16,
                    "source_generation_key": internal_key,
                    "checkpoint_hash": row["checkpoint_hash"],
                    "bank_lock_hash": row["bank_lock_hash"],
                    "generation_lock_hash": row["generation_lock_hash"],
                    "dataset_contract_hash": row["dataset_contract_hash"],
                    "evaluation_split": row["evaluation_split"],
                    "representation_id": row["representation_id"],
                    "backbone_identity_hash": row["backbone_identity_hash"],
                    "budget_per_class": row["budget_per_class"],
                    "output_sha256": row["output_sha256"],
                },
                sort_keys=True,
            )
        ),
    )

    with pytest.raises(ProtocolError, match="embedded metadata"):
        validation_module._validate_source_block_member(
            member,
            row=row,
            expected_internal_key=internal_key,
            expected_output_sha256=str(row["output_sha256"]),
        )


def _write_valid_bundle(root: Path) -> None:
    (root / "config.resolved.yaml").write_text("schema_version: stage70-test\n", encoding="utf-8")
    _write_json(root / "provenance/input_artifacts.json", {"schema_version": "test"})

    protocol = {
        "schema_version": "midogpp_stage70_descriptive_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": "1" * 16,
        "final_authorization_hash": AUTHORIZATION_HASH,
        "authorization_protocol_hash": AUTHORIZATION_PROTOCOL_HASH,
        "target_cache_content_hash": TARGET_CACHE_CONTENT_HASH,
        "target_cache_row_order_hash": TARGET_CACHE_ROW_ORDER_HASH,
        "dataset_contract_hash": SCORING_MANIFEST_SHA256,
        "scoring_manifest_sha256": SCORING_MANIFEST_SHA256,
        "representation_id": REPRESENTATION_ID,
        "backbone_identity_hash": BACKBONE_HASH,
        "policy_arms": list(POLICY_ARMS),
        "evaluation_split": "test_previously_consumed_for_representation_adoption",
        "fresh_confirmatory_evidence": False,
        "fresh_confirmatory_status": "BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT",
        "routing_policy_promotion_allowed": False,
        "deployment_claim_allowed": False,
        "target_support_used": False,
        "policy_or_seed_selection_performed": False,
        "predictions_persisted_before_labels_opened": True,
        "labels_used_for_scoring_only": True,
    }
    protocol["protocol_hash"] = stable_hash(protocol)
    _write_json(root / "manifests/protocol_manifest.json", protocol)

    evaluation_records = _evaluation_records()
    evaluation_plan = {
        "schema_version": "midogpp_stage70_descriptive_evaluation_plan_v1",
        "final_authorization_hash": AUTHORIZATION_HASH,
        "classifier_config_hash": CLASSIFIER_HASH,
        "policy_arms": list(POLICY_ARMS),
        "records": evaluation_records,
        "prediction_cells": 243,
        "training_seeds_retained": list(TRAINING_SEEDS),
        "generation_seeds_retained": list(GENERATION_SEEDS),
        "seed_selection": False,
        "target_labels_opened": False,
    }
    evaluation_plan["evaluation_plan_hash"] = stable_hash(evaluation_plan)
    _write_json(root / "manifests/evaluation_plan.json", evaluation_plan)
    identities_by_center = _target_identities_by_center()
    phase_01 = _phase_01_payload(evaluation_records, identities_by_center)
    _write_json(root / "reports/phase_01_authorization_complete.json", phase_01)

    source_records: list[dict[str, object]] = []
    for center in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                row, _ = _source_record(center, training_seed, generation_seed)
                source_records.append(row)
                member = root / str(row["artifact_member"])
                member.parent.mkdir(parents=True, exist_ok=True)
                member.write_bytes(_source_member_bytes(str(row["source_stream_id"])))
    _write_json(
        root / "manifests/source_block_index.json",
        {
            "schema_version": "midogpp_stage70_source_block_index_v1",
            "source_block_count": 81,
            "records": source_records,
            "target_labels_opened": False,
        },
    )

    evaluation_by_key = {_grid_key(row): row for row in evaluation_records}
    composition_records = _composition_records(evaluation_by_key)
    _write_json(
        root / "manifests/composition_index.json",
        {
            "schema_version": "midogpp_stage70_composition_index_v1",
            "composition_count": 243,
            "records": composition_records,
            "target_labels_opened": False,
        },
    )

    composition_by_key = {_grid_key(row): row for row in composition_records}
    prediction_records, arrays = _prediction_records(
        evaluation_by_key,
        composition_by_key,
        identities_by_center,
    )
    arrays_path = root / "arrays/target_predictions.npz"
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, **arrays)
    prediction_index = {
        "schema_version": "midogpp_stage70_prediction_index_v2",
        "phase": "PREDICTIONS_PERSISTED",
        "target_labels_opened": False,
        "cell_count": 243,
        "target_row_count": EXPECTED_TEST_ROWS,
        "phase_01_sha256": _sha256_file(
            root / "reports/phase_01_authorization_complete.json"
        ),
        "authorization_binding_hash": phase_01["authorization_binding_hash"],
        "prediction_metadata_hash": stable_hash(prediction_records),
        "records": prediction_records,
    }
    _write_json(root / "manifests/prediction_index.json", prediction_index)
    prediction_index_sha = _sha256_file(root / "manifests/prediction_index.json")
    arrays_sha = _sha256_file(arrays_path)
    seal = {
        "schema_version": "midogpp_stage70_prediction_seal_v2",
        "phase": "PREDICTIONS_PERSISTED",
        "phase_01_sha256": prediction_index["phase_01_sha256"],
        "authorization_binding_hash": phase_01["authorization_binding_hash"],
        "prediction_index_sha256": prediction_index_sha,
        "prediction_arrays_sha256": arrays_sha,
        "prediction_metadata_hash": prediction_index["prediction_metadata_hash"],
        "cell_count": 243,
        "target_row_count": EXPECTED_TEST_ROWS,
        "classifier_fit_count": 162,
        "prediction_reuse_count": 81,
        "target_labels_opened": False,
    }
    _write_json(root / "manifests/prediction_seal.json", seal)
    _write_json(
        root / "reports/phase_02_predictions_persisted.json",
        {**seal, "schema_version": "midogpp_stage70_phase_marker_v2"},
    )
    prediction_seal_sha = _sha256_file(root / "manifests/prediction_seal.json")
    phase_02_sha = _sha256_file(root / "reports/phase_02_predictions_persisted.json")
    _write_json(
        root / "reports/phase_03_labels_opened.json",
        {
            "schema_version": "midogpp_stage70_phase_marker_v1",
            "phase": "LABELS_OPENED_AFTER_PREDICTIONS_PERSISTED",
            "authorization_binding_hash": phase_01["authorization_binding_hash"],
            "final_authorization_hash": AUTHORIZATION_HASH,
            "target_cache_content_hash": TARGET_CACHE_CONTENT_HASH,
            "phase_01_sha256": prediction_index["phase_01_sha256"],
            "prediction_index_sha256": prediction_index_sha,
            "prediction_arrays_sha256": arrays_sha,
            "prediction_seal_sha256": prediction_seal_sha,
            "phase_02_sha256": phase_02_sha,
            "label_manifest_sha256": SCORING_MANIFEST_SHA256,
            "labels_used_for_scoring_only": True,
        },
    )
    _write_json(
        root / "reports/phase_04_scoring_complete.json",
        {
            "schema_version": "midogpp_stage70_phase_marker_v1",
            "phase": "SCORING_COMPLETE",
            "metric_row_count": 243,
            "authorization_binding_hash": phase_01["authorization_binding_hash"],
            "final_authorization_hash": AUTHORIZATION_HASH,
            "target_cache_content_hash": TARGET_CACHE_CONTENT_HASH,
            "phase_01_sha256": prediction_index["phase_01_sha256"],
            "prediction_index_sha256": prediction_index_sha,
            "prediction_arrays_sha256": arrays_sha,
            "prediction_seal_sha256": prediction_seal_sha,
            "phase_02_sha256": phase_02_sha,
            "label_manifest_sha256": SCORING_MANIFEST_SHA256,
        },
    )

    metric_rows, confusion_rows = _metric_and_confusion_rows(
        prediction_records,
        phase_01=phase_01,
        phase_01_sha256=str(prediction_index["phase_01_sha256"]),
        prediction_index_sha256=prediction_index_sha,
        prediction_arrays_sha256=arrays_sha,
        prediction_seal_sha256=prediction_seal_sha,
        phase_02_sha256=phase_02_sha,
    )
    _write_csv(root / "tables/target_metrics.csv", metric_rows)
    _write_csv(root / "tables/case_confusions.csv", confusion_rows)
    _write_csv(root / "tables/arm_summaries.csv", _summary_rows())
    _write_csv(root / "tables/paired_deltas.csv", _delta_rows())
    _write_csv(root / "tables/bootstrap_summary.csv", _bootstrap_rows())
    _write_json(
        root / "reports/leakage_report.json",
        {
            "schema_version": "midogpp_stage70_leakage_report_v1",
            "status": "PASS",
            "final_authorization_hash": AUTHORIZATION_HASH,
            "target_labels_opened_after_prediction_seal": True,
            "target_labels_used_for_fit_selection_or_prediction": False,
            "target_labels_used_for_scoring_only": True,
            "target_support_used": False,
            "routing_recomputed": False,
            "stage50_or_stage90_input_used": False,
            "fresh_confirmatory_evidence": False,
        },
    )
    _write_json(
        root / "reports/identity_overlap_report.json",
        {
            "schema_version": "midogpp_stage70_identity_overlap_v1",
            "status": "PASS",
            "target_expert_assignments": 0,
            "center_4_rows": 0,
            "legacy_label_encoded_identifiers_persisted": 0,
        },
    )
    _write_json(
        root / "reports/utility_control_equivalence.json",
        {
            "schema_version": "midogpp_stage70_utility_control_equivalence_v1",
            "status": "PASS",
            "cell_count": 81,
            "exact_metric_equivalence": True,
            "exact_prediction_and_probability_hash_equivalence": True,
            "independent_policy_hypothesis_test": False,
        },
    )
    _write_json(
        root / "reports/publication_decision.json",
        {
            "schema_version": "midogpp_stage70_publication_decision_v1",
            "status": "PASS",
            "decision": "DESCRIPTIVE_COMPARISON_COMPLETE",
            "claim_scope": CLAIM_SCOPE,
            "fresh_confirmatory_status": "BLOCKED_NO_UNCONSUMED_ELIGIBLE_SPLIT",
            "routing_policy_promoted": False,
            "deployment_utility_claimed": False,
            "new_center_generalization_claimed": False,
            "external_generalization_claimed": False,
        },
    )
    _write_json(
        root / "reports/run_state.json",
        {
            "schema_version": "midogpp_stage70_run_state_v1",
            "status": "COMPLETE",
            "phase": "SCORING_COMPLETE",
        },
    )
    write_content_index(root)
    checks = validate_frozen_policy_downstream_bundle(root, allow_pending=True)
    write_validation_report(root, checks)


def _evaluation_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for arm in POLICY_ARMS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for center in CENTERS:
                    records.append(
                        {
                            "policy_id": arm,
                            "target_center": center,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "replicate_id": stable_hash(
                                [center, training_seed, generation_seed]
                            ),
                            "policy_lock_hash": stable_hash([arm, "lock"]),
                            "policy_plan_hash": stable_hash([arm, "plan"]),
                            "assignment_table_hash": stable_hash([arm, "assignments"]),
                            "assignment_count": 8,
                            "synthetic_rows_per_class": 1024,
                            "target_expert_excluded": True,
                        }
                    )
    return records


def _target_identities_by_center() -> dict[str, list[dict[str, object]]]:
    identities: dict[str, list[dict[str, object]]] = {}
    global_index = 0
    for center in CENTERS:
        count = EXPECTED_TEST_ROWS_BY_CENTER[center]
        center_rows: list[dict[str, object]] = []
        for offset in range(count):
            center_rows.append(
                {
                    "evaluation_row_id": "eval_"
                    + hashlib.sha256(f"{center}:{offset}".encode("ascii")).hexdigest(),
                    "contract_row_index": global_index,
                    "case_id": f"case-{center}-{int(offset >= count // 2)}",
                }
            )
            global_index += 1
        identities[center] = center_rows
    assert global_index == EXPECTED_TEST_ROWS
    return identities


def _phase_01_payload(
    evaluation_records: list[dict[str, object]],
    identities_by_center: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    global_identities = sorted(
        (row for rows in identities_by_center.values() for row in rows),
        key=lambda row: int(row["contract_row_index"]),
    )
    authorized_cells = [
        {
            "policy_id": row["policy_id"],
            "target_center": row["target_center"],
            "training_seed": row["training_seed"],
            "generation_seed": row["generation_seed"],
            "replicate_id": row["replicate_id"],
        }
        for row in evaluation_records
    ]
    payload: dict[str, object] = {
        "schema_version": "midogpp_stage70_phase_01_authorization_binding_v2",
        "phase": "AUTHORIZATION_COMPLETE",
        "final_authorization_artifact_id": FINAL_AUTHORIZATION_OUTPUT_ARTIFACT_ID,
        "final_authorization_hash": AUTHORIZATION_HASH,
        "final_authorization_content_hash": "4" * 16,
        "authorization_protocol_hash": AUTHORIZATION_PROTOCOL_HASH,
        "identity_lock_hash": "5" * 16,
        "evaluation_plan_hash": "6" * 16,
        "reservation_content_hash": "7" * 16,
        "reservation_identity_lock_hash": "8" * 16,
        "target_evaluation_reservation_id": "reservation_stage70_test",
        "target_evaluation_reservation_protocol_hash": "9" * 16,
        "target_identity_table_hash": "a" * 16,
        "target_cache_artifact_id": CACHE_ARTIFACT_ID,
        "target_cache_content_hash": TARGET_CACHE_CONTENT_HASH,
        "target_cache_row_order_hash": TARGET_CACHE_ROW_ORDER_HASH,
        "target_cache_shard_sha256_by_center": {
            center: _sha_text([center, "shard"]) for center in CENTERS
        },
        "target_cache_rows_by_center": dict(EXPECTED_TEST_ROWS_BY_CENTER),
        "target_cache_row_count": EXPECTED_TEST_ROWS,
        "cache_extractor_protocol_hash": "b" * 16,
        "scoring_manifest_sha256": SCORING_MANIFEST_SHA256,
        "classifier_config_hash": CLASSIFIER_HASH,
        "authorized_cell_hash": stable_hash(authorized_cells),
        "target_identity_hash_by_center": {
            center: stable_hash(rows) for center, rows in identities_by_center.items()
        },
        "global_target_identity_hash": stable_hash(global_identities),
        "target_labels_opened": False,
    }
    payload["authorization_binding_hash"] = stable_hash(payload)
    return payload


def _source_record(
    center: str,
    training_seed: int,
    generation_seed: int,
    *,
    member_name: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    expert_lock_hash = stable_hash([center, training_seed, "expert"])
    identity = {
        "namespace": "uniform_b_v2_source_stream_v1",
        "bank_lock_hash": BANK_LOCK_HASH,
        "expert_lock_hash": expert_lock_hash,
        "source_center": center,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
    }
    stream_id = stable_hash(identity)
    class_seeds = {
        str(label): derived_generation_seed(
            namespace="uniform_b_v2_source_stream_v1",
            bank_lock_hash=BANK_LOCK_HASH,
            expert_lock_hash=expert_lock_hash,
            generation_seed=generation_seed,
            class_label=label,
        )
        for label in (0, 1)
    }
    key = SourceGenerationKey(
        source_center=center,
        training_seed=training_seed,
        generation_seed=generation_seed,
        expert_lock_hash=expert_lock_hash,
        stream_id=stream_id,
        class_seed_by_label=class_seeds,
    )
    cache_key = source_block_cache_key(
        key=key,
        dataset_contract_hash=SCORING_MANIFEST_SHA256,
        evaluation_split="test_previously_consumed_for_representation_adoption",
        representation_id=REPRESENTATION_ID,
        backbone_identity_hash=BACKBONE_HASH,
        checkpoint_hash=CHECKPOINT_HASH,
        bank_lock_hash=BANK_LOCK_HASH,
        generation_lock_hash=GENERATION_LOCK_HASH,
    )
    filename = member_name or f"{cache_key}.npz"
    record = {
        "schema_version": SOURCE_BLOCK_CACHE_SCHEMA,
        "cache_key": cache_key,
        "cache_status": "GENERATED",
        "source_center": center,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "source_stream_id": stream_id,
        "expert_lock_hash": expert_lock_hash,
        "checkpoint_hash": CHECKPOINT_HASH,
        "bank_lock_hash": BANK_LOCK_HASH,
        "generation_lock_hash": GENERATION_LOCK_HASH,
        "dataset_contract_hash": SCORING_MANIFEST_SHA256,
        "evaluation_split": "test_previously_consumed_for_representation_adoption",
        "representation_id": REPRESENTATION_ID,
        "backbone_identity_hash": BACKBONE_HASH,
        "budget_per_class": 1024,
        "output_sha256": "a" * 64,
        "path": filename,
        "persistent_path": filename,
        "member_path": f"arrays/source_blocks/{filename}",
        "member_sha256": hashlib.sha256(_source_member_bytes(stream_id)).hexdigest(),
        "artifact_member": f"arrays/source_blocks/{filename}",
    }
    return record, key.to_payload()


def _source_member_bytes(stream_id: str) -> bytes:
    return f"source:{stream_id}\n".encode("ascii")


def _composition_records(
    evaluation: dict[tuple[str, str, int, int], dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for key, plan in evaluation.items():
        arm, center, training_seed, generation_seed = key
        cell = [center, training_seed, generation_seed]
        records.append(
            {
                "policy_id": arm,
                "target_center": center,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "replicate_id": plan["replicate_id"],
                "policy_lock_hash": plan["policy_lock_hash"],
                "assignment_table_hash": plan["assignment_table_hash"],
                "composition_manifest_hash": stable_hash([arm, *cell, "composition"]),
                "train_content_sha256": _sha_text([*cell, "train"]),
                "pre_shuffle_sha256_by_label": {
                    "0": _sha_text([*cell, "pre", 0]),
                    "1": _sha_text([*cell, "pre", 1]),
                },
                "post_shuffle_sha256_by_label": {
                    "0": _sha_text([*cell, "post", 0]),
                    "1": _sha_text([*cell, "post", 1]),
                },
            }
        )
    return records


def _prediction_records(
    evaluation: dict[tuple[str, str, int, int], dict[str, object]],
    compositions: dict[tuple[str, str, int, int], dict[str, object]],
    identities_by_center: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    records: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    for ordinal, key in enumerate(evaluation):
        arm, center, training_seed, generation_seed = key
        identities = identities_by_center[center]
        count = len(identities)
        split = count // 2
        predictions = np.concatenate(
            (
                np.zeros(split, dtype=np.int64),
                np.ones(count - split, dtype=np.int64),
            )
        )
        probabilities = np.empty((count, 2), dtype=np.float64)
        probabilities[:split] = (0.9, 0.1)
        probabilities[split:] = (0.1, 0.9)
        prediction_key = f"prediction_{ordinal:03d}"
        probability_key = f"probability_{ordinal:03d}"
        arrays[prediction_key] = predictions
        arrays[probability_key] = probabilities
        composition = compositions[key]
        record = {
                "ordinal": ordinal,
                "policy_id": arm,
                "target_center": center,
                "training_seed": training_seed,
                "generation_seed": generation_seed,
                "replicate_id": evaluation[key]["replicate_id"],
                "row_count": count,
                "evaluation_row_ids": [row["evaluation_row_id"] for row in identities],
                "contract_row_indices": [row["contract_row_index"] for row in identities],
                "case_ids": [row["case_id"] for row in identities],
                "target_identity_hash": stable_hash(identities),
                "prediction_array_key": prediction_key,
                "probability_array_key": probability_key,
                "prediction_sha256": array_sha256(predictions),
                "probability_sha256": array_sha256(probabilities),
                "composition_manifest_hash": composition["composition_manifest_hash"],
                "train_content_sha256": composition["train_content_sha256"],
                "classifier_config_hash": CLASSIFIER_HASH,
                "scaler_state_hash": stable_hash([center, training_seed, generation_seed, "scaler"]),
                "target_row_order_hash": stable_hash([center, "rows"]),
                "reused_from_policy_id": CONTROL_ARM if arm == UTILITY_ARM else "",
            }
        record["target_row_order_hash"] = stable_hash(record["evaluation_row_ids"])
        record["prediction_cell_hash"] = stable_hash(record)
        records.append(record)
    return records, arrays


def _metric_and_confusion_rows(
    prediction_records: list[dict[str, object]],
    *,
    phase_01: dict[str, object],
    phase_01_sha256: str,
    prediction_index_sha256: str,
    prediction_arrays_sha256: str,
    prediction_seal_sha256: str,
    phase_02_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    confusions: list[dict[str, object]] = []
    for row in prediction_records:
        center = str(row["target_center"])
        row_count = int(row["row_count"])
        split = row_count // 2
        shard_hashes = phase_01["target_cache_shard_sha256_by_center"]
        assert isinstance(shard_hashes, dict)
        metrics.append(
            {
                "schema_version": "midogpp_stage70_target_metric_v1",
                "claim_scope": CLAIM_SCOPE,
                "claim_role": "descriptive_locked_policy_comparison",
                "row_role": "target_evaluation_metric",
                "policy_id": row["policy_id"],
                "target_center": row["target_center"],
                "training_seed": row["training_seed"],
                "generation_seed": row["generation_seed"],
                "replicate_id": row["replicate_id"],
                "n_eval": row_count,
                "n_cases": 2,
                "bacc": 1.0,
                "macro_f1": 1.0,
                "macro_f1_role": "secondary_descriptive_only",
                "prediction_sha256": row["prediction_sha256"],
                "probability_sha256": row["probability_sha256"],
                "prediction_cell_hash": row["prediction_cell_hash"],
                "target_identity_hash": row["target_identity_hash"],
                "composition_manifest_hash": row["composition_manifest_hash"],
                "train_content_sha256": row["train_content_sha256"],
                "classifier_config_hash": row["classifier_config_hash"],
                "scaler_state_hash": row["scaler_state_hash"],
                "target_row_order_hash": row["target_row_order_hash"],
                "label_manifest_sha256": SCORING_MANIFEST_SHA256,
                "reused_from_policy_id": row["reused_from_policy_id"],
                "authorization_binding_hash": phase_01["authorization_binding_hash"],
                "final_authorization_hash": phase_01["final_authorization_hash"],
                "authorization_protocol_hash": phase_01["authorization_protocol_hash"],
                "identity_lock_hash": phase_01["identity_lock_hash"],
                "evaluation_plan_hash": phase_01["evaluation_plan_hash"],
                "reservation_content_hash": phase_01["reservation_content_hash"],
                "target_evaluation_reservation_id": phase_01[
                    "target_evaluation_reservation_id"
                ],
                "target_evaluation_reservation_protocol_hash": phase_01[
                    "target_evaluation_reservation_protocol_hash"
                ],
                "target_cache_artifact_id": phase_01["target_cache_artifact_id"],
                "target_cache_content_hash": phase_01["target_cache_content_hash"],
                "target_cache_row_order_hash": phase_01["target_cache_row_order_hash"],
                "target_cache_shard_sha256": shard_hashes[center],
                "phase_01_sha256": phase_01_sha256,
                "prediction_index_sha256": prediction_index_sha256,
                "prediction_arrays_sha256": prediction_arrays_sha256,
                "prediction_seal_sha256": prediction_seal_sha256,
                "phase_02_sha256": phase_02_sha256,
                "target_labels_used_for_scoring_only": True,
                "fresh_confirmatory_evidence": False,
                "policy_or_seed_selection_performed": False,
            }
        )
        for suffix, counts in (
            ("0", (split, 0, 0, 0)),
            ("1", (0, 0, 0, row_count - split)),
        ):
            confusions.append(
                {
                    "schema_version": "midogpp_stage70_case_confusion_v1",
                    "policy_id": row["policy_id"],
                    "target_center": row["target_center"],
                    "training_seed": row["training_seed"],
                    "generation_seed": row["generation_seed"],
                    "case_id": f"case-{row['target_center']}-{suffix}",
                    "tn": counts[0],
                    "fp": counts[1],
                    "fn": counts[2],
                    "tp": counts[3],
                    "replicate_id": row["replicate_id"],
                    "prediction_sha256": row["prediction_sha256"],
                    "target_identity_hash": row["target_identity_hash"],
                    "label_manifest_sha256": SCORING_MANIFEST_SHA256,
                    "authorization_binding_hash": phase_01[
                        "authorization_binding_hash"
                    ],
                    "prediction_index_sha256": prediction_index_sha256,
                    "prediction_arrays_sha256": prediction_arrays_sha256,
                    "target_labels_used_for_scoring_only": True,
                }
            )
    return metrics, confusions


def _summary_rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": "midogpp_stage70_arm_summary_v1",
            "policy_id": arm,
            "equal_center_equal_seed_mean_bacc": 1.0,
            "equal_center_equal_seed_mean_macro_f1": 1.0,
            "minimum_cell_bacc": 1.0,
            "maximum_cell_bacc": 1.0,
            "cell_count": 81,
            "fresh_confirmatory_evidence": False,
        }
        for arm in POLICY_ARMS
    ]


def _delta_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for center in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                for arm, comparison, role in (
                    (
                        METADATA_ARM,
                        "metadata_max_tie_union_minus_equal_union",
                        "sole_predeclared_descriptive_policy_contrast",
                    ),
                    (
                        UTILITY_ARM,
                        "utility_regret_minus_equal_union",
                        "deterministic_fallback_equivalence_audit",
                    ),
                ):
                    rows.append(
                        {
                            "schema_version": "midogpp_stage70_paired_delta_v1",
                            "comparison_id": comparison,
                            "policy_id": arm,
                            "control_policy_id": CONTROL_ARM,
                            "target_center": center,
                            "training_seed": training_seed,
                            "generation_seed": generation_seed,
                            "bacc_delta": 0.0,
                            "macro_f1_delta": 0.0,
                            "role": role,
                            "paired": True,
                            "fresh_confirmatory_evidence": False,
                        }
                    )
    return rows


def _bootstrap_rows() -> list[dict[str, object]]:
    return [
        {
            "schema_version": "midogpp_stage70_descriptive_bootstrap_v1",
            "comparison_id": comparison,
            "observed_mean_bacc_delta": 0.0,
            "bootstrap_mean_bacc_delta": 0.0,
            "percentile_2_5": 0.0,
            "percentile_97_5": 0.0,
            "seed": 42,
            "valid_replicates": 10,
            "attempted_replicates": 10,
            "rejected_replicates": 0,
            "centers_resampled": True,
            "cases_resampled_within_center": True,
            "full_crossed_seed_grid_retained": True,
            "flattened_seed_pairs_resampled_as_iid": False,
            "invalid_class_denominator_draws_rejected": True,
            "interval_role": "descriptive_resampling_uncertainty_only",
            "fresh_confirmatory_inference": False,
        }
        for comparison in (
            "metadata_max_tie_union_minus_equal_union",
            "utility_regret_minus_equal_union_equivalence",
        )
    ]


def _grid_key(row: dict[str, object]) -> tuple[str, str, int, int]:
    return (
        str(row["policy_id"]),
        str(row["target_center"]),
        int(row["training_seed"]),
        int(row["generation_seed"]),
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_csv(path: Path, rows: list[dict[str, object]] | list[dict[str, str]]) -> None:
    assert rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _reindex(root: Path) -> None:
    write_content_index(root)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _semantic_hash_without(payload: dict[str, object], field: str) -> str:
    unhashed = dict(payload)
    unhashed.pop(field)
    return stable_hash(unhashed)
