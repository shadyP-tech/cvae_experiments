"""Input, seal, identity, and workstation checks for bundle validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .config import load_fixed_bank_disagreement_regret_prediction_only_config
from .constants import (
    CENTERS,
    EXPECTED_CLASSIFIER_FIT_COUNT,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_ROWS_BY_CENTER,
    EXPECTED_TASK_COUNT,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
)
from .development_actions import (
    DEVELOPMENT_ACTION_COUNT_PER_TASK,
    DEVELOPMENT_CLASSIFIER_FIT_COUNT,
    DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT,
    DEVELOPMENT_PHYSICAL_TASK_COUNT,
    development_action_library_payload,
)
from .experiment_contracts import (
    EXPERIMENT_ID,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
)
from .hashing import canonical_hash
from .validation_common import (
    EXPECTED_MODEL_BANK_COUNT,
    EXPECTED_SOURCE_CASE_COUNT,
    EXPECTED_SOURCE_FEATURE_ROWS,
    EXPECTED_TEST_CASE_COUNT,
    is_sha256,
    read_object,
)


def validate_resolved_config(
    root: Path, *, config: object, protocol_hash: str
) -> None:
    replay = load_fixed_bank_disagreement_regret_prediction_only_config(
        root / "config.resolved.yaml"
    )
    if (
        replay.contract_hash != str(getattr(config, "contract_hash"))
        or replay.experiment_id != EXPERIMENT_ID
        or replay.output_artifact_id != OUTPUT_ARTIFACT_ID
        or replay.input_artifact_ids != INPUT_ARTIFACT_IDS
        or replay.protocol.get("contract_hash") != protocol_hash
    ):
        raise ProtocolError("Prediction-only resolved config contract drifted.")


def validate_provenance(
    root: Path, *, config: object
) -> dict[str, Mapping[str, object]]:
    payload = read_object(root / "provenance/input_artifacts.json")
    rows = payload.get("input_artifacts")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
        or not isinstance(rows, list)
        or not all(isinstance(row, Mapping) for row in rows)
    ):
        raise ProtocolError("Prediction-only provenance header drifted.")
    identifiers = tuple(str(row.get("artifact_id")) for row in rows)
    if (
        identifiers != tuple(sorted(INPUT_ARTIFACT_IDS))
        or len(set(identifiers)) != len(identifiers)
        or tuple(getattr(config, "input_artifact_ids")) != INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Prediction-only provenance input fence drifted.")
    return {str(row["artifact_id"]): row for row in rows}


def validate_protocol_manifest(
    root: Path,
    *,
    config: object,
    protocol_hash: str,
    provenance: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    payload = read_object(root / "manifests/protocol_manifest.json")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_hash"}
    source = payload.get("source_frame_binding")
    target = payload.get("test_input_binding")
    firewall = payload.get("firewall")
    target_required = {
        "row_count": EXPECTED_TEST_ROWS,
        "feature_dim": 3_840,
        "cache_opened": False,
        "labels_available": False,
        "scoring_permitted": False,
        "admission_required_after_model_bank_seal": True,
    }
    if (
        payload.get("manifest_hash") != canonical_hash(unhashed)
        or payload.get("schema_version")
        != "midogpp_disagreement_regret_prediction_only_manifest_v1"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("output_artifact_id") != OUTPUT_ARTIFACT_ID
        or payload.get("config_contract_hash")
        != str(getattr(config, "contract_hash"))
        or payload.get("protocol_contract_hash") != protocol_hash
        or payload.get("input_artifact_ids") != list(INPUT_ARTIFACT_IDS)
        or payload.get("input_provenance_hash") != canonical_hash(dict(provenance))
        or not isinstance(source, Mapping)
        or source.get("row_count") != EXPECTED_SOURCE_ROWS
        or source.get("labels_in_typed_frame") is not False
        or source.get("source_label_field_accessed_by_projection_code") is not False
        or source.get("source_labels_physically_present_in_input_metadata")
        is not True
        or source.get("source_labels_are_posthoc") is not True
        or source.get("single_consumer_alias_only") is not True
        or not isinstance(target, Mapping)
        or any(target.get(key) != value for key, value in target_required.items())
        or not isinstance(firewall, Mapping)
        or payload.get("source_labels_opened") is not False
        or payload.get("test_cache_opened") is not False
        or payload.get("test_labels_available") is not False
        or payload.get("test_metrics_permitted") is not False
        or payload.get("prior_stage90_output_consumed") is not False
    ):
        raise ProtocolError("Prediction-only protocol manifest drifted.")
    return payload


def validate_prelabel_seal(
    root: Path, *, source_prediction_seal_hash: str
) -> Mapping[str, object]:
    payload = read_object(root / "manifests/prelabel_feature_seal.json")
    unhashed = {key: value for key, value in payload.items() if key != "seal_hash"}
    if (
        payload.get("seal_hash") != canonical_hash(unhashed)
        or payload.get("schema_version")
        != "midogpp_disagreement_regret_prelabel_feature_seal_v1"
        or payload.get("source_prediction_seal_hash") != source_prediction_seal_hash
        or not is_sha256(payload.get("prelabel_feature_seal_hash"))
        or payload.get("surface_count") != EXPECTED_MODEL_BANK_COUNT
        or payload.get("row_count") != EXPECTED_SOURCE_FEATURE_ROWS
        or payload.get("table_sha256")
        != sha256_file(root / "tables/source_case_features.csv")
        or payload.get("source_labels_opened") is not False
        or payload.get("test_cache_opened") is not False
        or payload.get("test_labels_opened") is not False
    ):
        raise ProtocolError("Prediction-only prelabel feature seal drifted.")
    return payload


def validate_prelabel_prediction_chain(
    root: Path,
    *,
    config_contract_hash: str,
    source_stream_lock_hash: str,
    target_action_library_hash: str,
    strict_source_predictions: object,
    target_classifier_bank: object,
    composite_seal: object,
) -> None:
    """Validate both prelabel classifier banks and their composite gate."""

    classifier_index = read_object(
        root / "manifests/action_classifier_bank_index.json"
    )
    classifier_seal = read_object(
        root / "manifests/action_classifier_bank_seal.json"
    )
    strict_bank = getattr(strict_source_predictions, "classifier_bank", None)
    strict_store = getattr(strict_source_predictions, "source_store", None)
    strict_payload = dict(getattr(strict_source_predictions, "seal_payload", {}))
    composite_payload = dict(getattr(composite_seal, "seal_payload", {}))
    source_binding_hash = getattr(strict_store, "frame_cache_binding_hash", None)
    target_seal_hash = getattr(target_classifier_bank, "seal_hash", None)
    strict_bank_seal_hash = getattr(strict_bank, "seal_hash", None)
    required_classifier = {
        "schema_version": "midogpp_prediction_only_action_classifier_bank_seal_v1",
        "status": "SEALED_1458_SOURCE_ONLY_ACTION_CLASSIFIERS",
        "config_contract_hash": config_contract_hash,
        "classifier_bank_hash": classifier_index.get("classifier_bank_hash"),
        "classifier_bank_index_sha256": sha256_file(
            root / "manifests/action_classifier_bank_index.json"
        ),
        "source_stream_lock_hash": source_stream_lock_hash,
        "action_library_hash": target_action_library_hash,
        "source_cache_binding_hash": source_binding_hash,
        "fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "task_count": EXPECTED_TASK_COUNT,
        "physical_action_count_per_task": PHYSICAL_ACTION_COUNT_PER_TARGET,
        "source_labels_available_during_fit": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
        "classifier_refit_required_for_test": False,
        "float64_frozen_parameter_arrays": True,
    }
    if (
        classifier_seal.get("classifier_bank_seal_hash")
        != target_seal_hash
        or any(
            classifier_seal.get(key) != value
            for key, value in required_classifier.items()
        )
        or classifier_index.get("fit_count") != EXPECTED_CLASSIFIER_FIT_COUNT
        or classifier_index.get("config_contract_hash") != config_contract_hash
        or classifier_index.get("source_stream_lock_hash") != source_stream_lock_hash
        or classifier_index.get("action_library_hash") != target_action_library_hash
        or classifier_index.get("source_cache_binding_hash")
        != source_binding_hash
        or classifier_index.get("source_labels_available_during_fit") is not False
        or classifier_index.get("test_cache_admitted") is not False
        or getattr(target_classifier_bank, "source_stream_lock_hash", None)
        != source_stream_lock_hash
    ):
        raise ProtocolError(
            "Prediction-only target-compatible classifier seal drifted."
        )

    source_library = read_object(root / "manifests/source_oof_action_library.json")
    if source_library != development_action_library_payload():
        raise ProtocolError("Strict source-OOF action library differs from replay.")
    strict_required = {
        "schema_version": "midogpp_strict_source_oof_prediction_seal_v1",
        "status": (
            "SEALED_5184_PHYSICAL_STRICT_SOURCE_OOF_FITS_"
            "10368_LOGICAL_PREDICTIONS"
        ),
        "config_contract_hash": config_contract_hash,
        "classifier_bank_seal_hash": strict_bank_seal_hash,
        "source_prediction_store_hash": getattr(strict_store, "store_hash", None),
        "source_prediction_array_sha256": sha256_file(
            root / "arrays/source_oof_action_probabilities.npz"
        ),
        "source_prediction_index_sha256": sha256_file(
            root / "manifests/source_oof_prediction_index.json"
        ),
        "physical_fit_count": DEVELOPMENT_CLASSIFIER_FIT_COUNT,
        "logical_source_prediction_cell_count": (
            DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        ),
        "predictions_only_for_q_rows": True,
        "unordered_excluded_pair_fit_reuse": True,
        "query_excluded_from_every_composition": True,
        "outer_target_excluded_from_every_composition": True,
        "source_labels_opened": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
    }
    if (
        any(strict_payload.get(key) != value for key, value in strict_required.items())
        or getattr(strict_bank, "source_stream_lock_hash", None)
        != source_stream_lock_hash
        or getattr(strict_bank, "source_cache_binding_hash", None)
        != source_binding_hash
        or getattr(strict_bank, "action_library_hash", None)
        != source_library["action_library_hash"]
        or getattr(strict_store, "action_library_hash", None)
        != source_library["action_library_hash"]
    ):
        raise ProtocolError("Strict source-OOF prediction seal drifted.")
    physical_by_key = getattr(strict_bank, "by_key", {})
    rows_by_query = getattr(strict_store, "rows_by_query", {})
    for cell in getattr(strict_store, "cells", ()):
        physical_key = (
            *tuple(sorted((cell.outer_target, cell.query_center))),
            cell.action_id,
            cell.training_seed,
            cell.generation_seed,
        )
        physical = physical_by_key.get(physical_key)
        if (
            physical is None
            or cell.classifier_parameter_sha256 != physical.parameter_sha256
            or cell.action_hash != physical.action_hash
            or cell.row_identity_hash
            != canonical_hash(list(rows_by_query[cell.query_center]))
        ):
            raise ProtocolError(
                "Strict source-OOF logical cell escaped its physical H/q fit."
            )

    composite_expected = {
        "schema_version": "midogpp_composite_prelabel_prediction_seal_v1",
        "status": (
            "SEALED_STRICT_SOURCE_OOF_AND_TARGET_CLASSIFIER_BANK_BEFORE_LABELS"
        ),
        "strict_source_prediction_seal_hash": getattr(
            strict_source_predictions, "seal_hash", None
        ),
        "strict_source_oof_classifier_bank_seal_hash": strict_bank_seal_hash,
        "strict_source_oof_prediction_store_hash": getattr(
            strict_store, "store_hash", None
        ),
        "target_classifier_bank_seal_hash": target_seal_hash,
        "source_cache_binding_hash": source_binding_hash,
        "strict_source_physical_fit_count": DEVELOPMENT_CLASSIFIER_FIT_COUNT,
        "strict_source_logical_prediction_cell_count": (
            DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        ),
        "target_classifier_fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "total_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT + EXPECTED_CLASSIFIER_FIT_COUNT
        ),
        "query_excluded_from_every_source_composition": True,
        "outer_target_excluded_from_every_source_composition": True,
        "unordered_excluded_pair_fit_reuse": True,
        "source_labels_opened": False,
        "test_cache_admitted": False,
        "target_labels_available": False,
    }
    expected_composite = {
        **composite_expected,
        "composite_prelabel_prediction_seal_hash": canonical_hash(
            composite_expected
        ),
    }
    if (
        composite_payload != expected_composite
        or getattr(composite_seal, "source_store", None) is not strict_store
        or getattr(composite_seal, "target_classifier_bank", None)
        is not target_classifier_bank
        or DEVELOPMENT_PHYSICAL_TASK_COUNT != 324
        or DEVELOPMENT_ACTION_COUNT_PER_TASK != 16
    ):
        raise ProtocolError("Composite prelabel prediction seal drifted.")


def validate_source_capability(
    root: Path,
    *,
    source_prediction_seal_hash: str,
    source_oof_classifier_bank_seal_hash: str,
    target_classifier_bank_seal_hash: str,
) -> Mapping[str, object]:
    payload = read_object(root / "manifests/source_label_capability_report.json")
    unhashed = {key: value for key, value in payload.items() if key != "access_report_hash"}
    required = {
        "schema_version": "midogpp_prediction_only_source_label_capability_v1",
        "status": "OPEN_SOURCE_ONLY",
        "source_prediction_seal_hash": source_prediction_seal_hash,
        "source_oof_classifier_bank_seal_hash": (
            source_oof_classifier_bank_seal_hash
        ),
        "target_classifier_bank_seal_hash": target_classifier_bank_seal_hash,
        "source_row_count": EXPECTED_SOURCE_ROWS,
        "outer_targets_accessed": list(CENTERS),
        "outer_target_label_excluded": True,
        "query_excluded_from_every_source_action_composition": True,
        "source_labels_opened": True,
        "source_labels_opened_after_complete_prediction_seal": True,
        "source_oof_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "source_oof_oriented_prediction_cell_count": (
            DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        ),
        "target_compatible_classifier_fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "test_manifest_opened": False,
        "test_labels_opened": False,
        "test_labels_available": False,
        "raw_source_labels_persisted": False,
        "raw_sample_ids_persisted": False,
    }
    if (
        set(unhashed) != set(required)
        or any(payload.get(key) != value for key, value in required.items())
        or payload.get("access_report_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("Prediction-only source label capability drifted.")
    return payload


def validate_source_identity_topology(source_store: object) -> dict[str, object]:
    rows_by_query = getattr(source_store, "rows_by_query", {})
    cases_by_query = getattr(source_store, "case_ids_by_query", {})
    if tuple(rows_by_query) != CENTERS or tuple(cases_by_query) != CENTERS:
        raise ProtocolError("Strict source-OOF query identity topology drifted.")
    source_rows = tuple(row for query in CENTERS for row in rows_by_query[query])
    source_cases = tuple(case for query in CENTERS for case in cases_by_query[query])
    source_queries = tuple(
        query for query in CENTERS for _row in rows_by_query[query]
    )
    source_cases_by_query: dict[str, tuple[str, ...]] = {}
    source_sample_counts: dict[tuple[str, str], int] = {}
    for query in CENTERS:
        if len(rows_by_query[query]) != len(cases_by_query[query]):
            raise ProtocolError("Strict source-OOF query row alignment drifted.")
        cases = tuple(sorted(set(cases_by_query[query])))
        source_cases_by_query[query] = cases
        for case in cases:
            source_sample_counts[(query, case)] = tuple(
                cases_by_query[query]
            ).count(case)
    if (
        len(source_rows) != EXPECTED_SOURCE_ROWS
        or len(set(source_rows)) != EXPECTED_SOURCE_ROWS
        or any(not value.startswith("src_") for value in source_rows)
        or dict(Counter(source_queries)) != EXPECTED_SOURCE_ROWS_BY_CENTER
        or len(set(source_cases)) != EXPECTED_SOURCE_CASE_COUNT
        or sum(len(value) for value in source_cases_by_query.values())
        != EXPECTED_SOURCE_CASE_COUNT
    ):
        raise ProtocolError("Prediction-only exact 9648-row source identity drifted.")
    return {
        "source_rows": source_rows,
        "source_cases": source_cases,
        "source_cases_by_query": source_cases_by_query,
        "source_sample_counts": source_sample_counts,
    }


def validate_test_identity_topology(test_store: object) -> dict[str, object]:
    test_rows: list[str] = []
    test_case_values: list[str] = []
    test_cases_by_query: dict[str, tuple[str, ...]] = {}
    for target in CENTERS:
        rows = tuple(test_store.rows_by_outer_target[target])
        cases = tuple(test_store.case_ids_by_outer_target[target])
        queries = tuple(test_store.query_ids_by_outer_target[target])
        if (
            len(rows) != EXPECTED_TEST_ROWS_BY_CENTER[target]
            or len(cases) != len(rows)
            or queries != (target,) * len(rows)
        ):
            raise ProtocolError("Prediction-only test center topology drifted.")
        test_rows.extend(rows)
        test_case_values.extend(cases)
        test_cases_by_query[target] = tuple(sorted(set(cases)))
    if (
        len(test_rows) != EXPECTED_TEST_ROWS
        or len(set(test_rows)) != EXPECTED_TEST_ROWS
        or any(not value.startswith("eval_") for value in test_rows)
        or len(set(test_case_values)) != EXPECTED_TEST_CASE_COUNT
        or sum(len(value) for value in test_cases_by_query.values())
        != EXPECTED_TEST_CASE_COUNT
    ):
        raise ProtocolError("Prediction-only exact 9928-row test identity drifted.")
    return {
        "test_rows": tuple(test_rows),
        "test_cases": tuple(test_case_values),
        "test_cases_by_query": test_cases_by_query,
    }


def validate_identity_topology(source_store: object, test_store: object) -> dict[str, object]:
    source = validate_source_identity_topology(source_store)
    test = validate_test_identity_topology(test_store)
    if set(source["source_rows"]).intersection(test["test_rows"]) or set(
        source["source_cases"]
    ).intersection(test["test_cases"]):
        raise ProtocolError("Prediction-only train/test identity overlap detected.")
    return {
        "source_cases_by_query": source["source_cases_by_query"],
        "source_sample_counts": source["source_sample_counts"],
        "test_cases_by_query": test["test_cases_by_query"],
    }


def validate_test_prediction_chain(
    test_predictions: object,
    *,
    target_classifier_bank: object,
    composite_prediction_seal_hash: str,
) -> None:
    store = getattr(test_predictions, "test_store", None)
    admission = getattr(test_predictions, "admission", None)
    bank_by_key = getattr(target_classifier_bank, "by_key", {})
    if (
        store is None
        or getattr(test_predictions, "classifier_bank", None)
        is not target_classifier_bank
        or getattr(admission, "source_prediction_seal_hash", None)
        != composite_prediction_seal_hash
        or getattr(admission, "action_classifier_bank_seal_hash", None)
        != getattr(target_classifier_bank, "seal_hash", None)
    ):
        raise ProtocolError("Prediction-only test admission lineage drifted.")
    for cell in store.cells:
        classifier = bank_by_key.get(cell.key)
        row_hash = canonical_hash(
            list(store.rows_by_outer_target[cell.target_center])
        )
        if (
            classifier is None
            or cell.action_hash != classifier.action_hash
            or cell.classifier_parameter_sha256 != classifier.parameter_sha256
            or cell.row_identity_hash != row_hash
        ):
            raise ProtocolError(
                "Prediction-only test cell escaped its frozen target classifier."
            )


def validate_preflight(root: Path, *, runtime: Mapping[str, object]) -> None:
    payload = read_object(root / "reports/workstation_preflight.json")
    required = {
        "status": "PASS",
        "prediction_only_phase_order": [
            "generated_source_streams",
            "target_compatible_classifier_bank_seal",
            "strict_H_q_source_oof_fit_and_prediction_seal",
            "composite_prelabel_prediction_seal",
            "source_label_capability",
            "regret_model_bank_seal",
            "test_cache_admission",
            "frozen_test_inference",
        ],
        "source_oof_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT
        ),
        "source_oof_oriented_prediction_cell_count": (
            DEVELOPMENT_LOGICAL_PREDICTION_CELL_COUNT
        ),
        "target_compatible_classifier_fit_count": EXPECTED_CLASSIFIER_FIT_COUNT,
        "total_physical_classifier_fit_count": (
            DEVELOPMENT_CLASSIFIER_FIT_COUNT + EXPECTED_CLASSIFIER_FIT_COUNT
        ),
        "test_phase_classifier_fit_count": 0,
        "persistent_a5000_gpu_worker_count": 2,
        "cpu_classifier_worker_count": 4,
        "blas_threads_per_classifier_worker": 3,
        "maximum_dense_fit_bytes": 536_870_912,
    }
    if (
        payload.get("schema_version") != "midogpp_label_free_workstation_preflight_v1"
        or any(payload.get(key) != value for key, value in required.items())
        or tuple(runtime.get("source_generation_devices", ())) != ("cuda:0", "cuda:1")
        or runtime.get("cpu_workers") != 4
        or runtime.get("threads_per_worker") != 3
    ):
        raise ProtocolError("Prediction-only workstation preflight drifted.")


def validate_run_state(root: Path, *, validation_exists: bool) -> None:
    payload = read_object(root / "reports/run_state.json")
    status = payload.get("status")
    phase = payload.get("phase")
    valid_running = {
        "CLOSED_WORLD_VALIDATION",
        "CLOSED_WORLD_CONTENT_FIRST_VALIDATION",
        "CLOSED_WORLD_PREDICTION_ONLY_VALIDATION",
        "VALIDATION_RECOVERY",
    }
    if (
        payload.get("schema_version")
        != "midogpp_disagreement_regret_prediction_only_run_state_v1"
        or payload.get("prediction_only") is not True
        or payload.get("test_labels_opened") is not False
        or status not in {"RUNNING", "COMPLETE"}
        or (status == "RUNNING" and phase not in valid_running)
        or (status == "COMPLETE" and phase != "COMPLETE")
        or (status == "COMPLETE" and not validation_exists)
        or "error" in payload
    ):
        raise ProtocolError("Prediction-only run state is not validatable.")


__all__ = tuple(name for name in globals() if name.startswith("validate_"))
