"""Independent reconstruction of a completed exact-tail scientific bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ..utility_aligned import CandidateFeatureRow
from .config import (
    INPUT_ARTIFACT_IDS,
    ExactTailUtilitySurfaceConfig,
    load_exact_tail_utility_surface_config,
)
from .contracts import (
    CENTERS,
    EXPECTED_COARSE_TASK_COUNT,
    EXPECTED_SOURCE_STREAM_COUNT,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
    expected_coarse_task_keys,
    expected_prediction_keys,
    expected_utility_keys,
)
from .features import validate_aligned_candidate_features
from .label_access import open_globally_sealed_development_labels
from .production_inputs import parse_development_partition
from .scoring import (
    ScoredExactTailUtilityRow,
    SealedPredictionSurface,
    array_sha256,
    score_exact_tail_utility_surface,
)
from .seals import PredictionCellSeal, build_global_prediction_seal


def reconstruct_surface_bundle(
    root: Path,
    *,
    config: ExactTailUtilitySurfaceConfig | None,
    lock: object,
) -> tuple[object, tuple[ScoredExactTailUtilityRow, ...], tuple[CandidateFeatureRow, ...]]:
    """Rebuild rows, arrays, index, seal, and reports from independent inputs."""

    from .bundle import build_surface_lock, sha256_file

    resolved = load_exact_tail_utility_surface_config(root / "config.resolved.yaml")
    effective = resolved if config is None else config
    if config is not None and resolved != config:
        raise ProtocolError("Exact-tail resolved config drifted from the running config.")
    if effective.contract_hash != lock.config_contract_hash:
        raise ProtocolError("Exact-tail resolved config escaped the surface lock.")
    _validate_provenance(root, effective)
    partitions, reservation = _load_partitions(root)
    seal, predictions = _reconstruct_predictions(root, effective, partitions, lock)
    utility_rows = _load_utility_rows(root / "tables/exact_tail_utility.csv")
    feature_rows = _load_feature_rows(root / "tables/candidate_features.csv")
    validate_aligned_candidate_features(feature_rows, utility_rows)

    labels = open_globally_sealed_development_labels(
        effective.development_manifest_path,
        partitions,
        seal=seal,
        seal_path=root / "manifests/global_prediction_seal.json",
        prediction_index_path=root / "manifests/prediction_index.json",
        prediction_arrays_path=root / "arrays/exact_tail_predictions.npz",
    )
    rescored = score_exact_tail_utility_surface(predictions, labels, partitions)
    if [row.to_payload() for row in utility_rows] != [
        row.to_payload() for row in rescored
    ]:
        raise ProtocolError("Exact-tail utility table drifted from sealed rescoring.")

    rebuilt = build_surface_lock(
        seal=seal,
        rows=utility_rows,
        feature_rows=feature_rows,
        utility_table_sha256=sha256_file(root / "tables/exact_tail_utility.csv"),
        feature_table_sha256=sha256_file(root / "tables/candidate_features.csv"),
        member_sha256=lock.member_sha256,
    )
    if rebuilt.to_payload() != lock.to_payload():
        raise ProtocolError("Exact-tail surface lock drifted from reconstruction.")
    _validate_supporting_tables(root, partitions)
    _validate_protocol_and_reports(root, effective, reservation, seal, lock)
    return rebuilt, utility_rows, feature_rows


def _load_partitions(root: Path) -> tuple[dict[str, object], Mapping[str, object]]:
    raw = _json(root / "manifests/development_reservation.json")
    expected = {
        "schema_version", "status", "dataset_family", "center_universe",
        "partitions", "metadata_similarity_by_query_source",
        "metadata_profile_sha256", "reservation_cache_and_index_contain_labels",
        "whole_case_support_evaluation_disjoint",
        "development_target_evaluation_disjoint", "reservation_hash",
    }
    unhashed = {key: value for key, value in raw.items() if key != "reservation_hash"}
    if (
        set(raw) != expected
        or raw.get("schema_version") != "midogpp_utility_aligned_development_reservation_v1"
        or raw.get("status") != "READY"
        or raw.get("dataset_family") != "MIDOG++"
        or raw.get("center_universe") != list(CENTERS)
        or raw.get("reservation_cache_and_index_contain_labels") is not False
        or raw.get("whole_case_support_evaluation_disjoint") is not True
        or raw.get("development_target_evaluation_disjoint") is not True
        or raw.get("reservation_hash") != stable_hash(unhashed)
    ):
        raise ProtocolError("Persisted exact-tail reservation drifted.")
    values = raw.get("partitions")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ProtocolError("Persisted exact-tail partitions are absent.")
    partitions: dict[str, object] = {}
    for value in values:
        partition = parse_development_partition(value)
        if partition.center in partitions:
            raise ProtocolError("Persisted exact-tail partition is duplicated.")
        partitions[partition.center] = partition
    if tuple(partitions) != CENTERS:
        raise ProtocolError("Persisted exact-tail partition coverage drifted.")
    return partitions, raw


def _reconstruct_predictions(
    root: Path,
    config: ExactTailUtilitySurfaceConfig,
    partitions: Mapping[str, object],
    lock: object,
):
    from .bundle import sha256_file

    index_path = root / "manifests/prediction_index.json"
    arrays_path = root / "arrays/exact_tail_predictions.npz"
    index = _json(index_path)
    expected_index = {
        "schema_version", "array_member", "array_file_sha256",
        "allowed_array_keys", "prediction_dtype", "probability_dtype",
        "offset_dtype", "cell_count", "cells", "labels_stored",
        "all_predictions_materialized_before_development_labels",
        "prediction_index_hash",
    }
    unhashed = {key: value for key, value in index.items() if key != "prediction_index_hash"}
    if (
        set(index) != expected_index
        or index.get("schema_version") != "midogpp_exact_tail_prediction_index_v1"
        or index.get("array_member") != "arrays/exact_tail_predictions.npz"
        or index.get("array_file_sha256") != sha256_file(arrays_path)
        or index.get("allowed_array_keys") != ["predictions", "probabilities", "offsets"]
        or index.get("prediction_dtype") != "uint8"
        or index.get("probability_dtype") != "float32"
        or index.get("offset_dtype") != "int64"
        or index.get("labels_stored") is not False
        or index.get("all_predictions_materialized_before_development_labels") is not True
        or index.get("prediction_index_hash") != stable_hash(unhashed)
    ):
        raise ProtocolError("Exact-tail prediction index drifted.")
    raw_cells = index.get("cells")
    if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
        raise ProtocolError("Exact-tail prediction cells are absent.")
    expected_keys = expected_prediction_keys()
    if len(raw_cells) != len(expected_keys) or index.get("cell_count") != len(expected_keys):
        raise ProtocolError("Exact-tail prediction cell count drifted.")
    with np.load(arrays_path, allow_pickle=False) as payload:
        if set(payload.files) != {"predictions", "probabilities", "offsets"}:
            raise ProtocolError("Exact-tail prediction NPZ schema drifted.")
        flat_predictions = np.asarray(payload["predictions"])
        flat_probabilities = np.asarray(payload["probabilities"])
        offsets = np.asarray(payload["offsets"])
    if (
        flat_predictions.ndim != 1 or flat_predictions.dtype != np.uint8
        or not np.isin(flat_predictions, (0, 1)).all()
        or flat_probabilities.ndim != 1 or flat_probabilities.dtype != np.float32
        or not np.isfinite(flat_probabilities).all()
        or np.any((flat_probabilities < 0.0) | (flat_probabilities > 1.0))
        or offsets.dtype != np.int64 or offsets.shape != (len(expected_keys) + 1,)
        or offsets[0] != 0 or offsets[-1] != len(flat_predictions)
        or len(flat_probabilities) != len(flat_predictions)
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ProtocolError("Exact-tail prediction arrays drifted.")
    cells: list[PredictionCellSeal] = []
    prediction_map: dict[tuple[str, str, str, int, int], np.ndarray] = {}
    cell_keys = {
        "cell_ordinal", "outer_target", "pseudo_query", "action_id",
        "training_seed", "generation_seed", "action_hash",
        "evaluation_row_identity_hash", "prediction_sha256", "probability_sha256",
        "composition_sha256", "classifier_config_hash",
        "evaluation_labels_available_to_fit_or_predict", "support_labels_used",
        "target_labels_used", "seed_selection_performed", "array_start",
        "array_stop", "scaler_state_hash", "labels_stored",
    }
    for ordinal, (expected_key, raw) in enumerate(zip(expected_keys, raw_cells, strict=True)):
        if not isinstance(raw, Mapping) or set(raw) != cell_keys:
            raise ProtocolError("Exact-tail prediction cell schema drifted.")
        start, stop = int(raw["array_start"]), int(raw["array_stop"])
        if (
            raw.get("cell_ordinal") != ordinal
            or start != int(offsets[ordinal]) or stop != int(offsets[ordinal + 1])
            or raw.get("labels_stored") is not False
            or raw.get("evaluation_labels_available_to_fit_or_predict") is not False
            or raw.get("support_labels_used") is not False
            or raw.get("target_labels_used") is not False
            or raw.get("seed_selection_performed") is not False
        ):
            raise ProtocolError("Exact-tail prediction cell boundary drifted.")
        cell = PredictionCellSeal(
            outer_target=str(raw["outer_target"]), pseudo_query=str(raw["pseudo_query"]),
            action_id=str(raw["action_id"]), training_seed=int(raw["training_seed"]),
            generation_seed=int(raw["generation_seed"]), action_hash=str(raw["action_hash"]),
            evaluation_row_identity_hash=str(raw["evaluation_row_identity_hash"]),
            prediction_sha256=str(raw["prediction_sha256"]),
            probability_sha256=str(raw["probability_sha256"]),
            composition_sha256=str(raw["composition_sha256"]),
            classifier_config_hash=str(raw["classifier_config_hash"]),
        )
        pred = np.ascontiguousarray(flat_predictions[start:stop])
        prob = np.ascontiguousarray(flat_probabilities[start:stop])
        if cell.key != expected_key or array_sha256(pred) != cell.prediction_sha256 or array_sha256(prob) != cell.probability_sha256:
            raise ProtocolError("Exact-tail indexed prediction bytes drifted.")
        cells.append(cell)
        prediction_map[cell.key] = pred
    seal = build_global_prediction_seal(
        config_contract_hash=config.contract_hash,
        reservation_index_hash=lock.reservation_index_hash,
        development_cache_binding_hash=lock.development_cache_binding_hash,
        development_manifest_sha256=lock.development_manifest_sha256,
        target_evaluation_binding_hash=lock.target_evaluation_binding_hash,
        prediction_index_sha256=sha256_file(index_path),
        prediction_arrays_sha256=sha256_file(arrays_path),
        partitions=partitions,
        cells=cells,
    )
    persisted = _json(root / "manifests/global_prediction_seal.json")
    if persisted != seal.to_payload() or seal.seal_hash != lock.prediction_seal_hash:
        raise ProtocolError("Exact-tail global seal drifted from arrays/index/partitions.")
    return seal, SealedPredictionSurface(prediction_map, seal)


def _load_utility_rows(path: Path) -> tuple[ScoredExactTailUtilityRow, ...]:
    rows = []
    for raw in _csv(path):
        expected = {
            "schema_version", "outer_target", "pseudo_query", "candidate_source",
            "training_seed", "generation_seed", "replicate_id", "base_bacc",
            "tail_bacc", "delta_bacc", "evaluation_row_count",
            "evaluation_case_count", "evaluation_row_hash", "support_partition_hash",
            "prediction_seal_hash", "base_prediction_sha256", "tail_prediction_sha256",
            "primary_metric", "response_semantics",
            "development_labels_used_for_scoring_only", "target_support_labels_used",
            "target_evaluation_labels_used", "seed_selection_performed", "utility_row_hash",
        }
        if set(raw) != expected or raw["schema_version"] != "midogpp_exact_additive_tail_utility_row_v1" or not _true(raw["development_labels_used_for_scoring_only"]):
            raise ProtocolError("Exact-tail utility CSV schema drifted.")
        row = ScoredExactTailUtilityRow(
            outer_target=raw["outer_target"], pseudo_query=raw["pseudo_query"],
            candidate_source=raw["candidate_source"], training_seed=int(raw["training_seed"]),
            generation_seed=int(raw["generation_seed"]), base_bacc=float(raw["base_bacc"]),
            tail_bacc=float(raw["tail_bacc"]), delta_bacc=float(raw["delta_bacc"]),
            evaluation_row_count=int(raw["evaluation_row_count"]),
            evaluation_case_count=int(raw["evaluation_case_count"]),
            evaluation_row_hash=raw["evaluation_row_hash"],
            support_partition_hash=raw["support_partition_hash"],
            prediction_seal_hash=raw["prediction_seal_hash"],
            base_prediction_sha256=raw["base_prediction_sha256"],
            tail_prediction_sha256=raw["tail_prediction_sha256"],
            utility_row_hash=raw["utility_row_hash"], primary_metric=raw["primary_metric"],
            response_semantics=raw["response_semantics"],
            target_support_labels_used=_true(raw["target_support_labels_used"]),
            target_evaluation_labels_used=_true(raw["target_evaluation_labels_used"]),
            seed_selection_performed=_true(raw["seed_selection_performed"]),
        )
        if raw["replicate_id"] != row.replicate_id:
            raise ProtocolError("Exact-tail utility replicate identity drifted.")
        rows.append(row)
    if tuple((r.outer_target, r.pseudo_query, r.candidate_source, r.training_seed, r.generation_seed) for r in rows) != expected_utility_keys():
        raise ProtocolError("Exact-tail utility CSV key grid drifted.")
    return tuple(rows)


def _load_feature_rows(path: Path) -> tuple[CandidateFeatureRow, ...]:
    rows = []
    core_keys = {
        "schema_version", "role", "outer_target_id", "query_id", "candidate_source",
        "training_seed", "generation_seed", "replicate_id", "candidate_source_count",
        "support_partition_hash", "support_case_count", "reconstruction_mean",
        "reconstruction_std", "reconstruction_q25", "reconstruction_q50",
        "reconstruction_q75", "kl_mean", "kl_std", "kl_q25", "kl_q50", "kl_q75",
        "replica_disagreement", "distribution_mmd", "metadata_similarity",
        "feature_semantics", "row_hash", "distribution_mmd_semantics",
    }
    for raw in _csv(path):
        if set(raw) != core_keys or raw["schema_version"] != "midogpp_utility_aligned_candidate_feature_row_v1" or raw["distribution_mmd_semantics"] != "linear_kernel_mmd_squared":
            raise ProtocolError("Exact-tail candidate-feature CSV schema drifted.")
        row = CandidateFeatureRow(
            role=raw["role"], outer_target_id=raw["outer_target_id"], query_id=raw["query_id"],
            candidate_source=raw["candidate_source"], training_seed=int(raw["training_seed"]),
            generation_seed=int(raw["generation_seed"]), candidate_source_count=int(raw["candidate_source_count"]),
            support_partition_hash=raw["support_partition_hash"], support_case_count=int(raw["support_case_count"]),
            reconstruction_mean=float(raw["reconstruction_mean"]), reconstruction_std=float(raw["reconstruction_std"]),
            reconstruction_q25=float(raw["reconstruction_q25"]), reconstruction_q50=float(raw["reconstruction_q50"]), reconstruction_q75=float(raw["reconstruction_q75"]),
            kl_mean=float(raw["kl_mean"]), kl_std=float(raw["kl_std"]), kl_q25=float(raw["kl_q25"]), kl_q50=float(raw["kl_q50"]), kl_q75=float(raw["kl_q75"]),
            replica_disagreement=float(raw["replica_disagreement"]), distribution_mmd=float(raw["distribution_mmd"]),
            metadata_similarity=float(raw["metadata_similarity"]), feature_semantics=raw["feature_semantics"],
        )
        if raw["replicate_id"] != row.replicate_id or raw["row_hash"] != row.row_hash:
            raise ProtocolError("Exact-tail candidate-feature row hash drifted.")
        rows.append(row)
    if tuple(row.row_key for row in rows) != expected_utility_keys():
        raise ProtocolError("Exact-tail candidate-feature key grid drifted.")
    return tuple(rows)


def _validate_supporting_tables(root: Path, partitions: Mapping[str, object]) -> None:
    source_rows = _csv(root / "tables/source_streams.csv")
    source_keys = tuple((row["source_center"], int(row["training_seed"]), int(row["generation_seed"])) for row in source_rows)
    expected_sources = tuple((center, train, gen) for center in CENTERS for train in (17, 42, 101) for gen in (17, 42, 101))
    if len(source_rows) != EXPECTED_SOURCE_STREAM_COUNT or source_keys != expected_sources:
        raise ProtocolError("Exact-tail source-stream table grid drifted.")
    task_rows = _csv(root / "tables/coarse_prediction_tasks.csv")
    task_keys = tuple((row["outer_target"], row["pseudo_query"], int(row["training_seed"]), int(row["generation_seed"])) for row in task_rows)
    if len(task_rows) != EXPECTED_COARSE_TASK_COUNT or task_keys != expected_coarse_task_keys():
        raise ProtocolError("Exact-tail coarse-task table grid drifted.")
    evaluation = _csv(root / "tables/evaluation_rows.csv")
    expected_eval = [row.identity_payload() for center in CENTERS for row in partitions[center].evaluation_rows]
    if len(evaluation) != len(expected_eval):
        raise ProtocolError("Exact-tail evaluation-row table count drifted.")
    for raw, expected in zip(evaluation, expected_eval, strict=True):
        if raw.get("schema_version") != "midogpp_exact_tail_evaluation_row_v1" or not _false(raw.get("label_present")) or any(str(raw.get(key)) != str(value) for key, value in expected.items()):
            raise ProtocolError("Exact-tail evaluation-row identity drifted.")


def _validate_provenance(root: Path, config: ExactTailUtilitySurfaceConfig) -> None:
    raw = _json(root / "provenance/input_artifacts.json")
    required = {"schema_version", "dataset_id", "experiment_id", "stage", "claim_scope", "selection_used_target_eval_artifacts", "input_artifacts", "repository_revision", "repository_dirty", "repository_status_hash"}
    rows = raw.get("input_artifacts")
    if set(raw) != required or raw.get("schema_version") != "midogpp_input_artifacts_v2" or raw.get("dataset_id") != "midogpp" or raw.get("experiment_id") != EXPERIMENT_ID or raw.get("stage") != "60_routing_and_composition" or raw.get("claim_scope") != "routing_and_composition" or raw.get("selection_used_target_eval_artifacts") is not False or not isinstance(rows, Sequence):
        raise ProtocolError("Exact-tail workspace provenance drifted.")
    if tuple(str(row.get("artifact_id")) for row in rows if isinstance(row, Mapping)) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Exact-tail workspace input graph drifted.")


def _validate_protocol_and_reports(root: Path, config: ExactTailUtilitySurfaceConfig, reservation: Mapping[str, object], seal: object, lock: object) -> None:
    protocol = _json(root / "manifests/protocol_manifest.json")
    required_protocol = {"schema_version", "experiment_id", "output_artifact_id", "config_contract_hash", "reservation_hash", "generated_source_cache_hash", "generation_lock_hash", "bank_lock_hash", "metadata_profile_sha256", "prediction_seal_hash", "inner_geometry", "distribution_mmd_semantics", "minimum_independent_support_cases_per_query", "uncertainty_units", "seed_cells_are_uncertainty_units", "all_predictions_sealed_before_development_labels", "dedicated_scoring_manifest_contains_exactly_sealed_rows", "target_support_labels_used", "target_evaluation_labels_used", "source_experts_updated", "seed_selection_performed"}
    if set(protocol) != required_protocol or protocol.get("schema_version") != "midogpp_exact_tail_protocol_manifest_v1" or protocol.get("experiment_id") != EXPERIMENT_ID or protocol.get("output_artifact_id") != OUTPUT_ARTIFACT_ID or protocol.get("config_contract_hash") != config.contract_hash or protocol.get("reservation_hash") != reservation["reservation_hash"] or protocol.get("prediction_seal_hash") != seal.seal_hash or protocol.get("inner_geometry") != "seven_by_144_base_plus_126_single_source_tail" or protocol.get("distribution_mmd_semantics") != "linear_kernel_mmd_squared" or protocol.get("minimum_independent_support_cases_per_query") != 8 or protocol.get("uncertainty_units") != ["query_cluster", "case_cluster"] or protocol.get("seed_cells_are_uncertainty_units") is not False or any(protocol.get(key) is not False for key in ("target_support_labels_used", "target_evaluation_labels_used", "source_experts_updated", "seed_selection_performed")) or protocol.get("all_predictions_sealed_before_development_labels") is not True or protocol.get("dedicated_scoring_manifest_contains_exactly_sealed_rows") is not True:
        raise ProtocolError("Exact-tail protocol manifest drifted.")
    leakage = _json(root / "reports/leakage_report.json")
    if leakage.get("status") != "PASS" or leakage.get("surface_lock_hash") != lock.surface_lock_hash or leakage.get("target_support_labels_used") is not False or leakage.get("target_evaluation_labels_used") is not False:
        raise ProtocolError("Exact-tail leakage report drifted.")
    state = _json(root / "reports/run_state.json")
    if state.get("status") != "COMPLETE" or state.get("surface_lock_hash") != lock.surface_lock_hash or state.get("prediction_cell_count") != 5184 or state.get("utility_row_count") != 4536 or state.get("feature_row_count") != 4536:
        raise ProtocolError("Exact-tail run-state report drifted.")


def validation_report_payload(surface_lock_hash: str) -> dict[str, object]:
    return {
        "schema_version": "midogpp_exact_tail_validation_report_v1",
        "status": "PASS",
        "surface_lock_hash": surface_lock_hash,
        "bundle_member_hashes_validated": True,
        "config_and_input_bindings_validated": True,
        "prediction_npz_index_and_global_seal_reconstructed": True,
        "utility_rows_rescored_from_sealed_predictions": True,
        "candidate_feature_rows_reconstructed": True,
        "utility_feature_key_alignment_validated": True,
        "predictions_sealed_before_labels": True,
    }


def _csv(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ProtocolError(f"Exact-tail CSV header drifted: {path}.")
            return tuple(dict(row) for row in reader)
    except OSError as exc:
        raise ProtocolError(f"Cannot read exact-tail CSV: {path}.") from exc


def _json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read exact-tail JSON: {path}.") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("Exact-tail JSON must be an object.")
    return raw


def _true(value: object) -> bool:
    if value not in (True, "True"):
        if value in (False, "False"):
            return False
        raise ProtocolError("Exact-tail persisted boolean is malformed.")
    return True


def _false(value: object) -> bool:
    return not _true(value)


__all__ = ("reconstruct_surface_bundle", "validation_report_payload")
