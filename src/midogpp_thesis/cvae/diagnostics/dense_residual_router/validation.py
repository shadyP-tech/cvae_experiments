"""Independent reconstruction for the consumed-data residual diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    source_generation_plan,
    validate_generation_bundle,
)
from ...generation.generation import derived_composition_seed
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import (
    build_hamilton_allocation,
    residual_soft_weights,
)
from .bundle import (
    CONTENT_INDEX_MEMBERS,
    REQUIRED_FILES,
    action_library_payload,
    label_access_report_payload,
    leakage_report_payload,
    phase_01_support_and_compatibility_payload,
    phase_02_development_complete_payload,
    phase_03_target_predictions_complete_payload,
    phase_04_scoring_complete_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    run_state_payload,
)
from .config import (
    DenseResidualDiagnosticConfig,
    load_dense_residual_diagnostic_config,
)
from .contracts import (
    ACTION_IDS,
    CENTERS,
    CONTROL_ACTION_ID,
    DEVELOPMENT_TOTAL_PER_CLASS,
    GENERATION_SEEDS,
    INPUT_ARTIFACT_IDS,
    MAX_SOURCE_WEIGHT,
    MIN_EFFECTIVE_SOURCE_COUNT,
    RHO_VALUES,
    TEMPERATURE,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    development_queries,
    legal_sources,
    row_identity_hash,
    target_sources,
)
from .execution import (
    COMPATIBILITY_CASE_COLUMNS,
    COMPATIBILITY_SCORE_COLUMNS,
    SUPPORT_PARTITION_COLUMNS,
    TARGET_ASSIGNMENT_COLUMNS,
    TARGET_WEIGHT_PLAN_COLUMNS,
    build_partition_surface,
    compute_compatibility_surface,
    load_label_free_validation_frame,
)
from .label_access import open_development_labels, open_target_labels
from .prediction_io import (
    DEVELOPMENT_ARRAY_MEMBER,
    PREDICTION_INDEX_COLUMNS,
    TARGET_ARRAY_MEMBER,
    read_prediction_store,
    sha256_file,
)
from .seals import (
    AllActionTargetPredictionSeal,
    DevelopmentPredictionSeal,
    DiagnosticDecisionSeal,
    PredictionCellSeal,
    TargetPredictionSeal,
)
from .selection import (
    ACTION_SUMMARY_COLUMNS,
    METRIC_COLUMNS,
    PAIRED_DELTA_COLUMNS,
    SELECTION_COLUMNS,
    choose_diagnostic_action,
    paired_target_deltas,
    score_prediction_cells,
    summarize_development_actions,
)


def validate_dense_residual_router_bundle(
    root: str | Path,
    *,
    config: DenseResidualDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Rebuild label gates, metrics, decisions, and every claim-bound report."""

    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(relative for relative in required if not (path / relative).is_file())
    if missing:
        raise ProtocolError(f"Dense residual artifact is incomplete: {missing}.")
    _validate_closed_world(path, allow_pending=allow_pending)
    resolved = load_dense_residual_diagnostic_config(path / "config.resolved.yaml")
    if (
        resolved.contract_hash != config.contract_hash
        or resolved.artifact_root.resolve() != config.artifact_root.resolve()
        or resolved.expert_bank_root.resolve() != config.expert_bank_root.resolve()
        or resolved.generation_lock_root.resolve()
        != config.generation_lock_root.resolve()
        or resolved.validation_cache_root.resolve()
        != config.validation_cache_root.resolve()
        or resolved.validation_manifest_path.resolve()
        != config.validation_manifest_path.resolve()
    ):
        raise ProtocolError("Dense residual resolved config drifted.")

    provenance = _validate_provenance(path, config=config)
    frame = load_label_free_validation_frame(config)
    expected_protocol = protocol_manifest_payload(
        config,
        input_artifact_hashes={
            artifact_id: stable_hash(provenance[artifact_id])
            for artifact_id in INPUT_ARTIFACT_IDS
        },
        validation_cache_binding_hash=frame.cache_binding_hash,
    )
    _require_payload(path / "manifests/protocol_manifest.json", expected_protocol)
    _require_payload(
        path / "manifests/action_library.json",
        action_library_payload(config),
    )

    partitions = build_partition_surface(frame, config_contract_hash=config.contract_hash)
    observed_partitions = _read_csv_exact(
        path / "tables/support_partitions.csv", SUPPORT_PARTITION_COLUMNS
    )
    if observed_partitions != _rows_as_strings(partitions.table_rows):
        raise ProtocolError("Dense residual support partition table drifted.")
    _require_payload(
        path / "manifests/support_partition_lock.json",
        partitions.lock_payload,
    )

    compatibility_cases = _read_csv_exact(
        path / "tables/compatibility_case_energy.csv", COMPATIBILITY_CASE_COLUMNS
    )
    compatibility_scores = _read_csv_exact(
        path / "tables/compatibility_scores.csv", COMPATIBILITY_SCORE_COLUMNS
    )
    reconstructed_compatibility = compute_compatibility_surface(
        config,
        frame,
        partitions,
    )
    if (
        compatibility_cases != _rows_as_strings(reconstructed_compatibility.case_rows)
        or compatibility_scores
        != _rows_as_strings(reconstructed_compatibility.score_rows)
        or any(row["query_partition_role"] != "support" for row in compatibility_cases)
        or any(row["labels_used"] != "False" for row in compatibility_cases)
        or any(row["query_support_labels_used"] != "False" for row in compatibility_scores)
    ):
        raise ProtocolError(
            "Dense residual compatibility escaped support-only semantics or failed "
            "independent reconstruction."
        )
    compatibility_index = _json(path / "manifests/compatibility_index.json")
    expected_index = {
        "schema_version": "midogpp_dense_residual_compatibility_index_v1",
        "config_contract_hash": config.contract_hash,
        "case_energy_member": "tables/compatibility_case_energy.csv",
        "case_energy_sha256": sha256_file(path / "tables/compatibility_case_energy.csv"),
        "case_energy_row_count": len(compatibility_cases),
        "score_member": "tables/compatibility_scores.csv",
        "score_sha256": sha256_file(path / "tables/compatibility_scores.csv"),
        "score_row_count": len(compatibility_scores),
        "query_rows_consumed": "support_partition_rows_only",
        "query_evaluation_embeddings_consumed": False,
        "support_labels_used": False,
        "exact_nelbo_claimed": False,
        "replica_aggregation": "arithmetic_mean_all_three_no_seed_selection",
    }
    expected_index["compatibility_index_hash"] = stable_hash(expected_index)
    if compatibility_index != expected_index:
        raise ProtocolError("Dense residual compatibility index drifted.")
    _require_payload(
        path / "reports/phase_01_support_and_compatibility_complete.json",
        phase_01_support_and_compatibility_payload(
            config,
            support_partition_lock_hash=partitions.lock_hash,
            compatibility_index_hash=str(expected_index["compatibility_index_hash"]),
            support_partition_row_count=len(partitions.table_rows),
            compatibility_score_row_count=len(compatibility_scores),
        ),
    )

    development_index = _read_csv_exact(
        path / "tables/development_prediction_index.csv", PREDICTION_INDEX_COLUMNS
    )
    target_index = _read_csv_exact(
        path / "tables/target_prediction_index.csv", PREDICTION_INDEX_COLUMNS
    )
    development_store = read_prediction_store(
        path / DEVELOPMENT_ARRAY_MEMBER, development_index
    )
    target_store = read_prediction_store(path / TARGET_ARRAY_MEMBER, target_index)
    all_action_target_seal = _read_all_action_target_seal(
        path / "manifests/all_action_target_prediction_seal.json"
    )
    _validate_all_action_target_seal(
        all_action_target_seal,
        config=config,
        path=path,
        partitions=partitions,
        compatibility_index_hash=str(expected_index["compatibility_index_hash"]),
        validation_cache_binding_hash=frame.cache_binding_hash,
        target_index=target_index,
    )
    generation_lock = _load_validated_generation_lock(config)
    _validate_development_prediction_mechanism(
        config,
        development_index,
        partitions=partitions,
        compatibility=reconstructed_compatibility,
        generation_lock=generation_lock,
    )
    _validate_target_prediction_mechanism(
        config,
        target_index,
        partitions=partitions,
        compatibility=reconstructed_compatibility,
        generation_lock=generation_lock,
    )
    development_seals = _read_development_seals(
        path / "manifests/development_prediction_seals.json"
    )
    decision_seals = _read_decision_seals(
        path / "manifests/diagnostic_decision_seals.json"
    )
    target_seals = _read_target_seals(path / "manifests/target_prediction_seals.json")
    if tuple(seal.outer_target for seal in development_seals) != CENTERS:
        raise ProtocolError("Dense residual development seal order/coverage drifted.")
    if tuple(seal.outer_target for seal in decision_seals) != CENTERS:
        raise ProtocolError("Dense residual decision seal order/coverage drifted.")
    if tuple(seal.outer_target for seal in target_seals) != CENTERS:
        raise ProtocolError("Dense residual target seal order/coverage drifted.")
    _validate_seal_disk_bindings(
        path,
        development_seals=development_seals,
        target_seals=target_seals,
    )

    expected_development_metrics: list[Mapping[str, object]] = []
    expected_action_summaries: list[Mapping[str, object]] = []
    expected_selections: list[Mapping[str, object]] = []
    development_label_hashes: dict[str, str] = {}
    for outer_target, prediction_seal, decision_seal in zip(
        CENTERS, development_seals, decision_seals, strict=True
    ):
        requested_rows = tuple(
            row
            for query in development_queries(outer_target)
            for row in partitions.evaluation_rows_by_center[query]
        )
        opened = open_development_labels(
            config.validation_manifest_path,
            requested_rows,
            seal=prediction_seal,
            all_action_target_seal=all_action_target_seal,
            all_action_target_seal_path=path
            / "manifests/all_action_target_prediction_seal.json",
            prediction_index_path=path / "tables/development_prediction_index.csv",
            prediction_arrays_path=path / DEVELOPMENT_ARRAY_MEMBER,
            target_prediction_index_path=path
            / "tables/target_prediction_index.csv",
            target_prediction_arrays_path=path / TARGET_ARRAY_MEMBER,
            expected_manifest_sha256=config.expected_manifest_sha256,
        )
        labels = dict(
            zip((row.sample_id for row in opened.rows), opened.labels, strict=True)
        )
        metrics = score_prediction_cells(
            development_store,
            labels_by_sample_id=labels,
            phase="development",
            outer_target=outer_target,
        )
        summaries = summarize_development_actions(
            metrics,
            development_store.index_rows,
            outer_target=outer_target,
        )
        selection = choose_diagnostic_action(summaries, outer_target=outer_target)
        if (
            decision_seal.development_prediction_seal_hash != prediction_seal.seal_hash
            or decision_seal.development_label_vector_hash != opened.label_vector_hash
            or decision_seal.development_metrics_sha256 != _rows_sha256(metrics)
            or decision_seal.action_summaries_sha256 != _rows_sha256(summaries)
            or decision_seal.selected_action_id != selection["selected_action_id"]
            or decision_seal.selected_rho != selection["selected_rho"]
            or decision_seal.selected_mean_paired_bacc_delta_vs_control
            != selection["selected_mean_paired_bacc_delta_vs_control"]
            or decision_seal.fallback_applied != selection["fallback_applied"]
            or decision_seal.fallback_reason != selection["fallback_reason"]
        ):
            raise ProtocolError("Dense residual diagnostic decision seal drifted.")
        expected_development_metrics.extend(metrics)
        expected_action_summaries.extend(summaries)
        expected_selections.append(selection)
        development_label_hashes[outer_target] = opened.label_vector_hash

    _require_csv_rows(
        path / "tables/development_metrics.csv",
        METRIC_COLUMNS,
        expected_development_metrics,
    )
    _require_csv_rows(
        path / "tables/action_summaries.csv",
        ACTION_SUMMARY_COLUMNS,
        expected_action_summaries,
    )
    _require_csv_rows(
        path / "tables/diagnostic_selections.csv",
        SELECTION_COLUMNS,
        expected_selections,
    )
    _require_payload(
        path / "reports/phase_02_development_complete.json",
        phase_02_development_complete_payload(
            config,
            development_seals=development_seals,
            all_action_target_seal=all_action_target_seal,
            decision_seals=decision_seals,
        ),
    )

    weight_plans = _read_csv_exact(
        path / "tables/target_weight_plans.csv", TARGET_WEIGHT_PLAN_COLUMNS
    )
    assignments = _read_csv_exact(
        path / "tables/target_assignments.csv", TARGET_ASSIGNMENT_COLUMNS
    )
    _validate_target_plan_tables(
        weight_plans,
        assignments,
        target_index=target_index,
        compatibility=reconstructed_compatibility,
        generation_lock=generation_lock,
    )
    for target_seal, decision_seal in zip(target_seals, decision_seals, strict=True):
        if (
            target_seal.diagnostic_decision_hash != decision_seal.decision_hash
            or target_seal.selected_action_id != decision_seal.selected_action_id
        ):
            raise ProtocolError("Dense residual target seal/decision binding drifted.")
    _require_payload(
        path / "reports/phase_03_target_predictions_complete.json",
        phase_03_target_predictions_complete_payload(
            config,
            target_seals=target_seals,
            all_action_target_seal=all_action_target_seal,
            all_action_prediction_cell_count=len(target_index),
        ),
    )

    expected_target_metrics: list[Mapping[str, object]] = []
    target_label_hashes: dict[str, str] = {}
    for target_center, target_seal in zip(CENTERS, target_seals, strict=True):
        rows = partitions.evaluation_rows_by_center[target_center]
        opened = open_target_labels(
            config.validation_manifest_path,
            rows,
            seal=target_seal,
            prediction_index_path=path / "tables/target_prediction_index.csv",
            prediction_arrays_path=path / TARGET_ARRAY_MEMBER,
            expected_manifest_sha256=config.expected_manifest_sha256,
        )
        labels = dict(
            zip((row.sample_id for row in opened.rows), opened.labels, strict=True)
        )
        expected_target_metrics.extend(
            score_prediction_cells(
                target_store,
                labels_by_sample_id=labels,
                phase="target",
                outer_target=target_center,
                selected_target_action_id=target_seal.selected_action_id,
            )
        )
        target_label_hashes[target_center] = opened.label_vector_hash
    expected_deltas = paired_target_deltas(
        expected_target_metrics, expected_selections
    )
    _require_csv_rows(
        path / "tables/target_metrics.csv", METRIC_COLUMNS, expected_target_metrics
    )
    _require_csv_rows(
        path / "tables/paired_deltas.csv", PAIRED_DELTA_COLUMNS, expected_deltas
    )
    _require_payload(
        path / "reports/phase_04_scoring_complete.json",
        phase_04_scoring_complete_payload(
            config,
            target_metrics_sha256=sha256_file(path / "tables/target_metrics.csv"),
            paired_deltas_sha256=sha256_file(path / "tables/paired_deltas.csv"),
            target_metric_row_count=len(expected_target_metrics),
            paired_delta_row_count=len(expected_deltas),
        ),
    )
    unique_eval_rows = tuple(
        row
        for center in CENTERS
        for row in partitions.evaluation_rows_by_center[center]
    )
    _require_payload(
        path / "reports/label_access_report.json",
        label_access_report_payload(
            development_label_vector_hash_by_outer_target=development_label_hashes,
            target_label_vector_hash_by_outer_target=target_label_hashes,
            consumed_row_count=len(unique_eval_rows),
            consumed_case_count=len(
                {(row.center, row.case_id) for row in unique_eval_rows}
            ),
        ),
    )
    _require_payload(path / "reports/leakage_report.json", leakage_report_payload())
    _require_payload(
        path / "reports/publication_decision.json",
        publication_decision_payload(
            descriptive_summary_hash=stable_hash(list(expected_deltas))
        ),
    )
    _require_payload(path / "reports/run_state.json", run_state_payload("COMPLETE"))
    _validate_content_index(path)

    checks: dict[str, object] = {
        "status": "PASS",
        "publication_status": "EXPLORATORY_CONSUMED_DATA_ONLY",
        "support_partition_row_count": len(partitions.table_rows),
        "query_evaluation_embeddings_consumed_for_compatibility": False,
        "compatibility_score_row_count": len(compatibility_scores),
        "compatibility_reconstructed_from_frozen_support_and_experts": True,
        "development_prediction_cell_count": len(development_index),
        "development_metric_cell_count": len(expected_development_metrics),
        "diagnostic_decision_count": len(decision_seals),
        "target_prediction_cell_count": len(target_index),
        "target_metric_cell_count": len(expected_target_metrics),
        "paired_delta_count": len(expected_deltas),
        "all_nine_seed_cells_retained": True,
        "development_rho0_exact_144_per_source": True,
        "target_rho0_exact_128_per_source": True,
        "compatibility_to_weight_and_allocation_chain_reconstructed": True,
        "all_action_target_predictions_materialized_before_any_label_access": True,
        "seed_selection_performed": False,
        "support_labels_used": False,
        "routing_quality_claimed": False,
        "may_feed_stage60": False,
        "may_feed_stage70": False,
        "may_feed_deployable_selection": False,
        "metrics_reconstructed_from_hash_bound_predictions": True,
    }
    if not allow_pending:
        _require_payload(
            path / "reports/validation_report.json",
            {
                "schema_version": "midogpp_dense_residual_validation_v1",
                "status": "PASS",
                "validator": "validate_dense_residual_router_bundle",
                "checks": checks,
            },
        )
    return checks


def _read_development_seals(path: Path) -> tuple[DevelopmentPredictionSeal, ...]:
    payload = _seal_collection(path)
    return tuple(_development_seal(row) for row in payload["seals"])


def _read_decision_seals(path: Path) -> tuple[DiagnosticDecisionSeal, ...]:
    payload = _seal_collection(path)
    return tuple(_decision_seal(row) for row in payload["seals"])


def _read_target_seals(path: Path) -> tuple[TargetPredictionSeal, ...]:
    payload = _seal_collection(path)
    return tuple(_target_seal(row) for row in payload["seals"])


def _read_all_action_target_seal(path: Path) -> AllActionTargetPredictionSeal:
    payload = _json(path)
    seal = AllActionTargetPredictionSeal(
        config_contract_hash=str(payload["config_contract_hash"]),
        action_library_hash=str(payload["action_library_hash"]),
        support_partition_lock_hash=str(payload["support_partition_lock_hash"]),
        compatibility_index_hash=str(payload["compatibility_index_hash"]),
        validation_cache_binding_hash=str(payload["validation_cache_binding_hash"]),
        validation_manifest_sha256=str(payload["validation_manifest_sha256"]),
        prediction_index_sha256=str(payload["prediction_index_sha256"]),
        prediction_arrays_sha256=str(payload["prediction_arrays_sha256"]),
        evaluation_row_ids_by_target={
            str(key): tuple(str(value) for value in values)
            for key, values in payload["evaluation_row_ids_by_target"].items()  # type: ignore[union-attr]
        },
        evaluation_row_identity_hash_by_target={
            str(key): str(value)
            for key, value in payload[
                "evaluation_row_identity_hash_by_target"
            ].items()  # type: ignore[union-attr]
        },
        cells=tuple(_prediction_cell(row) for row in payload["cells"]),  # type: ignore[arg-type]
        seal_hash=str(payload["seal_hash"]),
        status=str(payload["status"]),
    )
    if payload != seal.to_payload():
        raise ProtocolError("Dense residual all-target-action seal payload drifted.")
    return seal


def _validate_all_action_target_seal(
    seal: AllActionTargetPredictionSeal,
    *,
    config: DenseResidualDiagnosticConfig,
    path: Path,
    partitions: object,
    compatibility_index_hash: str,
    validation_cache_binding_hash: str,
    target_index: Sequence[Mapping[str, str]],
) -> None:
    seal.verify_complete()
    if (
        seal.config_contract_hash != config.contract_hash
        or seal.support_partition_lock_hash != partitions.lock_hash
        or seal.compatibility_index_hash != compatibility_index_hash
        or seal.validation_cache_binding_hash != validation_cache_binding_hash
        or seal.validation_manifest_sha256 != config.expected_manifest_sha256
        or seal.prediction_index_sha256
        != sha256_file(path / "tables/target_prediction_index.csv")
        or seal.prediction_arrays_sha256 != sha256_file(path / TARGET_ARRAY_MEMBER)
    ):
        raise ProtocolError("Dense residual all-target-action seal binding drifted.")
    for target in CENTERS:
        rows = partitions.evaluation_rows_by_center[target]
        if (
            seal.evaluation_row_ids_by_target[target]
            != tuple(row.sample_id for row in rows)
            or seal.evaluation_row_identity_hash_by_target[target]
            != row_identity_hash(tuple(rows))
        ):
            raise ProtocolError(
                "Dense residual all-target-action seal row identity drifted."
            )
    observed = {cell.all_action_target_key: cell for cell in seal.cells}
    expected_cells = tuple(_prediction_cell_from_index(row) for row in target_index)
    expected = {cell.all_action_target_key: cell for cell in expected_cells}
    if (
        len(observed) != len(seal.cells)
        or len(expected) != len(target_index)
        or observed != expected
    ):
        raise ProtocolError(
            "Dense residual all-target-action seal differs from target prediction index."
        )


def _seal_collection(path: Path) -> Mapping[str, object]:
    payload = _json(path)
    observed_hash = payload.get("seal_collection_hash")
    unhashed = {key: value for key, value in payload.items() if key != "seal_collection_hash"}
    if (
        payload.get("status") != "COMPLETE"
        or observed_hash != stable_hash(unhashed)
        or not isinstance(payload.get("seals"), list)
    ):
        raise ProtocolError("Dense residual seal collection drifted.")
    return payload


def _prediction_cell(payload: Mapping[str, object]) -> PredictionCellSeal:
    return PredictionCellSeal(
        phase=str(payload["phase"]),
        outer_target=str(payload["outer_target"]),
        query_center=str(payload["query_center"]),
        action_id=str(payload["action_id"]),
        arm_role=str(payload["arm_role"]),
        candidate_sources=tuple(str(value) for value in payload["candidate_sources"]),  # type: ignore[arg-type]
        training_seed=int(payload["training_seed"]),
        generation_seed=int(payload["generation_seed"]),
        evaluation_row_ids=tuple(str(value) for value in payload["evaluation_row_ids"]),  # type: ignore[arg-type]
        evaluation_row_identity_hash=str(payload["evaluation_row_identity_hash"]),
        prediction_sha256=str(payload["prediction_sha256"]),
        probability_sha256=str(payload["probability_sha256"]),
        composition_hash=str(payload["composition_hash"]),
        classifier_config_hash=str(payload["classifier_config_hash"]),
    )


def _prediction_cell_from_index(
    payload: Mapping[str, str],
) -> PredictionCellSeal:
    candidate_sources = _json_value(payload["candidate_sources_json"])
    evaluation_row_ids = _json_value(payload["evaluation_row_ids_json"])
    if (
        not isinstance(candidate_sources, list)
        or any(not isinstance(value, str) for value in candidate_sources)
        or not isinstance(evaluation_row_ids, list)
        or any(not isinstance(value, str) for value in evaluation_row_ids)
    ):
        raise ProtocolError("Dense residual prediction-index identities are malformed.")
    return PredictionCellSeal(
        phase=payload["phase"],
        outer_target=payload["outer_target"],
        query_center=payload["query_center"],
        action_id=payload["action_id"],
        arm_role=payload["arm_role"],
        candidate_sources=tuple(candidate_sources),
        training_seed=int(payload["training_seed"]),
        generation_seed=int(payload["generation_seed"]),
        evaluation_row_ids=tuple(evaluation_row_ids),
        evaluation_row_identity_hash=payload["evaluation_row_identity_hash"],
        prediction_sha256=payload["prediction_sha256"],
        probability_sha256=payload["probability_sha256"],
        composition_hash=payload["composition_hash"],
        classifier_config_hash=payload["classifier_config_hash"],
    )


def _development_seal(payload: Mapping[str, object]) -> DevelopmentPredictionSeal:
    return DevelopmentPredictionSeal(
        outer_target=str(payload["outer_target"]),
        config_contract_hash=str(payload["config_contract_hash"]),
        action_library_hash=str(payload["action_library_hash"]),
        support_partition_lock_hash=str(payload["support_partition_lock_hash"]),
        validation_cache_binding_hash=str(payload["validation_cache_binding_hash"]),
        validation_manifest_sha256=str(payload["validation_manifest_sha256"]),
        prediction_index_sha256=str(payload["prediction_index_sha256"]),
        prediction_arrays_sha256=str(payload["prediction_arrays_sha256"]),
        evaluation_row_ids_by_query={
            str(key): tuple(str(value) for value in values)
            for key, values in payload["evaluation_row_ids_by_query"].items()  # type: ignore[union-attr]
        },
        evaluation_row_identity_hash_by_query={
            str(key): str(value)
            for key, value in payload["evaluation_row_identity_hash_by_query"].items()  # type: ignore[union-attr]
        },
        cells=tuple(_prediction_cell(row) for row in payload["cells"]),  # type: ignore[arg-type]
        seal_hash=str(payload["seal_hash"]),
        status=str(payload["status"]),
    )


def _decision_seal(payload: Mapping[str, object]) -> DiagnosticDecisionSeal:
    return DiagnosticDecisionSeal(
        outer_target=str(payload["outer_target"]),
        config_contract_hash=str(payload["config_contract_hash"]),
        development_prediction_seal_hash=str(
            payload["development_prediction_seal_hash"]
        ),
        development_label_vector_hash=str(payload["development_label_vector_hash"]),
        development_metrics_sha256=str(payload["development_metrics_sha256"]),
        action_summaries_sha256=str(payload["action_summaries_sha256"]),
        selected_action_id=str(payload["selected_action_id"]),
        selected_rho=float(payload["selected_rho"]),
        selected_mean_paired_bacc_delta_vs_control=float(
            payload["selected_mean_paired_bacc_delta_vs_control"]
        ),
        fallback_applied=bool(payload["fallback_applied"]),
        fallback_reason=str(payload["fallback_reason"]),
        decision_hash=str(payload["decision_hash"]),
        status=str(payload["status"]),
    )


def _target_seal(payload: Mapping[str, object]) -> TargetPredictionSeal:
    return TargetPredictionSeal(
        outer_target=str(payload["outer_target"]),
        config_contract_hash=str(payload["config_contract_hash"]),
        diagnostic_decision_hash=str(payload["diagnostic_decision_hash"]),
        selected_action_id=str(payload["selected_action_id"]),
        validation_cache_binding_hash=str(payload["validation_cache_binding_hash"]),
        validation_manifest_sha256=str(payload["validation_manifest_sha256"]),
        prediction_index_sha256=str(payload["prediction_index_sha256"]),
        prediction_arrays_sha256=str(payload["prediction_arrays_sha256"]),
        evaluation_row_ids=tuple(str(value) for value in payload["evaluation_row_ids"]),  # type: ignore[arg-type]
        evaluation_row_identity_hash=str(payload["evaluation_row_identity_hash"]),
        cells=tuple(_prediction_cell(row) for row in payload["cells"]),  # type: ignore[arg-type]
        seal_hash=str(payload["seal_hash"]),
        status=str(payload["status"]),
    )


def _validate_seal_disk_bindings(
    path: Path,
    *,
    development_seals: Sequence[DevelopmentPredictionSeal],
    target_seals: Sequence[TargetPredictionSeal],
) -> None:
    dev_index = sha256_file(path / "tables/development_prediction_index.csv")
    dev_arrays = sha256_file(path / DEVELOPMENT_ARRAY_MEMBER)
    target_index = sha256_file(path / "tables/target_prediction_index.csv")
    target_arrays = sha256_file(path / TARGET_ARRAY_MEMBER)
    if any(
        seal.prediction_index_sha256 != dev_index
        or seal.prediction_arrays_sha256 != dev_arrays
        for seal in development_seals
    ) or any(
        seal.prediction_index_sha256 != target_index
        or seal.prediction_arrays_sha256 != target_arrays
        for seal in target_seals
    ):
        raise ProtocolError("Dense residual prediction seal disk binding drifted.")


def _validate_development_prediction_mechanism(
    config: DenseResidualDiagnosticConfig,
    rows: Sequence[Mapping[str, str]],
    *,
    partitions: object,
    compatibility: object,
    generation_lock: object,
) -> None:
    action_rho = dict(zip(ACTION_IDS, RHO_VALUES, strict=True))
    expected_keys = {
        (outer, query, action, training_seed, generation_seed)
        for outer in CENTERS
        for query in development_queries(outer)
        for action in ACTION_IDS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }
    observed: set[tuple[str, str, str, int, int]] = set()
    for row in rows:
        outer = row["outer_target"]
        query = row["query_center"]
        action = row["action_id"]
        training_seed = int(row["training_seed"])
        generation_seed = int(row["generation_seed"])
        key = (outer, query, action, training_seed, generation_seed)
        if key in observed:
            raise ProtocolError("Dense residual development prediction key duplicated.")
        observed.add(key)
        candidates = legal_sources(outer_target=outer, query_center=query)
        calibrated = {
            source: compatibility.calibrated_energy_by_query[query][source]
            for source in candidates
        }
        weights = residual_soft_weights(
            calibrated,
            rho=action_rho[action],
            temperature=TEMPERATURE,
            max_source_weight=MAX_SOURCE_WEIGHT,
            minimum_effective_sources=MIN_EFFECTIVE_SOURCE_COUNT,
        )
        allocation = build_hamilton_allocation(
            weights.weights,
            total=DEVELOPMENT_TOTAL_PER_CLASS,
            minimum_per_source=1,
        )
        eval_rows = partitions.evaluation_rows_by_center[query]
        expected_shuffle = {
            str(label): derived_composition_seed(
                generation_lock_hash=generation_lock.generation_lock_hash,
                target_center=query,
                training_seed=training_seed,
                generation_seed=generation_seed,
                class_label=label,
            )
            for label in (0, 1)
        }
        if (
            row["phase"] != "development"
            or row["arm_role"] != "development_action"
            or _json_value(row["candidate_sources_json"]) != list(candidates)
            or _json_value(row["calibrated_energy_json"]) != calibrated
            or float(row["requested_rho"]) != action_rho[action]
            or float(row["applied_rho"]) != weights.applied_rho
            or float(row["effective_source_count"])
            != weights.effective_source_count
            or _json_value(row["active_constraints_json"])
            != list(weights.active_constraints)
            or _json_value(row["weights_json"]) != dict(weights.weights)
            or _json_value(row["allocations_json"])
            != dict(allocation.allocations)
            or _json_value(row["shuffle_seed_by_class_json"]) != expected_shuffle
            or _json_value(row["evaluation_row_ids_json"])
            != [item.sample_id for item in eval_rows]
            or row["evaluation_row_identity_hash"]
            != row_identity_hash(tuple(eval_rows))
            or row["classifier_config_hash"] != config.classifier.config_hash
            or row["labels_available_to_fit_or_predict"] != "False"
            or row["seed_selection_performed"] != "False"
        ):
            raise ProtocolError(
                "Dense residual development compatibility-to-routing plan drifted."
            )
        if action == CONTROL_ACTION_ID and (
            any(value != 1.0 / 7.0 for value in weights.weights.values())
            or any(value != 144 for value in allocation.allocations.values())
        ):
            raise ProtocolError(
                "Dense residual development rho0 is not exact seven-source equal union."
            )
    if observed != expected_keys or len(rows) != len(expected_keys):
        raise ProtocolError("Dense residual development prediction geometry drifted.")


def _validate_target_prediction_mechanism(
    config: DenseResidualDiagnosticConfig,
    rows: Sequence[Mapping[str, str]],
    *,
    partitions: object,
    compatibility: object,
    generation_lock: object,
) -> None:
    action_rho = dict(zip(ACTION_IDS, RHO_VALUES, strict=True))
    expected_keys = {
        (target, "selected", action, training_seed, generation_seed)
        for target in CENTERS
        for action in ACTION_IDS
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
    }.union(
        {
            (target, "control", CONTROL_ACTION_ID, training_seed, generation_seed)
            for target in CENTERS
            for training_seed in TRAINING_SEEDS
            for generation_seed in GENERATION_SEEDS
        }
    )
    observed: set[tuple[str, str, str, int, int]] = set()
    for row in rows:
        target = row["outer_target"]
        role = row["arm_role"]
        action = row["action_id"]
        training_seed = int(row["training_seed"])
        generation_seed = int(row["generation_seed"])
        key = (target, role, action, training_seed, generation_seed)
        if key in observed:
            raise ProtocolError("Dense residual target prediction key duplicated.")
        observed.add(key)
        candidates = target_sources(target)
        calibrated = {
            source: compatibility.calibrated_energy_by_query[target][source]
            for source in candidates
        }
        weights = residual_soft_weights(
            calibrated,
            rho=action_rho[action],
            temperature=TEMPERATURE,
            max_source_weight=MAX_SOURCE_WEIGHT,
            minimum_effective_sources=MIN_EFFECTIVE_SOURCE_COUNT,
        )
        allocation = build_hamilton_allocation(
            weights.weights,
            total=TOTAL_PER_CLASS,
            minimum_per_source=1,
        )
        eval_rows = partitions.evaluation_rows_by_center[target]
        expected_shuffle = {
            str(label): derived_composition_seed(
                generation_lock_hash=generation_lock.generation_lock_hash,
                target_center=target,
                training_seed=training_seed,
                generation_seed=generation_seed,
                class_label=label,
            )
            for label in (0, 1)
        }
        if (
            row["phase"] != "target"
            or row["query_center"] != target
            or _json_value(row["candidate_sources_json"]) != list(candidates)
            or _json_value(row["calibrated_energy_json"]) != calibrated
            or float(row["requested_rho"]) != action_rho[action]
            or float(row["applied_rho"]) != weights.applied_rho
            or float(row["effective_source_count"])
            != weights.effective_source_count
            or _json_value(row["active_constraints_json"])
            != list(weights.active_constraints)
            or _json_value(row["weights_json"]) != dict(weights.weights)
            or _json_value(row["allocations_json"])
            != dict(allocation.allocations)
            or _json_value(row["shuffle_seed_by_class_json"]) != expected_shuffle
            or _json_value(row["evaluation_row_ids_json"])
            != [item.sample_id for item in eval_rows]
            or row["evaluation_row_identity_hash"]
            != row_identity_hash(tuple(eval_rows))
            or row["classifier_config_hash"] != config.classifier.config_hash
            or row["labels_available_to_fit_or_predict"] != "False"
            or row["seed_selection_performed"] != "False"
        ):
            raise ProtocolError(
                "Dense residual target compatibility-to-routing plan drifted."
            )
        if action == CONTROL_ACTION_ID and (
            any(value != 0.125 for value in weights.weights.values())
            or any(value != 128 for value in allocation.allocations.values())
        ):
            raise ProtocolError(
                "Dense residual target rho0 is not exact eight-source equal union."
            )
    if observed != expected_keys or len(rows) != len(expected_keys):
        raise ProtocolError("Dense residual target all-action geometry drifted.")
    by_key = {
        (
            row["outer_target"],
            row["arm_role"],
            row["action_id"],
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ): row
        for row in rows
    }
    alias_bound_fields = (
        "prediction_sha256",
        "probability_sha256",
        "composition_hash",
        "calibrated_energy_json",
        "weights_json",
        "allocations_json",
        "shuffle_seed_by_class_json",
    )
    for target in CENTERS:
        for training_seed in TRAINING_SEEDS:
            for generation_seed in GENERATION_SEEDS:
                selected = by_key[
                    (
                        target,
                        "selected",
                        CONTROL_ACTION_ID,
                        training_seed,
                        generation_seed,
                    )
                ]
                control = by_key[
                    (
                        target,
                        "control",
                        CONTROL_ACTION_ID,
                        training_seed,
                        generation_seed,
                    )
                ]
                if any(selected[field] != control[field] for field in alias_bound_fields):
                    raise ProtocolError(
                        "Dense residual target control alias differs from selected rho0."
                    )


def _validate_target_plan_tables(
    plans: Sequence[Mapping[str, str]],
    assignments: Sequence[Mapping[str, str]],
    *,
    target_index: Sequence[Mapping[str, str]],
    compatibility: object,
    generation_lock: object,
) -> None:
    expected_plan_keys = {
        (target, "selected", action)
        for target in CENTERS
        for action in ACTION_IDS
    }.union({(target, "control", CONTROL_ACTION_ID) for target in CENTERS})
    by_plan_key: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in plans:
        key = (row["target_center"], row["arm_role"], row["action_id"])
        if key in by_plan_key:
            raise ProtocolError("Dense residual target weight plan duplicated.")
        by_plan_key[key] = row
        candidates = target_sources(row["target_center"])
        calibrated = {
            source: compatibility.calibrated_energy_by_query[row["target_center"]][source]
            for source in candidates
        }
        rho = dict(zip(ACTION_IDS, RHO_VALUES, strict=True))[row["action_id"]]
        weights = residual_soft_weights(
            calibrated,
            rho=rho,
            temperature=TEMPERATURE,
            max_source_weight=MAX_SOURCE_WEIGHT,
            minimum_effective_sources=MIN_EFFECTIVE_SOURCE_COUNT,
        )
        allocation = build_hamilton_allocation(
            weights.weights,
            total=TOTAL_PER_CLASS,
            minimum_per_source=1,
        )
        if (
            _json_value(row["candidate_sources_json"]) != list(candidates)
            or _json_value(row["calibrated_energy_json"]) != calibrated
            or _json_value(row["weights_json"]) != dict(weights.weights)
            or _json_value(row["allocations_per_class_json"])
            != dict(allocation.allocations)
            or float(row["requested_rho"]) != rho
            or float(row["applied_rho"]) != weights.applied_rho
            or float(row["effective_source_count"])
            != weights.effective_source_count
            or _json_value(row["active_constraints_json"])
            != list(weights.active_constraints)
            or int(row["total_per_class"]) != TOTAL_PER_CLASS
            or row["support_labels_used"] != "False"
            or row["diagnostic_only"] != "True"
        ):
            raise ProtocolError("Dense residual target weight plan drifted.")
    if set(by_plan_key) != expected_plan_keys or len(plans) != len(expected_plan_keys):
        raise ProtocolError("Dense residual target weight-plan geometry drifted.")

    source_plan = {
        (key.source_center, key.training_seed, key.generation_seed): key
        for key in source_generation_plan(generation_lock)
    }
    expected_assignment_keys = {
        (target, role, action, training_seed, generation_seed, label, source)
        for target, role, action in expected_plan_keys
        for training_seed in TRAINING_SEEDS
        for generation_seed in GENERATION_SEEDS
        for label in (0, 1)
        for source in target_sources(target)
    }
    observed_assignment_keys = set()
    for row in assignments:
        key = (
            row["target_center"],
            row["arm_role"],
            row["action_id"],
            int(row["training_seed"]),
            int(row["generation_seed"]),
            int(row["class_label"]),
            row["source_center"],
        )
        if key in observed_assignment_keys:
            raise ProtocolError("Dense residual target assignment duplicated.")
        observed_assignment_keys.add(key)
        target, role, action, training_seed, generation_seed, _, source = key
        plan = by_plan_key[(target, role, action)]
        weights = _json_value(plan["weights_json"])
        allocations = _json_value(plan["allocations_per_class_json"])
        source_key = source_plan[(source, training_seed, generation_seed)]
        if (
            source == target
            or row["source_stream_id"] != source_key.stream_id
            or int(row["prefix_count"]) != int(allocations[source])
            or float(row["source_weight"]) != float(weights[source])
            or row["target_expert_excluded"] != "True"
            or row["seed_selected"] != "False"
        ):
            raise ProtocolError("Dense residual target assignment drifted.")
    if (
        observed_assignment_keys != expected_assignment_keys
        or len(assignments) != len(expected_assignment_keys)
    ):
        raise ProtocolError("Dense residual target assignment geometry drifted.")

    # Every target prediction index row must point to its exact seed-independent
    # plan; this cross-binds plans, assignments, and sealed predictions.
    for row in target_index:
        plan = by_plan_key[
            (row["outer_target"], row["arm_role"], row["action_id"])
        ]
        if (
            _json_value(row["weights_json"]) != _json_value(plan["weights_json"])
            or _json_value(row["allocations_json"])
            != _json_value(plan["allocations_per_class_json"])
            or _json_value(row["calibrated_energy_json"])
            != _json_value(plan["calibrated_energy_json"])
        ):
            raise ProtocolError(
                "Dense residual target prediction index is not bound to its plan."
            )


def _validate_content_index(path: Path) -> None:
    observed = _json(path / "manifests/content_index.json")
    records = []
    for relative in CONTENT_INDEX_MEMBERS:
        member = path / relative
        records.append(
            {
                "relative_path": relative,
                "sha256": sha256_file(member),
                "size_bytes": member.stat().st_size,
            }
        )
    expected: dict[str, object] = {
        "schema_version": "midogpp_dense_residual_content_index_v1",
        "records": records,
    }
    expected["content_hash"] = stable_hash(expected)
    if observed != expected:
        raise ProtocolError("Dense residual content index drifted.")


def _load_validated_generation_lock(config: DenseResidualDiagnosticConfig) -> object:
    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    if generation_config.bank_root.resolve() != config.expert_bank_root.resolve():
        raise ProtocolError("Dense residual bank/GenerationLock roots disagree.")
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    lock = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        lock.generation_lock_hash != config.expected_generation_lock_hash
        or lock.bank_lock_hash != config.expected_bank_lock_hash
    ):
        raise ProtocolError("Dense residual GenerationLock identity drifted.")
    return lock


def _validate_provenance(
    root: Path,
    *,
    config: DenseResidualDiagnosticConfig,
) -> dict[str, Mapping[str, object]]:
    payload = _json(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id")
        != "midogpp.oracle.uniform_b_v2_consumed_validation_dense_residual_router.v1"
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
    ):
        raise ProtocolError("Dense residual provenance header drifted.")
    rows = payload.get("input_artifacts")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ProtocolError("Dense residual provenance inputs are malformed.")
    by_id = {str(row.get("artifact_id", "")): row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(INPUT_ARTIFACT_IDS):
        raise ProtocolError("Dense residual provenance input set drifted.")
    expected_paths = {
        config.expert_bank_artifact_id: config.expert_bank_root,
        config.generation_lock_artifact_id: config.generation_lock_root,
        config.validation_cache_artifact_id: config.validation_cache_root,
        config.validation_manifest_artifact_id: config.validation_manifest_path.parent,
    }
    for artifact_id in INPUT_ARTIFACT_IDS:
        row = by_id[artifact_id]
        if (
            Path(str(row.get("resolved_path", ""))).resolve()
            != expected_paths[artifact_id].resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"Dense residual provenance drifted: {artifact_id}.")
    return {artifact_id: by_id[artifact_id] for artifact_id in INPUT_ARTIFACT_IDS}


def _read_csv_exact(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(columns):
                raise ProtocolError(f"Dense residual CSV columns drifted: {path.name}.")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise ProtocolError(f"Cannot read dense residual CSV: {path}.") from exc


def _json_value(value: object) -> object:
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ProtocolError("Dense residual table JSON field is malformed.") from exc


def _require_csv_rows(
    path: Path,
    columns: Sequence[str],
    expected: Sequence[Mapping[str, object]],
) -> None:
    if _read_csv_exact(path, columns) != _rows_as_strings(expected):
        raise ProtocolError(f"Dense residual reconstructed table drifted: {path.name}.")


def _rows_as_strings(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    return [{str(key): str(value) for key, value in row.items()} for row in rows]


def _require_payload(path: Path, expected: Mapping[str, object]) -> None:
    if _json(path) != dict(expected):
        raise ProtocolError(f"Dense residual JSON payload drifted: {path.name}.")


def _validate_closed_world(path: Path, *, allow_pending: bool) -> None:
    actual = {
        member.relative_to(path).as_posix()
        for member in path.rglob("*")
        if member.is_file()
    }
    expected = set(REQUIRED_FILES)
    if allow_pending:
        expected.remove("reports/validation_report.json")
    if actual != expected:
        raise ProtocolError(
            "Dense residual closed-world member set drifted: "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}."
        )


def _rows_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    import hashlib

    encoded = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read dense residual JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"Dense residual JSON must be an object: {path}.")
    return payload


__all__ = ("validate_dense_residual_router_bundle",)
