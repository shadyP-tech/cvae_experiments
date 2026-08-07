"""Independent closed-world validator for residual top-up diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .artifact_io import assert_closed_world, read_json
from .bundle import REQUIRED_FILES, validate_content_index
from .calibration import calibrate_outer_actions
from .config import (
    ResidualTopupDiagnosticConfig,
    load_residual_topup_config,
)
from .contracts import (
    CENTERS,
    EXPECTED_DEVELOPMENT_TASK_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    EXPECTED_TARGET_TASK_COUNT,
)
from .partitions import (
    SUPPORT_PARTITION_COLUMNS,
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_workspace_provenance,
)
from .prediction_store import read_prediction_store
from .predictions import validate_prediction_store_binding
from .reports import (
    action_library_payload,
    leakage_report_payload,
    phase_completion_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    scoring_summary_payload,
)
from .scoring import (
    development_paired_gains,
    score_prediction_store,
    target_paired_deltas,
    target_probability_ensemble_metrics,
)
from .seals import (
    open_evaluation_labels_after_global_seal,
    validate_global_prediction_seal,
)
from .source_cache import (
    load_source_cache,
    validate_source_cache_lock,
)
from .source_cache_validation import validate_source_cache_contents
from .target_plans import build_plan_surface
from .validation_checks import (
    compare_csv_rows,
    require_json_equal,
    validate_final_state,
    validate_phase_reports,
    validate_runtime_report,
)


def validate_residual_topup_router_bundle(
    root: str | Path,
    *,
    config: ResidualTopupDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    """Reconstruct every scientific surface and claim from sealed bytes."""

    path = Path(root)
    required = set(REQUIRED_FILES)
    if allow_pending:
        required.remove("reports/validation_report.json")
    missing = sorted(member for member in required if not (path / member).is_file())
    if missing:
        raise ProtocolError(f"Residual top-up artifact is incomplete: {missing}.")
    assert_closed_world(path, required_files=REQUIRED_FILES, allow_incomplete=False)

    resolved = load_residual_topup_config(path / "config.resolved.yaml")
    if resolved.contract_hash != config.contract_hash:
        raise ProtocolError("Residual top-up resolved config contract drifted.")
    provenance = validate_workspace_provenance(path, resolved)
    locks = load_validated_locks(resolved)
    frame = load_label_free_validation_frame(resolved)
    partitions = build_partition_surface(
        frame,
        config_contract_hash=resolved.contract_hash,
    )
    require_json_equal(
        path / "manifests/support_partition_lock.json",
        partitions.lock_payload,
        role="support partition lock",
    )
    compare_csv_rows(
        path / "tables/support_partitions.csv",
        partitions.table_rows,
        role="support partition table",
    )

    input_hashes = {
        artifact_id: stable_hash(dict(provenance[artifact_id]))
        for artifact_id in resolved.input_artifact_ids
    }
    require_json_equal(
        path / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            resolved,
            input_artifact_hashes=input_hashes,
            validation_cache_binding_hash=frame.cache_binding_hash,
        ),
        role="protocol manifest",
    )
    require_json_equal(
        path / "manifests/action_library.json",
        action_library_payload(resolved),
        role="action library",
    )

    source_cache = load_source_cache(path)
    source_checks = validate_source_cache_contents(
        source_cache,
        generation_lock=locks.generation,
        partitions=partitions,
    )
    source_lock = validate_source_cache_lock(
        path,
        config=resolved,
        generation_lock=locks.generation,
        frame=frame,
        partitions=partitions,
        source_cache=source_cache,
    )
    source_lock_hash = str(source_lock["source_cache_lock_hash"])

    plans = build_plan_surface(
        resolved,
        source_cache,
        source_cache_lock_hash=source_lock_hash,
        support_partition_lock_hash=partitions.lock_hash,
    )
    require_json_equal(
        path / "manifests/router_plan_lock.json",
        plans.lock_payload,
        role="router plan lock",
    )
    compare_csv_rows(
        path / "tables/action_plans.csv",
        plans.table_rows,
        role="action plan table",
    )
    compare_csv_rows(
        path / "tables/action_assignments.csv",
        plans.assignment_rows,
        role="action assignment table",
    )

    predictions = read_prediction_store(path)
    validate_prediction_store_binding(
        predictions,
        config=resolved,
        generation_lock_hash=locks.generation.generation_lock_hash,
        source_cache=source_cache,
        source_cache_lock_hash=source_lock_hash,
        plans=plans,
        partitions=partitions,
    )
    prediction_seal = validate_global_prediction_seal(
        resolved,
        partitions,
        plans,
        predictions,
        root=path,
    )
    labels_by_sample, label_report = open_evaluation_labels_after_global_seal(
        resolved,
        partitions,
        plans,
        predictions,
        root=path,
    )
    require_json_equal(
        path / "reports/label_access_report.json",
        label_report,
        role="label access report",
    )

    metric_rows = score_prediction_store(
        predictions,
        labels_by_sample_id=labels_by_sample,
    )
    gain_rows = development_paired_gains(metric_rows)
    query_rows, selections, calibration_lock = calibrate_outer_actions(
        gain_rows,
        config_contract_hash=resolved.contract_hash,
        global_prediction_seal_hash=str(prediction_seal["seal_hash"]),
    )
    target_deltas = target_paired_deltas(metric_rows, selections)
    selected_by_target = {
        str(row["outer_target"]): str(row["selected_action_id"])
        for row in selections
    }
    ensemble_rows = target_probability_ensemble_metrics(
        predictions,
        labels_by_sample_id=labels_by_sample,
        selected_action_by_target=selected_by_target,
    )
    summary = scoring_summary_payload(
        metric_rows,
        target_deltas,
        ensemble_rows,
        selections,
    )

    compare_csv_rows(
        path / "tables/all_action_metrics.csv",
        metric_rows,
        role="all-action metric table",
    )
    compare_csv_rows(
        path / "tables/development_paired_gains.csv",
        gain_rows,
        role="development paired-gain table",
    )
    compare_csv_rows(
        path / "tables/query_cluster_gains.csv",
        query_rows,
        role="query-cluster gain table",
    )
    compare_csv_rows(
        path / "tables/diagnostic_selections.csv",
        selections,
        role="diagnostic selection table",
    )
    compare_csv_rows(
        path / "tables/target_paired_deltas.csv",
        target_deltas,
        role="target paired-delta table",
    )
    compare_csv_rows(
        path / "tables/probability_ensemble_metrics.csv",
        ensemble_rows,
        role="probability ensemble table",
    )
    require_json_equal(
        path / "manifests/calibration_lock.json",
        calibration_lock,
        role="calibration lock",
    )

    phase_reports = _expected_phase_reports(
        resolved,
        source_cache=source_cache,
        source_lock_hash=source_lock_hash,
        plans=plans,
        predictions=predictions,
        prediction_seal_hash=str(prediction_seal["seal_hash"]),
        gain_rows=gain_rows,
        query_rows=query_rows,
        selections=selections,
        calibration_lock_hash=str(calibration_lock["calibration_lock_hash"]),
        metric_rows=metric_rows,
        target_deltas=target_deltas,
        ensemble_rows=ensemble_rows,
    )
    validate_phase_reports(path, phase_reports)
    require_json_equal(
        path / "reports/leakage_report.json",
        leakage_report_payload(
            support_partition_lock_hash=partitions.lock_hash,
            source_cache_lock_hash=source_lock_hash,
            router_plan_lock_hash=plans.lock_hash,
            global_prediction_seal_hash=str(prediction_seal["seal_hash"]),
            calibration_lock_hash=str(calibration_lock["calibration_lock_hash"]),
        ),
        role="leakage report",
    )
    require_json_equal(
        path / "reports/publication_decision.json",
        publication_decision_payload(summary),
        role="publication decision",
    )
    runtime = validate_runtime_report(path)
    validate_content_index(path, config_contract_hash=resolved.contract_hash)

    checks: dict[str, object] = {
        "status": "PASS",
        "config_contract_hash": resolved.contract_hash,
        "generation_lock_hash": locks.generation.generation_lock_hash,
        "equal_union_policy_lock_hash": locks.equal_union.policy_lock_hash,
        "support_partition_lock_hash": partitions.lock_hash,
        "source_cache_lock_hash": source_lock_hash,
        "router_plan_lock_hash": plans.lock_hash,
        "global_prediction_seal_hash": str(prediction_seal["seal_hash"]),
        "calibration_lock_hash": str(calibration_lock["calibration_lock_hash"]),
        "source_block_count": int(source_checks["source_block_count"]),
        "plan_count": len(plans.table_rows),
        "prediction_cell_count": len(predictions.index_rows),
        "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
        "metric_row_count": len(metric_rows),
        "development_gain_row_count": len(gain_rows),
        "target_delta_row_count": len(target_deltas),
        "all_source_block_hashes_verified": True,
        "all_actions_sealed_before_label_access": True,
        "target_H_labels_excluded_from_own_selection": True,
        "all_seed_cells_retained": True,
        "content_index_verified": True,
        "closed_world_verified": True,
        "runtime_contract_verified": runtime.get("status") == "PASS",
        "routing_quality_claimed": False,
        "promotion_eligible": False,
    }
    validate_final_state(path, allow_pending=allow_pending)
    if not allow_pending:
        report = read_json(path / "reports/validation_report.json")
        if report.get("checks") != checks:
            raise ProtocolError("Residual top-up validation report checks drifted.")
    return checks


def _expected_phase_reports(
    config: ResidualTopupDiagnosticConfig,
    *,
    source_cache: object,
    source_lock_hash: str,
    plans: object,
    predictions: object,
    prediction_seal_hash: str,
    gain_rows: object,
    query_rows: object,
    selections: object,
    calibration_lock_hash: str,
    metric_rows: object,
    target_deltas: object,
    ensemble_rows: object,
) -> dict[str, Mapping[str, object]]:
    return {
        "phase_01_source_cache_complete.json": phase_completion_payload(
            "phase_01_source_cache_complete",
            config_contract_hash=config.contract_hash,
            bindings={"source_cache_lock_hash": source_lock_hash},
            counts={
                "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
                "source_block_count": EXPECTED_SOURCE_BLOCK_COUNT,
                "compatibility_case_row_count": len(
                    getattr(source_cache, "compatibility_case_rows")
                ),
            },
            labels_opened=False,
        ),
        "phase_02_all_actions_sealed.json": phase_completion_payload(
            "phase_02_all_actions_sealed",
            config_contract_hash=config.contract_hash,
            bindings={
                "router_plan_lock_hash": getattr(plans, "lock_hash"),
                "global_prediction_seal_hash": prediction_seal_hash,
            },
            counts={
                "plan_count": len(getattr(plans, "table_rows")),
                "assignment_count": len(getattr(plans, "assignment_rows")),
                "prediction_task_count": EXPECTED_DEVELOPMENT_TASK_COUNT
                + EXPECTED_TARGET_TASK_COUNT,
                "prediction_cell_count": EXPECTED_PREDICTION_CELL_COUNT,
                "unique_classifier_fit_count": int(
                    getattr(predictions, "unique_classifier_fit_count")
                ),
            },
            labels_opened=False,
        ),
        "phase_03_calibration_complete.json": phase_completion_payload(
            "phase_03_calibration_complete",
            config_contract_hash=config.contract_hash,
            bindings={
                "global_prediction_seal_hash": prediction_seal_hash,
                "calibration_lock_hash": calibration_lock_hash,
            },
            counts={
                "development_gain_row_count": len(gain_rows),
                "query_gain_row_count": len(query_rows),
                "selection_row_count": len(selections),
            },
            labels_opened=True,
        ),
        "phase_04_scoring_complete.json": phase_completion_payload(
            "phase_04_scoring_complete",
            config_contract_hash=config.contract_hash,
            bindings={
                "global_prediction_seal_hash": prediction_seal_hash,
                "calibration_lock_hash": calibration_lock_hash,
            },
            counts={
                "metric_row_count": len(metric_rows),
                "target_delta_row_count": len(target_deltas),
                "ensemble_metric_row_count": len(ensemble_rows),
            },
            labels_opened=True,
        ),
    }


__all__ = ("validate_residual_topup_router_bundle",)
