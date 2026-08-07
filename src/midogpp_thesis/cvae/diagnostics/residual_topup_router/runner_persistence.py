"""Phase-bound artifact writers for the residual top-up runner.

The runner owns scientific ordering and seal/label capabilities.  This module
owns only the durable surfaces emitted at each boundary.  Keeping these
writes together makes the resume contract visible without introducing a
generic experiment framework that could blur protocol phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .artifact_io import atomic_write_json
from .calibration import QUERY_GAIN_COLUMNS, SELECTION_COLUMNS
from .contracts import (
    EXPECTED_DEVELOPMENT_TASK_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    EXPECTED_TARGET_TASK_COUNT,
)
from .partitions import SUPPORT_PARTITION_COLUMNS
from .reports import (
    action_library_payload,
    leakage_report_payload,
    phase_completion_payload,
    protocol_manifest_payload,
    publication_decision_payload,
    runtime_summary_payload,
    run_state_payload,
)
from .resume import persist_or_validate_csv, persist_or_validate_json
from .scoring import (
    DEVELOPMENT_GAIN_COLUMNS,
    ENSEMBLE_METRIC_COLUMNS,
    METRIC_COLUMNS,
    TARGET_DELTA_COLUMNS,
)
from .target_plans import ASSIGNMENT_COLUMNS, PLAN_COLUMNS


_PREDICTION_TASK_COUNT = EXPECTED_DEVELOPMENT_TASK_COUNT + EXPECTED_TARGET_TASK_COUNT


def persist_initial_artifacts(
    root: Path,
    *,
    config: object,
    provenance: Mapping[str, object],
    frame: object,
    partitions: object,
) -> None:
    """Persist or validate label-free manifests and the frozen partition surface."""

    input_hashes = {
        artifact_id: stable_hash(provenance[artifact_id])
        for artifact_id in getattr(config, "input_artifact_ids")
    }
    persist_or_validate_json(
        root / "manifests/protocol_manifest.json",
        protocol_manifest_payload(
            config,
            input_artifact_hashes=input_hashes,
            validation_cache_binding_hash=str(getattr(frame, "cache_binding_hash")),
        ),
    )
    persist_or_validate_json(
        root / "manifests/action_library.json",
        action_library_payload(config),
    )
    persist_or_validate_csv(
        root / "tables/support_partitions.csv",
        getattr(partitions, "table_rows"),
        columns=SUPPORT_PARTITION_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/support_partition_lock.json",
        getattr(partitions, "lock_payload"),
    )


def persist_source_phase_completion(
    root: Path,
    *,
    config: object,
    source_cache: object,
    source_lock_hash: str,
) -> None:
    """Commit the source-cache phase report after its lock has validated."""

    persist_or_validate_json(
        root / "reports/phase_01_source_cache_complete.json",
        phase_completion_payload(
            "phase_01_source_cache_complete",
            config_contract_hash=str(getattr(config, "contract_hash")),
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
    )


def persist_plan_artifacts(root: Path, *, plans: object) -> None:
    """Persist or validate the complete label-free action-plan surface."""

    persist_or_validate_csv(
        root / "tables/action_plans.csv",
        getattr(plans, "table_rows"),
        columns=PLAN_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/action_assignments.csv",
        getattr(plans, "assignment_rows"),
        columns=ASSIGNMENT_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/router_plan_lock.json",
        getattr(plans, "lock_payload"),
    )


def persist_prediction_phase_completion(
    root: Path,
    *,
    config: object,
    plans: object,
    predictions: object,
    seal_hash: str,
) -> None:
    """Commit the global all-action seal phase report after validation."""

    persist_or_validate_json(
        root / "reports/phase_02_all_actions_sealed.json",
        phase_completion_payload(
            "phase_02_all_actions_sealed",
            config_contract_hash=str(getattr(config, "contract_hash")),
            bindings={
                "router_plan_lock_hash": str(getattr(plans, "lock_hash")),
                "global_prediction_seal_hash": seal_hash,
            },
            counts={
                "plan_count": len(getattr(plans, "table_rows")),
                "assignment_count": len(getattr(plans, "assignment_rows")),
                "prediction_task_count": _PREDICTION_TASK_COUNT,
                "prediction_cell_count": EXPECTED_PREDICTION_CELL_COUNT,
                "unique_classifier_fit_count": int(
                    getattr(predictions, "unique_classifier_fit_count")
                ),
            },
            labels_opened=False,
        ),
    )


def persist_label_access_report(root: Path, label_report: Mapping[str, object]) -> None:
    """Persist the seal-bound label-access record before scoring begins."""

    persist_or_validate_json(root / "reports/label_access_report.json", label_report)


def persist_calibration_artifacts(
    root: Path,
    *,
    config: object,
    global_prediction_seal_hash: str,
    metric_rows: Sequence[Mapping[str, object]],
    development_gains: Sequence[Mapping[str, object]],
    query_gains: Sequence[Mapping[str, object]],
    selections: Sequence[Mapping[str, object]],
    calibration_lock: Mapping[str, object],
) -> str:
    """Persist scored rows and the fixed outer-fold calibration lock."""

    persist_or_validate_csv(
        root / "tables/all_action_metrics.csv",
        metric_rows,
        columns=METRIC_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/development_paired_gains.csv",
        development_gains,
        columns=DEVELOPMENT_GAIN_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/query_cluster_gains.csv",
        query_gains,
        columns=QUERY_GAIN_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/diagnostic_selections.csv",
        selections,
        columns=SELECTION_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/calibration_lock.json",
        calibration_lock,
    )
    calibration_lock_hash = str(calibration_lock["calibration_lock_hash"])
    persist_or_validate_json(
        root / "reports/phase_03_calibration_complete.json",
        phase_completion_payload(
            "phase_03_calibration_complete",
            config_contract_hash=str(getattr(config, "contract_hash")),
            bindings={
                "global_prediction_seal_hash": global_prediction_seal_hash,
                "calibration_lock_hash": calibration_lock_hash,
            },
            counts={
                "development_gain_row_count": len(development_gains),
                "query_gain_row_count": len(query_gains),
                "selection_row_count": len(selections),
            },
            labels_opened=True,
        ),
    )
    return calibration_lock_hash


def persist_scoring_artifacts(
    root: Path,
    *,
    config: object,
    global_prediction_seal_hash: str,
    calibration_lock_hash: str,
    metric_rows: Sequence[Mapping[str, object]],
    target_deltas: Sequence[Mapping[str, object]],
    ensemble_metrics: Sequence[Mapping[str, object]],
) -> None:
    """Persist terminal target scoring and its completion report."""

    persist_or_validate_csv(
        root / "tables/target_paired_deltas.csv",
        target_deltas,
        columns=TARGET_DELTA_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/probability_ensemble_metrics.csv",
        ensemble_metrics,
        columns=ENSEMBLE_METRIC_COLUMNS,
    )
    persist_or_validate_json(
        root / "reports/phase_04_scoring_complete.json",
        phase_completion_payload(
            "phase_04_scoring_complete",
            config_contract_hash=str(getattr(config, "contract_hash")),
            bindings={
                "global_prediction_seal_hash": global_prediction_seal_hash,
                "calibration_lock_hash": calibration_lock_hash,
            },
            counts={
                "metric_row_count": len(metric_rows),
                "target_delta_row_count": len(target_deltas),
                "ensemble_metric_row_count": len(ensemble_metrics),
            },
            labels_opened=True,
        ),
    )


def persist_leakage_and_publication_reports(
    root: Path,
    *,
    support_partition_lock_hash: str,
    source_cache_lock_hash: str,
    router_plan_lock_hash: str,
    global_prediction_seal_hash: str,
    calibration_lock_hash: str,
    summary: Mapping[str, object],
) -> None:
    """Persist report surfaces whose values are bound to all terminal locks."""

    persist_or_validate_json(
        root / "reports/leakage_report.json",
        leakage_report_payload(
            support_partition_lock_hash=support_partition_lock_hash,
            source_cache_lock_hash=source_cache_lock_hash,
            router_plan_lock_hash=router_plan_lock_hash,
            global_prediction_seal_hash=global_prediction_seal_hash,
            calibration_lock_hash=calibration_lock_hash,
        ),
    )
    persist_or_validate_json(
        root / "reports/publication_decision.json",
        publication_decision_payload(summary),
    )


def persist_runtime_summary(
    root: Path,
    *,
    preflight: Mapping[str, object],
    source_task_count: int,
    source_block_count: int,
    prediction_task_count: int,
    prediction_cell_count: int,
    unique_classifier_fit_count: int,
) -> None:
    """Write the launch-runtime snapshot (a recomputable derived member)."""

    atomic_write_json(
        root / "reports/runtime_summary.json",
        runtime_summary_payload(
            preflight,
            source_task_count=source_task_count,
            source_block_count=source_block_count,
            prediction_task_count=prediction_task_count,
            prediction_cell_count=prediction_cell_count,
            unique_classifier_fit_count=unique_classifier_fit_count,
        ),
    )


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    """Persist the independent validation result without overwriting drift."""

    persist_or_validate_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_residual_topup_validation_report_v1",
            "status": "PASS",
            "validator": "validate_residual_topup_router_bundle",
            "checks": dict(checks),
        },
    )


def write_run_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
) -> None:
    """Persist mutable crash state; unlike scientific members this advances."""

    atomic_write_json(
        root / "reports/run_state.json",
        run_state_payload(status, phase, error=error),
    )


__all__ = (
    "persist_calibration_artifacts",
    "persist_initial_artifacts",
    "persist_label_access_report",
    "persist_leakage_and_publication_reports",
    "persist_plan_artifacts",
    "persist_prediction_phase_completion",
    "persist_runtime_summary",
    "persist_scoring_artifacts",
    "persist_source_phase_completion",
    "persist_validation_report",
    "write_run_state",
)
