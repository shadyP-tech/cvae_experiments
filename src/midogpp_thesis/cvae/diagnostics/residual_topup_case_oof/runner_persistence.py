"""Durable phase writers for the case-OOF runner."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from ....common.hashing import stable_hash
from .artifact_io import (
    atomic_write_json,
    persist_or_validate_csv,
    persist_or_validate_json,
)
from .contracts import (
    EXPECTED_FROZEN_ACTION_COUNT,
    EXPECTED_PROXY_SCORE_COUNT,
    EXPECTED_SEALED_PREDICTION_CELL_COUNT,
)
from .inference import (
    CONTRAST_INFERENCE_COLUMNS,
    ORACLE_HXE_COLUMNS,
    PRIMARY_CONTRAST_COLUMNS,
)
from .inputs import SUPPORT_PARTITION_COLUMNS
from .partitions import CASE_OOF_FOLD_COLUMNS
from .reports import (
    ACTION_ASSIGNMENT_COLUMNS,
    ACTION_PLAN_COLUMNS,
    PROXY_BALLOT_COLUMNS,
    PROXY_RANK_COLUMNS,
    action_assignment_rows,
    action_plan_rows,
    phase_completion_payload,
    protocol_manifest_payload,
    proxy_ballot_rows,
    proxy_rank_rows,
    run_state_payload,
)
from .scoring import CENTER_ENSEMBLE_METRIC_COLUMNS, CENTER_SEED_METRIC_COLUMNS
from .source_cache import EXPECTED_SOURCE_BLOCK_COUNT, EXPECTED_SOURCE_TASK_COUNT


def persist_initial_surfaces(
    root: Path,
    *,
    config: object,
    provenance: Mapping[str, Mapping[str, object]],
    frame: object,
    pre_gpu_firewall: Mapping[str, object],
    base: object,
    crossfit: object,
) -> None:
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
            pre_gpu_firewall=pre_gpu_firewall,
        ),
    )
    persist_or_validate_csv(
        root / "tables/support_partitions.csv",
        getattr(base, "table_rows"),
        columns=SUPPORT_PARTITION_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/support_partition_lock.json",
        getattr(base, "lock_payload"),
    )
    persist_or_validate_csv(
        root / "tables/crossfit_folds.csv",
        getattr(crossfit, "table_rows"),
        columns=CASE_OOF_FOLD_COLUMNS,
    )
    persist_or_validate_json(
        root / "manifests/crossfit_fold_lock.json",
        getattr(crossfit, "lock_payload"),
    )


def persist_rank_and_plan_surfaces(
    root: Path, *, rank_surface: Mapping[str, object], plan: object
) -> None:
    persist_or_validate_json(
        root / "manifests/action_library.json",
        getattr(plan, "action_library_payload"),
    )
    persist_or_validate_json(
        root / "manifests/router_plan_lock.json", getattr(plan, "lock_payload")
    )
    persist_or_validate_csv(
        root / "tables/proxy_case_ballots.csv",
        proxy_ballot_rows(rank_surface),
        columns=PROXY_BALLOT_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/proxy_rank_actions.csv",
        proxy_rank_rows(rank_surface),
        columns=PROXY_RANK_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/action_plans.csv",
        action_plan_rows(plan),
        columns=ACTION_PLAN_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/action_assignments.csv",
        action_assignment_rows(plan),
        columns=ACTION_ASSIGNMENT_COLUMNS,
    )


def persist_source_phase(
    root: Path,
    *,
    config_contract_hash: str,
    source_cache: object,
    source_cache_lock_hash: str,
) -> None:
    persist_or_validate_json(
        root / "reports/phase_01_source_cache_complete.json",
        phase_completion_payload(
            "phase_01_source_cache_complete",
            config_contract_hash=config_contract_hash,
            bindings={"source_cache_lock_hash": source_cache_lock_hash},
            counts={
                "source_task_count": EXPECTED_SOURCE_TASK_COUNT,
                "source_block_count": EXPECTED_SOURCE_BLOCK_COUNT,
                "compatibility_case_row_count": len(
                    getattr(source_cache, "compatibility_case_rows")
                ),
                "expanded_proxy_score_count": EXPECTED_PROXY_SCORE_COUNT,
            },
            labels_opened=False,
        ),
    )


def persist_prediction_phase(
    root: Path,
    *,
    config_contract_hash: str,
    plan: object,
    predictions: object,
    seal_hash: str,
) -> None:
    persist_or_validate_json(
        root / "reports/phase_02_all_predictions_sealed.json",
        phase_completion_payload(
            "phase_02_all_predictions_sealed",
            config_contract_hash=config_contract_hash,
            bindings={
                "router_plan_lock_hash": str(getattr(plan, "lock_hash")),
                "global_prediction_seal_hash": seal_hash,
            },
            counts={
                "frozen_action_count": EXPECTED_FROZEN_ACTION_COUNT,
                "prediction_cell_count": EXPECTED_SEALED_PREDICTION_CELL_COUNT,
                "unique_classifier_fit_count": int(
                    getattr(predictions, "unique_classifier_fit_count")
                ),
            },
            labels_opened=False,
        ),
    )


def persist_terminal_surfaces(
    root: Path,
    *,
    config_contract_hash: str,
    label_report: Mapping[str, object],
    center_seed_rows: Sequence[Mapping[str, object]],
    ensemble_rows: Sequence[Mapping[str, object]],
    contrast_rows: Sequence[Mapping[str, object]],
    inference_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    leakage_report: Mapping[str, object],
    publication_decision: Mapping[str, object],
    runtime_summary: Mapping[str, object],
    seal_hash: str,
) -> None:
    persist_or_validate_json(root / "reports/label_access_report.json", label_report)
    persist_or_validate_csv(
        root / "tables/center_seed_metrics.csv",
        center_seed_rows,
        columns=CENTER_SEED_METRIC_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/center_ensemble_metrics.csv",
        ensemble_rows,
        columns=CENTER_ENSEMBLE_METRIC_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/primary_contrasts.csv",
        contrast_rows,
        columns=PRIMARY_CONTRAST_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/contrast_inference.csv",
        inference_rows,
        columns=CONTRAST_INFERENCE_COLUMNS,
    )
    persist_or_validate_csv(
        root / "tables/oracle_hxe_diagnostics.csv",
        oracle_rows,
        columns=ORACLE_HXE_COLUMNS,
    )
    persist_or_validate_json(root / "reports/leakage_report.json", leakage_report)
    persist_or_validate_json(
        root / "reports/publication_decision.json", publication_decision
    )
    atomic_write_json(root / "reports/runtime_summary.json", runtime_summary)
    persist_or_validate_json(
        root / "reports/phase_03_terminal_scoring_complete.json",
        phase_completion_payload(
            "phase_03_terminal_scoring_complete",
            config_contract_hash=config_contract_hash,
            bindings={"global_prediction_seal_hash": seal_hash},
            counts={
                "center_seed_metric_count": len(center_seed_rows),
                "center_ensemble_metric_count": len(ensemble_rows),
                "center_contrast_count": len(contrast_rows),
                "contrast_inference_count": len(inference_rows),
                "oracle_target_count": len(oracle_rows),
            },
            labels_opened=True,
        ),
    )


def persist_validation_report(root: Path, checks: Mapping[str, object]) -> None:
    persist_or_validate_json(
        root / "reports/validation_report.json",
        {
            "schema_version": "midogpp_residual_topup_case_oof_validation_report_v1",
            "status": "PASS",
            "validator": "validate_residual_topup_case_oof_bundle",
            "checks": dict(checks),
        },
    )


def write_run_state(
    root: Path, *, status: str, phase: str, error: str | None = None
) -> None:
    atomic_write_json(
        root / "reports/run_state.json",
        run_state_payload(status, phase, error=error),
    )


__all__ = (
    "persist_initial_surfaces",
    "persist_prediction_phase",
    "persist_rank_and_plan_surfaces",
    "persist_source_phase",
    "persist_terminal_surfaces",
    "persist_validation_report",
    "write_run_state",
)
