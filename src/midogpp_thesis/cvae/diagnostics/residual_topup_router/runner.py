"""Orchestrate the sealed residual top-up Stage-90 diagnostic.

The scientific work is deliberately delegated to small phase modules.  This
runner owns only ordering, durable phase boundaries, crash state, and final
independent validation.  In particular, the label-bearing validation manifest
is not handed to a callable until every development and target prediction has
been materialized and covered by the global prediction seal.
"""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .artifact_io import (
    assert_closed_world,
    exclusive_run_lock,
    prune_stale_temp_files,
    read_json,
)
from .bundle import REQUIRED_FILES, write_content_index
from .calibration import calibrate_outer_actions
from .config import ResidualTopupDiagnosticConfig
from .contracts import (
    CENTERS,
    EXPECTED_DEVELOPMENT_TASK_COUNT,
    EXPECTED_PREDICTION_CELL_COUNT,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    EXPECTED_TARGET_TASK_COUNT,
)
from .partitions import (
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_workspace_provenance,
)
from .predictions import (
    materialize_all_action_predictions,
    validate_prediction_store_binding,
)
from .reports import scoring_summary_payload
from .runtime_preflight import run_workstation_preflight
from .scoring import (
    development_paired_gains,
    score_prediction_store,
    target_paired_deltas,
    target_probability_ensemble_metrics,
)
from .seals import (
    GLOBAL_PREDICTION_SEAL_MEMBER,
    build_global_prediction_seal,
    open_evaluation_labels_after_global_seal,
    validate_global_prediction_seal,
)
from .source_cache import (
    materialize_source_cache,
    validate_source_cache_lock,
)
from .target_plans import build_plan_surface
from .runner_persistence import (
    persist_calibration_artifacts,
    persist_initial_artifacts,
    persist_label_access_report,
    persist_leakage_and_publication_reports,
    persist_plan_artifacts,
    persist_prediction_phase_completion,
    persist_runtime_summary,
    persist_scoring_artifacts,
    persist_source_phase_completion,
    persist_validation_report,
    write_run_state,
)


_PREDICTION_TASK_COUNT = EXPECTED_DEVELOPMENT_TASK_COUNT + EXPECTED_TARGET_TASK_COUNT


def run_residual_topup_router_diagnostic(
    config: ResidualTopupDiagnosticConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Run the terminal consumed-validation diagnostic and validate its bundle."""

    root = Path(artifact_root or config.artifact_root)
    _assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)
    assert_closed_world(root, required_files=REQUIRED_FILES, allow_incomplete=True)

    with exclusive_run_lock(root):
        prune_stale_temp_files(root)
        state_path = root / "reports/run_state.json"
        if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
            _validate_bundle(root, config=config)
            return root

        phase = "INITIALIZING"
        _write_state(root, status="RUNNING", phase=phase)
        try:
            provenance = validate_workspace_provenance(root, config)
            locks = load_validated_locks(config)
            preflight = run_workstation_preflight(root, runtime=config.runtime)

            # Label-free identities and whole-case support partitions are the
            # only validation data available before the global prediction seal.
            frame = load_label_free_validation_frame(config)
            partitions = build_partition_surface(
                frame,
                config_contract_hash=config.contract_hash,
            )
            persist_initial_artifacts(
                root,
                config=config,
                provenance=provenance,
                frame=frame,
                partitions=partitions,
            )

            phase = "SOURCE_CACHE"
            _write_state(root, status="RUNNING", phase=phase)
            source_cache = materialize_source_cache(
                config,
                locks.generation,
                frame,
                partitions,
                root=root,
            )
            source_lock = validate_source_cache_lock(
                root,
                config=config,
                generation_lock=locks.generation,
                frame=frame,
                partitions=partitions,
                source_cache=source_cache,
            )
            source_lock_hash = str(source_lock["source_cache_lock_hash"])
            persist_source_phase_completion(
                root,
                config=config,
                source_cache=source_cache,
                source_lock_hash=source_lock_hash,
            )

            phase = "ALL_ACTION_PREDICTIONS"
            _write_state(root, status="RUNNING", phase=phase)
            plans = build_plan_surface(
                config,
                source_cache,
                source_cache_lock_hash=source_lock_hash,
                support_partition_lock_hash=partitions.lock_hash,
            )
            persist_plan_artifacts(root, plans=plans)
            predictions = materialize_all_action_predictions(
                config,
                locks.generation.generation_lock_hash,
                source_cache,
                plans,
                frame,
                partitions,
                source_cache_lock_hash=source_lock_hash,
                root=root,
            )
            validate_prediction_store_binding(
                predictions,
                config=config,
                generation_lock_hash=locks.generation.generation_lock_hash,
                source_cache=source_cache,
                source_cache_lock_hash=source_lock_hash,
                plans=plans,
                partitions=partitions,
            )
            if (root / GLOBAL_PREDICTION_SEAL_MEMBER).is_file():
                seal = validate_global_prediction_seal(
                    config,
                    partitions,
                    plans,
                    predictions,
                    root=root,
                )
            else:
                seal = build_global_prediction_seal(
                    config,
                    partitions,
                    plans,
                    predictions,
                    root=root,
                )
                validate_global_prediction_seal(
                    config,
                    partitions,
                    plans,
                    predictions,
                    root=root,
                )
            seal_hash = str(seal["seal_hash"])
            persist_prediction_phase_completion(
                root,
                config=config,
                plans=plans,
                predictions=predictions,
                seal_hash=seal_hash,
            )

            # First and only label-capable boundary.  The capability itself
            # revalidates the durable seal and streams evaluation rows only.
            phase = "POST_SEAL_CALIBRATION"
            _write_state(root, status="RUNNING", phase=phase)
            labels_by_sample, label_report = open_evaluation_labels_after_global_seal(
                config,
                partitions,
                plans,
                predictions,
                root=root,
            )
            # Persist the seal-bound access state before any calibration or
            # scoring work so an interruption cannot erase the access record.
            persist_label_access_report(root, label_report)
            metric_rows = score_prediction_store(
                predictions,
                labels_by_sample_id=labels_by_sample,
            )
            development_gains = development_paired_gains(metric_rows)
            query_gains, selections, calibration_lock = calibrate_outer_actions(
                development_gains,
                config_contract_hash=config.contract_hash,
                global_prediction_seal_hash=seal_hash,
            )
            calibration_lock_hash = persist_calibration_artifacts(
                root,
                config=config,
                global_prediction_seal_hash=seal_hash,
                metric_rows=metric_rows,
                development_gains=development_gains,
                query_gains=query_gains,
                selections=selections,
                calibration_lock=calibration_lock,
            )

            phase = "TERMINAL_SCORING"
            _write_state(root, status="RUNNING", phase=phase)
            selected_by_target = {
                str(row["outer_target"]): str(row["selected_action_id"])
                for row in selections
            }
            if set(selected_by_target) != set(CENTERS):
                raise ProtocolError("Residual top-up selection coverage drifted.")
            target_deltas = target_paired_deltas(metric_rows, selections)
            ensemble_metrics = target_probability_ensemble_metrics(
                predictions,
                labels_by_sample_id=labels_by_sample,
                selected_action_by_target=selected_by_target,
            )
            summary = scoring_summary_payload(
                metric_rows,
                target_deltas,
                ensemble_metrics,
                selections,
            )
            persist_scoring_artifacts(
                root,
                config=config,
                global_prediction_seal_hash=seal_hash,
                calibration_lock_hash=calibration_lock_hash,
                metric_rows=metric_rows,
                target_deltas=target_deltas,
                ensemble_metrics=ensemble_metrics,
            )
            persist_leakage_and_publication_reports(
                root,
                support_partition_lock_hash=partitions.lock_hash,
                source_cache_lock_hash=source_lock_hash,
                router_plan_lock_hash=plans.lock_hash,
                global_prediction_seal_hash=seal_hash,
                calibration_lock_hash=calibration_lock_hash,
                summary=summary,
            )
            persist_runtime_summary(
                root,
                preflight=preflight,
                source_task_count=EXPECTED_SOURCE_TASK_COUNT,
                source_block_count=EXPECTED_SOURCE_BLOCK_COUNT,
                prediction_task_count=_PREDICTION_TASK_COUNT,
                prediction_cell_count=EXPECTED_PREDICTION_CELL_COUNT,
                unique_classifier_fit_count=(
                    predictions.unique_classifier_fit_count
                ),
            )

            phase = "VALIDATING"
            _write_state(root, status="RUNNING", phase=phase)
            write_content_index(root, config_contract_hash=config.contract_hash)
            checks = _validate_bundle(
                root,
                config=config,
                allow_pending=True,
            )
            persist_validation_report(root, checks)
            _write_state(root, status="COMPLETE", phase="COMPLETE")
            _validate_bundle(root, config=config)
            return root
        except BaseException as exc:
            _write_state(
                root,
                status="FAILED",
                phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _validate_bundle(
    root: Path,
    *,
    config: ResidualTopupDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    from .validation import validate_residual_topup_router_bundle

    return validate_residual_topup_router_bundle(
        root,
        config=config,
        allow_pending=allow_pending,
    )


def _assert_workspace_resolved_paths(
    config: ResidualTopupDiagnosticConfig,
    *,
    root: Path,
) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": config.artifact_root,
        "expert-bank root": config.expert_bank_root,
        "GenerationLock root": config.generation_lock_root,
        "equal-union policy root": config.equal_union_policy_root,
        "validation-cache root": config.validation_cache_root,
        "validation manifest": config.validation_manifest_path,
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved:
        raise ProtocolError(
            "Residual top-up execution requires workspace-resolved paths; run the "
            "registered experiment with `python -m midogpp_thesis workspace run`. "
            f"Unresolved paths: {unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        relative
        for relative in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / relative).is_file()
    ]
    if missing:
        raise ProtocolError(
            f"Residual top-up workspace launch files are missing: {missing}."
        )


def _write_state(
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
) -> None:
    write_run_state(root, status=status, phase=phase, error=error)


__all__ = ("run_residual_topup_router_diagnostic",)
