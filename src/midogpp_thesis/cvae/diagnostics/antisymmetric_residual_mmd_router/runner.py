"""Run the modular, resumable antisymmetric residual-MMD diagnostic."""

from __future__ import annotations

from pathlib import Path
import time

from ...protocol import ProtocolError
from .artifact_io import (
    assert_closed_world,
    atomic_write_csv_rows,
    atomic_write_json,
    exclusive_run_lock,
    prune_stale_temp_files,
    read_json,
)
from .bundle import REQUIRED_FILES
from .config import AntisymmetricResidualMMDDiagnosticConfig
from .partitions import CROSSFIT_FOLD_COLUMNS, build_case_crossfit_surface
from .planning import build_antisymmetric_router_plans
from .prediction import materialize_case_crossfit_predictions
from .reports import (
    _leakage_report,
    _phase_payload,
    _protocol_manifest,
    _publication_decision,
    _runtime_summary,
    _write_content_index,
    _write_state,
)
from .runtime_preflight import run_workstation_preflight
from .scoring import (
    PAIRED_DELTA_COLUMNS,
    TARGET_METRIC_COLUMNS,
    score_case_crossfit_predictions,
)
from .seals import (
    build_global_crossfit_prediction_seal,
    open_crossfit_evaluation_labels,
)
from ..mmd_kmm_router.inputs import (
    SUPPORT_PARTITION_COLUMNS,
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_workspace_provenance,
)
from ..mmd_kmm_router.source_products import (
    materialize_source_products,
    validate_source_products_lock,
)


def run_antisymmetric_residual_mmd_router_diagnostic(
    config: AntisymmetricResidualMMDDiagnosticConfig,
    *,
    artifact_root: str | Path | None = None,
) -> Path:
    """Execute all phases and independently validate the closed-world bundle."""

    _assert_resolved_paths(config)
    root = Path(artifact_root or config.artifact_root)
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
        started = time.monotonic()
        phase = "INITIALIZING"
        _write_state(root, status="RUNNING", phase=phase)
        try:
            provenance = validate_workspace_provenance(root, config)
            locks = load_validated_locks(config)
            workstation_preflight = run_workstation_preflight(
                root,
                runtime=config.runtime,
            )
            frame = load_label_free_validation_frame(config)
            base_partitions = build_partition_surface(
                frame,
                config_contract_hash=config.contract_hash,
            )
            crossfit = build_case_crossfit_surface(
                base_partitions,
                config_contract_hash=config.contract_hash,
            )
            atomic_write_json(
                root / "manifests/protocol_manifest.json",
                _protocol_manifest(
                    config,
                    provenance=provenance,
                    validation_cache_binding_hash=frame.cache_binding_hash,
                    support_partition_lock_hash=base_partitions.lock_hash,
                    crossfit_surface_lock_hash=crossfit.lock_hash,
                ),
            )
            atomic_write_json(
                root / "manifests/support_partition_lock.json",
                base_partitions.lock_payload,
            )
            atomic_write_csv_rows(
                root / "tables/support_partitions.csv",
                base_partitions.table_rows,
                columns=SUPPORT_PARTITION_COLUMNS,
            )
            atomic_write_json(
                root / "manifests/crossfit_surface_lock.json",
                crossfit.lock_payload,
            )
            atomic_write_csv_rows(
                root / "tables/crossfit_folds.csv",
                crossfit.table_rows,
                columns=CROSSFIT_FOLD_COLUMNS,
            )

            phase = "SOURCE_PRODUCTS"
            _write_state(root, status="RUNNING", phase=phase)
            source_products = materialize_source_products(
                config,
                locks.generation,
                frame,
                base_partitions,
                root=root,
            )
            source_lock = validate_source_products_lock(
                root,
                config=config,
                generation_lock=locks.generation,
                frame=frame,
                partitions=base_partitions,
                source_products=source_products,
            )
            atomic_write_json(
                root / "reports/phase_01_source_products_complete.json",
                _phase_payload(
                    "PHASE_01_SOURCE_PRODUCTS_COMPLETE",
                    source_products_hash=source_products.source_products_hash,
                    source_block_count=len(source_products.index_rows),
                    compatibility_score_count=len(
                        source_products.compatibility_score_rows
                    ),
                    compatibility_scores_used_by_router=False,
                    manifest_labels_opened=False,
                ),
            )

            phase = "ROUTER_PLANS"
            _write_state(root, status="RUNNING", phase=phase)
            plans = build_antisymmetric_router_plans(
                config,
                source_products,
                frame,
                base_partitions,
                crossfit,
                source_products_lock_hash=str(
                    source_lock["source_products_lock_hash"]
                ),
                root=root,
            )
            atomic_write_json(
                root / "reports/phase_02_router_plans_complete.json",
                _phase_payload(
                    "PHASE_02_CASE_CROSSFIT_ROUTER_PLANS_COMPLETE",
                    router_plan_lock_hash=plans.lock_hash,
                    target_workspace_count=9,
                    crossfit_fold_count=len(plans.plans_by_fold),
                    nonuniform_plan_count=sum(
                        not bool(plan["used_uniform_fallback"])
                        for plan in plans.plans_by_fold.values()
                    ),
                    target_labels_used=False,
                    heldout_case_embeddings_used_for_own_route=False,
                    cohort_embeddings_used_for_other_case_routes=True,
                ),
            )

            phase = "TARGET_PREDICTIONS"
            _write_state(root, status="RUNNING", phase=phase)
            predictions = materialize_case_crossfit_predictions(
                config,
                locks.generation.generation_lock_hash,
                source_products,
                plans,
                frame,
                crossfit,
                source_products_lock_hash=str(
                    source_lock["source_products_lock_hash"]
                ),
                root=root,
            )
            seal = build_global_crossfit_prediction_seal(
                config,
                crossfit,
                plans,
                predictions,
                root=root,
            )
            atomic_write_json(
                root / "reports/phase_03_predictions_sealed.json",
                _phase_payload(
                    "PHASE_03_ALL_CASE_CROSSFIT_PREDICTIONS_SEALED",
                    global_prediction_seal_hash=seal["seal_hash"],
                    prediction_cell_count=len(predictions.index_rows),
                    unique_classifier_fit_count=predictions.unique_classifier_fit_count,
                    evaluation_labels_opened=False,
                ),
            )

            phase = "SCORING"
            _write_state(root, status="RUNNING", phase=phase)
            labels_by_sample, label_report = open_crossfit_evaluation_labels(
                config,
                crossfit,
                root=root,
            )
            metrics, deltas, scoring = score_case_crossfit_predictions(
                predictions,
                crossfit,
                labels_by_sample_id=labels_by_sample,
            )
            atomic_write_csv_rows(
                root / "tables/target_metrics.csv",
                metrics,
                columns=TARGET_METRIC_COLUMNS,
            )
            atomic_write_csv_rows(
                root / "tables/paired_deltas.csv",
                deltas,
                columns=PAIRED_DELTA_COLUMNS,
            )
            atomic_write_json(root / "reports/label_access_report.json", label_report)
            atomic_write_json(root / "reports/phase_04_scoring_complete.json", scoring)
            atomic_write_json(root / "reports/leakage_report.json", _leakage_report())
            atomic_write_json(
                root / "reports/publication_decision.json",
                _publication_decision(scoring, plans=plans.plans_by_fold),
            )
            atomic_write_json(
                root / "reports/runtime_summary.json",
                _runtime_summary(
                    config,
                    elapsed_seconds=time.monotonic() - started,
                    unique_classifier_fit_count=predictions.unique_classifier_fit_count,
                    workstation_preflight=workstation_preflight,
                ),
            )

            phase = "VALIDATING"
            _write_state(root, status="RUNNING", phase=phase)
            _write_content_index(root)
            checks = _validate_bundle(root, config=config, allow_pending=True)
            atomic_write_json(
                root / "reports/validation_report.json",
                {
                    "schema_version": (
                        "midogpp_antisymmetric_residual_mmd_validation_report_v1"
                    ),
                    "status": "PASS",
                    "validator": (
                        "validate_antisymmetric_residual_mmd_router_bundle"
                    ),
                    "checks": checks,
                },
            )
            _write_state(root, status="COMPLETE", phase="COMPLETE")
            _validate_bundle(root, config=config)
            return root
        except BaseException as exc:
            _write_state(
                root,
                status="FAILED",
                phase=phase,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise


def _validate_bundle(
    root: Path,
    *,
    config: AntisymmetricResidualMMDDiagnosticConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    from .validation import validate_antisymmetric_residual_mmd_router_bundle

    return validate_antisymmetric_residual_mmd_router_bundle(
        root,
        config=config,
        allow_pending=allow_pending,
    )


def _assert_resolved_paths(
    config: AntisymmetricResidualMMDDiagnosticConfig,
) -> None:
    paths = (
        config.artifact_root,
        config.expert_bank_root,
        config.generation_lock_root,
        config.equal_union_policy_root,
        config.validation_cache_root,
        config.validation_manifest_path,
    )
    if any(not path.is_absolute() for path in paths):
        raise ProtocolError(
            "Antisymmetric execution requires workspace-resolved paths; use "
            "`python -m midogpp_thesis workspace run`."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        relative
        for relative in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / relative).is_file()
    ]
    if missing:
        raise ProtocolError(
            f"Antisymmetric workspace launch files are missing: {missing}."
        )


__all__ = ("run_antisymmetric_residual_mmd_router_diagnostic",)
