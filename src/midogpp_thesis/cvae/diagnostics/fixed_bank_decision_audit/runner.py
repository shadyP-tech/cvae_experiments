"""Thin dependency-injected phase orchestrator for the fixed-bank audit."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .artifact_io import read_json
from .bundle import assert_closed_world, write_content_index
from .config import FixedBankDecisionAuditConfig
from .execution_adapter import (
    materialize_development_predictions,
    open_globally_sealed_development_labels,
    run_workstation_preflight,
    validate_global_development_seal,
)
from .feature_production import (
    build_fixed_bank_feature_lock,
    produce_label_free_fixed_bank_features,
)
from .features import build_fixed_bank_dataset
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_metadata_similarity,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .partitions import build_fixed_test_partition_surface
from .persistence import (
    persist_initial_surfaces,
    persist_postseal_audit,
    persist_prelabel_surfaces,
    persist_validation_report,
    write_run_state,
)
from .reports import leakage_report_payload
from .response_production import (
    build_exact_crossfit_lock,
    build_fixed_bank_response_lock,
    build_smooth_descriptive_crossfit_lock,
    produce_fixed_bank_responses,
)
from .source_cache import (
    materialize_source_cache,
    stage_source_cache_for_cpu,
    validate_source_cache_lock,
)


@dataclass(frozen=True)
class FixedBankDecisionAuditRunnerDependencies:
    validate_inputs: Callable[..., object] | None = None
    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    build_partitions: Callable[..., object] | None = None
    persist_initial: Callable[..., None] | None = None
    preflight: Callable[..., object] | None = None
    materialize_source: Callable[..., object] | None = None
    validate_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    load_metadata: Callable[..., object] | None = None
    materialize_development: Callable[..., object] | None = None
    validate_development_seal: Callable[..., object] | None = None
    produce_features: Callable[..., object] | None = None
    build_feature_lock: Callable[..., object] | None = None
    persist_prelabel: Callable[..., None] | None = None
    open_development_labels: Callable[..., object] | None = None
    produce_responses: Callable[..., object] | None = None
    build_response_lock: Callable[..., object] | None = None
    build_dataset: Callable[..., object] | None = None
    run_core: Callable[..., object] | None = None
    build_exact_lock: Callable[..., object] | None = None
    build_smooth_lock: Callable[..., object] | None = None
    persist_postseal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


def run_fixed_bank_decision_audit(
    config: FixedBankDecisionAuditConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: FixedBankDecisionAuditRunnerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or FixedBankDecisionAuditRunnerDependencies()
    _assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)
    assert_closed_world(root, allow_incomplete=True)
    with _exclusive_run_lock(root):
        state_path = root / "reports/run_state.json"
        if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
            assert_closed_world(root, allow_incomplete=False)
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root

        phase = "INITIALIZING"
        _write_state(deps, root, status="RUNNING", phase=phase)
        try:
            _observe(deps, "input_fence")
            (deps.validate_inputs or assert_input_fence)(config)
            _observe(deps, "workspace")
            workspace = (
                deps.validate_workspace or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(
                root, config
            )
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_test_frame)(config)
            firewall_fn = deps.validate_firewall or validate_pre_gpu_firewall
            firewall_value = firewall_fn(config, frame, locks)
            firewall = {**firewall_value, "workspace_binding": workspace}
            partitions = (deps.build_partitions or build_fixed_test_partition_surface)(
                frame,
                config_contract_hash=config.contract_hash,
                support_case_count=config.fixed_support_case_count_per_center,
                split_seed=int(config.protocol["support_split_seed"]),
                namespace=str(config.protocol["support_partition_namespace"]),
            )
            (deps.persist_initial or persist_initial_surfaces)(
                root,
                config=config,
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                partitions=partitions,
            )

            phase = "WORKSTATION_PREFLIGHT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "preflight")
            preflight = (deps.preflight or run_workstation_preflight)(
                root, runtime=config.runtime
            )

            phase = "SOURCE_CACHE_270"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "source_cache")
            canonical_cache = (deps.materialize_source or materialize_source_cache)(
                config, locks.generation, frame, partitions, root=root
            )
            source_lock = (deps.validate_source or validate_source_cache_lock)(
                root,
                config=config,
                generation_lock=locks.generation,
                frame=frame,
                partitions=partitions,
                source_cache=canonical_cache,
            )
            source_lock_hash = str(source_lock["source_cache_lock_hash"])
            cpu_cache = canonical_cache
            staging: dict[str, object] = {
                "attempted": True,
                "used": False,
                "status": "CANONICAL_FALLBACK",
            }
            try:
                cpu_cache = (deps.stage_source or stage_source_cache_for_cpu)(
                    canonical_cache,
                    scratch_root=Path(str(config.runtime["scratch_preference"][0])),
                    canonical_root=root,
                )
            except (OSError, ProtocolError) as exc:
                staging["failure"] = f"{type(exc).__name__}: {exc}"
            else:
                staging.update(
                    {
                        "used": cpu_cache is not canonical_cache,
                        "status": (
                            "STAGED_LOCAL_CPU_CACHE"
                            if cpu_cache is not canonical_cache
                            else "CANONICAL_ALREADY_LOCAL"
                        ),
                    }
                )

            phase = "GLOBAL_LABEL_FREE_PREDICTION_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "development_predictions")
            development = (
                deps.materialize_development or materialize_development_predictions
            )(
                config,
                locks.generation,
                cpu_cache,
                frame,
                partitions,
                source_cache_lock_hash=source_lock_hash,
                root=root,
            )
            (deps.validate_development_seal or validate_global_development_seal)(
                development
            )

            phase = "SEALED_PRELABEL_FIXED_BANK_FEATURE_SURFACE"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "fixed_bank_features")
            metadata = (deps.load_metadata or load_metadata_similarity)(config)
            features = (
                deps.produce_features or produce_label_free_fixed_bank_features
            )(cpu_cache, frame, partitions, metadata, development)
            feature_lock = (
                deps.build_feature_lock or build_fixed_bank_feature_lock
            )(
                features,
                partition_lock_hash=partitions.lock_hash,
                development_prediction_seal_hash=(
                    development.seal.prediction_seal_hash
                ),
            )
            (deps.persist_prelabel or persist_prelabel_surfaces)(
                root,
                config_contract_hash=config.contract_hash,
                source_cache_lock_hash=source_lock_hash,
                development_prediction_seal_hash=(
                    development.seal.prediction_seal_hash
                ),
                feature_surface=features,
                feature_lock=feature_lock,
            )
            persisted_feature_lock = read_json(
                root / "manifests/fixed_bank_feature_lock.json"
            )
            if persisted_feature_lock != feature_lock:
                raise ProtocolError("Persisted feature lock differs before label access.")

            phase = "POSTSEAL_EXACT_AND_DESCRIPTIVE_AUDIT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "test_labels")
            labels = (
                deps.open_development_labels
                or open_globally_sealed_development_labels
            )(config.test_manifest_path, partitions, capability=development)
            _observe(deps, "responses")
            responses = (
                deps.produce_responses or produce_fixed_bank_responses
            )(
                features,
                persisted_feature_lock,
                development,
                labels,
                partitions,
            )
            response_lock = (
                deps.build_response_lock or build_fixed_bank_response_lock
            )(
                responses,
                feature_lock_hash=str(
                    feature_lock["fixed_bank_feature_lock_hash"]
                ),
                prediction_seal_hash=development.seal.prediction_seal_hash,
            )
            dataset = (deps.build_dataset or build_fixed_bank_dataset)(
                features.rows, responses.rows
            )
            _observe(deps, "fixed_bank_core")
            audit = (deps.run_core or _run_core)(
                dataset, include_smooth_descriptive=True
            )
            exact_lock = (deps.build_exact_lock or build_exact_crossfit_lock)(
                audit.exact_crossfit
            )
            if audit.smooth_descriptive_crossfit is None:
                raise ProtocolError("Canonical fixed-bank audit lacks smooth description.")
            smooth_lock = (
                deps.build_smooth_lock or build_smooth_descriptive_crossfit_lock
            )(
                audit.smooth_descriptive_crossfit,
                exact_crossfit_hash=audit.exact_crossfit.result_hash,
            )
            leakage = leakage_report_payload(
                support_partition_lock_hash=partitions.lock_hash,
                development_prediction_seal_hash=(
                    development.seal.prediction_seal_hash
                ),
                feature_lock_hash=str(
                    feature_lock["fixed_bank_feature_lock_hash"]
                ),
                response_lock_hash=str(
                    response_lock["fixed_bank_response_lock_hash"]
                ),
                exact_crossfit_lock_hash=str(
                    exact_lock["exact_crossfit_lock_hash"]
                ),
                smooth_crossfit_lock_hash=str(
                    smooth_lock["smooth_descriptive_crossfit_lock_hash"]
                ),
            )
            runtime_summary = _runtime_summary(
                config=config,
                preflight=preflight,
                staging=staging,
                canonical_cache=canonical_cache,
                development=development,
                features=features,
                responses=responses,
                audit=audit,
            )
            (deps.persist_postseal or persist_postseal_audit)(
                root,
                development_labels=labels,
                responses=responses,
                response_lock=response_lock,
                exact_crossfit_lock=exact_lock,
                smooth_crossfit_lock=smooth_lock,
                audit=audit,
                leakage_report=leakage,
                runtime_summary=runtime_summary,
            )

            phase = "CLOSED_WORLD_VALIDATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            (deps.write_index or write_content_index)(
                root, config_contract_hash=config.contract_hash
            )
            checks = (deps.validate_bundle or _validate_bundle)(
                root, config=config, allow_pending=True
            )
            (deps.persist_validation or persist_validation_report)(root, checks)
            _write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root
        except BaseException as exc:
            _write_state(
                deps,
                root,
                status="FAILED",
                phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _run_core(*args: object, **kwargs: object) -> object:
    from .audit import run_fixed_bank_decision_core

    return run_fixed_bank_decision_core(*args, **kwargs)


def _runtime_summary(
    *,
    config: object,
    preflight: object,
    staging: Mapping[str, object],
    canonical_cache: object,
    development: object,
    features: object,
    responses: object,
    audit: object,
) -> dict[str, object]:
    runtime = getattr(config, "runtime")
    smooth = audit.smooth_descriptive_crossfit
    return {
        "schema_version": "midogpp_stage90_fixed_bank_runtime_summary_v1",
        "workstation_preflight": dict(preflight),
        "source_cache_staging": dict(staging),
        "source_stream_count": len(canonical_cache.source_records),
        "development_prediction_cell_count": len(development.store.cells),
        "feature_row_count": len(features.rows),
        "response_row_count": len(responses.rows),
        "descriptive_seed_row_count": len(responses.descriptive_seed_rows),
        "exact_prediction_row_count": len(audit.exact_crossfit.predictions),
        "exact_fold_audit_row_count": len(audit.exact_crossfit.fold_audits),
        "exact_query_metric_row_count": len(audit.query_metrics),
        "exact_outer_metric_row_count": len(audit.outer_metrics),
        "exact_family_summary_row_count": len(audit.family_summaries),
        "abstention_decision_row_count": len(audit.abstention_decisions),
        "abstention_summary_row_count": len(audit.abstention_summaries),
        "smooth_prediction_row_count": len(smooth.predictions),
        "smooth_fold_audit_row_count": len(smooth.fold_audits),
        "target_task_count": 0,
        "target_action_count": 0,
        "test_labels_opened_after_prediction_and_feature_seals": True,
        "support_labels_opened": False,
        "generation_devices": list(runtime["generation_devices"]),
        "persistent_source_workers": bool(runtime["persistent_source_workers"]),
        "classifier_workers": int(runtime["classifier_workers"]),
        "classifier_threads_per_worker": int(
            runtime["classifier_threads_per_worker"]
        ),
        "scratch_preference": list(runtime["scratch_preference"]),
        "hash_validated_resume": True,
        "gpu_and_cpu_phases_disjoint": True,
        "features_persisted_before_test_label_access": True,
        "smooth_isolated_from_exact_decision": True,
        "terminal_consumed_test_diagnostic_only": True,
        "action_selection_authorized": False,
        "policy_update_authorized": False,
    }


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_fixed_bank_decision_audit_bundle

    return validate_fixed_bank_decision_audit_bundle(root, **kwargs)


def _observe(deps: FixedBankDecisionAuditRunnerDependencies, phase: str) -> None:
    if deps.phase_observer is not None:
        deps.phase_observer(phase)


def _write_state(
    deps: FixedBankDecisionAuditRunnerDependencies,
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
) -> None:
    (deps.write_state or write_run_state)(
        root, status=status, phase=phase, error=error
    )


@contextmanager
def _exclusive_run_lock(root: Path):
    path = root / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError("Fixed-bank audit is already running.") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _assert_workspace_resolved_paths(
    config: FixedBankDecisionAuditConfig, *, root: Path
) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": config.artifact_root,
        "expert-bank root": config.expert_bank_root,
        "generation-lock root": config.generation_lock_root,
        "test-cache root": config.test_cache_root,
        "test manifest": config.test_manifest_path,
        "test-consumption ledger": config.test_consumption_ledger_path,
        "ledger amendment": config.ledger_amendment_path,
        "metadata-profile root": config.metadata_profile_root,
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved or root.resolve() != config.artifact_root.resolve():
        raise ProtocolError(
            "Fixed-bank audit requires workspace-resolved paths; "
            f"unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Fixed-bank launch files are absent: {missing}.")


__all__ = (
    "FixedBankDecisionAuditRunnerDependencies",
    "run_fixed_bank_decision_audit",
)
