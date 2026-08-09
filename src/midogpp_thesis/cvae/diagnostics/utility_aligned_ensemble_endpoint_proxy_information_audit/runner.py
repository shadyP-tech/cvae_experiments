"""Thin phase orchestrator for the independent proxy-information audit."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .artifact_io import read_json
from .bundle import assert_closed_world, write_content_index
from .config import ProxyInformationAuditConfig
from .execution_adapter import (
    materialize_development_predictions,
    open_globally_sealed_development_labels,
    produce_label_free_seed_features,
    run_workstation_preflight,
    score_development_ensemble_endpoints,
)
from .feature_production import (
    build_proxy_feature_lock,
    produce_label_free_proxy_feature_payloads,
)
from .inputs import (
    load_label_free_validation_frame,
    load_metadata_similarity,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .partitions import build_fixed_partition_surface
from .persistence import (
    persist_initial_surfaces,
    persist_postseal_audit,
    persist_prelabel_surfaces,
    persist_validation_report,
    write_run_state,
)
from .reports import leakage_report_payload
from .source_cache import (
    materialize_source_cache,
    stage_source_cache_for_cpu,
    validate_source_cache_lock,
)


@dataclass(frozen=True)
class ProxyAuditRunnerDependencies:
    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    build_partitions: Callable[..., object] | None = None
    preflight: Callable[..., object] | None = None
    materialize_source: Callable[..., object] | None = None
    validate_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    load_metadata: Callable[..., object] | None = None
    produce_seed_features: Callable[..., object] | None = None
    materialize_development: Callable[..., object] | None = None
    produce_proxy_features: Callable[..., object] | None = None
    build_proxy_lock: Callable[..., object] | None = None
    open_development_labels: Callable[..., object] | None = None
    score_development: Callable[..., object] | None = None
    run_audit: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_initial: Callable[..., None] | None = None
    persist_prelabel: Callable[..., None] | None = None
    persist_postseal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


def run_utility_aligned_ensemble_endpoint_proxy_information_audit(
    config: ProxyInformationAuditConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: ProxyAuditRunnerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or ProxyAuditRunnerDependencies()
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
            _observe(deps, "workspace")
            workspace = (
                deps.validate_workspace
                or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(
                root, config
            )
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_validation_frame)(config)
            firewall = {
                **(deps.validate_firewall or validate_pre_gpu_firewall)(config, frame),
                "workspace_binding": workspace,
            }
            partitions = (deps.build_partitions or build_fixed_partition_surface)(
                frame, config_contract_hash=config.contract_hash
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
                    scratch_root=Path("/data/local"),
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

            phase = "SEALED_PRELABEL_PROXY_SURFACE"
            _write_state(deps, root, status="RUNNING", phase=phase)
            metadata = (deps.load_metadata or load_metadata_similarity)(config)
            seed_features = (
                deps.produce_seed_features or produce_label_free_seed_features
            )(cpu_cache, frame, partitions, metadata)
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
            _observe(deps, "proxy_features")
            proxy_payloads = (
                deps.produce_proxy_features or produce_label_free_proxy_feature_payloads
            )(seed_features, development, partitions)
            proxy_lock = (deps.build_proxy_lock or build_proxy_feature_lock)(
                proxy_payloads,
                partition_lock_hash=partitions.lock_hash,
                development_prediction_seal_hash=development.seal.prediction_seal_hash,
            )
            (deps.persist_prelabel or persist_prelabel_surfaces)(
                root,
                config_contract_hash=config.contract_hash,
                source_cache_lock_hash=source_lock_hash,
                development_prediction_seal_hash=development.seal.prediction_seal_hash,
                proxy_rows=proxy_payloads,
                proxy_feature_lock=proxy_lock,
            )

            phase = "POSTSEAL_ENDPOINT_SCORING_AND_PROXY_AUDIT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "development_labels")
            labels = (
                deps.open_development_labels
                or open_globally_sealed_development_labels
            )(config.validation_manifest_path, partitions, capability=development)
            utility, seed_rows = (
                deps.score_development or score_development_ensemble_endpoints
            )(development, labels, partitions)
            _observe(deps, "proxy_information_audit")
            audit = (deps.run_audit or _run_audit_core)(
                proxy_payloads, utility.rows, ridge_alpha=float(config.model["ridge_alpha"])
            )
            leakage = leakage_report_payload(
                support_partition_lock_hash=partitions.lock_hash,
                development_prediction_seal_hash=development.seal.prediction_seal_hash,
                proxy_feature_lock_hash=str(proxy_lock["proxy_feature_lock_hash"]),
                crossfit_fold_lock_hash=str(
                    audit.fold_lock["crossfit_fold_lock_hash"]
                ),
            )
            runtime_summary = {
                "schema_version": "midogpp_stage90_proxy_information_runtime_summary_v1",
                "workstation_preflight": dict(preflight),
                "source_cache_staging": staging,
                "source_stream_count": len(canonical_cache.source_records),
                "development_prediction_cell_count": len(development.store.cells),
                "proxy_feature_row_count": len(proxy_payloads),
                "endpoint_response_count": len(utility.rows),
                "descriptive_seed_row_count": len(seed_rows),
                "target_task_count": 0,
                "target_labels_opened": False,
                "generation_devices": ["cuda:0", "cuda:1"],
                "classifier_workers": 4,
                "classifier_threads_per_worker": 3,
                "hash_validated_resume": True,
            }
            (deps.persist_postseal or persist_postseal_audit)(
                root,
                development_labels=labels,
                utility_surface=utility,
                descriptive_seed_rows=seed_rows,
                fold_lock=audit.fold_lock,
                audit_result=audit.result_payload,
                crossfit_rows=audit.crossfit_rows,
                query_rows=audit.query_metric_rows,
                outer_rows=audit.outer_metric_rows,
                family_rows=audit.family_summary_rows,
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


def _run_audit_core(*args: object, **kwargs: object) -> object:
    from .audit_adapter import run_persistable_proxy_information_audit

    return run_persistable_proxy_information_audit(*args, **kwargs)


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_proxy_information_audit_bundle

    return validate_proxy_information_audit_bundle(root, **kwargs)


def _observe(deps: ProxyAuditRunnerDependencies, phase: str) -> None:
    if deps.phase_observer is not None:
        deps.phase_observer(phase)


def _write_state(
    deps: ProxyAuditRunnerDependencies,
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
        raise ProtocolError("Proxy-information audit is already running.") from exc
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
    config: ProxyInformationAuditConfig, *, root: Path
) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": config.artifact_root,
        "expert-bank root": config.expert_bank_root,
        "generation-lock root": config.generation_lock_root,
        "validation-cache root": config.validation_cache_root,
        "validation manifest": config.validation_manifest_path,
        "metadata-profile root": config.metadata_profile_root,
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved or root.resolve() != config.artifact_root.resolve():
        raise ProtocolError(
            f"Proxy-information audit requires workspace-resolved paths; unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Proxy-information launch files are absent: {missing}.")


__all__ = (
    "ProxyAuditRunnerDependencies",
    "run_utility_aligned_ensemble_endpoint_proxy_information_audit",
)
