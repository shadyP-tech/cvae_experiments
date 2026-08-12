"""Phase-ordered runner for the consumed-test endpoint-router diagnostic."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .artifact_io import atomic_json, read_json
from .bundle import assert_closed_world, write_content_index
from .config import ConsumedTestEndpointRouterConfig
from .inputs import (
    load_label_free_test_frame, load_metadata_compatibility, load_validated_locks,
    validate_active_diagnostic_workspace_binding, validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .label_capabilities import admit_manifest_without_labels
from .partitions import build_consumed_test_partitions
from .persistence import (
    persist_development_surfaces, persist_initial_surfaces,
    persist_model_and_plan_surfaces, persist_terminal_surfaces,
    persist_validation_report,
)
from .prediction_planning import (
    build_development_prediction_plan, build_target_prediction_plan,
    cleanup_staged_target_embeddings, stage_target_embeddings,
)
from .protocol import (
    assert_consumed_test_diagnostic_only, canonical_consumed_test_protocol,
)
from .reports import run_state_payload
from .run_lock import exclusive_run_lock as _exclusive_run_lock
from .runner_dependencies import (
    ConsumedTestEndpointRouterDependencies,
    ConsumedTestEndpointRouterRunnerDependencies,
)
from .runtime_preflight import run_endpoint_router_workstation_preflight
from .source_cache import (
    cleanup_staged_source_cache, enter_cuda_free_cpu_phase,
    materialize_source_cache, stage_source_cache_for_cpu,
)
from .development_runtime import materialize_development_predictions
from .feature_execution import (
    cleanup_feature_runtime_checkpoints, materialize_label_free_seed_features,
    materialize_label_free_support_shifts,
)
from .runner_science import run_prelabel_science, run_terminal_science
from .target_runtime import materialize_target_predictions


def run_utility_aligned_consumed_test_endpoint_router(
    config: ConsumedTestEndpointRouterConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: ConsumedTestEndpointRouterRunnerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or ConsumedTestEndpointRouterRunnerDependencies()
    _assert_workspace_resolved_paths(config, root=root)
    _assert_launch_files(root)
    for member in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / member).mkdir(parents=True, exist_ok=True)
    assert_closed_world(root, allow_incomplete=True)
    with _exclusive_run_lock(root):
        state_path = root / "reports/run_state.json"
        state = read_json(state_path) if state_path.is_file() else {}
        if state.get("status") == "COMPLETE":
            assert_closed_world(root, allow_incomplete=False)
            _validate_bundle(root, config=config)
            return root
        if (
            state.get("status") in {"RUNNING", "FAILED"}
            and state.get("phase") == "CLOSED_WORLD_VALIDATION"
        ):
            protocol = canonical_consumed_test_protocol()
            assert_consumed_test_diagnostic_only(protocol)
            validate_active_diagnostic_workspace_binding(config)
            validate_workspace_provenance(root, config)
            return _finalize_bundle(root, config=config, protocol=protocol)
        phase = "INITIALIZING"
        staged = None
        target_embeddings = None
        _write_state(root, status="RUNNING", phase=phase)
        try:
            protocol = canonical_consumed_test_protocol()
            assert_consumed_test_diagnostic_only(protocol)
            if deps.phase_observer is not None:
                deps.phase_observer("workspace_and_inputs")
            workspace = validate_active_diagnostic_workspace_binding(config)
            provenance = validate_workspace_provenance(root, config)
            locks = load_validated_locks(config)
            frame = load_label_free_test_frame(config)
            admission = admit_manifest_without_labels(
                config.test_manifest_path, expected_sha256=config.expected_manifest_sha256
            )
            firewall = dict(validate_pre_gpu_firewall(config, frame))
            firewall.pop("firewall_hash", None)
            firewall["workspace_binding"] = workspace
            firewall["firewall_hash"] = canonical_sha256(firewall)
            partitions = build_consumed_test_partitions(frame.rows)
            metadata = load_metadata_compatibility(
                config, manifest_admission=admission
            )
            persist_initial_surfaces(
                root, config=config, protocol=protocol, provenance=provenance,
                cache_binding_hash=frame.cache_binding_hash,
                manifest_admission_hash=str(admission["manifest_admission_hash"]),
                firewall=firewall, support_partition=partitions,
                action_library=config.action_library,
            )

            phase = "WORKSTATION_PREFLIGHT"
            _phase(deps, root, phase)
            preflight = run_endpoint_router_workstation_preflight(
                root, runtime=config.runtime
            )

            phase = "SOURCE_AND_LABEL_FREE_FEATURES"
            _phase(deps, root, phase)
            source = (deps.materialize_source or materialize_source_cache)(
                config, locks.generation, root=root
            )
            staged = (deps.stage_source or stage_source_cache_for_cpu)(
                source, artifact_root=root, runtime=config.runtime
            )
            feature_producer = (
                deps.produce_candidate_feature_rows or materialize_label_free_seed_features
            )
            feature_kwargs = {"root": root}
            if deps.produce_candidate_feature_rows is None:
                feature_kwargs["retain_checkpoints"] = True
            seed_features = feature_producer(
                config, staged.cache, frame, partitions, metadata, **feature_kwargs
            )

            phase = "GLOBAL_DEVELOPMENT_PREDICTION_SEAL"
            _phase(deps, root, phase)
            enter_cuda_free_cpu_phase()
            target_embeddings = (deps.stage_target_embeddings or stage_target_embeddings)(
                frame, scratch_root=staged.scratch_root
            )
            development_plan = (
                deps.build_development_plan or build_development_prediction_plan
            )(
                config, frame=frame, partitions=partitions, source_cache=staged.cache,
                target_embeddings=target_embeddings,
                checkpoint_root=root / "checkpoints/development_predictions",
            )
            development = (deps.materialize_development or materialize_development_predictions)(
                development_plan, root=root, workers=4
            )

            phase = "TARGET_PHYSICAL_PREDICTIONS"
            _phase(deps, root, phase)
            target_plan = (deps.build_target_plan or build_target_prediction_plan)(
                config, frame=frame, partitions=partitions, source_cache=staged.cache,
                target_embeddings=target_embeddings,
                checkpoint_root=root / "checkpoints/target_predictions",
            )
            target_store = (deps.materialize_target or materialize_target_predictions)(
                target_plan, root=root, workers=4
            )
            cleanup_staged_target_embeddings(target_embeddings)
            target_embeddings = None
            shifts = (deps.produce_support_shifts or materialize_label_free_support_shifts)(
                seed_features, development, target_store, partitions, root=root
            )

            phase = "DEVELOPMENT_MODELS_AND_TARGET_PLANS"
            _phase(deps, root, phase)
            prelabel = (deps.run_prelabel_science or run_prelabel_science)(
                config=config, root=root, partitions=partitions,
                development=development, seed_features=seed_features,
                shifts=shifts, target_store=target_store, frame=frame,
            )
            target_capability = prelabel.seal_target(target_store, root=root)
            persist_development_surfaces(
                root, **prelabel.development_persistence
            )
            persist_model_and_plan_surfaces(
                root, **prelabel.model_plan_persistence(target_capability)
            )

            phase = "TERMINAL_TARGET_SCORING"
            _phase(deps, root, phase)
            terminal = (deps.run_terminal_science or run_terminal_science)(
                config=config, root=root, partitions=partitions,
                development=development, target_capability=target_capability,
                prelabel=prelabel, preflight=preflight,
                source_cache_staging=staged.report_payload(),
            )
            persist_terminal_surfaces(
                root, **terminal.persistence
            )
            phase = "CLOSED_WORLD_VALIDATION"
            _phase(deps, root, phase)
            cleanup_staged_source_cache(staged)
            staged = None
            return _finalize_bundle(root, config=config, protocol=protocol)
        except BaseException as exc:
            _write_state(
                root, status="FAILED", phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            if target_embeddings is not None:
                cleanup_staged_target_embeddings(target_embeddings)
            if staged is not None:
                cleanup_staged_source_cache(staged)


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_utility_aligned_consumed_test_endpoint_router_bundle
    return validate_utility_aligned_consumed_test_endpoint_router_bundle(root, **kwargs)


def _finalize_bundle(root: Path, *, config: object, protocol: object) -> Path:
    cleanup_feature_runtime_checkpoints(root)
    write_content_index(
        root, config_contract_hash=config.contract_hash,
        protocol_contract_hash=protocol.contract_hash,
    )
    checks = _validate_bundle(root, config=config, allow_pending=True)
    persist_validation_report(root, checks)
    _write_state(root, status="COMPLETE", phase="COMPLETE")
    _validate_bundle(root, config=config)
    return root


def _phase(
    deps: ConsumedTestEndpointRouterRunnerDependencies, root: Path, phase: str
) -> None:
    _write_state(root, status="RUNNING", phase=phase)
    if deps.phase_observer is not None:
        deps.phase_observer(phase.lower())


def _write_state(root: Path, *, status: str, phase: str, error: str | None = None) -> None:
    atomic_json(root / "reports/run_state.json", run_state_payload(status, phase, error=error))


def _assert_workspace_resolved_paths(
    config: ConsumedTestEndpointRouterConfig, *, root: Path
) -> None:
    paths = {
        "root": root, "configured root": config.artifact_root,
        "expert bank": config.expert_bank_root,
        "generation lock": config.generation_lock_root,
        "test cache": config.test_cache_root,
        "test manifest": config.test_manifest_path,
        "domain mapping": config.domain_mapping_path,
        "consumption ledger": config.test_consumption_ledger_path,
        "ledger amendment": config.ledger_amendment_path,
    }
    unresolved = [name for name, value in paths.items() if not Path(value).is_absolute()]
    if unresolved or root.resolve() != config.artifact_root.resolve():
        raise ProtocolError(
            f"Endpoint router requires workspace-resolved paths; unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Endpoint-router launch files are absent: {missing}.")


__all__ = ("ConsumedTestEndpointRouterDependencies",
           "ConsumedTestEndpointRouterRunnerDependencies",
           "run_utility_aligned_consumed_test_endpoint_router")
