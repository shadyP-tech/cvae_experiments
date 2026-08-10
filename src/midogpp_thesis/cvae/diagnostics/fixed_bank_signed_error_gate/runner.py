"""Phase orchestrator for the terminal signed sample-error diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.artifact_io import read_json
from .bundle import (
    assert_closed_world,
    assert_terminal_phase_complete,
    cleanup_owned_atomic_temps,
    write_content_index,
)
from .execution import (
    build_signed_fold_products,
    build_signed_prelabel_products,
    fit_all_target_families,
)
from .execution_adapter import (
    build_case_partition,
    materialize_probabilities,
    materialize_sources,
    run_label_free_workstation_preflight,
    runtime_summary_payload,
    seed_probability_rows,
    stage_sources_for_cpu,
)
from .label_capabilities import SignedErrorLabelCapabilityManager
from .persistence import (
    persist_and_validate_fold_products,
    persist_and_validate_models,
    persist_initial_surfaces,
    persist_postseal_results,
    persist_prelabel_surfaces,
    persist_validation_report,
    write_run_state,
)
from .probability_surface import aggregate_exact_nine_probabilities
from .protocol import canonical_consumed_test_protocol
from .reports import leakage_report_payload
from .sealing import record_durable_fold_seals, record_durable_model_seals
from .terminal import evaluate_sealed_fold_products


@dataclass(frozen=True)
class FixedBankSignedErrorGateDependencies:
    """Narrow injectable seams used by phase-order integration tests."""

    validate_inputs: Callable[..., object] | None = None
    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    build_partition: Callable[..., object] | None = None
    persist_initial: Callable[..., None] | None = None
    preflight: Callable[..., object] | None = None
    materialize_source: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    build_seed_rows: Callable[..., object] | None = None
    aggregate_probabilities: Callable[..., object] | None = None
    build_prelabel: Callable[..., object] | None = None
    persist_prelabel: Callable[..., object] | None = None
    build_label_manager: Callable[..., object] | None = None
    fit_models: Callable[..., object] | None = None
    persist_models: Callable[..., object] | None = None
    record_models: Callable[..., None] | None = None
    fit_folds: Callable[..., object] | None = None
    persist_folds: Callable[..., object] | None = None
    record_folds: Callable[..., None] | None = None
    evaluate: Callable[..., object] | None = None
    persist_postseal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


def run_fixed_bank_signed_error_gate(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    dependencies: FixedBankSignedErrorGateDependencies | None = None,
) -> Path:
    """Run once under the standalone consumed-test signed-error authorization."""

    root = Path(artifact_root or getattr(config, "artifact_root"))
    deps = dependencies or FixedBankSignedErrorGateDependencies()
    protocol = canonical_consumed_test_protocol()
    _assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)

    from .inputs import (
        assert_input_fence,
        load_label_free_test_frame,
        load_validated_locks,
        validate_active_diagnostic_workspace_binding,
        validate_pre_gpu_firewall,
        validate_workspace_provenance,
    )

    with _exclusive_run_lock(root):
        cleanup_owned_atomic_temps(root)
        assert_closed_world(root, allow_incomplete=True)
        state_path = root / "reports/run_state.json"
        if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
            assert_closed_world(root, allow_incomplete=False)
            _enter_cuda_free_cpu_phase()
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root
        if (root / "manifests/sealed_terminal_evaluation.json").is_file() and not (
            root / "manifests/content_index.json"
        ).is_file():
            assert_terminal_phase_complete(root)
            _write_state(
                deps,
                root,
                status="RUNNING",
                phase="TERMINAL_PHASE_VALIDATION_RECOVERY",
            )
            (deps.write_index or write_content_index)(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.contract_hash,
            )
            _enter_cuda_free_cpu_phase()
            checks = (deps.validate_bundle or _validate_bundle)(root, config=config)
            (deps.persist_validation or persist_validation_report)(root, checks)
            _write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root
        if (root / "manifests/content_index.json").is_file():
            assert_closed_world(
                root,
                allow_incomplete=False,
                allow_pending_validation=not (
                    root / "reports/validation_report.json"
                ).is_file(),
            )
            _write_state(
                deps,
                root,
                status="RUNNING",
                phase="CLOSED_WORLD_CONTENT_FIRST_VALIDATION_RECOVERY",
            )
            _enter_cuda_free_cpu_phase()
            checks = (deps.validate_bundle or _validate_bundle)(root, config=config)
            (deps.persist_validation or persist_validation_report)(root, checks)
            _write_state(deps, root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root

        phase = "INITIALIZING"
        _write_state(deps, root, status="RUNNING", phase=phase)
        try:
            _observe(deps, "input_fence")
            (deps.validate_inputs or assert_input_fence)(config)
            workspace = (
                deps.validate_workspace
                or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(
                root, config
            )
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_test_frame)(config)
            firewall = dict(
                (deps.validate_firewall or validate_pre_gpu_firewall)(
                    config, frame, locks
                )
            )
            firewall["workspace_binding"] = workspace
            partition = (deps.build_partition or build_case_partition)(
                frame, config=config
            )
            (deps.persist_initial or persist_initial_surfaces)(
                root,
                config=config,
                protocol=protocol,
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                partition=partition,
            )

            phase = "WORKSTATION_PREFLIGHT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "preflight")
            preflight = (deps.preflight or run_label_free_workstation_preflight)(
                root, runtime=getattr(config, "runtime")
            )

            phase = "FROZEN_SOURCE_STREAMS_TWO_GPU"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "gpu_source_streams")
            canonical_source = (deps.materialize_source or materialize_sources)(
                config, locks.generation, root=root
            )
            source_for_cpu = canonical_source
            staging: dict[str, object] = {
                "attempted": True,
                "used": False,
                "status": "CANONICAL_FALLBACK",
            }
            try:
                source_for_cpu = (deps.stage_source or stage_sources_for_cpu)(
                    canonical_source, config=config, root=root
                )
            except (OSError, ProtocolError) as exc:
                staging["failure"] = f"{type(exc).__name__}: {exc}"
            else:
                staging.update(
                    {
                        "used": source_for_cpu is not canonical_source,
                        "status": (
                            "STAGED_LOCAL_CPU_CACHE"
                            if source_for_cpu is not canonical_source
                            else "CANONICAL_ALREADY_LOCAL"
                        ),
                    }
                )

            _enter_cuda_free_cpu_phase()
            phase = "GLOBAL_PROBABILITY_AND_SIGNED_PRELABEL_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "cuda_free_probability_and_prelabel_seal")
            prediction_capability = (
                deps.materialize_predictions or materialize_probabilities
            )(config, source_for_cpu, frame, partition, root=root)
            seed_rows = (deps.build_seed_rows or seed_probability_rows)(
                prediction_capability
            )
            probabilities, probability_surface_hash = (
                deps.aggregate_probabilities or aggregate_exact_nine_probabilities
            )(seed_rows)
            prelabel = (deps.build_prelabel or build_signed_prelabel_products)(
                probabilities, protocol=protocol
            )
            (deps.persist_prelabel or persist_prelabel_surfaces)(
                root,
                prediction_capability=prediction_capability,
                seed_rows=seed_rows,
                probabilities=probabilities,
                probability_surface_hash=probability_surface_hash,
                prelabel=prelabel,
            )
            persisted_prelabel = read_json(
                root / "manifests/signed_prelabel_feature_seal.json"
            )
            if persisted_prelabel.get("feature_surface_hash") != prelabel.feature_surface_hash:
                raise ProtocolError("Signed-error prelabel feature seal drifted.")

            manager = (deps.build_label_manager or SignedErrorLabelCapabilityManager)(
                getattr(config, "test_manifest_path"),
                frame,
                partition,
                global_prediction_seal_hash=prediction_capability.seal_hash,
                label_free_feature_seal_hash=prelabel.feature_surface_hash,
            )

            phase = "STRICT_OUTER_H_NESTED_QUERY_G_R_P_MODELS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "loco_donor_labels_and_signed_models")
            models = (deps.fit_models or fit_all_target_families)(
                probabilities=probabilities,
                prelabel=prelabel,
                label_manager=manager,
                protocol=protocol,
                worker_count=int(getattr(config, "runtime")["model_workers"]),
                threads_per_worker=int(
                    getattr(config, "runtime")["model_threads_per_worker"]
                ),
            )
            (deps.persist_models or persist_and_validate_models)(
                root, products=models
            )
            (deps.record_models or record_durable_model_seals)(manager, models)
            _observe(deps, "all_nine_G_R_P_model_families_durable_before_support")

            phase = "FORTY_FIVE_SUPPORT_DECISIONS_AND_270_METHOD_SEALS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "support_calibrations_and_six_method_decisions")
            folds = (deps.fit_folds or build_signed_fold_products)(
                probabilities=probabilities,
                model_products=models,
                partition=partition,
                label_manager=manager,
                protocol=protocol,
            )
            (deps.persist_folds or persist_and_validate_fold_products)(
                root, products=folds
            )
            (deps.record_folds or record_durable_fold_seals)(manager, folds)
            _observe(deps, "all_270_decisions_and_permutation_durable_before_eval")

            phase = "TERMINAL_POOLED_EXACT_BACC_EVALUATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "terminal_evaluation_labels")
            terminal_labels = manager.open_oof_evaluation_labels()
            capability_report = manager.access_report()
            evaluation = (deps.evaluate or evaluate_sealed_fold_products)(
                fold_products=folds,
                capability_report=capability_report,
                terminal_labels=terminal_labels,
                protocol=protocol,
                bootstrap_replicates=int(
                    getattr(config, "evaluation")[
                        "whole_case_cluster_bootstrap_replicates"
                    ]
                ),
                bootstrap_seed=int(
                    getattr(config, "evaluation")["whole_case_cluster_bootstrap_seed"]
                ),
                bootstrap_workers=int(
                    getattr(config, "runtime")["bootstrap_workers"]
                ),
                bootstrap_threads_per_worker=int(
                    getattr(config, "runtime")["bootstrap_threads_per_worker"]
                ),
                multiprocessing_start_method=str(
                    getattr(config, "runtime")["multiprocessing_start_method"]
                ),
            )
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction_capability.seal_hash,
                feature_seal_hash=prelabel.feature_surface_hash,
                model_family_count=len(models.target_fits),
                decision_count=len(folds.decisions) * 6,
                capability_report=capability_report,
            )
            runtime_summary = runtime_summary_payload(
                source_cache=canonical_source,
                prediction_capability=prediction_capability,
                local_staging={**staging, "workstation_preflight": dict(preflight)},
                runtime=getattr(config, "runtime"),
            )
            (deps.persist_postseal or persist_postseal_results)(
                root,
                evaluation=evaluation,
                capability_report=capability_report,
                leakage_report=leakage,
                runtime_summary=runtime_summary,
            )

            phase = "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "validation")
            (deps.write_index or write_content_index)(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.contract_hash,
            )
            checks = (deps.validate_bundle or _validate_bundle)(root, config=config)
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


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_fixed_bank_signed_error_gate_bundle

    return validate_fixed_bank_signed_error_gate_bundle(root, **kwargs)


def _enter_cuda_free_cpu_phase() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _observe(deps: FixedBankSignedErrorGateDependencies, phase: str) -> None:
    if deps.phase_observer is not None:
        deps.phase_observer(phase)


def _write_state(
    deps: FixedBankSignedErrorGateDependencies,
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
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("Signed-error diagnostic is already running.") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": getattr(config, "artifact_root"),
        "expert-bank root": getattr(config, "expert_bank_root"),
        "generation-lock root": getattr(config, "generation_lock_root"),
        "test-cache root": getattr(config, "test_cache_root"),
        "test manifest": getattr(config, "test_manifest_path"),
        "test-consumption ledger": getattr(config, "test_consumption_ledger_path"),
        "ledger amendment": getattr(config, "ledger_amendment_path"),
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved or root.resolve() != Path(getattr(config, "artifact_root")).resolve():
        raise ProtocolError(
            f"Signed-error runner requires workspace-resolved paths; unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Signed-error launch files are absent: {missing}.")


__all__ = (
    "FixedBankSignedErrorGateDependencies",
    "run_fixed_bank_signed_error_gate",
)
