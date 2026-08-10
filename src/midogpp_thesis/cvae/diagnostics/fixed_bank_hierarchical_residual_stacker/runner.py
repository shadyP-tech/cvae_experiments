"""Thin phase orchestrator for the terminal hierarchical residual stacker."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .artifact_io import read_json
from .bundle import assert_closed_world, write_content_index
from .config import FixedBankHierarchicalResidualStackerConfig
from .execution_adapter import (
    materialize_probabilities,
    materialize_sources,
    run_label_free_workstation_preflight,
    runtime_summary_payload,
    seed_probability_rows,
    stage_sources_for_cpu,
)
from .execution_phases import (
    build_fold_decisions,
    build_prelabel_products,
    evaluate_terminal_predictions,
    fit_all_loco_models,
)
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .execution_adapter import build_case_partition
from .label_capabilities import LabelCapabilityManager
from .persistence import (
    persist_and_validate_loco_models,
    persist_and_validate_preevaluation_seals,
    persist_initial_surfaces,
    persist_postseal_results,
    persist_prediction_and_feature_surfaces,
    persist_validation_report,
    write_run_state,
)
from .reports import leakage_report_payload


@dataclass(frozen=True)
class FixedBankHierarchicalResidualStackerDependencies:
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
    build_prelabel: Callable[..., object] | None = None
    persist_prelabel: Callable[..., object] | None = None
    build_label_manager: Callable[..., object] | None = None
    fit_loco_models: Callable[..., object] | None = None
    persist_loco_models: Callable[..., object] | None = None
    fit_fold_decisions: Callable[..., object] | None = None
    persist_preevaluation: Callable[..., object] | None = None
    evaluate: Callable[..., object] | None = None
    persist_postseal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


def run_fixed_bank_hierarchical_residual_stacker(
    config: FixedBankHierarchicalResidualStackerConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: FixedBankHierarchicalResidualStackerDependencies | None = None,
) -> Path:
    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or FixedBankHierarchicalResidualStackerDependencies()
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
            workspace = (
                deps.validate_workspace or validate_active_diagnostic_workspace_binding
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
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                partition=partition,
            )

            phase = "WORKSTATION_PREFLIGHT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "preflight")
            preflight = (deps.preflight or run_label_free_workstation_preflight)(
                root, runtime=config.runtime
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

            # No later phase may initialize CUDA in the parent or CPU children.
            _enter_cuda_free_cpu_phase()
            phase = "GLOBAL_PREDICTION_AND_LABEL_FREE_FEATURE_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "cuda_free_probability_and_feature_seal")
            prediction_capability = (
                deps.materialize_predictions or materialize_probabilities
            )(config, source_for_cpu, frame, partition, root=root)
            seed_rows = (deps.build_seed_rows or seed_probability_rows)(
                prediction_capability
            )
            prelabel_products = (deps.build_prelabel or build_prelabel_products)(
                seed_rows
            )
            prelabel_seal = (
                deps.persist_prelabel or persist_prediction_and_feature_surfaces
            )(
                root,
                prediction_capability=prediction_capability,
                seed_rows=seed_rows,
                probabilities=prelabel_products.probabilities,
                probability_surface_hash=prelabel_products.probability_surface_hash,
                case_features=prelabel_products.features,
                source_controls=prelabel_products.source_controls,
            )
            feature_hash = str(prelabel_seal["feature_surface_hash"])
            if (
                read_json(
                    root / "reports/phase_01_prediction_and_feature_seal_complete.json"
                ).get("feature_surface_hash")
                != feature_hash
            ):
                raise ProtocolError("Residual-stacker prelabel feature seal drifted.")

            manager = (deps.build_label_manager or LabelCapabilityManager)(
                config.test_manifest_path,
                frame,
                partition,
                global_prediction_seal_hash=prediction_capability.seal_hash,
                label_free_feature_seal_hash=feature_hash,
            )

            phase = "STRICT_LOCO_G_R_P_MODEL_FAMILIES"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "loco_labels_and_models")
            loco = (deps.fit_loco_models or fit_all_loco_models)(
                probabilities=prelabel_products.probabilities,
                features=prelabel_products.features,
                label_manager=manager,
                worker_count=int(config.runtime["model_workers"]),
            )
            (deps.persist_loco_models or persist_and_validate_loco_models)(
                root,
                donor_responses=loco.donor_response_records,
                models=loco.models,
            )
            for bundle in loco.bundles:
                manager.record_loco_model_seals(
                    bundle.target_center,
                    bundle.global_model.model_hash,
                    bundle.residual_model.model_hash,
                    bundle.permuted_model.model_hash,
                )
            _observe(deps, "all_G_R_P_models_durable_before_support")

            phase = "FORTY_FIVE_SUPPORT_CALIBRATIONS_AND_225_METHOD_DECISIONS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "support_calibrations_and_decisions")
            fold_products = (deps.fit_fold_decisions or build_fold_decisions)(
                probabilities=prelabel_products.probabilities,
                features=prelabel_products.features,
                bundles=loco.bundles,
                partition=partition,
                label_manager=manager,
            )
            decision_seal, permutation_seal = (
                deps.persist_preevaluation
                or persist_and_validate_preevaluation_seals
            )(
                root,
                calibrations=fold_products.calibrations,
                decisions=fold_products.decisions,
                permutation_provenance=fold_products.permutation_provenance,
                config_contract_hash=config.contract_hash,
            )
            manager.record_preevaluation_seals(
                str(decision_seal["decision_seal_hash"]),
                str(permutation_seal["permutation_provenance_hash"]),
                decision_count=len(fold_products.decisions),
            )
            _observe(deps, "all_225_decisions_and_permutation_durable_before_eval")

            phase = "TERMINAL_POOLED_EXACT_BACC_EVALUATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "terminal_evaluation_labels")
            evaluation_labels = manager.open_oof_evaluation_labels()
            evaluation_products = (deps.evaluate or evaluate_terminal_predictions)(
                predictions_by_method=fold_products.predictions_by_method,
                labels=evaluation_labels,
                calibrations=fold_products.calibrations,
                bootstrap_replicates=int(
                    config.evaluation["whole_case_cluster_bootstrap_replicates"]
                ),
                bootstrap_seed=int(
                    config.evaluation["whole_case_cluster_bootstrap_seed"]
                ),
                bootstrap_workers=int(config.runtime["bootstrap_workers"]),
            )
            capability_report = manager.access_report()
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction_capability.seal_hash,
                feature_seal_hash=feature_hash,
                model_count=len(loco.models),
                decision_count=len(fold_products.decisions),
                capability_report=capability_report,
            )
            runtime_summary = runtime_summary_payload(
                source_cache=canonical_source,
                prediction_capability=prediction_capability,
                local_staging={**staging, "workstation_preflight": dict(preflight)},
            )
            (deps.persist_postseal or persist_postseal_results)(
                root,
                evaluation=evaluation_products.evaluation,
                confusion_rows=evaluation_products.confusion_rows,
                metric_rows=evaluation_products.metric_rows,
                contrast_rows=evaluation_products.contrast_rows,
                capability_report=capability_report,
                leakage_report=leakage,
                runtime_summary=runtime_summary,
            )

            phase = "CLOSED_WORLD_CONTENT_FIRST_VALIDATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "validation")
            (deps.write_index or write_content_index)(
                root, config_contract_hash=config.contract_hash
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
    from .validation import validate_fixed_bank_hierarchical_residual_stacker_bundle

    return validate_fixed_bank_hierarchical_residual_stacker_bundle(root, **kwargs)


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


def _observe(
    deps: FixedBankHierarchicalResidualStackerDependencies, phase: str
) -> None:
    if deps.phase_observer is not None:
        deps.phase_observer(phase)


def _write_state(
    deps: FixedBankHierarchicalResidualStackerDependencies,
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
        raise ProtocolError("Residual-stacker diagnostic is already running.") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": config.artifact_root,
        "expert-bank root": config.expert_bank_root,
        "generation-lock root": config.generation_lock_root,
        "test-cache root": config.test_cache_root,
        "test manifest": config.test_manifest_path,
        "test-consumption ledger": config.test_consumption_ledger_path,
        "ledger amendment": config.ledger_amendment_path,
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved or root.resolve() != Path(config.artifact_root).resolve():
        raise ProtocolError(
            f"Residual stacker requires workspace-resolved paths; unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Residual-stacker launch files are absent: {missing}.")


__all__ = (
    "FixedBankHierarchicalResidualStackerDependencies",
    "run_fixed_bank_hierarchical_residual_stacker",
)
