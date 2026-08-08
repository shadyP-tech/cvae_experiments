"""Phase-ordered runner for the terminal utility-aligned Stage-90 diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .actions import build_exact_tail_action_library
from .artifact_io import read_json
from .bundle import assert_closed_world, write_content_index
from .config import UtilityAlignedExactTailRouterConfig
from .development_label_access import open_globally_sealed_development_labels
from .development_scoring import score_exact_tail_development_utility
from .development_seal import materialize_development_predictions
from .feature_production import produce_label_free_features
from .inference import build_center_contrasts, infer_center_contrasts
from .inputs import (
    load_label_free_validation_frame,
    load_metadata_similarity,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .modeling import fit_stage90_models
from .partitions import build_case_fold_surface, build_fixed_partition_surface
from .r2_policy import build_stage90_r2_plan_set
from .reports import (
    leakage_report_payload,
    publication_decision_payload,
    runtime_summary_payload,
    scoring_summary_payload,
)
from .runner_persistence import (
    persist_development_and_router_surfaces,
    persist_initial_surfaces,
    persist_source_and_feature_surfaces,
    persist_target_seal_phase,
    persist_terminal_surfaces,
    persist_validation_report,
    write_run_state,
)
from .runtime_preflight import run_workstation_preflight
from .scoring import (
    build_hxe_oracle_diagnostics,
    score_target_probability_ensembles,
    score_target_seed_cells,
)
from .source_cache import materialize_source_cache, stage_source_cache_for_cpu
from .source_cache_validation import validate_source_cache_lock
from .target_label_access import open_target_labels_after_global_seal
from .target_prediction_store import materialize_target_predictions
from .target_seal import build_global_target_prediction_seal


@dataclass(frozen=True)
class UtilityAlignedRunnerDependencies:
    """Injection seams for order, failure, resume, and workstation tests."""

    validate_workspace: Callable[..., object] | None = None
    validate_provenance: Callable[..., object] | None = None
    load_locks: Callable[..., object] | None = None
    load_frame: Callable[..., object] | None = None
    validate_firewall: Callable[..., object] | None = None
    build_partitions: Callable[..., object] | None = None
    build_case_folds: Callable[..., object] | None = None
    run_preflight: Callable[..., object] | None = None
    materialize_source: Callable[..., object] | None = None
    validate_source_lock: Callable[..., object] | None = None
    stage_source: Callable[..., object] | None = None
    load_metadata: Callable[..., object] | None = None
    produce_features: Callable[..., object] | None = None
    materialize_development: Callable[..., object] | None = None
    open_development_labels: Callable[..., object] | None = None
    score_development: Callable[..., object] | None = None
    fit_models: Callable[..., object] | None = None
    build_plans: Callable[..., object] | None = None
    build_actions: Callable[..., object] | None = None
    materialize_target: Callable[..., object] | None = None
    build_target_seal: Callable[..., object] | None = None
    open_target_labels: Callable[..., object] | None = None
    score_seed_cells: Callable[..., object] | None = None
    score_ensembles: Callable[..., object] | None = None
    build_contrasts: Callable[..., object] | None = None
    infer_contrasts: Callable[..., object] | None = None
    build_oracle: Callable[..., object] | None = None
    validate_bundle: Callable[..., object] | None = None
    persist_initial: Callable[..., None] | None = None
    persist_source_features: Callable[..., None] | None = None
    persist_development_router: Callable[..., None] | None = None
    persist_target_seal: Callable[..., None] | None = None
    persist_terminal: Callable[..., None] | None = None
    write_index: Callable[..., object] | None = None
    persist_validation: Callable[..., None] | None = None
    write_state: Callable[..., None] | None = None
    phase_observer: Callable[[str], None] | None = None


def run_utility_aligned_exact_tail_router_diagnostic(
    config: UtilityAlignedExactTailRouterConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: UtilityAlignedRunnerDependencies | None = None,
) -> Path:
    """Execute the frozen protocol and validate the resulting closed world."""

    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or UtilityAlignedRunnerDependencies()
    _assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)
    assert_closed_world(root, allow_incomplete=True)

    with _exclusive_run_lock(root):
        state_path = root / "reports/run_state.json"
        if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
            # A COMPLETE marker is never trusted as a substitute for inventory.
            assert_closed_world(root, allow_incomplete=False)
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root

        phase = "INITIALIZING"
        _write_state(deps, root, status="RUNNING", phase=phase)
        try:
            _observe(deps, "workspace")
            workspace = (deps.validate_workspace or validate_active_diagnostic_workspace_binding)(config)
            _observe(deps, "provenance")
            provenance = (deps.validate_provenance or validate_workspace_provenance)(root, config)
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_validation_frame)(config)
            _observe(deps, "firewall")
            firewall = {
                **(deps.validate_firewall or validate_pre_gpu_firewall)(config, frame),
                "workspace_binding": workspace,
            }
            _observe(deps, "partitions")
            partitions = (deps.build_partitions or build_fixed_partition_surface)(
                frame, config_contract_hash=config.contract_hash
            )
            case_folds = (deps.build_case_folds or build_case_fold_surface)(
                partitions, config_contract_hash=config.contract_hash
            )
            (deps.persist_initial or persist_initial_surfaces)(
                root,
                config=config,
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                partitions=partitions,
                case_folds=case_folds,
            )

            phase = "WORKSTATION_PREFLIGHT"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "preflight")
            preflight = (deps.run_preflight or run_workstation_preflight)(
                root, runtime=config.runtime
            )

            phase = "SOURCE_CACHE_270"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "source_cache")
            canonical_cache = (deps.materialize_source or materialize_source_cache)(
                config, locks.generation, frame, partitions, root=root
            )
            source_lock = (deps.validate_source_lock or validate_source_cache_lock)(
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
                "canonical_root": str(root.resolve()),
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
                        "active_root": str(cpu_cache.root.resolve()),
                    }
                )

            phase = "LABEL_FREE_FEATURES"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "features")
            metadata = (deps.load_metadata or load_metadata_similarity)(config)
            production = (deps.produce_features or produce_label_free_features)(
                cpu_cache, frame, partitions, metadata
            )
            (deps.persist_source_features or persist_source_and_feature_surfaces)(
                root,
                config_contract_hash=config.contract_hash,
                source_cache_lock_hash=source_lock_hash,
                production=production,
            )

            phase = "GLOBAL_INNER_PREDICTION_SEAL"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "inner_predictions")
            development = (
                deps.materialize_development or materialize_development_predictions
            )(
                config,
                locks.generation,
                cpu_cache,
                frame,
                partitions,
                root=root,
            )

            phase = "DEVELOPMENT_SCORING_AND_ROUTER_LOCK"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "development_labels")
            development_labels = (
                deps.open_development_labels
                or open_globally_sealed_development_labels
            )(
                config.validation_manifest_path,
                partitions,
                capability=development,
            )
            _observe(deps, "development_scoring")
            utility_rows = (deps.score_development or score_exact_tail_development_utility)(
                development, development_labels, partitions
            )
            _observe(deps, "models")
            models = (deps.fit_models or fit_stage90_models)(
                production.surfaces, utility_rows
            )
            _observe(deps, "plans")
            plans = (deps.build_plans or build_stage90_r2_plan_set)(
                models, production.surfaces
            )
            _observe(deps, "actions")
            actions = (deps.build_actions or build_exact_tail_action_library)(plans)
            (deps.persist_development_router or persist_development_and_router_surfaces)(
                root,
                config_contract_hash=config.contract_hash,
                development_labels=development_labels,
                utility_rows=utility_rows,
                models=models,
                plans=plans,
                actions=actions,
                development_prediction_seal_hash=development.seal.prediction_seal_hash,
            )

            phase = "TARGET_PREDICTIONS"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "target_predictions")
            predictions = (deps.materialize_target or materialize_target_predictions)(
                config,
                cpu_cache,
                actions,
                frame,
                partitions,
                case_folds,
                source_cache_lock_hash=source_lock_hash,
                root=root,
            )
            _observe(deps, "target_seal")
            target_seal = (deps.build_target_seal or build_global_target_prediction_seal)(
                root,
                config_contract_hash=config.contract_hash,
                source_cache_lock_hash=source_lock_hash,
                partitions=partitions,
                case_folds=case_folds,
                library=actions,
                predictions=predictions,
            )
            (deps.persist_target_seal or persist_target_seal_phase)(
                root,
                config_contract_hash=config.contract_hash,
                action_library_hash=actions.action_library_hash,
                target_seal=target_seal,
                prediction_cell_count=len(predictions.cells),
                unique_classifier_fit_count=predictions.unique_classifier_fit_count,
            )

            phase = "TERMINAL_TARGET_SCORING"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "target_labels")
            labels_by_sample, target_label_report = (
                deps.open_target_labels or open_target_labels_after_global_seal
            )(config, partitions, root=root)
            _observe(deps, "terminal_scoring")
            seed_rows = (deps.score_seed_cells or score_target_seed_cells)(
                predictions, labels_by_sample, partitions
            )
            ensemble_rows = (
                deps.score_ensembles or score_target_probability_ensembles
            )(predictions, labels_by_sample, partitions)
            center_rows = (deps.build_contrasts or build_center_contrasts)(ensemble_rows)
            inference_rows = (deps.infer_contrasts or infer_center_contrasts)(center_rows)
            oracle_rows = (deps.build_oracle or build_hxe_oracle_diagnostics)(
                ensemble_rows, plans
            )
            scoring_summary = scoring_summary_payload(
                ensemble_rows, inference_rows, oracle_rows
            )
            leakage = leakage_report_payload(
                support_partition_lock_hash=partitions.lock_hash,
                case_fold_lock_hash=case_folds.lock_hash,
                development_prediction_seal_hash=(
                    development.seal.prediction_seal_hash
                ),
                model_set_hash=models.model_set_hash,
                plan_set_hash=plans.plan_set_hash,
                action_library_hash=actions.action_library_hash,
                target_prediction_seal_hash=str(target_seal["seal_hash"]),
                firewall=firewall,
            )
            counts = {
                "source_stream_count": len(canonical_cache.source_records),
                "support_component_count": len(canonical_cache.component_records),
                "inner_feature_row_count": len(production.inner_rows),
                "target_feature_row_count": len(production.target_rows),
                "development_prediction_cell_count": development.seal.cell_count,
                "exact_tail_utility_row_count": len(utility_rows),
                "target_action_count": actions.action_count,
                "target_prediction_cell_count": len(predictions.cells),
                "unique_classifier_fit_count": predictions.unique_classifier_fit_count,
            }
            runtime_summary = runtime_summary_payload(
                preflight, counts=counts, source_cache_staging=staging
            )
            (deps.persist_terminal or persist_terminal_surfaces)(
                root,
                config_contract_hash=config.contract_hash,
                target_label_report=target_label_report,
                seed_rows=seed_rows,
                ensemble_rows=ensemble_rows,
                center_contrasts=center_rows,
                inference_rows=inference_rows,
                oracle_rows=oracle_rows,
                leakage_report=leakage,
                scoring_summary=scoring_summary,
                publication_decision=publication_decision_payload(scoring_summary),
                runtime_summary=runtime_summary,
                target_seal_hash=str(target_seal["seal_hash"]),
            )

            phase = "CLOSED_WORLD_VALIDATION"
            _write_state(deps, root, status="RUNNING", phase=phase)
            _observe(deps, "validation")
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


def _validate_bundle(root: Path, **kwargs: object) -> Mapping[str, object]:
    from .validation import validate_utility_aligned_exact_tail_router_bundle

    return validate_utility_aligned_exact_tail_router_bundle(root, **kwargs)


def _observe(deps: UtilityAlignedRunnerDependencies, phase: str) -> None:
    if deps.phase_observer is not None:
        deps.phase_observer(phase)


def _write_state(
    deps: UtilityAlignedRunnerDependencies,
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
        raise ProtocolError("Utility-aligned diagnostic is already running.") from exc
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
    config: UtilityAlignedExactTailRouterConfig, *, root: Path
) -> None:
    paths = {
        "artifact root": root,
        "configured artifact root": config.artifact_root,
        "expert-bank root": config.expert_bank_root,
        "generation-lock root": config.generation_lock_root,
        "equal-union root": config.equal_union_policy_root,
        "validation-cache root": config.validation_cache_root,
        "validation manifest": config.validation_manifest_path,
        "metadata-profile root": config.metadata_profile_root,
    }
    unresolved = [role for role, path in paths.items() if not Path(path).is_absolute()]
    if unresolved or root.resolve() != config.artifact_root.resolve():
        raise ProtocolError(
            "Utility-aligned diagnostic requires workspace-resolved paths; "
            f"unresolved={unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(
            f"Utility-aligned workspace launch files are missing: {missing}."
        )


__all__ = (
    "UtilityAlignedRunnerDependencies",
    "run_utility_aligned_exact_tail_router_diagnostic",
)
