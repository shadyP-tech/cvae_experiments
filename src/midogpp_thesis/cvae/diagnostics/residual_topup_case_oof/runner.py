"""Thin phase orchestrator for the terminal consumed-validation case-OOF run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from .artifact_io import (
    assert_closed_world,
    exclusive_run_lock,
    persist_or_validate_json,
    prune_stale_temp_files,
    read_json,
)
from .bundle import REQUIRED_FILES, write_content_index
from .config import ResidualTopupCaseOOFConfig
from .contracts import (
    EXPECTED_PROXY_SCORE_COUNT,
    EXPECTED_SEALED_PREDICTION_CELL_COUNT,
)
from .inference import (
    build_center_contrasts,
    build_oracle_hxe_diagnostics,
    infer_center_contrasts,
)
from .inputs import (
    build_partition_surface,
    load_label_free_validation_frame,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .label_access import open_evaluation_labels_after_global_seal
from .partitions import build_case_oof_surface
from .planning import build_case_oof_plan
from .prediction_execution import materialize_all_action_predictions
from .prediction_planning import EXPECTED_PREDICTION_TASK_COUNT
from .prediction_validation import validate_prediction_store_binding
from .ranking import build_rank_surface
from .reports import (
    leakage_report_payload,
    publication_decision_payload,
    runtime_summary_payload,
    scoring_summary_payload,
)
from .runner_persistence import (
    persist_initial_surfaces,
    persist_prediction_phase,
    persist_rank_and_plan_surfaces,
    persist_source_phase,
    persist_terminal_surfaces,
    persist_validation_report,
    write_run_state,
)
from .runtime_preflight import run_workstation_preflight
from .scoring import score_center_probability_ensembles, score_center_seed_cells
from .seals import (
    GLOBAL_PREDICTION_SEAL_MEMBER,
    build_global_prediction_seal,
    validate_global_prediction_seal,
)
from .source_cache import (
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPECTED_SOURCE_TASK_COUNT,
    materialize_source_cache,
    validate_source_cache_lock,
)


@dataclass(frozen=True)
class CaseOOFRunnerDependencies:
    """Narrow injection seam for protocol-order and failure-boundary tests."""

    validate_workspace: Callable[[object], Mapping[str, object]] | None = None
    validate_provenance: Callable[..., Mapping[str, Mapping[str, object]]] | None = None
    load_locks: Callable[[object], object] | None = None
    load_frame: Callable[[object], object] | None = None
    validate_firewall: Callable[[object, object], Mapping[str, object]] | None = None
    run_preflight: Callable[..., Mapping[str, object]] | None = None
    materialize_source: Callable[..., object] | None = None
    materialize_predictions: Callable[..., object] | None = None
    open_labels: Callable[..., tuple[dict[str, int], Mapping[str, object]]] | None = None
    validate_bundle: Callable[..., dict[str, object]] | None = None


def run_residual_topup_case_oof_diagnostic(
    config: ResidualTopupCaseOOFConfig,
    *,
    artifact_root: str | Path | None = None,
    dependencies: CaseOOFRunnerDependencies | None = None,
) -> Path:
    """Run, seal, score, and independently validate the Stage-90 diagnostic."""

    root = Path(artifact_root or config.artifact_root)
    deps = dependencies or CaseOOFRunnerDependencies()
    _assert_workspace_resolved_paths(config, root=root)
    for relative in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    _assert_launch_files(root)
    assert_closed_world(root, required_files=REQUIRED_FILES, allow_incomplete=True)

    with exclusive_run_lock(root):
        prune_stale_temp_files(root)
        state_path = root / "reports/run_state.json"
        if state_path.is_file() and read_json(state_path).get("status") == "COMPLETE":
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root

        phase = "INITIALIZING"
        write_run_state(root, status="RUNNING", phase=phase)
        try:
            workspace_binding = (
                deps.validate_workspace or validate_active_diagnostic_workspace_binding
            )(config)
            provenance = (deps.validate_provenance or validate_workspace_provenance)(
                root, config
            )
            locks = (deps.load_locks or load_validated_locks)(config)
            frame = (deps.load_frame or load_label_free_validation_frame)(config)
            firewall = {
                **(deps.validate_firewall or validate_pre_gpu_firewall)(config, frame),
                "workspace_binding": workspace_binding,
            }
            base = build_partition_surface(
                frame, config_contract_hash=config.contract_hash
            )
            crossfit = build_case_oof_surface(
                base, config_contract_hash=config.contract_hash
            )
            persist_initial_surfaces(
                root,
                config=config,
                provenance=provenance,
                frame=frame,
                pre_gpu_firewall=firewall,
                base=base,
                crossfit=crossfit,
            )

            # This is deliberately after the source/validation split firewall.
            preflight = (deps.run_preflight or run_workstation_preflight)(
                root, runtime=config.runtime
            )

            phase = "SOURCE_CACHE"
            write_run_state(root, status="RUNNING", phase=phase)
            source_cache = (deps.materialize_source or materialize_source_cache)(
                config, locks.generation, frame, crossfit, root=root
            )
            source_lock = validate_source_cache_lock(
                root,
                config=config,
                generation_lock=locks.generation,
                frame=frame,
                crossfit=crossfit,
                source_cache=source_cache,
            )
            source_lock_hash = str(source_lock["source_cache_lock_hash"])
            proxy_rows = source_cache.proxy_score_rows(crossfit)
            if len(proxy_rows) != EXPECTED_PROXY_SCORE_COUNT:
                raise ProtocolError("Case-OOF proxy-score coverage drifted.")
            rank_surface = build_rank_surface(proxy_rows, crossfit)
            plan = build_case_oof_plan(
                rank_surface, crossfit, config_contract_hash=config.contract_hash
            )
            persist_source_phase(
                root,
                config_contract_hash=config.contract_hash,
                source_cache=source_cache,
                source_cache_lock_hash=source_lock_hash,
            )
            persist_rank_and_plan_surfaces(
                root, rank_surface=rank_surface, plan=plan
            )

            phase = "ALL_ACTION_PREDICTIONS"
            write_run_state(root, status="RUNNING", phase=phase)
            predictions = (
                deps.materialize_predictions or materialize_all_action_predictions
            )(
                config,
                locks.generation.generation_lock_hash,
                source_cache,
                plan,
                frame,
                crossfit,
                source_cache_lock_hash=source_lock_hash,
                root=root,
            )
            validate_prediction_store_binding(
                predictions,
                config=config,
                generation_lock_hash=locks.generation.generation_lock_hash,
                source_cache=source_cache,
                source_cache_lock_hash=source_lock_hash,
                plan=plan,
                crossfit=crossfit,
            )
            if (root / GLOBAL_PREDICTION_SEAL_MEMBER).is_file():
                seal = validate_global_prediction_seal(
                    config,
                    crossfit,
                    plan,
                    predictions,
                    source_cache_lock_hash=source_lock_hash,
                    root=root,
                )
            else:
                seal = build_global_prediction_seal(
                    config,
                    crossfit,
                    plan,
                    predictions,
                    source_cache_lock_hash=source_lock_hash,
                    root=root,
                )
                validate_global_prediction_seal(
                    config,
                    crossfit,
                    plan,
                    predictions,
                    source_cache_lock_hash=source_lock_hash,
                    root=root,
                )
            seal_hash = str(seal["seal_hash"])
            persist_prediction_phase(
                root,
                config_contract_hash=config.contract_hash,
                plan=plan,
                predictions=predictions,
                seal_hash=seal_hash,
            )

            # The only label-capable operation; it revalidates the durable seal.
            phase = "TERMINAL_SCORING"
            write_run_state(root, status="RUNNING", phase=phase)
            labels, label_report = (
                deps.open_labels or open_evaluation_labels_after_global_seal
            )(
                config,
                crossfit,
                plan,
                predictions,
                source_cache_lock_hash=source_lock_hash,
                root=root,
            )
            persist_or_validate_json(
                root / "reports/label_access_report.json", label_report
            )
            center_seed_rows = score_center_seed_cells(
                predictions, labels_by_sample_id=labels, crossfit=crossfit
            )
            ensemble_rows = score_center_probability_ensembles(
                predictions, labels_by_sample_id=labels, crossfit=crossfit
            )
            contrast_rows = build_center_contrasts(ensemble_rows)
            inference_rows = infer_center_contrasts(contrast_rows)
            oracle_rows = build_oracle_hxe_diagnostics(
                ensemble_rows, rank_surface=rank_surface
            )
            summary = scoring_summary_payload(
                ensemble_rows, inference_rows, oracle_rows
            )
            leakage = leakage_report_payload(
                support_partition_lock_hash=base.lock_hash,
                crossfit_fold_lock_hash=crossfit.lock_hash,
                source_cache_lock_hash=source_lock_hash,
                router_plan_lock_hash=plan.lock_hash,
                global_prediction_seal_hash=seal_hash,
                pre_gpu_firewall=firewall,
            )
            runtime_summary = runtime_summary_payload(
                preflight,
                source_task_count=EXPECTED_SOURCE_TASK_COUNT,
                source_block_count=EXPECTED_SOURCE_BLOCK_COUNT,
                prediction_task_count=EXPECTED_PREDICTION_TASK_COUNT,
                prediction_cell_count=EXPECTED_SEALED_PREDICTION_CELL_COUNT,
                unique_classifier_fit_count=predictions.unique_classifier_fit_count,
            )
            persist_terminal_surfaces(
                root,
                config_contract_hash=config.contract_hash,
                label_report=label_report,
                center_seed_rows=center_seed_rows,
                ensemble_rows=ensemble_rows,
                contrast_rows=contrast_rows,
                inference_rows=inference_rows,
                oracle_rows=oracle_rows,
                leakage_report=leakage,
                publication_decision=publication_decision_payload(summary),
                runtime_summary=runtime_summary,
                seal_hash=seal_hash,
            )

            phase = "VALIDATING"
            write_run_state(root, status="RUNNING", phase=phase)
            write_content_index(root, config_contract_hash=config.contract_hash)
            checks = (deps.validate_bundle or _validate_bundle)(
                root, config=config, allow_pending=True
            )
            persist_validation_report(root, checks)
            write_run_state(root, status="COMPLETE", phase="COMPLETE")
            (deps.validate_bundle or _validate_bundle)(root, config=config)
            return root
        except BaseException as exc:
            write_run_state(
                root,
                status="FAILED",
                phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise


def _validate_bundle(
    root: Path,
    *,
    config: ResidualTopupCaseOOFConfig,
    allow_pending: bool = False,
) -> dict[str, object]:
    from .validation import validate_residual_topup_case_oof_bundle

    return validate_residual_topup_case_oof_bundle(
        root, config=config, allow_pending=allow_pending
    )


def _assert_workspace_resolved_paths(
    config: ResidualTopupCaseOOFConfig, *, root: Path
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
    if unresolved or root.resolve() != config.artifact_root.resolve():
        raise ProtocolError(
            "Case-OOF requires canonical workspace-resolved paths and output binding. "
            f"Unresolved paths: {unresolved}."
        )


def _assert_launch_files(root: Path) -> None:
    missing = [
        member
        for member in ("config.resolved.yaml", "provenance/input_artifacts.json")
        if not (root / member).is_file()
    ]
    if missing:
        raise ProtocolError(f"Case-OOF workspace launch files are missing: {missing}.")


__all__ = (
    "CaseOOFRunnerDependencies",
    "run_residual_topup_case_oof_diagnostic",
)
