"""Thin phase orchestrator for the terminal support-static S4 diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ...protocol import ProtocolError
from .runner_dependencies import SupportStaticRouterDependencies


def run_fixed_bank_support_static_router(
    config: object, *, artifact_root: Path | None = None
) -> Path:
    root = Path(artifact_root or getattr(config, "artifact_root"))
    return _run(config, artifact_root=root, deps=SupportStaticRouterDependencies())


def _run(
    config: object,
    *,
    artifact_root: Path,
    deps: SupportStaticRouterDependencies,
) -> Path:
    """Execute the complete workflow; imports are local to keep CLI startup cheap."""

    from .bundle import (
        assert_closed_world,
        cleanup_owned_atomic_temps,
        relative_files,
        write_content_index,
    )
    from .decisions import (
        build_null_selection_plan,
        make_route_decision,
        seal_global_static_selections,
        seal_route_decisions,
        select_global_static_action,
        select_support_static_action,
    )
    from .execution_adapter import (
        build_case_partition,
        cleanup_validated_local_stage,
        enter_cuda_free_cpu_phase,
        materialize_probabilities,
        materialize_sources,
        run_workstation_preflight,
        runtime_summary_payload,
        stage_sources_for_cpu,
    )
    from .fresh_process_validation import run_two_fresh_process_replays
    from .inputs import (
        assert_input_fence,
        load_label_free_test_frame,
        load_validated_locks,
        validate_pre_gpu_firewall,
    )
    from .label_capabilities import LabelCapabilityManager
    from .persistence import (
        persist_fold_plans,
        persist_fresh_process_report,
        persist_global_static,
        persist_initial_surfaces,
        persist_probability_surface,
        persist_route_decisions,
        persist_terminal_checkpoint,
        persist_validation_report,
        finalize_terminal_checkpoint,
        remove_validated_terminal_checkpoint,
    )
    from .probability_surfaces import (
        build_exact_nine_surface,
        build_prediction_row_index,
        prediction_rows,
        probability_surface_seal_payload,
    )
    from .protocol import canonical_consumed_test_protocol
    from .reports import leakage_report_payload, publication_decision_payload
    from .runner_runtime import (
        assert_launch_files,
        assert_workspace_resolved_paths,
        exclusive_run_lock,
        observe,
        recover_if_possible,
        write_state,
    )
    from .scoring import score_case_action_counts
    from .terminal import (
        evaluate_terminal,
        load_null_selection_plan_seal,
        seal_null_selection_plans,
    )
    from .validation import (
        assert_completed_bundle_binding,
        validate_fixed_bank_support_static_router_bundle,
    )
    from .workspace_inputs import (
        validate_active_diagnostic_workspace_binding,
        validate_workspace_provenance,
    )

    root = Path(artifact_root)
    protocol = canonical_consumed_test_protocol()
    assert_launch_files(root, config)
    assert_workspace_resolved_paths(config, root=root)
    canonical_source = None
    with exclusive_run_lock(root):
        recovered = recover_if_possible(root, config=config, protocol=protocol)
        if recovered is not None:
            return recovered
        if (root / "reports/run_state.json").exists():
            raise ProtocolError(
                "S4 existing run state is not an exact recovery boundary."
            )
        cleanup_owned_atomic_temps(root)
        assert_closed_world(root, allow_incomplete=True)
        if not (root / "reports/run_state.json").exists() and set(relative_files(root)) != {
            "config.resolved.yaml",
            "provenance/input_artifacts.json",
        }:
            raise ProtocolError("S4 partial root lacks a registered run state.")
        phase = "ADMISSION"
        try:
            _phase(root, deps, phase)
            assert_input_fence(config)
            workspace = validate_active_diagnostic_workspace_binding(config)
            provenance = validate_workspace_provenance(root, config)
            locks = load_validated_locks(config)
            frame = load_label_free_test_frame(config)
            firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
            firewall["workspace_binding"] = workspace
            partition = build_case_partition(frame, config=config)
            preflight = run_workstation_preflight(root, runtime=getattr(config, "runtime"))

            phase = "INITIAL_SURFACES"
            _phase(root, deps, phase)
            persist_initial_surfaces(
                root,
                config=config,
                protocol=protocol,
                provenance=provenance,
                frame=frame,
                firewall=firewall,
                partition=partition,
            )

            phase = "SOURCE_GENERATION"
            _phase(root, deps, phase)
            canonical_source = (deps.materialize_source or materialize_sources)(
                config, locks.generation, root=root
            )

            phase = "PREDICTION_MATERIALIZATION"
            _phase(root, deps, phase)
            enter_cuda_free_cpu_phase()
            staged_source = (deps.stage_source or stage_sources_for_cpu)(
                canonical_source, config=config, root=root
            )
            prediction = (deps.materialize_predictions or materialize_probabilities)(
                config, staged_source, frame, partition, root=root
            )
            probability = build_exact_nine_surface(prediction)
            persist_probability_surface(
                root, probability_surface_seal_payload(probability)
            )
            predictions = prediction_rows(probability)
            prediction_index = build_prediction_row_index(
                predictions, surface_hash=probability.surface_hash
            )

            phase = "FOLD_PLAN_SEALING"
            _phase(root, deps, phase)
            manager = LabelCapabilityManager(
                Path(getattr(config, "test_manifest_path")),
                frame,
                partition,
                probability_seal_hash=prediction.seal_hash,
            )
            plans = manager.seal_all_fold_plans()
            persist_fold_plans(root, plans)

            phase = "GLOBAL_STATIC_SEALING"
            _phase(root, deps, phase)
            g_selections = []
            for target in tuple(getattr(partition, "folds"))[::5]:
                heldout = target.target_center
                donor_counts = {}
                for source in _candidate_sources(heldout):
                    grant = manager.open_g_static_donor_labels(heldout, source)
                    action = _a1_action_id(source)
                    scoped = score_case_action_counts(
                        prediction_index.for_labels(grant.labels), grant.labels
                    )
                    donor_counts[action] = tuple(
                        row for row in scoped if row.action_id in {"B", action}
                    )
                prerequisite = manager.g_static_donor_grant_seal(heldout)
                selection = select_global_static_action(
                    heldout,
                    donor_counts,
                    prerequisite_seal_hash=prerequisite,
                )
                manager.record_g_static_selection(selection)
                g_selections.append(selection)
            g_seal = seal_global_static_selections(
                g_selections, probability_seal_hash=prediction.seal_hash
            )
            persist_global_static(
                root,
                selections=g_selections,
                seal_payload=g_seal.to_payload(),
            )

            phase = "ROUTE_DECISION_AND_NULL_SEALING"
            _phase(root, deps, phase)
            decisions = []
            null_plans = []
            for fold in partition.folds:
                support_grant = manager.open_fold_support_labels(
                    fold.target_center, fold.fold_ordinal
                )
                support_counts = score_case_action_counts(
                    prediction_index.for_labels(support_grant.labels),
                    support_grant.labels,
                )
                s4 = select_support_static_action(
                    fold,
                    support_counts,
                    prerequisite_seal_hash=support_grant.grant_hash,
                )
                decision = make_route_decision(
                    fold,
                    g_static_seal=g_seal,
                    s4_selection=s4,
                    probability_seal_hash=prediction.seal_hash,
                )
                manager.record_route_decision(decision)
                null = build_null_selection_plan(
                    fold,
                    support_counts,
                    prerequisite_seal_hash=support_grant.grant_hash,
                )
                manager.record_route_null_selection(null)
                decisions.append(decision)
                null_plans.append(null)
            decision_seal = seal_route_decisions(
                decisions,
                partition=partition,
                probability_seal_hash=prediction.seal_hash,
            )
            persist_route_decisions(root, decision_seal)
            null_plan_seal, _null_action_matrix = seal_null_selection_plans(
                root,
                plans=null_plans,
                decision_seal_hash=decision_seal.decision_seal_hash,
                partition_hash=partition.partition_hash,
            )
            # Re-open the durable NPZ and verify its exact member, dtype,
            # shape, semantic array hash, file hash, route plan hashes, and
            # aggregate JSON hash before binding the label capability barrier.
            null_plan_seal, _null_action_matrix = load_null_selection_plan_seal(
                root,
                plans=null_plans,
                decision_seal_hash=decision_seal.decision_seal_hash,
                partition_hash=partition.partition_hash,
            )
            # This is the global protocol barrier: the label manager verifies
            # the exact persisted aggregate surfaces before any route may open
            # evaluation labels that could support another route.
            manager.record_pre_evaluation_aggregate_seals(
                decision_seal, null_plan_seal
            )

            phase = "ROUTE_EVALUATION"
            _phase(root, deps, phase)
            evaluation_counts = []
            for fold in partition.folds:
                grant = manager.open_route_evaluation_labels(
                    fold.target_center, fold.fold_ordinal
                )
                evaluation_counts.extend(
                    score_case_action_counts(
                        prediction_index.for_labels(grant.labels), grant.labels
                    )
                )
            terminal = evaluate_terminal(
                root=root,
                partition=partition,
                decision_seal=decision_seal,
                null_plans=null_plans,
                evaluation_counts=evaluation_counts,
            )
            capability = dict(manager.access_report())
            null_seal = terminal["action_identity_null_seal"]
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction.seal_hash,
                probability_surface_hash=probability.surface_hash,
                capability_report=capability,
                global_static_seal_hash=g_seal.seal_hash,
                decision_seal_hash=decision_seal.decision_seal_hash,
                null_seal_hash=str(null_seal["null_seal_hash"]),
            )
            terminal_seal = terminal["sealed_terminal_evaluation"]
            publication = publication_decision_payload(
                str(terminal_seal["sealed_result_hash"])
            )
            runtime = runtime_summary_payload(
                source_cache=canonical_source,
                prediction=prediction,
                preflight=preflight,
                staged_source=staged_source,
                artifact_root=root,
            )
            persist_terminal_checkpoint(
                root,
                result=terminal,
                capability_report=capability,
                leakage_report=leakage,
                publication_decision=publication,
                runtime_summary=runtime,
            )
            finalize_terminal_checkpoint(root)
            remove_validated_terminal_checkpoint(root)

            phase = "FINALIZATION"
            _phase(root, deps, phase)
            _finalize_bundle(
                root,
                config=config,
                protocol=protocol,
                write_content_index_fn=write_content_index,
                validate_bundle_fn=validate_fixed_bank_support_static_router_bundle,
                run_fresh_fn=run_two_fresh_process_replays,
                persist_fresh_fn=persist_fresh_process_report,
                persist_validation_fn=persist_validation_report,
                write_state_fn=write_state,
                assert_completed_fn=assert_completed_bundle_binding,
            )
        except BaseException as exc:
            write_state(
                root,
                status="FAILED",
                phase=phase,
                error=str(exc) or type(exc).__name__,
                error_class=type(exc).__name__,
            )
            raise
    (deps.cleanup_staging or cleanup_validated_local_stage)(
        config, canonical_source=canonical_source
    )
    return root


def _phase(root: Path, deps: SupportStaticRouterDependencies, phase: str) -> None:
    from .runner_runtime import observe, write_state

    write_state(root, status="RUNNING", phase=phase)
    observe(deps, phase)


def _candidate_sources(target: str) -> tuple[str, ...]:
    from .constants import candidate_sources

    return candidate_sources(target)


def _a1_action_id(source: str) -> str:
    from .constants import a1_action_id

    return a1_action_id(source)


def _finalize_bundle(
    root: Path,
    *,
    config: object,
    protocol: object,
    write_content_index_fn: Callable[..., object],
    validate_bundle_fn: Callable[..., object],
    run_fresh_fn: Callable[..., object],
    persist_fresh_fn: Callable[..., object],
    persist_validation_fn: Callable[..., object],
    write_state_fn: Callable[..., object],
    assert_completed_fn: Callable[..., object],
) -> object:
    """Publish only after pending, fresh-process, and final binding checks.

    Callables are explicit so the finalization order is directly testable
    without launching the 810-cell workstation runtime.
    """

    write_content_index_fn(
        root,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=str(getattr(protocol, "contract_hash")),
    )
    validate_bundle_fn(
        root,
        config=config,
        allow_pending_validation=True,
        skip_fresh_process_report=True,
    )
    fresh = run_fresh_fn(
        root, config_path=Path(getattr(config, "source_path"))
    )
    persist_fresh_fn(root, fresh)
    checks = validate_bundle_fn(
        root,
        config=config,
        allow_pending_validation=True,
    )
    persist_validation_fn(root, checks)
    write_state_fn(root, status="COMPLETE", phase="COMPLETE")
    assert_completed_fn(root, config=config, expected_checks=checks)
    return checks


__all__ = ("run_fixed_bank_support_static_router",)
