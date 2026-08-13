"""Thin phase orchestrator for the terminal multi-challenger diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import (
    assert_closed_world,
    cleanup_owned_atomic_temps,
    relative_files,
    write_content_index,
)
from .execution_adapter import (
    build_case_partition,
    cleanup_validated_local_stage,
    materialize_probabilities,
    materialize_sources,
    run_workstation_preflight,
    runtime_summary_payload,
    stage_sources_for_cpu,
)
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_active_diagnostic_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .label_capabilities import MultiChallengerLabelCapabilityManager
from .persistence import (
    finalize_terminal_checkpoint,
    persist_donor_models,
    persist_fold_decisions,
    persist_fold_plans,
    persist_initial_surfaces,
    persist_prelabel_surfaces,
    persist_terminal_checkpoint,
    persist_validation_report,
    remove_validated_terminal_checkpoint,
)
from .probability_surfaces import (
    aggregate_exact_nine,
    build_prelabel_surface,
    seed_probability_rows,
)
from .protocol import canonical_consumed_test_protocol
from .recovery import recovery_capability
from .recovery_provenance import (
    assert_repair_repository_state_unchanged,
    current_repair_repository_state,
    fresh_recovery_audit_payload,
    original_repository_state_from_provenance,
    recovery_audit_payload,
    sealed_recovery_input_hashes,
)
from .reports import leakage_report_payload, publication_decision_payload
from .runner_dependencies import MultiChallengerRouterDependencies
from .runner_runtime import (
    assert_completed_binding,
    assert_launch_files,
    assert_workspace_resolved_paths,
    enter_cuda_free_cpu_phase,
    exclusive_run_lock,
    observe,
    recover_if_possible,
    validate_bundle,
    write_state,
)


def run_fixed_bank_multi_challenger_hierarchical_flip_router(
    config: object, *, artifact_root: Path
) -> Path:
    return _run(
        config,
        artifact_root=artifact_root,
        deps=MultiChallengerRouterDependencies(),
    )


def _run(
    config: object,
    *,
    artifact_root: Path,
    deps: MultiChallengerRouterDependencies,
) -> Path:
    root = Path(artifact_root)
    protocol = canonical_consumed_test_protocol()
    assert_launch_files(root, config)
    assert_workspace_resolved_paths(config, root=root)
    with exclusive_run_lock(root):
        recovery_audit = _launch_recovery_audit(root)
        recovered = recover_if_possible(root, config=config, protocol=protocol)
        if recovered is not None:
            (deps.cleanup_staging or cleanup_validated_local_stage)(config)
            return recovered
        if recovery_audit is None:
            recovery_audit = _fresh_launch_audit(root)
        cleanup_owned_atomic_temps(root)
        assert_closed_world(root, allow_incomplete=True)
        initial_members = {
            "config.resolved.yaml",
            "provenance/input_artifacts.json",
        }
        if (
            not (root / "reports/run_state.json").exists()
            and set(relative_files(root)) != initial_members
        ):
            raise ProtocolError(
                "Multi-challenger partial root lacks a durable run state."
            )
        phase = "ADMISSION"
        try:
            write_state(root, status="RUNNING", phase=phase)
            observe(deps, phase)
            assert_input_fence(config)
            workspace = validate_active_diagnostic_workspace_binding(config)
            provenance = validate_workspace_provenance(root, config)
            locks = load_validated_locks(config)
            frame = load_label_free_test_frame(config)
            firewall = dict(validate_pre_gpu_firewall(config, frame, locks))
            firewall["workspace_binding"] = workspace
            partition = build_case_partition(frame, config=config)
            preflight = run_workstation_preflight(
                root, runtime=getattr(config, "runtime")
            )

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
            prediction = (
                deps.materialize_predictions or materialize_probabilities
            )(
                config,
                staged_source,
                frame,
                partition,
                root=root,
            )

            phase = "PRELABEL_SEALING"
            _phase(root, deps, phase)
            seeds = seed_probability_rows(prediction)
            probabilities = aggregate_exact_nine(seeds)
            prelabel = build_prelabel_surface(
                probabilities, prediction_seal_hash=prediction.seal_hash
            )
            persist_prelabel_surfaces(
                root,
                prediction=prediction,
                seed_rows=seeds,
                probability_surface=probabilities,
                prelabel=prelabel,
            )

            phase = "FOLD_PLAN_SEALING"
            _phase(root, deps, phase)
            manager = MultiChallengerLabelCapabilityManager(
                Path(getattr(config, "test_manifest_path")),
                frame,
                partition,
                prediction_seal_hash=prediction.seal_hash,
                feature_seal_hash=prelabel.feature_surface_hash,
            )
            persist_fold_plans(root, manager.seal_all_fold_plans())

            phase = "DONOR_MODEL_FITTING"
            _phase(root, deps, phase)
            donor = (deps.build_donor_models or _fit_donor_phase)(
                probability_surface=probabilities,
                prelabel=prelabel,
                partition=partition,
                manager=manager,
                config=config,
            )
            persist_donor_models(root, donor)

            phase = "FOLD_DECISION_SEALING"
            _phase(root, deps, phase)
            decisions = (deps.build_fold_decisions or _fit_decision_phase)(
                probability_surface=probabilities,
                prelabel=prelabel,
                partition=partition,
                manager=manager,
                donor_phase=donor,
                config=config,
            )
            persist_fold_decisions(root, decisions)

            phase = "TERMINAL_EVALUATION"
            _phase(root, deps, phase)
            terminal_labels = manager.open_terminal_evaluation_labels()
            terminal = (deps.evaluate_terminal or _evaluate_terminal_phase)(
                probability_surface=probabilities,
                partition=partition,
                terminal_labels=terminal_labels,
                decision_phase=decisions,
                config=config,
            )
            capability = manager.report_payload()
            leakage = leakage_report_payload(
                prediction_seal_hash=prediction.seal_hash,
                feature_seal_hash=prelabel.feature_surface_hash,
                capability_report=capability,
            )
            sealed = terminal.get("sealed_terminal_evaluation")
            if not isinstance(sealed, Mapping):
                raise ProtocolError("Multi-challenger terminal phase omitted its seal.")
            gate = sealed.get("diagnostic_routing_gate")
            if not isinstance(gate, Mapping):
                raise ProtocolError("Multi-challenger terminal gate is absent.")
            publication = publication_decision_payload(
                str(sealed["sealed_result_hash"]), diagnostic_gate=gate
            )
            local_staging = {
                "attempted": True,
                "used": staged_source.root.resolve() != root.resolve(),
                "status": (
                    "STAGED_LOCAL_CPU_CACHE"
                    if staged_source.root.resolve() != root.resolve()
                    else "CANONICAL_ALREADY_LOCAL"
                ),
                "workstation_preflight": dict(preflight),
            }
            runtime_summary = runtime_summary_payload(
                source_cache=canonical_source,
                prediction=prediction,
                preflight=preflight,
                local_staging=local_staging,
                recovery_audit=recovery_audit,
            )
            persist_terminal_checkpoint(
                root,
                result=terminal,
                capability_report=capability,
                leakage_report=leakage,
                publication_decision=publication,
                runtime_summary=runtime_summary,
            )
            finalize_terminal_checkpoint(root)
            remove_validated_terminal_checkpoint(root)

            phase = "FINALIZATION"
            _phase(root, deps, phase)
            write_content_index(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.contract_hash,
            )
            checks = validate_bundle(
                root, config=config, allow_pending_validation=True
            )
            persist_validation_report(root, checks)
            write_state(root, status="COMPLETE", phase="COMPLETE")
            assert_completed_binding(root, config=config, expected_checks=checks)
            if recovery_audit["recovery_used"] is True:
                assert_repair_repository_state_unchanged(recovery_audit)
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


def _launch_recovery_audit(root: Path) -> Mapping[str, object] | None:
    """Capture exact recovery lineage before the FAILED state is overwritten."""

    state_path = root / "reports/run_state.json"
    if not state_path.exists():
        return _fresh_launch_audit(root)
    if state_path.is_symlink() or not state_path.is_file():
        raise ProtocolError("Multi-challenger run state is absent or unsafe.")
    state = read_json(state_path)
    if state.get("status") != "FAILED":
        return None
    capability = recovery_capability(root)
    if capability is None:
        raise ProtocolError(
            "Multi-challenger refuses an unregistered FAILED partial root."
        )
    # The terminal-header defect is a validation-only B->C continuation.  Its
    # exact recovery path runs before any fresh/scientific phase and carries a
    # separate report-level audit, so it must not mint a second A->C
    # mappingproxy audit here.
    if capability.mode == "FINALIZATION_VALIDATION":
        return None
    return recovery_audit_payload(
        original_repository_state=original_repository_state_from_provenance(root),
        repair_repository_state=current_repair_repository_state(),
        **sealed_recovery_input_hashes(root),
    )


def _fresh_launch_audit(root: Path) -> Mapping[str, object]:
    original = dict(original_repository_state_from_provenance(root))
    current = dict(current_repair_repository_state())
    if original != current:
        raise ProtocolError(
            "Multi-challenger fresh run repository state differs from provenance."
        )
    return fresh_recovery_audit_payload()


def _phase(
    root: Path, deps: MultiChallengerRouterDependencies, phase: str
) -> None:
    write_state(root, status="RUNNING", phase=phase)
    observe(deps, phase)


def _fit_donor_phase(**kwargs: object) -> object:
    from .science_donor import fit_h_specific_donor_phase

    return fit_h_specific_donor_phase(**kwargs)


def _fit_decision_phase(**kwargs: object) -> object:
    from .science_decisions import build_fold_decision_phase

    return build_fold_decision_phase(**kwargs)


def _evaluate_terminal_phase(**kwargs: object) -> Mapping[str, object]:
    from .science_terminal import evaluate_terminal_phase

    return evaluate_terminal_phase(**kwargs)


__all__ = (
    "_run",
    "run_fixed_bank_multi_challenger_hierarchical_flip_router",
)
