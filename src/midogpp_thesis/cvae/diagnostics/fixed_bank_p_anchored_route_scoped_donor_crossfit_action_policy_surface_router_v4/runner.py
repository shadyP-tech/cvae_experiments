"""Phase-ordered one-shot workstation runner for authorized P-DCAPS v4."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.lifecycle import (
    PDCAPSLabelLifecycle,
)
from .config import PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config
from .fresh_process_validation import (
    require_two_fresh_final_validations,
    require_two_fresh_preterminal_validations,
)
from .input_contracts import build_source_snapshot_payload
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_pre_gpu_firewall,
)
from .lineage import build_six_input_binding
from .manifest_labels import read_scoped_manifest_labels
from .gpu_phase import materialize_gpu_phase
from .outer_chunks import persist_and_verify_outer_chunks
from .outer_execution import execute_outer_runtime
from .persistence import (
    FINAL_REPORT_MEMBERS,
    persist_durable_attestation,
    persist_preterminal_bundle,
    persist_terminal_bundle,
)
from .reports import (
    diagnostic_summary_payload,
    label_capability_report_payload,
    leakage_report_payload,
    publication_decision_payload,
    runtime_summary_payload,
    validation_report_payload,
)
from .route_planning import build_route_plan_inventory
from .route_runtime import build_route_runtime, open_all_pseudo_responses
from .run_admission import assert_read_only_run_admission, exclusive_run_lock
from .run_state import write_run_state
from .scratch import (
    PREDICTION_DIRECTORY,
    SOURCE_DIRECTORY,
    ScratchLease,
    cleanup_scratch,
    create_scratch,
)
from .terminal import TerminalEvaluationResult, evaluate_terminal
from .validation import verify_durable_preterminal_attestation
from .workstation import (
    assert_cuda_free_cpu_phase,
    cpu_phase_environment,
    enter_cuda_free_cpu_phase,
    load_validated_workstation_preflight,
    run_workstation_preflight,
)
from .workspace_inputs import validate_workspace_provenance


PhaseObserver = Callable[[str], None]


def run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4(
    config: PAnchoredRouteScopedDonorCrossfitActionPolicySurfaceRouterV4Config,
    *,
    artifact_root: str | Path,
    phase_observer: PhaseObserver | None = None,
) -> Path:
    """Execute exactly one terminal consumed-test diagnostic attempt."""

    root = Path(artifact_root).resolve()
    admission = assert_read_only_run_admission(config, root=root)
    scratch: ScratchLease | None = None
    phase = "BEGIN"
    hashes: dict[str, str] = {}
    with exclusive_run_lock(root, admission=admission):
        try:
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            assert_input_fence(config)
            provenance = validate_workspace_provenance(root, config)
            input_binding = build_six_input_binding(config, provenance)
            hashes["input_binding"] = input_binding.binding_hash
            locks = load_validated_locks(config)
            frame = load_label_free_test_frame(config)
            pre_gpu = validate_pre_gpu_firewall(config, frame, locks)
            source_snapshot = build_source_snapshot_payload()

            phase = "WORKSTATION_PREFLIGHT"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            run_workstation_preflight(root, runtime=config.runtime)
            preflight = load_validated_workstation_preflight(
                root, runtime=config.runtime
            )
            scratch = create_scratch(root, config.runtime)

            phase = "FRESH_PHYSICAL_810_CELLS"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            physical = materialize_gpu_phase(
                config,
                locks.generation,
                frame,
                root=scratch.root / SOURCE_DIRECTORY,
                prediction_scratch_root=scratch.root / PREDICTION_DIRECTORY,
            )
            hashes["physical_surface"] = physical.surface.physical_surface_hash
            enter_cuda_free_cpu_phase()
            assert_cuda_free_cpu_phase()

            inventory = ExpectedRouteInventory.from_label_free_keys(
                tuple(
                    (row.center, row.case_id, row.sample_id) for row in frame.rows
                ),
                manifest_sha256=config.expected_manifest_sha256,
                row_order_hash=config.expected_test_cache_row_order_hash,
            )
            lifecycle = PDCAPSLabelLifecycle(
                lambda keys, role: read_scoped_manifest_labels(
                    config,
                    frame,
                    allowed_keys=frozenset(keys),
                    role=role,
                ),
                protocol_hash=str(config.protocol["protocol_hash"]),
                expected_inventory=inventory,
                require_derived_response_denominators=True,
                require_durable_terminal_attestation=True,
            )

            phase = "ROUTE_SURFACES_AND_PSEUDO_RESPONSES"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            plans = build_route_plan_inventory(inventory, physical.surface)
            routes = build_route_runtime(
                physical_surface=physical.surface,
                lifecycle=lifecycle,
                route_plans=plans,
            )
            pseudo = open_all_pseudo_responses(lifecycle, routes)
            hashes["route_runtime"] = routes.runtime_hash
            hashes["pseudo_runtime"] = pseudo.runtime_hash

            phase = "FOUR_SPAWN_OUTER_H_WORKERS"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            with cpu_phase_environment(1):
                outer = execute_outer_runtime(
                    routes,
                    pseudo,
                    use_processes=True,
                    max_workers=int(config.runtime["outer_process_workers"]),
                )
            assert_cuda_free_cpu_phase()
            outer_chunks = persist_and_verify_outer_chunks(
                scratch, outer.results
            )
            hashes["outer_science"] = outer.science_hash
            hashes["outer_runtime"] = outer.runtime_hash
            hashes["outer_chunk_manifest"] = str(
                outer_chunks["manifest_hash"]
            )

            phase = "DURABLE_PRETERMINAL_BARRIER"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            preterminal = outer.preterminal
            preterminal_seal = lifecycle.attest_preterminal(
                preterminal.output_hashes
            )
            hashes["preterminal_output"] = (
                preterminal.output_hashes.output_bundle_hash
            )
            hashes["preterminal_seal"] = str(preterminal_seal["seal_hash"])
            preterminal_receipt = persist_preterminal_bundle(
                root,
                surface_set=routes.surface_set,
                identity_results=tuple(
                    row.identity_result for row in preterminal.outer_results
                ),
                cyclic_results=tuple(
                    row.cyclic_result for row in preterminal.outer_results
                ),
                identity_legacy_controls=tuple(
                    row.identity_legacy_control
                    for row in preterminal.outer_results
                ),
                cyclic_legacy_controls=tuple(
                    row.cyclic_legacy_control
                    for row in preterminal.outer_results
                ),
                identity_admissions=outer.identity_admissions,
                cyclic_admissions=outer.cyclic_admissions,
                method_decisions=tuple(
                    decision
                    for row in preterminal.outer_results
                    for decision in row.decisions
                ),
                method_compositions=tuple(
                    composition
                    for row in preterminal.outer_results
                    for composition in row.compositions
                ),
                output_hashes=preterminal.output_hashes,
                preterminal_seal=preterminal_seal,
                lifecycle_audit=lifecycle.audit_payload(),
                config_hash=config.config_hash,
                input_binding=input_binding.to_payload(),
                source_snapshot=source_snapshot,
            )
            hashes["preterminal_content_index"] = str(
                preterminal_receipt["preterminal_content_index_hash"]
            )

            phase = "TWO_FRESH_PRETERMINAL_VALIDATORS"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            durable = require_two_fresh_preterminal_validations(root)
            persist_durable_attestation(root, durable)
            barrier = verify_durable_preterminal_attestation(root, durable)
            hashes["durable_preterminal_attestation"] = durable.attestation_hash
            hashes["durable_preterminal_barrier"] = str(
                barrier["preterminal_content_index_hash"]
            )

            phase = "TERMINAL_LABELS_AND_DIAGNOSTICS"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            lifecycle.begin_terminal_evaluation(durable)
            capabilities = tuple(
                lifecycle.open_terminal_center_labels(center)
                for center in inventory.centers
            )
            result = evaluate_terminal(
                identity_results=tuple(
                    row.identity_result for row in preterminal.outer_results
                ),
                surface_set=routes.surface_set,
                compositions=tuple(
                    composition
                    for row in preterminal.outer_results
                    for composition in row.compositions
                ),
                capabilities=capabilities,
                preterminal_seal_hash=str(preterminal_seal["seal_hash"]),
            )
            lifecycle_audit = lifecycle.audit_payload()
            final_reports = _final_reports(
                result=result,
                lifecycle_audit=lifecycle_audit,
                input_binding=input_binding.to_payload(),
                pre_gpu=pre_gpu,
                source_snapshot=source_snapshot,
                preflight=preflight,
                physical_surface_hash=physical.surface.physical_surface_hash,
                route_runtime_hash=routes.runtime_hash,
                pseudo_runtime_hash=pseudo.runtime_hash,
                outer_execution={
                    "execution_mode": outer.execution_mode,
                    "worker_count": outer.worker_count,
                    "runtime_hash": outer.runtime_hash,
                    "science_hash": outer.science_hash,
                },
                outer_chunk_manifest=outer_chunks,
            )
            terminal_receipt = persist_terminal_bundle(
                root, result, final_reports=final_reports
            )
            hashes["terminal_result"] = result.result_hash
            hashes["final_content_index"] = str(
                terminal_receipt["final_content_index_hash"]
            )

            phase = "TWO_FRESH_FINAL_VALIDATORS"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )
            final_attestation = require_two_fresh_final_validations(root)
            validation_report = validation_report_payload(final_attestation)
            atomic_json(
                root / "reports/validation_report.json", validation_report
            )
            hashes["final_fresh_process_attestation"] = str(
                final_attestation["attestation_hash"]
            )
            hashes["validation_report"] = str(validation_report["report_hash"])

            cleanup_scratch(scratch, artifact_root=root)
            scratch = None
            phase = "COMPLETE"
            _advance(
                root,
                config=config,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
                status="COMPLETE",
            )
            return root
        except BaseException as exc:
            cleanup_error = _cleanup_failed_scratch(scratch, root=root)
            detail = str(exc)
            if cleanup_error is not None:
                detail = f"{detail}; scratch_cleanup={cleanup_error}"
            write_run_state(
                root,
                config_hash=config.config_hash,
                status="FAILED",
                phase=phase,
                bound_hashes=hashes,
                error_class=type(exc).__name__,
                error=detail,
            )
            raise


def _final_reports(
    *,
    result: TerminalEvaluationResult,
    lifecycle_audit: Mapping[str, object],
    input_binding: Mapping[str, object],
    pre_gpu: Mapping[str, object],
    source_snapshot: Mapping[str, object],
    preflight: Mapping[str, object],
    physical_surface_hash: str,
    route_runtime_hash: str,
    pseudo_runtime_hash: str,
    outer_execution: Mapping[str, object],
    outer_chunk_manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    reports: dict[str, Mapping[str, object]] = {
        "reports/diagnostic_summary.json": diagnostic_summary_payload(result),
        "reports/label_capability_report.json": label_capability_report_payload(
            lifecycle_audit
        ),
        "reports/leakage_report.json": leakage_report_payload(
            input_binding=input_binding,
            pre_gpu_firewall=pre_gpu,
            lifecycle_audit=lifecycle_audit,
            source_snapshot=source_snapshot,
        ),
        "reports/publication_decision.json": publication_decision_payload(result),
        "reports/runtime_summary.json": runtime_summary_payload(
            preflight=preflight,
            physical_surface_hash=physical_surface_hash,
            route_runtime_hash=route_runtime_hash,
            pseudo_runtime_hash=pseudo_runtime_hash,
            outer_execution=outer_execution,
            outer_chunk_manifest=outer_chunk_manifest,
        ),
    }
    if set(reports) != set(FINAL_REPORT_MEMBERS):
        raise ProtocolError("P-DCAPS v4 runner report inventory drifted.")
    return reports


def _advance(
    root: Path,
    *,
    config: object,
    phase: str,
    hashes: Mapping[str, str],
    observer: PhaseObserver | None,
    status: str = "RUNNING",
) -> None:
    if observer is not None:
        observer(phase)
    print(f"[pdcaps-v4] phase={phase}", flush=True)
    write_run_state(
        root,
        config_hash=str(getattr(config, "config_hash")),
        status=status,
        phase=phase,
        bound_hashes=hashes,
    )


def _cleanup_failed_scratch(
    scratch: ScratchLease | None, *, root: Path
) -> str | None:
    if scratch is None or not scratch.root.exists():
        return None
    try:
        cleanup_scratch(scratch, artifact_root=root)
    except BaseException as exc:  # preserve the scientific failure as primary
        return f"{type(exc).__name__}: {exc}"
    return None


__all__ = (
    "run_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4",
)
