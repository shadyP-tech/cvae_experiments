"""Thin, phase-ordered workstation runner for repaired CBPUPR v3."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import write_content_index, write_preterminal_content_index
from .engine import build_preterminal_result
from .evaluation import evaluate_terminal
from .execution_admission import assert_v3_execution_authorized
from .fresh_process_validation import (
    require_two_fresh_preterminal_process_validations,
    require_two_fresh_process_validations,
    validation_report_payload,
)
from .hashing import canonical_hash
from .inputs import (
    assert_input_fence,
    load_label_free_test_frame,
    load_validated_locks,
    validate_active_workspace_binding,
    validate_pre_gpu_firewall,
    validate_workspace_provenance,
)
from .manifest_labels import read_scoped_manifest_labels
from .persistence import (
    persist_admission,
    persist_label_capability_report,
    persist_physical_surface,
    persist_preterminal,
    persist_terminal,
    persist_validation_report,
)
from .physical_runtime import (
    build_surface,
    materialize_physical_inputs,
    probability_index_rows,
    runtime_summary_payload,
)
from .preflight import run_workstation_preflight
from .preterminal_gate import (
    persist_preterminal_capability_report,
    persist_preterminal_validation_report,
    persist_preterminal_validation_seal,
    preterminal_validation_report_payload,
    validate_preterminal_gate_artifacts,
)
from .preterminal_validation import (
    PRETERMINAL_VALIDATION_PHASE,
    validate_preterminal_bundle,
    verify_preterminal_attested_bundle,
)
from .reports import leakage_report_payload, publication_decision_payload
from .run_admission import (
    assert_launch_files,
    assert_no_partial_state,
    assert_workspace_resolved_paths,
    exclusive_run_lock,
    reject_failed_predecessor_execution,
    reject_existing_run_state,
    write_state,
)
from .scratch import cleanup_scratch
from .terminal_access_journal import (
    persist_terminal_label_access_intent,
    persist_terminal_label_access_opened_receipt,
)
from .validation import (
    validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle,
    verify_completed_attested_bundle,
)
from .workstation import assert_cuda_free_cpu_phase


def run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    phase_observer: Callable[[str], None] | None = None,
) -> Path:
    reject_failed_predecessor_execution(config)
    root = Path(artifact_root or getattr(config, "artifact_root")).resolve()
    assert_workspace_resolved_paths(config, root=root)
    assert_launch_files(root, config)
    assert_v3_execution_authorized(config)
    for directory in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    with exclusive_run_lock(root):
        reject_existing_run_state(root)
        assert_no_partial_state(root)
        phase = "INPUT_ADMISSION"
        write_state(root, status="RUNNING", phase=phase)
        try:
            _observe(phase_observer, phase)
            assert_input_fence(config)
            validate_active_workspace_binding(config)
            provenance = validate_workspace_provenance(root, config)
            locks = load_validated_locks(config)
            frame = load_label_free_test_frame(config)
            pre_gpu = validate_pre_gpu_firewall(config, frame, locks)
            persist_admission(
                root,
                config=config,
                provenance=provenance,
                frame=frame,
                pre_gpu_firewall=pre_gpu,
            )

            phase = "WORKSTATION_PREFLIGHT"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            preflight = run_workstation_preflight(
                root, runtime=getattr(config, "runtime")
            )

            phase = "FROZEN_GENERATION_AND_EXACT_810_PHYSICAL_CELLS"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            physical = materialize_physical_inputs(
                config, locks.generation, frame, root=root
            )
            assert_cuda_free_cpu_phase()
            surface = build_surface(physical)
            persist_physical_surface(
                root,
                physical=physical,
                surface=surface,
                probability_index=probability_index_rows(physical.prediction),
            )

            phase = "ROUTE_ENDPOINTS_436_POSTERIORS_AND_CANDIDATE_SEAL"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            preterminal = build_preterminal_result(
                surface,
                lambda allowed, role: read_scoped_manifest_labels(
                    config, frame, allowed_keys=allowed, role=role
                ),
                use_processes=True,
            )

            phase = "DURABLE_PRETERMINAL_BARRIER"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            persist_preterminal(root, preterminal)

            phase = PRETERMINAL_VALIDATION_PHASE
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            persist_preterminal_capability_report(
                root, preterminal.candidates.firewall.audit_payload()
            )
            write_preterminal_content_index(root)
            preterminal_checks = validate_preterminal_bundle(
                root, require_attested=False
            )
            preterminal_attestation = (
                require_two_fresh_preterminal_process_validations(
                    root, expected_checks=preterminal_checks
                )
            )
            preterminal_report = preterminal_validation_report_payload(
                preterminal_checks, preterminal_attestation
            )
            persist_preterminal_validation_report(root, preterminal_report)
            persist_preterminal_validation_seal(
                root,
                checks=preterminal_checks,
                attestation=preterminal_attestation,
                report=preterminal_report,
            )
            verify_preterminal_attested_bundle(
                root, expected_checks=preterminal_checks
            )

            phase = "TERMINAL_LABELS_METRICS_AND_CONTROLS"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            labels, capability = _open_terminal_after_durable_preterminal(
                root, preterminal, expected_checks=preterminal_checks
            )
            terminal = evaluate_terminal(
                probabilities=preterminal.decisions.probabilities,
                sample_ids=preterminal.decisions.sample_ids,
                labels=labels,
                aggregate_seal_hash=preterminal.decisions.aggregate_seal_hash,
                diagnostic_summary=preterminal.diagnostic_summary(),
            )
            leakage = leakage_report_payload(
                probability_surface_hash=surface.surface_hash,
                plan_seal_hash=preterminal.candidates.plan_seal.seal_hash,
                aggregate_seal_hash=preterminal.decisions.aggregate_seal_hash,
                capability_report=capability,
            )
            publication = publication_decision_payload(terminal.diagnostic_summary)
            runtime = runtime_summary_payload(
                physical,
                preflight=preflight,
                runtime=getattr(config, "runtime"),
            )
            persist_terminal(
                root,
                terminal=terminal,
                leakage_report=leakage,
                publication_decision=publication,
                runtime_summary=runtime,
                aggregate_seal_hash=preterminal.decisions.aggregate_seal_hash,
            )

            phase = "CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            write_content_index(root)
            checks = validate_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router_bundle(
                root, require_final=False
            )
            attestation = require_two_fresh_process_validations(
                root, expected_checks=checks
            )
            persist_validation_report(
                root, validation_report_payload(checks, attestation)
            )
            cleanup_scratch(physical.scratch, config=config, artifact_root=root)
            write_state(root, status="COMPLETE", phase="COMPLETE")
            verify_completed_attested_bundle(root, expected_checks=checks)
            return root
        except BaseException as exc:
            write_state(
                root,
                status="FAILED",
                phase=phase,
                error=str(exc),
                error_class=type(exc).__name__,
            )
            raise


def _observe(callback: Callable[[str], None] | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


def _open_terminal_after_durable_preterminal(
    root: Path,
    preterminal: object,
    *,
    expected_checks: Mapping[str, object],
) -> tuple[tuple[object, ...], dict[str, object]]:
    """Open terminal labels only after the attested barrier revalidates."""

    barrier = root / "manifests/preterminal_aggregate_seal.json"
    decision_barrier = root / "manifests/decision_barrier.json"
    if not barrier.is_file() or not decision_barrier.is_file():
        raise ProtocolError("CBPUPR terminal opening requires durable barrier files.")
    aggregate_payload = read_json(barrier)
    decision_payload = read_json(decision_barrier)
    candidates = preterminal.candidates
    decisions = preterminal.decisions
    decision_unhashed = {
        key: value
        for key, value in decision_payload.items()
        if key != "decision_barrier_hash"
    }
    if (
        aggregate_payload
        != {
            "schema_version": "fixed_bank_cbpupr_preterminal_aggregate_seal_v1",
            "aggregate_seal_hash": decisions.aggregate_seal_hash,
            "preterminal_hash": preterminal.preterminal_hash,
            "target_evaluation_opened": False,
        }
        or decision_unhashed
        != {
            "schema_version": "fixed_bank_cbpupr_decision_barrier_v1",
            "candidate_seal_hash": candidates.target_candidate_seal_hash,
            "pre_evaluation_seal_hash": candidates.pre_evaluation_seal_hash,
            "replay_calibration_seal_hash": decisions.replay_calibration_seal_hash,
            "pseudo_evaluation_opened_after_candidate_seal": True,
            "target_evaluation_opened": False,
        }
        or decision_payload.get("decision_barrier_hash")
        != canonical_hash(decision_unhashed)
        or expected_checks.get("preterminal_hash")
        != preterminal.preterminal_hash
    ):
        raise ProtocolError("CBPUPR durable preterminal barrier lineage drifted.")
    # The complete scientific replay ran while the recorded phase was still
    # preterminal. Recheck the durable attestation/report/seal after the sole
    # intervening run-state transition and immediately before label access.
    validate_preterminal_gate_artifacts(root, expected_checks=expected_checks)
    access_intent = persist_terminal_label_access_intent(
        root, expected_checks=expected_checks
    )
    firewall = candidates.firewall
    labels = tuple(firewall.open_target_terminal_labels())
    persist_terminal_label_access_opened_receipt(
        root, intent=access_intent, labels=labels
    )
    capability = firewall.audit_payload()
    persist_label_capability_report(root, capability)
    return labels, capability


__all__ = (
    "run_p_anchored_route_scoped_center_balanced_posterior_utility_prefix_router",
)
