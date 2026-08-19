"""Thin, phase-ordered runner for the terminal PUMR diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .bundle import write_content_index
from .engine import build_preterminal_result
from .evaluation import evaluate_terminal
from .fresh_process_validation import require_two_fresh_process_validations
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
from .protocol import build_frozen_protocol
from .reports import leakage_report_payload, publication_decision_payload
from .run_admission import (
    assert_launch_files,
    assert_no_partial_state,
    assert_workspace_resolved_paths,
    exclusive_run_lock,
    reject_existing_run_state,
    write_state,
)
from .scratch import cleanup_scratch
from .validation import validate_p_anchored_crossfit_posterior_utility_margin_router_bundle
from .workstation import assert_cuda_free_cpu_phase


def run_p_anchored_crossfit_posterior_utility_margin_router(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    phase_observer: Callable[[str], None] | None = None,
) -> Path:
    root = Path(artifact_root or getattr(config, "artifact_root")).resolve()
    protocol = build_frozen_protocol()
    assert_workspace_resolved_paths(config, root=root)
    assert_launch_files(root, config)
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
            pre_gpu = dict(validate_pre_gpu_firewall(config, frame, locks))
            persist_admission(
                root,
                config=config,
                protocol=protocol,
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

            phase = "TWO_PERSISTENT_A5000_WORKERS_THEN_EXACT_810_CLASSIFIER_FITS"
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

            phase = "OUTER_LOO_ENDPOINTS_FIVE_FOLD_POSTERIORS_AND_NESTED_MARGIN_ROUTING"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            preterminal = build_preterminal_result(
                surface,
                lambda allowed, role: read_scoped_manifest_labels(
                    config, frame, allowed_keys=allowed, role=role
                ),
                use_processes=True,
            )
            persist_preterminal(root, preterminal)

            phase = "TERMINAL_LABELS_METRICS_ORACLES_AND_CONTROLS"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            terminal = evaluate_terminal(preterminal)
            leakage = leakage_report_payload(
                probability_surface_hash=surface.surface_hash,
                preterminal=preterminal,
                capability_report=terminal.capability_report,
            )
            publication = publication_decision_payload(terminal)
            runtime_summary = runtime_summary_payload(
                physical,
                preflight=preflight,
                runtime=getattr(config, "runtime"),
            )
            persist_terminal(
                root,
                terminal=terminal,
                leakage_report=leakage,
                publication_decision=publication,
                runtime_summary=runtime_summary,
            )

            phase = "CONTENT_FIRST_TWO_FRESH_PROCESS_VALIDATION"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            write_content_index(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.protocol_hash,
            )
            checks = validate_p_anchored_crossfit_posterior_utility_margin_router_bundle(
                root, config=config, allow_pending_validation=True
            )
            attested = require_two_fresh_process_validations(
                root, expected_checks=checks
            )
            persist_validation_report(root, attested)
            validate_p_anchored_crossfit_posterior_utility_margin_router_bundle(
                root, config=config, allow_pending_validation=False
            )
            cleanup_scratch(
                physical.scratch,
                config=config,
                artifact_root=root,
            )
            write_state(root, status="COMPLETE", phase="COMPLETE")
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


run_pumr = run_p_anchored_crossfit_posterior_utility_margin_router


__all__ = ("run_p_anchored_crossfit_posterior_utility_margin_router", "run_pumr")
