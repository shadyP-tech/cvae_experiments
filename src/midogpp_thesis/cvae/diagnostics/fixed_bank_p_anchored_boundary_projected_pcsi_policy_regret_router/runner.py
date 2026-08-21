"""Thin, phase-ordered runner for the terminal PCSI-PARC diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

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
    persist_phase_telemetry,
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
from .reports import (
    assert_transport_authorization_lineage_valid,
    leakage_report_payload,
    publication_decision_payload,
)
from .run_admission import (
    assert_launch_files,
    assert_no_partial_state,
    assert_workspace_resolved_paths,
    exclusive_run_lock,
    reject_existing_run_state,
    write_state,
)
from .scratch import cleanup_scratch
from .telemetry import PhaseTelemetryRecorder
from .validation import (
    build_validation_checks_payload,
    verify_completed_attested_bundle,
)
from .workstation import assert_cuda_free_cpu_phase


def run_p_anchored_boundary_projected_pcsi_policy_regret_router(
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
        telemetry = PhaseTelemetryRecorder()
        telemetry.begin(phase)
        try:
            _observe(phase_observer, phase)
            assert_input_fence(config)
            workspace_binding = validate_active_workspace_binding(config)
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
            telemetry.transition(
                phase, {"input_artifact_count": len(provenance)}
            )
            preflight = run_workstation_preflight(
                root, runtime=getattr(config, "runtime")
            )

            phase = "TWO_PERSISTENT_A5000_WORKERS_THEN_EXACT_810_CLASSIFIER_FITS"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            telemetry.transition(phase)
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
            telemetry.finish(
                {
                    "physical_probability_cell_count": len(
                        physical.prediction.store.cells
                    )
                }
            )

            phase = "OUTER_LOO_ENDPOINTS_TARGET_POSTERIORS_AND_DONOR_VETO_ROUTING"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            preterminal = build_preterminal_result(
                surface,
                lambda allowed, role: read_scoped_manifest_labels(
                    config, frame, allowed_keys=allowed, role=role
                ),
                use_processes=True,
                phase_observer=_engine_phase_observer(
                    telemetry, external=phase_observer
                ),
            )
            assert_transport_authorization_lineage_valid(preterminal)
            persist_preterminal(root, preterminal)

            phase = "TERMINAL_LABELS_METRICS_ORACLES_AND_CONTROLS"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            telemetry.begin(phase)
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
            telemetry.finish(
                {"terminal_case_count": len(preterminal.plans.outer_plans)}
            )
            telemetry_payload = telemetry.payload()
            persist_phase_telemetry(root, telemetry_payload)

            phase = "CONTENT_FIRST_TWO_FRESH_PROCESS_VALIDATION"
            write_state(root, status="RUNNING", phase=phase)
            _observe(phase_observer, phase)
            content = write_content_index(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.protocol_hash,
            )
            checks = build_validation_checks_payload(
                content=content,
                config=config,
                protocol=protocol,
                workspace_binding=workspace_binding,
                provenance=provenance,
                pre_gpu=pre_gpu,
                preflight=preflight,
                telemetry=telemetry_payload,
                prediction=physical.prediction,
                preterminal=preterminal,
                terminal=terminal,
            )
            attested = require_two_fresh_process_validations(
                root, expected_checks=checks
            )
            persist_validation_report(root, attested)
            verify_completed_attested_bundle(
                root,
                config=config,
                expected_checks=checks,
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


def _engine_phase_observer(
    recorder: PhaseTelemetryRecorder,
    *,
    external: Callable[[str], None] | None,
) -> Callable[[str, Mapping[str, int]], None]:
    phase_names = {
        "physical": "PHYSICAL_FINGERPRINT_CONSTRUCTION",
        "endpoint": "OUTER_LOO_ENDPOINT_RECONSTRUCTION",
        "posterior": "TARGET_LOCAL_POSTERIOR_FITS",
        "donor_utility": "PROJECTED_RAW_AND_LEGACY_UTILITY_FITS",
        "policy_replay": "H_J_DOUBLE_EXCLUDED_POLICY_REPLAYS",
    }

    def observe(event: str, counts: Mapping[str, int]) -> None:
        _observe(external, event.upper())
        if event.endswith("_started"):
            family = event.removesuffix("_started")
            recorder.begin(phase_names[family])
            return
        if not event.endswith("_completed"):
            raise RuntimeError(f"Unknown PCSI-PARC engine phase event: {event}.")
        family = event.removesuffix("_completed")
        completed: dict[str, int] = {}
        if family == "endpoint":
            completed = {
                "whole_case_route_count": int(counts["outer_route_count"]),
                "outer_endpoint_model_fit_count": int(counts["model_fit_count"]),
            }
        elif family == "posterior":
            completed = {
                "target_posterior_model_fit_count": int(counts["model_fit_count"])
            }
        elif family == "donor_utility":
            completed = {"utility_model_fit_count": int(counts["model_fit_count"])}
        elif family == "policy_replay":
            completed = {"policy_replay_count": int(counts["policy_replay_count"])}
        recorder.finish(completed)

    return observe


run_pcsi = run_p_anchored_boundary_projected_pcsi_policy_regret_router


__all__ = ("run_p_anchored_boundary_projected_pcsi_policy_regret_router", "run_pcsi")
