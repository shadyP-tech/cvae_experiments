"""Thin, phase-ordered runner for the terminal OGDE diagnostic."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError
from .bundle import write_content_index
from .fresh_process_validation import require_two_fresh_process_validations
from .label_capabilities import DualEndpointLabelFirewall
from .persistence import (
    persist_terminal,
    persist_validation_report,
)
from .protocol import build_frozen_science_protocol
from .reports import leakage_report_payload, publication_decision_payload
from .runner_runtime import (
    assert_cuda_free_cpu_phase,
    assert_launch_files,
    assert_no_foreign_or_partial_state,
    assert_workspace_resolved_paths,
    enter_cuda_free_cpu_phase,
    exclusive_run_lock,
    observe,
    reject_existing_run_state,
    write_state,
)
from .runner_services import RunnerServices, read_scoped_manifest_labels
from .runtime_adapter import cleanup_validated_scratch, runtime_summary_payload


def run_fixed_bank_loo_opportunity_gated_dual_endpoint_router(
    config: object,
    *,
    artifact_root: str | Path | None = None,
    services: RunnerServices | None = None,
) -> Path:
    root = Path(artifact_root or getattr(config, "artifact_root"))
    svc = services or RunnerServices()
    protocol = build_frozen_science_protocol()
    assert_workspace_resolved_paths(config, root=root)
    assert_launch_files(root, config)
    for directory in ("arrays", "manifests", "provenance", "reports", "tables"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    with exclusive_run_lock(root):
        reject_existing_run_state(root)
        assert_no_foreign_or_partial_state(root)
        phase = "INPUT_ADMISSION"
        write_state(svc, root, status="RUNNING", phase=phase)
        try:
            observe(svc, "input_admission")
            admitted = svc.admission.admit(root, config, protocol)

            phase = "WORKSTATION_PREFLIGHT"
            write_state(svc, root, status="RUNNING", phase=phase)
            observe(svc, "workstation_preflight")
            preflight = svc.physical.preflight(root, getattr(config, "runtime"))

            phase = "TWO_PERSISTENT_A5000_SOURCE_WORKERS"
            write_state(svc, root, status="RUNNING", phase=phase)
            observe(svc, "source_generation")
            locks = admitted["locks"]
            source_caches = svc.physical.source_streams(
                config, getattr(locks, "generation"), root
            )

            enter_cuda_free_cpu_phase()
            assert_cuda_free_cpu_phase()
            phase = "FOUR_SPAWNED_CPU_CLASSIFIERS_EXACT_810_SEAL"
            write_state(svc, root, status="RUNNING", phase=phase)
            observe(svc, "physical_probability_seal")
            physical = svc.physical.probabilities(
                config,
                getattr(source_caches, "local", source_caches),
                admitted["frame"],
                root,
            )

            phase = "LABEL_FREE_218_PLAN_FEATURE_OPPORTUNITY_SEALS"
            write_state(svc, root, status="RUNNING", phase=phase)
            observe(svc, "label_free_plan_feature_seal")
            physical_seal = physical["seal"]
            label_free = svc.science.label_free(
                root,
                admitted["frame"],
                physical["surface"],
                str(physical_seal["seal_hash"]),
            )
            firewall = DualEndpointLabelFirewall(
                label_free["plan_seal"],
                lambda allowed: read_scoped_manifest_labels(
                    config, admitted["frame"], allowed_keys=allowed
                ),
            )

            phase = "DONOR_THEN_H_MINUS_C_I_R_CONTROLS_AND_218_SEAL"
            write_state(svc, root, status="RUNNING", phase=phase)
            observe(svc, "scoped_route_science")
            route = svc.science.route(
                root=root,
                config=config,
                surface=physical["surface"],
                plans=label_free["plans"],
                features=label_free["features"],
                label_firewall=firewall,
                persisted_plan_seal=label_free["persisted_plan_seal"],
                feature_seal=label_free["feature_seal"],
            )

            phase = "TERMINAL_LABELS_METRICS_ORACLES_SENSITIVITY"
            write_state(svc, root, status="RUNNING", phase=phase)
            observe(svc, "terminal_evaluation")
            terminal_labels = firewall.open_terminal_labels()
            aggregate = route["seals"]["aggregate"]
            terminal = svc.finalization.evaluate(
                surface=physical["surface"],
                plans=label_free["plans"],
                directional_support_gains=route["directional_support_gains"],
                identification_decisions=route["identification_decisions"],
                robust_arm_decisions=route["robust_arm_decisions"],
                method_predictions=route["method_predictions"],
                terminal_labels=terminal_labels,
                aggregate_seal_hash=str(aggregate["seal_hash"]),
                config=config,
            )
            capability = firewall.report_payload()
            terminal_seal = terminal.get("terminal_seal")
            if not isinstance(terminal_seal, dict):
                raise ProtocolError("Dual-endpoint terminal evaluation lacks a seal.")
            leakage = leakage_report_payload(
                physical_prelabel_seal_hash=str(physical_seal["seal_hash"]),
                feature_seal_hash=str(label_free["feature_seal"]["seal_hash"]),
                aggregate_plan_decision_seal_hash=str(aggregate["seal_hash"]),
                capability_report=capability,
            )
            publication = publication_decision_payload(
                str(terminal_seal["seal_hash"]),
                diagnostic_summary=dict(terminal.get("diagnostic_summary", {})),
            )
            runtime_summary = runtime_summary_payload(
                source_cache=source_caches,
                prediction=physical["prediction"],
                preflight=preflight,
                runtime=getattr(config, "runtime"),
            )
            persist_terminal(
                root,
                result=terminal,
                capability_report=capability,
                leakage_report=leakage,
                publication_decision=publication,
                runtime_summary=runtime_summary,
            )

            phase = "CONTENT_FIRST_TWO_FRESH_PROCESS_VALIDATION"
            write_state(svc, root, status="RUNNING", phase=phase)
            write_content_index(
                root,
                config_contract_hash=str(getattr(config, "contract_hash")),
                protocol_contract_hash=protocol.protocol_hash,
            )
            checks = svc.finalization.validate(root, config=config, pending=True)
            attested = require_two_fresh_process_validations(
                root, expected_checks=checks
            )
            persist_validation_report(root, attested)
            write_state(svc, root, status="COMPLETE", phase="COMPLETE")
            svc.finalization.validate(root, config=config, pending=False)
            cleanup_validated_scratch(config)
            return root
        except BaseException as exc:
            write_state(
                svc,
                root,
                status="FAILED",
                phase=phase,
                error=str(exc),
                error_class=type(exc).__name__,
            )
            raise


__all__ = ("run_fixed_bank_loo_opportunity_gated_dual_endpoint_router",)
