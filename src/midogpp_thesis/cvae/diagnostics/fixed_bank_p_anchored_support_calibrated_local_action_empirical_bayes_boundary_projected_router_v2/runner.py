"""Single-use workstation runner for executable SCALE-BP v2.

All heavyweight prediction work is completed label-free before any spawned
outer-center worker receives a scoped manifest capability.  If the frozen
pseudo-route admission gate fails, the run stops preterminally and the
authorization remains exhausted; terminal labels are never opened.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

from .authorization_lease import (
    AuthorizationLeaseClaim,
    claim_authorization_lease,
    record_authorization_outcome,
)
from .artifacts import persist_preterminal_bundle
from .artifacts.hashing import canonical_hash
from .artifacts.io import atomic_json, read_json_object
from .config import ScaleBPV2Config
from .execution import run_outer_center_tasks
from .execution_admission import admit_single_use_execution
from .fresh_process_validation import require_two_fresh_process_attestations
from .identity import (
    EXPECTED_CASE_COUNT,
    EXPERIMENT_ID,
    GovernanceError,
)
from .inputs import (
    load_label_free_test_frame,
    load_validated_inputs,
    validate_pre_gpu_firewall,
)
from .label_capabilities import LabelCapabilityJournal
from .label_identity import persist_label_identity_index
from .manifest_labels import ManifestLabelDecoder
from .orchestration import (
    assemble_method_probabilities,
    build_decision_seal_hash,
    build_outer_tasks,
    collect_outer_results,
    finalize_terminal_run,
    persist_launch_receipts,
    persist_preterminal_admission_abort,
    score_terminal_phase,
)
from .physical_memmaps import persist_physical_memmaps
from .protocol import validate_protocol_payload
from .run_state import (
    build_run_identity_hash,
    create_single_use_run,
    mark_failed_exhausted,
    read_run_state,
    transition_run,
)
from .workstation import (
    DETERMINISTIC_PARENT_ENVIRONMENT,
    canonical_workstation_payload,
    preflight_workstation,
)
from .worker_contract import validate_outer_worker_callback


def dry_run_scale_bp_v2(
    config: ScaleBPV2Config,
    *,
    artifact_root: str | Path | None = None,
) -> dict[str, object]:
    """Perform the complete read-only, label-free source/input admission path."""

    root = _requested_artifact_root(config, artifact_root)
    admission = admit_single_use_execution(config, root, config.scratch_root)
    _, worker_contract = validate_outer_worker_callback(spawn_probe=True)
    frame = load_label_free_test_frame(config)
    inputs = load_validated_inputs(config)
    firewall = validate_pre_gpu_firewall(config, frame, inputs)
    body = {
        "schema_version": "scale_bp_v2_no_label_dry_run_v1",
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "artifact_root": str(root),
        "scratch_root": str(config.scratch_root),
        "config_hash": config.contract_hash,
        "protocol_hash": str(config.protocol["protocol_hash"]),
        "frame_hash": frame.frame_hash,
        "cache_binding_hash": frame.cache_binding_hash,
        "input_binding_hash": admission.direct_input_binding_hash,
        "source_fence_receipt_hash": admission.source_fence_receipt_hash,
        "authorization_lease_path": admission.authorization_lease_path,
        "pre_gpu_firewall": dict(firewall),
        "workstation_plan_hash": canonical_workstation_payload()["plan_hash"],
        "worker_contract_receipt_hash": worker_contract["receipt_hash"],
        "spawn_import_probe_performed": True,
        "host_resource_probe_performed": False,
        "manifest_label_rows_decoded": 0,
        "target_labels_opened": False,
        "gpu_work_launched": False,
        "cpu_workers_launched": False,
        "mutation_performed": False,
        "authorization_consumed": False,
        "authorization_consumption_lease_created": False,
        "fresh_evidence": False,
        "terminal_diagnostic_only": True,
    }
    return {**body, "dry_run_hash": canonical_hash(body)}


def run_scale_bp_v2(
    config: ScaleBPV2Config,
    *,
    artifact_root: str | Path | None = None,
    use_processes: bool = True,
) -> Path:
    """Execute one complete authorized run; no retry or recovery is possible."""

    if not isinstance(config, ScaleBPV2Config):
        raise GovernanceError("SCALE-BP v2 runner requires its sealed config.")
    validate_protocol_payload(config.protocol)
    root = _requested_artifact_root(config, artifact_root)
    scratch = config.scratch_root
    protocol_hash = str(config.protocol["protocol_hash"])
    phase_started = time.monotonic()
    timings: dict[str, float] = {}
    run_created = False
    lease_claim: AuthorizationLeaseClaim | None = None

    # Verify the complete closed-world source and direct-input identity before
    # executing even the fresh-spawn import probe.  This is read-only and does
    # not consume the one-shot authority.
    admission = admit_single_use_execution(config, root, scratch)

    # Resolve, pickle, and fresh-spawn-import the canonical callback before the
    # external lease or either run-owned root can be created.
    worker_fn, worker_contract = validate_outer_worker_callback(spawn_probe=True)

    # Full direct-original input and cache validation is deliberately before
    # admission consumption or creation of either single-use root.
    frame = load_label_free_test_frame(config)
    inputs = load_validated_inputs(config)
    firewall = validate_pre_gpu_firewall(config, frame, inputs)
    workstation = preflight_workstation(root, scratch)
    run_identity_hash = build_run_identity_hash(
        config_hash=config.contract_hash,
        protocol_hash=protocol_hash,
        admission_receipt_hash=admission.receipt_hash,
    )
    try:
        lease_claim = claim_authorization_lease(
            admission,
            protocol_hash=protocol_hash,
            claim_boundary_hash=canonical_hash(dict(config.claim_boundary)),
            authorization_amendment_sha256=(
                config.expected_authorization_amendment_sha256
            ),
            run_identity_hash=run_identity_hash,
        )
        create_single_use_run(
            root,
            scratch,
            run_identity_hash=run_identity_hash,
            admission_receipt=admission,
            authorization_lease=lease_claim,
            config_hash=config.contract_hash,
            protocol_hash=protocol_hash,
        )
        run_created = True
        timings["read_only_input_and_host_preflight"] = time.monotonic() - phase_started
        persist_launch_receipts(
            root,
            config=config,
            inputs=inputs,
            frame=frame,
            firewall=firewall,
            workstation=workstation.to_payload(),
            admission_hash=admission.receipt_hash,
            source_fence_hash=admission.source_fence_receipt_hash,
            run_identity_hash=run_identity_hash,
            worker_contract=worker_contract,
            authorization_lease_claim_hash=lease_claim.claim_hash,
        )
        protocol_manifest = read_json_object(root / "manifests/protocol_manifest.json")
        transition_run(
            root,
            "INPUTS_SEALED",
            expected_phase="ADMITTED",
            evidence_hash=str(protocol_manifest["manifest_hash"]),
        )

        phase_started = time.monotonic()
        for key, value in DETERMINISTIC_PARENT_ENVIRONMENT.items():
            os.environ[key] = value
        from .physical_runtime import materialize_physical_bank

        physical = materialize_physical_bank(
            config,
            inputs.generation_lock,
            frame,
            root=root,
            scratch_root=scratch,
        )
        memmaps = persist_physical_memmaps(
            physical.store, root=scratch / "worker_physical_memmaps"
        )
        identity_frame = persist_label_identity_index(
            frame, path=scratch / "worker_label_identity_index.json"
        )
        physical_receipt = {
            **physical.physical_receipt,
            "worker_memmap_bundle_hash": memmaps.bundle_hash,
            "worker_memmap_index_hash": memmaps.index_hash,
            "worker_label_identity_hash": identity_frame.identity_hash,
            "worker_maps_persisted_in_final_artifact": False,
            "scratch_recovery_allowed": False,
        }
        physical_receipt = {
            **physical_receipt,
            "execution_receipt_hash": canonical_hash(physical_receipt),
        }
        atomic_json(root / "reports/physical_materialization.json", physical_receipt)
        del physical
        del inputs
        transition_run(
            root,
            "PHYSICAL_SURFACE_SEALED",
            expected_phase="INPUTS_SEALED",
            evidence_hash=str(physical_receipt["execution_receipt_hash"]),
        )
        timings["label_free_physical_materialization"] = (
            time.monotonic() - phase_started
        )

        phase_started = time.monotonic()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[key] = "1"
        journal = LabelCapabilityJournal(run_identity_hash)
        decoder = ManifestLabelDecoder(frame, config.test_manifest_path)
        tasks, delegations = build_outer_tasks(
            config=config,
            root=root,
            frame=frame,
            decoder=decoder,
            journal=journal,
            memmaps=memmaps,
            identity_index_path=scratch / "worker_label_identity_index.json",
            identity_hash=identity_frame.identity_hash,
            run_identity_hash=run_identity_hash,
            protocol_hash=protocol_hash,
        )
        results = run_outer_center_tasks(
            tasks, worker_fn, use_processes=use_processes
        )
        outer = collect_outer_results(
            root,
            results=results,
            journal=journal,
            delegations=delegations,
        )
        transition_run(
            root,
            "OUTER_CENTERS_COMPLETE",
            expected_phase="PHYSICAL_SURFACE_SEALED",
            evidence_hash=outer.outer_results_hash,
        )
        timings["spawned_outer_center_science"] = time.monotonic() - phase_started

        persist_preterminal_admission_abort(root, outer.route_payloads)

        _, probability_hashes = assemble_method_probabilities(
            outer.route_payloads
        )
        decision_seal_hash = build_decision_seal_hash(
            results, probability_hashes
        )
        journal.seal_decisions(
            decision_seal_hash=decision_seal_hash,
            route_count=EXPECTED_CASE_COUNT,
        )
        preterminal = persist_preterminal_bundle(
            root,
            decision_seal_hash=decision_seal_hash,
            route_count=EXPECTED_CASE_COUNT,
            center_manifests=outer.center_manifests,
            label_capability_journal=journal.audit_payload(),
            required_members=(
                "provenance/authorization_consumption_lease.json",
                "manifests/protocol_manifest.json",
                "reports/pre_gpu_firewall.json",
                "reports/workstation_preflight.json",
                "reports/worker_contract_preflight.json",
                "reports/physical_materialization.json",
            ),
        )
        transition_run(
            root,
            "PRETERMINAL_SEALED",
            expected_phase="OUTER_CENTERS_COMPLETE",
            evidence_hash=str(preterminal["aggregate_seal_hash"]),
        )
        preterminal_attestation = require_two_fresh_process_attestations(
            root, phase="preterminal"
        )
        transition_run(
            root,
            "PRETERMINAL_ATTESTED",
            expected_phase="PRETERMINAL_SEALED",
            evidence_hash=str(preterminal_attestation["attestation_hash"]),
        )

        # Reconstruct the vectors from immutable chunks after attestation so
        # terminal scoring consumes exactly the preterminally sealed bytes.
        terminal_phase = score_terminal_phase(
            root,
            results=results,
            journal=journal,
            decoder=decoder,
            decision_seal_hash=decision_seal_hash,
        )
        timings["aggregate_only_terminal_scoring"] = terminal_phase.elapsed_seconds
        finalize_terminal_run(
            root,
            protocol_hash=protocol_hash,
            decision_seal_hash=decision_seal_hash,
            preterminal=preterminal,
            terminal_phase=terminal_phase,
            results=results,
            phase_timings_seconds=timings,
            memmap_references=memmaps.references,
        )
        completed_state = read_run_state(root)
        record_authorization_outcome(
            lease_claim,
            status="COMPLETE",
            evidence_hash=str(completed_state["state_hash"]),
        )
        return root
    except BaseException as exc:
        failure_evidence_hash = canonical_hash(
            {
                "schema_version": "scale_bp_v2_runner_failure_v1",
                "run_identity_hash": run_identity_hash,
                "error": " ".join(str(exc).split())[:500],
                "error_class": type(exc).__name__,
            }
        )
        if run_created:
            try:
                state = read_run_state(root)
                if state.get("status") == "RUNNING":
                    failed = mark_failed_exhausted(
                        root,
                        error=str(exc),
                        error_class=type(exc).__name__,
                    )
                    failure_evidence_hash = str(failed["state_hash"])
            except BaseException:
                # Preserve the original scientific/governance exception; the
                # immutable lock still records authorization exhaustion.
                pass
        if lease_claim is not None:
            try:
                record_authorization_outcome(
                    lease_claim,
                    status="FAILED_EXHAUSTED",
                    evidence_hash=failure_evidence_hash,
                    error_class=type(exc).__name__,
                )
            except BaseException:
                # The claim directory itself is the irreversible exhaustion
                # transition; an absent/partial outcome never restores authority.
                pass
        raise


def _requested_artifact_root(
    config: ScaleBPV2Config, value: str | Path | None
) -> Path:
    requested = config.artifact_root if value is None else Path(value)
    if (
        not requested.is_absolute()
        or requested.resolve(strict=False) != config.artifact_root
    ):
        raise GovernanceError("SCALE-BP v2 requested artifact root drifted.")
    return requested.resolve(strict=False)


# Long-form alias mirrors the CLI/experiment identity without making cli.py a
# scientific composition root.
run_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2 = (
    run_scale_bp_v2
)


__all__ = (
    "dry_run_scale_bp_v2",
    "run_scale_bp_v2",
    "run_support_calibrated_local_action_empirical_bayes_boundary_projected_router_v2",
)
