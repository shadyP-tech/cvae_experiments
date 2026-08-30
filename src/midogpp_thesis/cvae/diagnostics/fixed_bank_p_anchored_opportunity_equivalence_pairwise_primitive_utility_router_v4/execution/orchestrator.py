"""Thin single-use orchestration for the executable OE-PPUR v4 router."""

from __future__ import annotations

from pathlib import Path

from ....protocol import ProtocolError
from ..authorization_lease import (
    AuthorizationAcquisitionFailureReceipt,
    AuthorizationLeaseClaim,
    assert_authorization_unclaimed,
    claim_authorization_lease,
    discover_authorization_acquisition,
    finalize_failed_authorization,
    record_authorization_outcome,
    record_completion_commit,
    validate_complete_run_bundle,
)
from ..capacity_preflight import preflight_resource_capacity
from ..complete_artifact_validation import build_complete_artifact_seal
from ..execution.authority import (
    LoadedExecutionLaunchAuthority,
    load_execution_launch_authority,
)
from ..execution.fresh_attestation import (
    attest_preterminal_artifact_twice,
    attest_terminal_aggregate_twice,
)
from ..execution.preparation_commit import commit_prepared_output
from ..execution.preterminal_artifact import (
    FINAL_ATTESTATION_MEMBER,
    PRETERMINAL_ATTESTATION_MEMBER,
    persist_attestation_json_exclusive,
    persist_preterminal_artifact,
)
from ..execution.sealed_replay import (
    SealedExecutionReplay,
    build_resolved_config_bundle,
    replay_sealed_execution,
)
from ..execution.services import (
    CanonicalRouterExecutionRequest,
    ServicePreflightRequest,
)
from ..hashing import canonical_hash
from ..identity import (
    EXPECTED_CASE_COUNT,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
)
from ..output_artifact import assemble_final_aggregate_bundle
from ..physical import (
    PhysicalRuntimeConfig,
    load_label_free_test_frame,
    load_validated_upstream_inputs,
    materialize_physical_inputs,
    project_workstation_topology,
)
from ..run_admission import admit_seven_input_execution
from ..run_state import (
    build_run_identity_hash,
    commit_complete_run_state,
    create_single_use_run,
    prepare_complete_run_state,
    transition_run,
    write_exclusive_json,
)
from ..service_factory import prepare_canonical_scientific_service_factory
from ..source_seal import build_source_seal
from ..source_supervision import load_immutable_source_training_surface
from ..terminal import (
    build_physical_manifest_label_reader,
    issue_terminal_aggregate_capability,
    seal_guarded_preterminal_boundary,
)


def run_sealed_oe_ppur_v4(
    repository_root: str | Path,
    *,
    preflight_receipt_path: str | Path,
    launch_authority_path: str | Path,
    scratch_root: str | Path,
    host_id: str | None = None,
) -> Path:
    """Execute the exactly replayed terminal diagnostic once."""

    replay = replay_sealed_execution(
        repository_root,
        preflight_receipt_path=preflight_receipt_path,
        scratch_root=scratch_root,
        host_id=host_id,
    )
    loaded_authority = load_execution_launch_authority(launch_authority_path)
    bundle = build_resolved_config_bundle(replay, loaded_authority)
    return run_admitted_oe_ppur_v4(
        replay,
        loaded_authority=loaded_authority,
        bundle=bundle,
    )


def run_admitted_oe_ppur_v4(
    replay: SealedExecutionReplay,
    *,
    loaded_authority: LoadedExecutionLaunchAuthority,
    bundle: object,
) -> Path:
    """Internal typed edge kept separate from command parsing and sealing."""

    from ..config import ResolvedV4ConfigBundle

    if (
        type(replay) is not SealedExecutionReplay
        or type(loaded_authority) is not LoadedExecutionLaunchAuthority
        or type(bundle) is not ResolvedV4ConfigBundle
    ):
        raise ProtocolError("OE-PPUR v4 orchestrator inputs are untyped.")

    config = bundle.config
    seal = build_source_seal(replay.context.repository_root)
    source_surface = load_immutable_source_training_surface(
        bundle.input_bindings[2].path
    )
    admission = admit_seven_input_execution(
        bundle,
        replay=replay,
        launch_authority=loaded_authority,
        source_seal=seal,
        source_surface=source_surface,
        scratch_root=replay.context.candidate.plan.topology.scratch_root,
    )
    upstream = load_validated_upstream_inputs(
        bundle.input_bindings[0].path,
        bundle.input_bindings[1].path,
    )
    frame = load_label_free_test_frame(bundle.input_bindings[3].path)
    workstation = project_workstation_topology(
        replay.context.candidate.plan.workstation
    )
    capacity = preflight_resource_capacity(
        admission.artifact_root,
        admission.scratch_root,
    )
    factory = prepare_canonical_scientific_service_factory(
        bundle,
        source_seal=seal,
        source_surface=source_surface,
        admission=admission,
    )
    service = factory.build()
    service_preflight = service.preflight(
        ServicePreflightRequest(
            seven_input_contract_hash=config.seven_input_contract_hash,
            protocol_hash=config.protocol_hash,
            source_seal_hash=seal.combined_source_sha256,
            workstation_receipt_hash=workstation.receipt_hash,
        )
    )
    assert_authorization_unclaimed(admission.artifact_root, admission.scratch_root)
    run_identity_hash = build_run_identity_hash(admission)

    lease: AuthorizationLeaseClaim | None = None
    root = admission.artifact_root
    try:
        lease = claim_authorization_lease(
            admission,
            run_identity_hash=run_identity_hash,
        )
        preparation = commit_prepared_output(
            bundle,
            replay=replay,
            launch_authority=loaded_authority,
            run_admission=admission,
            lease=lease,
        )
        create_single_use_run(admission, lease, run_identity_hash=run_identity_hash)
        launch_receipts = {
            "schema_version": "oe_ppur_v4_launch_receipts_v1",
            "seven_input_admission": admission.to_payload(),
            "sealed_execution_replay": replay.to_payload(),
            "prepared_output": preparation.to_payload(),
            "source_seal": seal.to_payload(),
            "source_training_surface_receipt_hash": source_surface.receipt.receipt_hash,
            "source_training_surface_hash": source_surface.surface_hash,
            "upstream_receipt_hash": upstream.receipt_hash,
            "test_frame_hash": frame.frame_hash,
            "workstation": workstation.to_payload(),
            "resource_capacity": capacity.to_payload(),
            "service_factory_identity": factory.identity.to_payload(),
            "service_preflight": service_preflight.to_payload(),
            "scientific_input_count": 7,
            "launch_authority_is_scientific_input": False,
            "target_labels_opened": False,
        }
        write_exclusive_json(root / "reports/launch_receipts.json", launch_receipts)
        transition_run(
            root,
            "INPUTS_SEALED",
            expected_phase="ADMITTED",
            evidence_hash=canonical_hash(launch_receipts),
        )

        physical = materialize_physical_inputs(
            PhysicalRuntimeConfig(upstream.expert_bank_root),
            upstream.generation_lock,
            frame,
            artifact_root=root,
            scratch_root=admission.scratch_root,
        )
        transition_run(
            root,
            "PHYSICAL_PROBABILITIES_MATERIALIZED",
            expected_phase="INPUTS_SEALED",
            evidence_hash=physical.prediction.seal_hash,
        )

        request = CanonicalRouterExecutionRequest(
            frame=frame,
            physical_inputs=physical,
            upstream_receipt_hash=upstream.receipt_hash,
            workstation_receipt=workstation,
        )
        preterminal = service.execute_label_free(request)
        persisted = persist_preterminal_artifact(root, preterminal, request)
        transition_run(
            root,
            "PRETERMINAL_DECISIONS_SEALED",
            expected_phase="PHYSICAL_PROBABILITIES_MATERIALIZED",
            evidence_hash=preterminal.result_hash,
        )

        attestations = attest_preterminal_artifact_twice(persisted)
        boundary = seal_guarded_preterminal_boundary(
            seven_input_contract_hash=config.seven_input_contract_hash,
            source_seal_hash=seal.combined_source_sha256,
            source_training_surface_receipt_hash=(
                source_surface.receipt.receipt_hash
            ),
            decision_ledger_receipt_hash=preterminal.decision_ledger.ledger_hash,
            attestations=attestations,
            case_inventory_sha256=EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
            case_count=EXPECTED_CASE_COUNT,
            exact_p_fallback_count=preterminal.decision_ledger.exact_p_count,
        )
        preterminal_attestation_payload = {
            "schema_version": "oe_ppur_v4_two_fresh_preterminal_attestations_v1",
            "artifact_file_sha256": persisted.artifact_file_sha256,
            "artifact_file_identity_sha256": persisted.artifact_file_identity_sha256,
            "attestations": [row.to_payload() for row in attestations],
            "guarded_boundary": boundary.to_payload(),
            "fresh_process_count": 2,
            "target_labels_opened": False,
        }
        persist_attestation_json_exclusive(
            root,
            PRETERMINAL_ATTESTATION_MEMBER,
            preterminal_attestation_payload,
        )
        transition_run(
            root,
            "PRETERMINAL_ATTESTED",
            expected_phase="PRETERMINAL_DECISIONS_SEALED",
            evidence_hash=boundary.receipt_hash,
        )

        reader = build_physical_manifest_label_reader(
            bundle,
            boundary=boundary,
            preterminal_result=preterminal,
            execution_request=request,
            persisted_artifact=persisted,
            attestations=attestations,
        )
        terminal = issue_terminal_aggregate_capability(
            boundary,
            reader=reader,
        ).score_aggregates()
        terminal_path = root / "reports/terminal_metrics.json"
        write_exclusive_json(terminal_path, terminal.to_payload())
        transition_run(
            root,
            "TERMINAL_AGGREGATES_SCORED",
            expected_phase="PRETERMINAL_ATTESTED",
            evidence_hash=terminal.receipt_hash,
        )

        final_attestation = attest_terminal_aggregate_twice(terminal_path, terminal)
        persist_attestation_json_exclusive(
            root,
            FINAL_ATTESTATION_MEMBER,
            final_attestation.to_payload(),
        )
        transition_run(
            root,
            "FINAL_ATTESTED",
            expected_phase="TERMINAL_AGGREGATES_SCORED",
            evidence_hash=final_attestation.receipt_hash,
        )
        final_bundle = assemble_final_aggregate_bundle(
            root,
            config=config,
            seven_input_contract=replay.input_contract,
            source_seal=seal,
            source_surface=source_surface,
            preterminal_boundary=boundary,
            preterminal_attestations=attestations,
            terminal_receipt=terminal,
            final_attestation=final_attestation,
            run_admission=admission,
            runtime_summary={
                "workstation_receipt_hash": workstation.receipt_hash,
                "resource_capacity_receipt_hash": capacity.receipt_hash,
                "service_factory_identity_hash": factory.identity.receipt_hash,
                "service_preflight_receipt_hash": service_preflight.receipt_hash,
                "prepared_output_receipt_hash": preparation.receipt_hash,
                "persistent_gpu_worker_count": 2,
                "spawn_cpu_worker_count": 4,
                "blas_threads_per_cpu_worker": 1,
                "prediction_storage_dtype": "<f4",
                "reduction_dtype": "<f8",
                "nested_process_pools_allowed": False,
                "cross_run_recovery_allowed": False,
            },
        )
        transition_run(
            root,
            "COMPLETION_PENDING",
            expected_phase="FINAL_ATTESTED",
            evidence_hash=final_bundle.receipt_hash,
        )
        prepared_complete = prepare_complete_run_state(root, final_bundle=final_bundle)
        complete_artifact_seal = build_complete_artifact_seal(
            root,
            expected_complete_state=prepared_complete,
        )
        completion_commit = record_completion_commit(
            lease,
            prepared_state=prepared_complete,
            final_bundle=final_bundle,
            complete_artifact_seal=complete_artifact_seal,
        )
        completed = commit_complete_run_state(
            prepared_complete,
            completion_commit=completion_commit,
        )
        outcome = record_authorization_outcome(
            lease,
            terminal_state=completed,
            final_bundle=final_bundle,
            prepared_state=prepared_complete,
            completion_commit=completion_commit,
            complete_artifact_seal=complete_artifact_seal,
        )
        validate_complete_run_bundle(
            lease,
            terminal_state=completed,
            final_bundle=final_bundle,
            prepared_state=prepared_complete,
            completion_commit=completion_commit,
            complete_artifact_seal=complete_artifact_seal,
            outcome=outcome,
        )
        return root
    except BaseException as exc:
        _finalize_runner_failure(
            lease=lease,
            root=root,
            scratch_root=admission.scratch_root,
            original_error=exc,
        )
        raise


def _finalize_runner_failure(
    *,
    lease: AuthorizationLeaseClaim | None,
    root: Path,
    scratch_root: Path,
    original_error: BaseException,
) -> object | None:
    failure_claim = lease
    if failure_claim is None:
        try:
            discovered = discover_authorization_acquisition(root, scratch_root)
        except BaseException as discovery_error:
            discovery_error.add_note(
                "Original OE-PPUR v4 post-admission failure: "
                f"{type(original_error).__name__}: {original_error}"
            )
            raise discovery_error from original_error
        if type(discovered) is AuthorizationLeaseClaim:
            failure_claim = discovered
        elif type(discovered) is AuthorizationAcquisitionFailureReceipt:
            return discovered
        elif discovered is not None:
            raise ProtocolError(
                "OE-PPUR v4 authorization discovery returned an untyped result."
            )
    if failure_claim is None:
        return None
    return finalize_failed_authorization(
        failure_claim,
        artifact_root=root,
        original_error=original_error,
    )


__all__ = ("run_admitted_oe_ppur_v4", "run_sealed_oe_ppur_v4")
