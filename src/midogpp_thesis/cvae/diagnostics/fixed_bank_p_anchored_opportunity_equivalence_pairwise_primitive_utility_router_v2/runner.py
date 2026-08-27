"""Thin single-use runner for the executable OE-PPUR v2 mechanics."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ...protocol import ProtocolError
from .authorization_lease import (
    AuthorizationLeaseClaim,
    assert_authorization_unclaimed,
    claim_authorization_lease,
    record_authorization_outcome,
)
from .config import ResolvedConfigBundle, RouterV2Config
from .execution.probability_matrix import parse_probability_matrix_shards
from .execution.decision_receipts import (
    validate_typed_preterminal_decision_ledger,
)
from .execution_admission import admit_six_input_execution
from .execution_services import ExecutionContext, RouterExecutionServices
from .fresh_process_validation import (
    require_two_fresh_artifact_attestations,
    validate_artifact_fresh_process_attestation,
)
from .hashing import canonical_hash, canonical_json_bytes
from .persistence import atomic_json
from .phase_contracts import (
    AggregateOnlyTerminalReceipt,
    OuterFoldExecutionReceipt,
    ProbabilityMaterializationReceipt,
    ServicePreflightReceipt,
    assert_aggregate_only_payload,
)
from .row_binding import derive_admitted_row_binding
from .run_state import (
    build_run_identity_hash,
    create_single_use_run,
    mark_complete,
    mark_failed_exhausted,
    read_run_state,
    transition_run,
)
from .source_seal import build_source_contract_receipt
from .service_factory import build_canonical_execution_services
from .terminal_capability import issue_terminal_aggregate_capability
from .workstation import preflight_workstation, workstation_plan_payload


def inspect_planned_router(config: RouterV2Config) -> dict[str, object]:
    """Return a path-free implementation receipt without granting authority."""

    if not isinstance(config, RouterV2Config):
        raise ProtocolError("OE-PPUR v2 inspection requires its exact config.")
    source = build_source_contract_receipt()
    body = {
        "schema_version": "oe_ppur_v2_implementation_inspection_v1",
        "experiment_id": config.experiment_id,
        "authorization_state": config.authorization_state,
        "execution_authorized": config.execution_authorized,
        "config_contract_hash": config.contract_hash,
        "protocol_hash": config.protocol["protocol_hash"],
        "current_source_contract_hash": source.combined_source_sha256,
        "current_source_receipt_hash": source.receipt_hash,
        "direct_input_count": len(config.input_artifact_ids),
        "direct_input_artifact_ids": list(config.input_artifact_ids),
        "parsed_probability_matrix_shape": [9928, 7],
        "parsed_probability_matrix_dtype": "<f4",
        "workstation_plan": workstation_plan_payload(),
        "mutation_performed": False,
        "authorization_consumed": False,
        "labels_opened": False,
        "experiment_launched": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def run_oe_ppur_v2(
    resolved: ResolvedConfigBundle | RouterV2Config,
    *,
    scratch_root: str | Path,
) -> Path:
    """Execute one resolved run through the nominal source-sealed service factory.

    ``RouterV2Config`` is accepted only so the checked-in planned contract can
    reject before source, service, or path access.  An authorized execution must
    arrive as the exact workspace-rendered :class:`ResolvedConfigBundle`; callers
    cannot inject paths or a structurally compatible science service.
    """

    # The checked-in planned config rejects before source, input, service, or
    # run-path access.  This is intentionally redundant with six-input admission.
    if isinstance(resolved, RouterV2Config):
        if not resolved.execution_authorized:
            raise ProtocolError(
                "OE-PPUR v2 execution is not authorized by a real amendment."
            )
        raise ProtocolError(
            "OE-PPUR v2 authorized execution requires config.resolved.yaml."
        )
    if type(resolved) is not ResolvedConfigBundle:
        raise ProtocolError("OE-PPUR v2 runner requires a resolved config bundle.")
    config = resolved.config
    if type(config) is not RouterV2Config or not config.execution_authorized:
        raise ProtocolError(
            "OE-PPUR v2 execution is not authorized by a real amendment."
        )

    source = build_source_contract_receipt()
    admission = admit_six_input_execution(
        config,
        input_bindings=resolved.input_bindings,
        artifact_root=resolved.artifact_root,
        scratch_root=scratch_root,
        source_contract_receipt=source,
    )
    row_binding = derive_admitted_row_binding(admission)
    # Hardware facts are always probed live on the production runner.  Tests
    # may replace this function boundary, but callers cannot inject a topology.
    workstation = preflight_workstation(
        admission.artifact_root,
        admission.scratch_root,
    )
    services = build_canonical_execution_services(
        resolved,
        admission=admission,
        source=source,
    )
    if not isinstance(services, RouterExecutionServices):
        raise ProtocolError("OE-PPUR v2 execution services are incomplete.")
    service_preflight = services.preflight(admission, source)
    if not isinstance(service_preflight, ServicePreflightReceipt):
        raise ProtocolError("OE-PPUR v2 service preflight is untyped.")
    assert_authorization_unclaimed(admission.artifact_root, admission.scratch_root)
    run_identity_hash = build_run_identity_hash(
        config_hash=admission.config_contract_hash,
        protocol_hash=admission.protocol_hash,
        source_contract_hash=admission.source_contract_hash,
        admission_receipt_hash=admission.receipt_hash,
    )

    lease: AuthorizationLeaseClaim | None = None
    run_created = False
    root = Path(admission.artifact_root)
    try:
        lease = claim_authorization_lease(
            admission, run_identity_hash=run_identity_hash
        )
        create_single_use_run(
            admission, lease, run_identity_hash=run_identity_hash
        )
        run_created = True
        context = ExecutionContext(
            artifact_root=root,
            scratch_root=Path(admission.scratch_root),
            run_identity_hash=run_identity_hash,
            admission=admission,
            source=source,
            workstation=workstation,
            lease=lease,
        )
        launch_receipts = {
            "schema_version": "oe_ppur_v2_launch_receipts_v1",
            "source": source.to_payload(),
            "six_input_admission": admission.to_payload(),
            "canonical_row_binding": row_binding.to_payload(),
            "service_preflight": service_preflight.to_payload(),
            "workstation": workstation.to_payload(),
            "labels_opened": False,
        }
        atomic_json(root / "reports/launch_receipts.json", launch_receipts)
        transition_run(
            root,
            "INPUTS_SEALED",
            expected_phase="ADMITTED",
            evidence_hash=canonical_hash(launch_receipts),
        )

        materialized = services.materialize_probability_matrix(context)
        if not isinstance(materialized, ProbabilityMaterializationReceipt):
            raise ProtocolError("OE-PPUR v2 probability materialization is untyped.")
        if (
            materialized.row_binding_hash != row_binding.receipt_hash
            or materialized.row_index_sha256 != row_binding.row_index_sha256
            or materialized.row_alignment_receipt_hash
            != row_binding.row_alignment_receipt_hash
        ):
            raise ProtocolError(
                "OE-PPUR v2 probability materialization row lineage drifted."
            )
        matrix = parse_probability_matrix_shards(
            materialized.shards,
            scratch_root=admission.scratch_root,
            row_binding=row_binding,
            gpu_prediction_batch_hash=materialized.gpu_prediction_batch_hash,
            gpu_result_surface_sha256=materialized.gpu_result_surface_sha256,
            ordered_gpu_worker_result_hashes=(
                materialized.ordered_gpu_worker_result_hashes
            ),
            ordered_gpu_result_file_hashes=(
                materialized.ordered_gpu_result_file_hashes
            ),
        )
        atomic_json(
            root / "reports/parsed_probability_matrix.json", matrix.to_payload()
        )
        transition_run(
            root,
            "PROBABILITY_MATRIX_SEALED",
            expected_phase="INPUTS_SEALED",
            evidence_hash=matrix.receipt_hash,
        )

        outer = services.run_outer_folds(context, matrix)
        if (
            not isinstance(outer, OuterFoldExecutionReceipt)
            or outer.parsed_probability_matrix_receipt_hash != matrix.receipt_hash
        ):
            raise ProtocolError("OE-PPUR v2 outer-fold matrix lineage drifted.")
        atomic_json(root / "reports/outer_fold_execution.json", outer.to_payload())
        transition_run(
            root,
            "OUTER_FOLDS_COMPLETE",
            expected_phase="PROBABILITY_MATRIX_SEALED",
            evidence_hash=outer.receipt_hash,
        )

        preterminal = services.seal_preterminal_decisions(context, matrix, outer)
        preterminal = validate_typed_preterminal_decision_ledger(
            preterminal,
            admission_receipt=admission,
            matrix_receipt=matrix,
            outer_fold_receipt=outer,
        )
        preterminal_payload = preterminal.to_payload()
        preterminal_path = (
            root / "manifests/preterminal_decision_ledger_receipt.json"
        )
        atomic_json(preterminal_path, preterminal_payload)
        transition_run(
            root,
            "PRETERMINAL_DECISIONS_SEALED",
            expected_phase="OUTER_FOLDS_COMPLETE",
            evidence_hash=preterminal.receipt_hash,
        )

        preterminal_file_sha256 = _persisted_payload_sha256(preterminal_payload)
        preterminal_attestation = require_two_fresh_artifact_attestations(
            preterminal_path,
            phase="preterminal",
            expected_sealed_receipt_hash=preterminal.receipt_hash,
            expected_file_sha256=preterminal_file_sha256,
        )
        preterminal_attestation = validate_artifact_fresh_process_attestation(
            preterminal_attestation,
            expected_phase="preterminal",
            expected_sealed_receipt_hash=preterminal.receipt_hash,
            expected_file_sha256=preterminal_file_sha256,
        )
        atomic_json(
            root / "reports/preterminal_fresh_process_attestation.json",
            preterminal_attestation.to_payload(),
        )
        transition_run(
            root,
            "PRETERMINAL_ATTESTED",
            expected_phase="PRETERMINAL_DECISIONS_SEALED",
            evidence_hash=preterminal_attestation.receipt_hash,
        )

        terminal_capability = issue_terminal_aggregate_capability(
            preterminal,
            preterminal_attestation,
            scorer=services.build_terminal_scorer(context),
        )
        terminal = terminal_capability.score_aggregates()
        if (
            not isinstance(terminal, AggregateOnlyTerminalReceipt)
            or terminal.preterminal_attestation_hash
            != preterminal_attestation.receipt_hash
        ):
            raise ProtocolError("OE-PPUR v2 terminal aggregate lineage drifted.")
        terminal_payload = terminal.to_payload()
        assert_aggregate_only_payload(terminal_payload)
        terminal_path = root / "reports/terminal_metrics.json"
        atomic_json(terminal_path, terminal_payload)
        transition_run(
            root,
            "TERMINAL_AGGREGATES_SCORED",
            expected_phase="PRETERMINAL_ATTESTED",
            evidence_hash=terminal.receipt_hash,
        )

        terminal_file_sha256 = _persisted_payload_sha256(terminal_payload)
        final_attestation = require_two_fresh_artifact_attestations(
            terminal_path,
            phase="final",
            expected_sealed_receipt_hash=terminal.receipt_hash,
            expected_file_sha256=terminal_file_sha256,
        )
        final_attestation = validate_artifact_fresh_process_attestation(
            final_attestation,
            expected_phase="final",
            expected_sealed_receipt_hash=terminal.receipt_hash,
            expected_file_sha256=terminal_file_sha256,
        )
        atomic_json(
            root / "reports/final_fresh_process_attestation.json",
            final_attestation.to_payload(),
        )
        transition_run(
            root,
            "FINAL_ATTESTED",
            expected_phase="TERMINAL_AGGREGATES_SCORED",
            evidence_hash=final_attestation.receipt_hash,
        )
        completed = mark_complete(
            root, final_attestation_hash=final_attestation.receipt_hash
        )
        record_authorization_outcome(
            lease,
            status="COMPLETE",
            evidence_hash=str(completed["state_hash"]),
        )
        return root
    except BaseException as exc:
        if lease is not None:
            failure_hash = canonical_hash(
                {
                    "schema_version": "oe_ppur_v2_runner_failure_v1",
                    "run_identity_hash": run_identity_hash,
                    "error_class": type(exc).__name__,
                    "authorization_exhausted": True,
                }
            )
            outcome_evidence = failure_hash
            if run_created:
                try:
                    state = read_run_state(root)
                    if state.get("status") == "RUNNING":
                        failed = mark_failed_exhausted(
                            root,
                            error_class=type(exc).__name__,
                            evidence_hash=failure_hash,
                        )
                        outcome_evidence = str(failed["state_hash"])
                except BaseException:
                    pass
            try:
                record_authorization_outcome(
                    lease,
                    status="FAILED_EXHAUSTED",
                    evidence_hash=outcome_evidence,
                    error_class=type(exc).__name__,
                )
            except BaseException:
                pass
        raise


run_opportunity_equivalence_pairwise_primitive_utility_router_v2 = run_oe_ppur_v2


def _persisted_payload_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload) + b"\n").hexdigest()


__all__ = (
    "inspect_planned_router",
    "run_oe_ppur_v2",
    "run_opportunity_equivalence_pairwise_primitive_utility_router_v2",
)
