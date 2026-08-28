"""Thin production orchestrator for the SCEPTRE v4 terminal diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Mapping

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import read_json

from .config import SceptreV4Config
from .execution.authorization_lease import AuthorizationLease, LEASE_MEMBER
from .execution.persistence import (
    persist_durable_attestation,
    persist_failure_report,
    persist_final_validation,
    persist_preterminal_bundle,
    persist_terminal_bundle,
)
from .execution.production_phases import (
    evaluate_terminal_policy,
    form_route_policy,
    freeze_development,
    materialize_physical_surfaces,
)
from .execution.run_state import write_run_state
from .execution.services import DEFAULT_PRODUCTION_SERVICES, ProductionServices
from .execution.validation import validate_complete_bundle
from .identity import (
    EXPERIMENT_ID,
    POLICY_TRANSITION,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    canonical_hash,
)
from .source_seal import source_snapshot_identity


IMPLEMENTED_COMPONENTS = (
    "strict_outer_nested_lodo_expert_ranking_prior",
    "full_eight_member_candidate_set_freeze",
    "zero_anchored_support_empirical_bayes_tournament",
    "support_selected_candidate_paired_whole_case_posterior",
    "same_member_or_exact_b_confirmation_gate",
    "global_45_fold_phase_manager",
    "manager_owned_role_label_broker",
    "fresh_v4_owned_physical_source_and_prediction_surfaces",
    "sealed_route_policy_round_trip",
    "two_fresh_process_preterminal_and_final_validation_barriers",
    "terminal_only_route_minus_exact_b_evaluator",
)


def inspect_sceptre_v4(config: SceptreV4Config) -> Mapping[str, object]:
    """Inspect executable identities without resolving or opening any input."""

    _require_executable(config)
    source = dict(source_snapshot_identity())
    body = {
        "schema_version": "sceptre_v4_executable_implementation_inspection_v1",
        "status": "EXECUTABLE_AUTHORIZED_UNPROBED",
        "experiment_id": EXPERIMENT_ID,
        "config_hash": config.config_hash,
        "protocol_hash": config.protocol["protocol_hash"],
        "source_snapshot_manifest_sha256": source[
            "source_snapshot_manifest_sha256"
        ],
        "source_snapshot_tree_sha256": source["source_snapshot_tree_sha256"],
        "source_snapshot_member_count": source["source_snapshot_member_count"],
        "implemented_components": list(IMPLEMENTED_COMPONENTS),
        "policy_transition": POLICY_TRANSITION,
        "execution_authorized": True,
        "consumed_test_reuse_authorized": True,
        "execution_amendment_declared": True,
        "authorization_lease_probed": False,
        "paths_resolved": False,
        "hardware_probed": False,
        "filesystem_mutations": 0,
        "target_cache_opened": False,
        "target_manifest_opened": False,
        "target_labels_opened": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "nelbo_compatibility_claimed": False,
    }
    return {**body, "inspection_hash": canonical_hash(body)}


def inspect_planned_sceptre_v4(config: SceptreV4Config) -> Mapping[str, object]:
    """Compatibility alias for the original CLI inspection switch."""

    return inspect_sceptre_v4(config)


def dry_run_sceptre_v4(
    config: SceptreV4Config,
    *,
    artifact_root: str | Path,
    services: ProductionServices = DEFAULT_PRODUCTION_SERVICES,
) -> Mapping[str, object]:
    """Run every read-only admission gate without claiming the one-shot lease."""

    _require_executable(config)
    root = _exact_artifact_root(config, artifact_root)
    admission, _validated, input_binding, runtime = services.admit(
        config, artifact_root=root
    )
    body = {
        "schema_version": "sceptre_v4_mutation_free_dry_run_v1",
        "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "artifact_root": str(root),
        "config_hash": config.config_hash,
        "admission_hash": admission.admission_hash,
        "input_binding_hash": canonical_hash(dict(input_binding)),
        "workstation_preflight_hash": admission.workstation_preflight_hash,
        "worker_runtime_smoke_hash": admission.worker_runtime_smoke_hash,
        "source_snapshot_tree_sha256": admission.source_snapshot_tree_sha256,
        "execution_amendment_sha256": admission.execution_amendment_sha256,
        "runtime_hash": canonical_hash(dict(runtime)),
        "authorization_lease_claimed": False,
        "filesystem_mutations": 0,
        "target_labels_opened": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "nelbo_compatibility_claimed": False,
    }
    return {**body, "dry_run_hash": canonical_hash(body)}


def run_sceptre_v4(
    config: SceptreV4Config,
    *,
    artifact_root: str | Path,
    phase_observer: Callable[[str, str], None] | None = None,
    services: ProductionServices = DEFAULT_PRODUCTION_SERVICES,
) -> str:
    """Execute the single-use, consumed-test SCEPTRE v4 diagnostic."""

    _require_executable(config)
    root = _exact_artifact_root(config, artifact_root)
    _announce("BEGIN_ADMISSION")
    admission, validated, input_binding, runtime = services.admit(
        config, artifact_root=root
    )

    lease: AuthorizationLease | None = None
    scratch = None
    bound_hashes: dict[str, str] = {
        "admission_hash": admission.admission_hash,
        "input_binding_hash": admission.input_binding_hash,
        "worker_runtime_smoke_hash": admission.worker_runtime_smoke_hash,
    }
    current_phase = "BEGIN"

    def advance(phase: str, role: str | None = None, digest: str | None = None) -> None:
        nonlocal current_phase
        if not isinstance(lease, AuthorizationLease):
            raise ProtocolError("SCEPTRE v4 run state lacks its typed lease.")
        if role is not None and digest is not None:
            bound_hashes[role] = digest
        write_run_state(
            root,
            authorization_lease=lease,
            config_hash=config.config_hash,
            status="RUNNING",
            phase=phase,
            bound_hashes=bound_hashes,
        )
        current_phase = phase
        _announce(phase)
        if phase_observer is not None:
            phase_observer(phase, "" if digest is None else digest)

    try:
        # This atomic lease-directory creation is deliberately the first mutation.
        claimed = services.claim_lease(
            config, admission_hash=admission.admission_hash
        )
        if not isinstance(claimed, AuthorizationLease):
            raise ProtocolError("SCEPTRE v4 lease service returned an invalid lease.")
        lease = claimed
        advance("BEGIN")
        scratch = services.create_scratch(
            root,
            config.runtime,
            authorization_lease=lease,
            admitted=admission.scratch,
        )
        advance(
            "WORKSTATION_PREFLIGHT",
            "workstation_preflight_hash",
            admission.workstation_preflight_hash,
        )

        partition, development = freeze_development(
            config, validated, services=services
        )
        advance(
            "SOURCE_INNER_DEVELOPMENT_FREEZE",
            "development_replay_hash",
            development.replay_hash,
        )

        physical = materialize_physical_surfaces(
            config,
            validated,
            root=root,
            scratch=scratch,
            attempt_id=lease.lease_hash,
            services=services,
            source_observer=lambda digest: advance(
                "FRESH_PHYSICAL_SOURCE_STREAMS", "source_store_hash", digest
            ),
        )
        advance(
            "FRESH_PHYSICAL_PREDICTION_SURFACE",
            "prediction_store_hash",
            physical.prediction_store_hash,
        )

        def observe_route_phase(phase: str, digest: str) -> None:
            role = {
                "ALL_PROPOSAL_SETS_SEALED": "proposal_set_seal_hash",
                "ALL_SUPPORT_DECISIONS_SEALED": "support_seal_hash",
                "ALL_CONFIRMATION_DECISIONS_SEALED": "policy_seal_hash",
            }.get(phase)
            if role is None:
                raise ProtocolError("SCEPTRE v4 routing emitted an unknown phase.")
            advance(phase, role, digest)

        manager, broker, phases = form_route_policy(
            config,
            validated,
            partition,
            development,
            physical,
            authorization_lease_hash=lease.lease_hash,
            services=services,
            phase_observer=observe_route_phase,
        )
        advance(
            "ROUTE_POLICY_SEALED",
            "route_policy_hash",
            phases.route_policy.policy_artifact_hash,
        )

        lease_payload = read_json(lease.root / LEASE_MEMBER)
        preterminal = persist_preterminal_bundle(
            root,
            config_hash=config.config_hash,
            admission=admission.to_payload(),
            input_binding=input_binding,
            authorization_lease=lease_payload,
            runtime=runtime,
            source_store=physical.source_binding,
            prediction_store=physical.prediction_binding,
            prediction_member_hashes=physical.prediction_member_hashes,
            partition=partition,
            development=development,
            phases=phases,
        )
        advance(
            "DURABLE_PRETERMINAL_BARRIER",
            "preterminal_content_index_hash",
            str(preterminal["content_index_hash"]),
        )
        durable = services.validate_preterminal(root)
        persist_durable_attestation(
            root,
            durable,
            preterminal_content_index_hash=str(preterminal["content_index_hash"]),
        )
        advance(
            "TWO_FRESH_PRETERMINAL_VALIDATORS",
            "durable_attestation_hash",
            durable.attestation_hash,
        )

        result = evaluate_terminal_policy(
            manager,
            broker,
            partition,
            development,
            phases,
            physical,
            durable,
            services=services,
        )
        persist_terminal_bundle(
            root,
            result,
            final_label_journal=broker.journal_payload(),
        )
        advance(
            "TERMINAL_LABELS_AND_DIAGNOSTICS",
            "terminal_result_hash",
            result.result_hash,
        )

        final_validations = services.validate_final(root)
        validation_report = persist_final_validation(
            root, validations=final_validations
        )
        advance(
            "TWO_FRESH_FINAL_VALIDATORS",
            "validation_report_hash",
            str(validation_report["report_hash"]),
        )
        complete = validate_complete_bundle(root)
        advance(
            "POSTVALIDATION_INDEX_AUTHENTICATED",
            "complete_reconstruction_hash",
            str(complete["reconstruction_hash"]),
        )
        advance("FINALIZING_AUTHORIZATION")
        services.cleanup_scratch(scratch, artifact_root=root)
        scratch = None
        completed_lease = services.complete_lease(lease)
        if not isinstance(completed_lease, AuthorizationLease):
            raise ProtocolError("SCEPTRE v4 completion returned an invalid lease.")
        lease = completed_lease
        write_run_state(
            root,
            authorization_lease=lease,
            config_hash=config.config_hash,
            status="COMPLETE",
            phase="COMPLETE",
            bound_hashes=bound_hashes,
        )
        _announce("COMPLETE")
        return str(root)
    except BaseException as exc:
        if not isinstance(lease, AuthorizationLease):
            # A partial external lease directory, if created, is itself exhausted.
            raise
        cleanup_error: BaseException | None = None
        try:
            persist_failure_report(
                root,
                config_hash=config.config_hash,
                authorization_lease=lease,
                phase=current_phase,
                bound_hashes=bound_hashes,
                error=exc,
                scratch_root=None if scratch is None else scratch.root,
            )
        except BaseException:
            # The external lease remains the authoritative exhaustion record.
            pass
        if scratch is not None and scratch.root.is_dir() and not scratch.root.is_symlink():
            try:
                services.cleanup_scratch(scratch, artifact_root=root)
            except BaseException as failure:
                cleanup_error = failure
        detail = str(exc)
        if cleanup_error is not None:
            detail = f"{detail}; scratch_cleanup={cleanup_error}"
        lease_error: BaseException | None = None
        if lease.status == "CLAIMED_IN_PROGRESS":
            try:
                failed_lease = services.fail_lease(lease, error=exc)
                if not isinstance(failed_lease, AuthorizationLease):
                    raise ProtocolError(
                        "SCEPTRE v4 failure finalization returned an invalid lease."
                    )
                lease = failed_lease
            except BaseException as failure:
                lease_error = failure
        state_status = (
            "FAILED" if lease.status == "FAILED_EXHAUSTED" else "FINALIZATION_ERROR"
        )
        state_phase = (
            current_phase if state_status == "FAILED" else "FINALIZING_AUTHORIZATION"
        )
        state_error: BaseException | None = None
        try:
            write_run_state(
                root,
                authorization_lease=lease,
                config_hash=config.config_hash,
                status=state_status,
                phase=state_phase,
                bound_hashes=bound_hashes,
                error_class=exc.__class__.__name__,
                error=detail,
            )
        except BaseException as failure:
            state_error = failure
        if lease_error is not None and hasattr(exc, "add_note"):
            exc.add_note(
                "SCEPTRE v4 failed to finalize its exhausted lease: "
                f"{lease_error.__class__.__name__}: {lease_error}"
            )
        if state_error is not None and hasattr(exc, "add_note"):
            exc.add_note(
                "SCEPTRE v4 failed to persist its failure state: "
                f"{state_error.__class__.__name__}: {state_error}"
            )
        raise


def _require_executable(config: object) -> None:
    if (
        not isinstance(config, SceptreV4Config)
        or config.experiment_id != EXPERIMENT_ID
        or config.execution_authorized is not True
        or config.protocol.get("experiment_id") != EXPERIMENT_ID
        or config.claim_boundary.get("publication_status") != PUBLICATION_STATUS
        or config.claim_boundary.get("terminal_decision") != TERMINAL_DECISION
    ):
        raise ProtocolError("SCEPTRE v4 executable identity drifted.")


def _exact_artifact_root(
    config: SceptreV4Config, artifact_root: str | Path
) -> Path:
    root = Path(artifact_root).resolve()
    configured = Path(config.artifact_root)
    if not configured.is_absolute() or configured.resolve() != root:
        raise ProtocolError(
            "SCEPTRE v4 CLI artifact root differs from config.resolved.yaml."
        )
    return root


def _announce(phase: str) -> None:
    print(f"[sceptre-v4] phase={phase}", file=sys.stderr, flush=True)


__all__ = (
    "IMPLEMENTED_COMPONENTS",
    "dry_run_sceptre_v4",
    "inspect_planned_sceptre_v4",
    "inspect_sceptre_v4",
    "run_sceptre_v4",
)
