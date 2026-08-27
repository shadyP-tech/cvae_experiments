"""One-shot workstation runner for the executable SCEPTRE v2 diagnostic."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    read_json,
    sha256_file,
)

from ..fixed_bank_sceptre_router.hashing import canonical_hash
from ..fixed_bank_sceptre_router.partitions import (
    CaseIdentity,
    build_three_role_partition,
)
from ..fixed_bank_sceptre_router.phase_order import SceptrePhaseManager
from .authorization_lease import (
    AuthorizationLease,
    LEASE_MEMBER,
    claim_authorization_lease,
    mark_authorization_complete,
    mark_authorization_failed,
)
from .config import SceptreV2Config
from .development_orchestrator import fit_and_freeze_development_router
from .execution_admission import DryRunAdmission, dry_run_admission
from .fresh_process_validation import (
    require_two_fresh_final_validations,
    require_two_fresh_preterminal_validations,
)
from .label_broker import RoleLabelBroker
from .outcome_builder import build_role_evidence
from .persistence import (
    FINAL_FRESH_ATTESTATION_MEMBER,
    VALIDATION_REPORT_MEMBER,
    persist_durable_attestation,
    persist_preterminal_bundle,
    persist_terminal_bundle,
    persist_validation_index,
)
from .phase_orchestrator import run_routing_phases
from .prediction_surface import (
    CANDIDATE_ARRAY_MEMBER,
    EXACT_B_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_RECEIPT_MEMBER,
    materialize_prediction_surface,
)
from .reports import (
    FINAL_REPORT_MEMBERS,
    build_final_reports,
    build_validation_report,
)
from .run_state import write_run_state
from .scratch import (
    SOURCE_DIRECTORY,
    ScratchLease,
    cleanup_scratch,
    create_scratch,
)
from .source_inner_surfaces import load_development_surfaces
from .source_seal import build_source_snapshot_payload
from .source_streams import materialize_source_streams
from .terminal_evaluation import evaluate_terminal_surfaces
from .validation import validate_publication_bundle
from .workspace_inputs import validate_workspace_provenance
from .workstation import run_workstation_preflight


PREDICTION_STORE_DIRECTORY = "prediction_store"
PhaseObserver = Callable[[str], None]


def dry_run_sceptre_v2(
    config: SceptreV2Config,
    *,
    artifact_root: str | Path,
) -> Mapping[str, object]:
    """Validate the full launch envelope without mutation or label access."""

    root = _artifact_root(config, artifact_root)
    admission = dry_run_admission(config)
    preflight = dict(
        run_workstation_preflight(
            root,
            admission.scratch.root.parent,
            runtime=config.runtime,
        )
    )
    provenance = validate_workspace_provenance(root, config)
    base = {
        "schema_version": "sceptre_v2_mutation_free_dry_run_v1",
        "status": "PASS",
        "experiment_id": config.experiment_id,
        "artifact_root": str(root),
        "config_hash": config.config_hash,
        "admission_hash": admission.admission_hash,
        "source_tree_sha256": admission.source_tree_sha256,
        "cache_binding_hash": admission.cache_binding_hash,
        "scratch_role": admission.scratch.role,
        "workstation_preflight_hash": canonical_hash(preflight),
        "workspace_provenance_hash": canonical_hash(provenance),
        "authorization_lease_claimed": False,
        "filesystem_mutations": 0,
        "target_labels_opened": False,
        "publication_status": "POST_HOC_CONSUMED_TEST_SENSITIVITY",
        "terminal_decision": "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE",
        "fresh_evidence": False,
    }
    return {**base, "dry_run_hash": canonical_hash(base)}


def run_sceptre_v2(
    config: SceptreV2Config,
    *,
    artifact_root: str | Path,
    phase_observer: PhaseObserver | None = None,
) -> Path:
    """Execute one irreversible terminal consumed-test diagnostic attempt."""

    root = _artifact_root(config, artifact_root)
    captured_inputs: dict[str, object] = {}

    def _capture_inputs(value: object):
        from .inputs import load_validated_inputs

        validated = load_validated_inputs(value)
        captured_inputs["validated"] = validated
        return validated

    # Every read-only launch, input, source, output, scratch and host gate runs
    # before the irreversible authorization directory is atomically created.
    admission = dry_run_admission(config, input_loader=_capture_inputs)
    preflight = dict(
        run_workstation_preflight(
            root,
            admission.scratch.root.parent,
            runtime=config.runtime,
        )
    )
    provenance = validate_workspace_provenance(root, config)
    validated = captured_inputs.get("validated")
    from .inputs import ValidatedInputs

    if not isinstance(validated, ValidatedInputs):
        raise ProtocolError("SCEPTRE v2 admitted input receipt was lost.")

    lease: AuthorizationLease | None = None
    scratch: ScratchLease | None = None
    phase = "BEGIN"
    hashes: dict[str, str] = {}
    try:
        # First mutation.  A failure at any later point permanently exhausts
        # this exact execution identity; output deletion never restores it.
        lease = claim_authorization_lease(
            config, admission_hash=admission.admission_hash
        )
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        scratch = create_scratch(
            root, config.runtime, authorization_lease=lease
        )
        hashes["workstation_preflight"] = canonical_hash(preflight)
        phase = "WORKSTATION_PREFLIGHT"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        partition = build_three_role_partition(
            tuple(
                CaseIdentity(row.center, row.case_id, row.sample_id)
                for row in validated.frame.rows
            )
        )
        source_development, source_prediction = load_development_surfaces(
            config.source_inner_root,
            receipt=validated.source_inner,
        )
        development = fit_and_freeze_development_router(
            source_development,
            source_prediction,
            generation_lock=validated.generation_lock,
            partition=partition,
        )
        hashes["partition"] = partition.partition_hash
        hashes["development_replay"] = development.replay_hash
        hashes["frozen_prelabel_router"] = development.router.full_router_sha256
        phase = "SOURCE_INNER_DEVELOPMENT_FREEZE"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        source_store = materialize_source_streams(
            config,
            validated.generation_lock,
            root=scratch.root / SOURCE_DIRECTORY,
        )
        hashes["source_stream_store"] = source_store.receipt_hash
        phase = "FRESH_PHYSICAL_SOURCE_STREAMS"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        prediction_root = root / PREDICTION_STORE_DIRECTORY
        prediction = materialize_prediction_surface(
            config,
            source_store,
            validated.frame,
            root=prediction_root,
        )
        prediction_store, prediction_members = _prediction_store_payload(
            root, prediction
        )
        candidate_probabilities = prediction.candidate_probabilities
        exact_b_probabilities = prediction.exact_b_probabilities
        hashes["prediction_store"] = prediction.receipt_hash
        phase = "FRESH_PHYSICAL_PREDICTION_SURFACE"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        manager = SceptrePhaseManager(partition, development.router)
        broker = RoleLabelBroker(
            manager=manager,
            partition=partition,
            frame=validated.frame,
            manifest_path=config.test_manifest_path,
            expected_manifest_sha256=config.expected_manifest_sha256,
            prediction_store_hash=prediction.receipt_hash,
            authorization_lease_hash=lease.lease_hash,
        )

        def _routing_progress(observed_phase: str, seal_hash: str) -> None:
            nonlocal phase
            hash_roles = {
                "ALL_G_DECISIONS_SEALED": "g_seal",
                "ALL_SELECTION_DECISIONS_SEALED": "selection_seal",
                "ALL_CALIBRATION_DECISIONS_SEALED": "policy_seal",
            }
            try:
                hash_role = hash_roles[observed_phase]
            except KeyError as exc:
                raise ProtocolError("SCEPTRE v2 routing phase callback drifted.") from exc
            hashes[hash_role] = seal_hash
            phase = observed_phase
            _advance(
                root,
                config=config,
                lease=lease,
                phase=phase,
                hashes=hashes,
                observer=phase_observer,
            )

        phases = run_routing_phases(
            development,
            partition=partition,
            manager=manager,
            broker=broker,
            candidate_probabilities=candidate_probabilities,
            exact_b_probabilities=exact_b_probabilities,
            candidate_source_order=prediction.geometry.centers,
            prediction_store_hash=prediction.receipt_hash,
            phase_observer=_routing_progress,
        )
        if (
            hashes.get("g_seal") != phases.g_seal.seal_hash
            or hashes.get("selection_seal") != phases.selection_seal.seal_hash
            or hashes.get("policy_seal") != phases.policy_seal.seal_hash
        ):
            raise ProtocolError("SCEPTRE v2 routing phase callback lost a seal.")
        hashes["route_policy"] = phases.route_policy.policy_artifact_hash
        phase = "ROUTE_POLICY_SEALED"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        source_snapshot = build_source_snapshot_payload()
        input_binding = _input_binding(
            config,
            provenance=provenance,
            admission=admission,
            validated=validated,
        )
        preterminal = persist_preterminal_bundle(
            root,
            config_hash=config.config_hash,
            input_binding=input_binding,
            source_snapshot=source_snapshot,
            authorization_lease=read_json(lease.root / LEASE_MEMBER),
            prediction_store=prediction_store,
            prediction_member_hashes=prediction_members,
            partition=partition,
            development=development,
            phases=phases,
        )
        hashes["preterminal_content_index"] = str(
            preterminal["content_index_hash"]
        )
        phase = "DURABLE_PRETERMINAL_BARRIER"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        durable = require_two_fresh_preterminal_validations(root)
        durable_payload = persist_durable_attestation(
            root,
            durable,
            preterminal_content_index_hash=str(preterminal["content_index_hash"]),
        )
        hashes["durable_preterminal_attestation"] = durable.attestation_hash
        phase = "TWO_FRESH_PRETERMINAL_VALIDATORS"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        terminal_capability = manager.begin_terminal_evaluation(durable)
        broker.activate_terminal(terminal_capability)
        evaluation_surfaces = []
        for target in prediction.geometry.centers:
            for fold_ordinal in range(5):
                fold = partition.fold(target, fold_ordinal)
                scoped = broker.open_evaluation(
                    target, fold_ordinal, terminal_capability
                )
                model = development.router.model_for_target(target)
                evidence = build_role_evidence(
                    scoped,
                    fold=fold,
                    partition_hash=partition.partition_hash,
                    candidate_probabilities=candidate_probabilities,
                    exact_b_probabilities=exact_b_probabilities,
                    candidate_source_order=prediction.geometry.centers,
                    prediction_store_hash=prediction.receipt_hash,
                    candidate_menu_hash=model.candidate_menu_hash,
                    exact_b_control_receipt_hash=(
                        model.exact_b_control_receipt_hash
                    ),
                    phase_capability=terminal_capability,
                )
                evaluation_surfaces.append(evidence.surface)
                del evidence, scoped
        result = evaluate_terminal_surfaces(
            phases.route_policy,
            tuple(evaluation_surfaces),
            prediction_store_hash=prediction.receipt_hash,
            terminal_capability_hash=terminal_capability.capability_hash,
        )
        del evaluation_surfaces
        final_journal = broker.journal_payload()
        runtime_receipt = {
            "schema_version": "sceptre_v2_runtime_receipt_v1",
            "workstation_preflight": dict(preflight),
            "workstation_preflight_hash": hashes["workstation_preflight"],
            "source_stream_receipt_hash": source_store.receipt_hash,
            "prediction_store_hash": prediction.receipt_hash,
            "scratch_role": scratch.role,
            "durable_attestation_hash": durable_payload["attestation_hash"],
        }
        reports = build_final_reports(
            result,
            input_binding=input_binding,
            source_snapshot=source_snapshot,
            label_journal=final_journal,
            runtime=runtime_receipt,
            prediction_store=prediction_store,
        )
        if tuple(reports) != FINAL_REPORT_MEMBERS:
            raise ProtocolError("SCEPTRE v2 final report inventory drifted.")
        final_index = persist_terminal_bundle(root, result, reports=reports)
        hashes["terminal_result"] = result.result_hash
        hashes["final_content_index"] = str(final_index["content_index_hash"])
        phase = "TERMINAL_LABELS_AND_DIAGNOSTICS"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        final_validation = require_two_fresh_final_validations(root)
        atomic_json(root / FINAL_FRESH_ATTESTATION_MEMBER, final_validation)
        validation_report = build_validation_report(final_validation)
        atomic_json(root / VALIDATION_REPORT_MEMBER, validation_report)
        hashes["final_fresh_process_attestation"] = str(
            final_validation["attestation_hash"]
        )
        hashes["validation_report"] = str(validation_report["report_hash"])
        phase = "TWO_FRESH_FINAL_VALIDATORS"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        validation_index = persist_validation_index(
            root,
            final_attestation_member=FINAL_FRESH_ATTESTATION_MEMBER,
            validation_report_member=VALIDATION_REPORT_MEMBER,
        )
        publication_validation = validate_publication_bundle(root)
        if (
            publication_validation["validation_index_hash"]
            != validation_index["validation_index_hash"]
        ):
            raise ProtocolError("SCEPTRE v2 publication validation drifted.")
        hashes["validation_index"] = str(
            validation_index["validation_index_hash"]
        )
        hashes["publication_reconstruction"] = str(
            publication_validation["publication_reconstruction_hash"]
        )
        phase = "POSTVALIDATION_INDEX_AUTHENTICATED"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )

        cleanup_scratch(scratch, artifact_root=root)
        scratch = None
        phase = "FINALIZING_AUTHORIZATION"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=phase_observer,
        )
        lease = mark_authorization_complete(lease)
        phase = "COMPLETE"
        _advance(
            root,
            config=config,
            lease=lease,
            phase=phase,
            hashes=hashes,
            observer=None,
            status="COMPLETE",
        )
        return root
    except BaseException as exc:
        cleanup_error = _cleanup_failed_scratch(scratch, root=root)
        if lease is not None:
            detail = str(exc)
            if cleanup_error is not None:
                detail = f"{detail}; scratch_cleanup={cleanup_error}"
            lease_error: BaseException | None = None
            if lease.status == "CLAIMED_IN_PROGRESS":
                try:
                    lease = mark_authorization_failed(lease, error=exc)
                except BaseException as failure:
                    lease_error = failure
            state_error: BaseException | None = None
            state_status = (
                "FAILED"
                if lease.status == "FAILED_EXHAUSTED"
                else "FINALIZATION_ERROR"
            )
            state_phase = (
                phase if state_status == "FAILED" else "FINALIZING_AUTHORIZATION"
            )
            try:
                write_run_state(
                    root,
                    authorization_lease=lease,
                    config_hash=config.config_hash,
                    status=state_status,
                    phase=state_phase,
                    bound_hashes=hashes,
                    error_class=type(exc).__name__,
                    error=detail,
                )
            except BaseException as failure:
                state_error = failure
            if lease_error is not None and hasattr(exc, "add_note"):
                exc.add_note(
                    "SCEPTRE v2 failed to finalize its exhausted lease: "
                    f"{type(lease_error).__name__}: {lease_error}"
                )
            if state_error is not None and hasattr(exc, "add_note"):
                exc.add_note(
                    "SCEPTRE v2 failed to persist its failure state: "
                    f"{type(state_error).__name__}: {state_error}"
                )
        raise


def _prediction_store_payload(
    artifact_root: Path, prediction: object
) -> tuple[dict[str, object], dict[str, str]]:
    paths = (
        Path(getattr(prediction, "candidate_array_path")),
        Path(getattr(prediction, "exact_b_array_path")),
        Path(getattr(prediction, "index_path")),
        Path(getattr(prediction, "receipt_path")),
    )
    expected_names = (
        CANDIDATE_ARRAY_MEMBER,
        EXACT_B_ARRAY_MEMBER,
        PREDICTION_INDEX_MEMBER,
        PREDICTION_RECEIPT_MEMBER,
    )
    members: dict[str, str] = {}
    for path, relative in zip(paths, expected_names, strict=True):
        expected = artifact_root / PREDICTION_STORE_DIRECTORY / relative
        if path.resolve() != expected.resolve() or path.is_symlink() or not path.is_file():
            raise ProtocolError("SCEPTRE v2 prediction member escaped its output root.")
        members[
            f"{PREDICTION_STORE_DIRECTORY}/{relative}"
        ] = sha256_file(path)
    receipt = dict(getattr(prediction, "receipt"))
    store_hash = str(getattr(prediction, "receipt_hash"))
    payload = {
        "schema_version": "sceptre_v2_durable_prediction_store_v1",
        "store_hash": store_hash,
        "physical_receipt": receipt,
        "member_sha256": members,
        "candidate_source_order": list(getattr(prediction, "geometry").centers),
        "manifest_opened": False,
        "outcomes_available": False,
        "raw_sample_paths_available": False,
    }
    return payload, members


def _input_binding(
    config: SceptreV2Config,
    *,
    provenance: Mapping[str, Mapping[str, object]],
    admission: DryRunAdmission,
    validated: object,
) -> dict[str, object]:
    frame = getattr(validated, "frame")
    generation = getattr(validated, "generation_lock")
    source_inner = getattr(validated, "source_inner")
    body = {
        "schema_version": "sceptre_v2_exact_eight_input_binding_v1",
        "experiment_id": config.experiment_id,
        "config_hash": config.config_hash,
        "direct_input_artifact_ids": list(config.input_artifact_ids),
        "direct_input_count": 8,
        "workspace_provenance": {
            artifact_id: dict(provenance[artifact_id])
            for artifact_id in config.input_artifact_ids
        },
        "admission_hash": admission.admission_hash,
        "cache_binding_hash": frame.cache_binding_hash,
        "generation_lock_hash": generation.generation_lock_hash,
        "source_inner_amendment_sha256": source_inner.amendment_sha256,
        "execution_amendment_sha256": (
            config.expected_execution_amendment_sha256
        ),
        "all_inputs_validated_before_authorization_claim": True,
        "target_labels_opened": False,
        "previous_stage90_output_used": False,
    }
    return {**body, "binding_hash": canonical_hash(body)}


def _artifact_root(config: SceptreV2Config, value: str | Path) -> Path:
    if not isinstance(config, SceptreV2Config):
        raise ProtocolError("SCEPTRE v2 runner requires its canonical config.")
    root = Path(value).resolve()
    if root != Path(config.artifact_root).resolve():
        raise ProtocolError("SCEPTRE v2 CLI/output artifact root drifted.")
    return root


def _advance(
    root: Path,
    *,
    config: SceptreV2Config,
    lease: AuthorizationLease,
    phase: str,
    hashes: Mapping[str, str],
    observer: PhaseObserver | None,
    status: str = "RUNNING",
) -> None:
    if observer is not None:
        observer(phase)
    print(f"[sceptre-v2] phase={phase}", flush=True)
    write_run_state(
        root,
        authorization_lease=lease,
        config_hash=config.config_hash,
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


__all__ = ("dry_run_sceptre_v2", "run_sceptre_v2")
