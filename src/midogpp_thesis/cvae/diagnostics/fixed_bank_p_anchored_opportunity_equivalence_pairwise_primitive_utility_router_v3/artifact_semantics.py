"""Semantic reopening and typed persisted-receipt reconstruction for OE-PPUR v3.

The final aggregate index intentionally covers only aggregate reports.  This
module closes the larger lifecycle boundary: every catalog member (except this
index's own bytes) is content hashed, all physical stores and preterminal
receipts are semantically reopened, and exact prepared COMPLETE bytes stand in
for the still-pending on-disk run state until the completion journal commits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
import hashlib
import os
from pathlib import Path
import stat

from ...protocol import ProtocolError
from ...runtime.fixed_bank_a1_prediction_contracts import validate_action_library
from ...runtime.fixed_bank_a1_prediction_store import load_global_prediction_seal
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .action_compiler import canonical_compiler_receipt
from .lease_claim import (
    AuthorizationLeaseClaim,
    validate_authorization_lease,
)
from .capacity_preflight import validate_capacity_observation
from .config import load_resolved_config
from .execution.inputs import build_authorized_seven_input_contract
from .execution.preterminal_artifact import (
    FINAL_ATTESTATION_MEMBER,
    MANIFEST_MEMBER,
    MATRIX_MEMBER,
    PRETERMINAL_ATTESTATION_MEMBER,
    _reconstruct_final_aggregate_attestation,
    _validate_preterminal_files,
)
from .execution.fresh_attestation import _validator_runtime_sha256
from .execution.services import (
    CanonicalRouterExecutionRequest,
    ServicePreflightRequest,
)
from .hashing import canonical_hash, require_sha256
from .identity import (
    EXPECTED_TEST_MANIFEST_SHA256,
    EXPERIMENT_ID,
    OUTPUT_ARTIFACT_ID,
)
from .lifecycle_lineage import (
    parse_complete_phase_evidence,
    validate_complete_lifecycle_evidence,
)
from .output_persistence import (
    COMPLETE_ARTIFACT_INDEX_MEMBER,
    COMPLETE_CATALOG_MEMBERS,
    COMPLETE_INTERNAL_MEMBERS,
    FINAL_BINDING_MEMBER,
    TERMINAL_METRICS_MEMBER,
    _fsync_directory,
    _read_json_object,
    _sha256_file,
    _write_json_exclusive,
)
from .output_validation import (
    FinalAggregateBundleReceipt,
    _issue_final_aggregate_bundle,
    validate_complete_artifact_inventory,
)
from .physical.actions import action_library_by_target
from .physical.cache_loader import load_label_free_test_frame
from .physical.prediction_runtime import (
    MaterializedPhysicalInputs,
    physical_partition_hash,
)
from .physical.runtime_config import PhysicalRuntimeConfig
from .physical.upstream import load_validated_upstream_inputs
from .run_admission import SevenInputRunAdmission, _ADMISSION_TOKEN
from .service_factory import prepare_canonical_scientific_service_factory
from .source_bundle import parse_source_training_bundle
from .source_production import canonical_held_action_library
from .source_seal import build_source_seal, validate_live_producer_seal_binding
from .terminal.authority import validate_resolved_terminal_authority
from .terminal.contracts import (
    ArtifactOnlyPreterminalAttestationReceipt,
    GuardedPreterminalBoundary,
    _ATTESTATION_TOKEN,
    _issue_artifact_only_preterminal_attestation,
    _reconstruct_persisted_aggregate_only_terminal_receipt,
    seal_guarded_preterminal_boundary,
)
from .workspace_provenance import validate_workspace_input_provenance
from .workstation import validate_workstation_observation


_COMPLETE_ARTIFACT_SEAL_TOKEN = object()
_RUN_STATE_MEMBER = "reports/run_state.json"
_RUN_LOCK_MEMBER = ".run.lock"


@dataclass(frozen=True, slots=True)
class _PreparedStateBinding:
    complete_payload: dict[str, object]
    complete_file_sha256: str
    state_hash: str
    receipt_hash: str
    final_bundle_receipt_hash: str


@dataclass(frozen=True, slots=True)
class _SemanticReopenResult:
    semantic_validation_hash: str
    source_seal_hash: str
    final_bundle_receipt_hash: str


def _validate_prepared_complete_state_for_build(
    root: Path,
    expected_complete_state: object,
) -> tuple[_PreparedStateBinding, FinalAggregateBundleReceipt]:
    from .run_state import (
        PreparedCompleteRunState,
        prepare_complete_run_state,
        validate_prepared_complete_run_state,
    )

    if type(expected_complete_state) is not PreparedCompleteRunState:
        raise ProtocolError("OE-PPUR v3 complete seal requires prepared state.")
    prepared = validate_prepared_complete_run_state(expected_complete_state)
    if prepared.artifact_root != root:
        raise ProtocolError("OE-PPUR v3 prepared state root drifted.")
    final_bundle = _issue_final_aggregate_bundle(root)
    rebuilt = prepare_complete_run_state(root, final_bundle=final_bundle)
    if rebuilt != prepared or prepared.final_bundle_receipt_hash != final_bundle.receipt_hash:
        raise ProtocolError("OE-PPUR v3 prepared state/final bundle drifted.")
    payload = _thaw_json(prepared.complete_payload)
    if not isinstance(payload, dict):  # pragma: no cover - type is factory guarded
        raise ProtocolError("OE-PPUR v3 prepared COMPLETE payload is malformed.")
    return (
        _PreparedStateBinding(
            complete_payload=payload,
            complete_file_sha256=hashlib.sha256(
                prepared.canonical_complete_bytes
            ).hexdigest(),
            state_hash=prepared.state_hash,
            receipt_hash=prepared.receipt_hash,
            final_bundle_receipt_hash=prepared.final_bundle_receipt_hash,
        ),
        final_bundle,
    )


def _require_prepared_complete_state_type(value: object) -> None:
    from .run_state import PreparedCompleteRunState

    if type(value) is not PreparedCompleteRunState:
        raise ProtocolError("OE-PPUR v3 complete seal requires prepared state.")


def _validate_committed_complete_state(
    root: Path,
    *,
    expected_complete_state: object | None,
) -> dict[str, object]:
    from .run_state import (
        PreparedCompleteRunState,
        read_terminal_run_state,
        validate_terminal_run_state,
    )

    terminal = validate_terminal_run_state(read_terminal_run_state(root))
    if terminal.status != "COMPLETE" or terminal.phase != "COMPLETE":
        raise ProtocolError("OE-PPUR v3 complete artifact state is not COMPLETE.")
    payload = _read_json_object(root / _RUN_STATE_MEMBER)
    if payload.get("state_hash") != terminal.state_hash:
        raise ProtocolError("OE-PPUR v3 complete state receipt drifted from bytes.")
    if expected_complete_state is not None:
        if type(expected_complete_state) is not PreparedCompleteRunState:
            raise ProtocolError("OE-PPUR v3 expected prepared state is untyped.")
        prepared_payload = _thaw_json(expected_complete_state.complete_payload)
        if (
            expected_complete_state.artifact_root != root
            or expected_complete_state.state_hash != terminal.state_hash
            or prepared_payload != payload
        ):
            raise ProtocolError("OE-PPUR v3 committed state differs from preparation.")
    return payload


def _semantic_reopen_complete_artifact(
    root: Path,
    *,
    complete_state_payload: Mapping[str, object],
    final_bundle: FinalAggregateBundleReceipt,
) -> _SemanticReopenResult:
    """Reopen every trust-bearing layer without terminal row-label access."""

    if type(final_bundle) is not FinalAggregateBundleReceipt:
        raise ProtocolError("OE-PPUR v3 semantic reopen final bundle is untyped.")
    resolved = load_resolved_config(root / "config.resolved.yaml")
    config = resolved.config
    if resolved.artifact_root != root:
        raise ProtocolError("OE-PPUR v3 semantic reopen config root drifted.")
    workspace = validate_workspace_input_provenance(root, resolved.input_bindings)
    source_seal = build_source_seal()
    compiler = canonical_compiler_receipt()
    held_library = canonical_held_action_library()
    source_surface = parse_source_training_bundle(
        resolved.input_bindings[2].path,
        compiler=compiler,
        expected_producer_source_seal_sha256=str(
            config.source_supervision_producer_seal_sha256
        ),
        expected_compiler_recomputation_receipt_sha256=str(
            config.source_supervision_recomputation_receipt_sha256
        ),
        expected_held_action_library_sha256=held_library.library_hash,
        expected_held_mass_policy_receipt_sha256=(
            held_library.mass_policy.receipt_hash
        ),
    )
    validate_live_producer_seal_binding(
        configured_sha256=config.source_supervision_producer_seal_sha256,
        parsed_sha256=(
            source_surface.receipt.contract.producer_source_seal_sha256
        ),
        source_seal=source_seal,
    )
    if (
        _sha256_file(resolved.input_bindings[4].path)
        != EXPECTED_TEST_MANIFEST_SHA256
    ):
        raise ProtocolError("OE-PPUR v3 test manifest bytes drifted.")
    lifecycle_source_seal = validate_resolved_terminal_authority(
        resolved,
        source_training_surface_receipt_hash=source_surface.receipt.receipt_hash,
    )
    admission = _reconstruct_run_admission(
        _read_json_object(root / "provenance/execution_admission.json")
    )
    claim = _reopen_authorization_claim(
        _read_json_object(
            root / "provenance/authorization_consumption_lease.json"
        )
    )
    run_lock_hash = _validate_run_lock(
        _read_json_object(root / _RUN_LOCK_MEMBER),
        admission=admission,
        claim=claim,
    )
    if (
        admission.artifact_root != root
        or claim.payload.get("artifact_root") != root.as_posix()
        or claim.payload.get("scratch_root") != admission.scratch_root.as_posix()
        or claim.payload.get("config_contract_hash") != admission.config_contract_hash
        or claim.payload.get("protocol_hash") != admission.protocol_hash
        or claim.payload.get("source_seal_hash") != admission.source_seal_hash
        or claim.payload.get("authorization_amendment_sha256")
        != admission.authorization_amendment_sha256
    ):
        raise ProtocolError("OE-PPUR v3 admission/lease roots or lineage drifted.")

    upstream = load_validated_upstream_inputs(
        resolved.input_bindings[0].path,
        resolved.input_bindings[1].path,
    )
    frame = load_label_free_test_frame(resolved.input_bindings[3].path)
    launch = _read_json_object(root / "reports/launch_receipts.json")
    workstation, capacity, factory, service_preflight = _reopen_launch_receipts(
        launch,
        config=config,
        source_seal=source_seal,
        source_surface=source_surface,
        admission=admission,
        upstream_receipt_hash=upstream.receipt_hash,
        frame_hash=frame.frame_hash,
    )

    physical_config = PhysicalRuntimeConfig(upstream.expert_bank_root)
    source_cache = load_frozen_source_streams(
        root / "physical/source_streams",
        expected_config_hash=physical_config.contract_hash,
        expected_generation_lock_hash=upstream.generation_lock.generation_lock_hash,
    )
    _library_payload, action_library_hash = validate_action_library(
        action_library_by_target()
    )
    partition_hash = physical_partition_hash(frame)
    prediction = load_global_prediction_seal(
        root / "physical/predictions",
        expected_config_hash=physical_config.contract_hash,
        expected_partition_hash=partition_hash,
        expected_source_lock_hash=source_cache.lock_hash,
        expected_action_library_hash=action_library_hash,
        expected_target_cache_binding_hash=frame.cache_binding_hash,
    )
    physical = MaterializedPhysicalInputs(
        source_cache=source_cache,
        prediction=prediction,
        partition_hash=partition_hash,
        source_root=root / "physical/source_streams",
        prediction_root=root / "physical/predictions",
    )
    request = CanonicalRouterExecutionRequest(
        frame=frame,
        physical_inputs=physical,
        upstream_receipt_hash=upstream.receipt_hash,
        workstation_receipt=workstation,
    )
    complete_state = dict(complete_state_payload)
    parsed_phase_evidence = parse_complete_phase_evidence(
        complete_state.get("transitions")
    )
    evidence_by_phase = dict(parsed_phase_evidence)

    manifest = _read_json_object(root / MANIFEST_MEMBER)
    attestation_payload = _read_json_object(root / PRETERMINAL_ATTESTATION_MEMBER)
    attestations, boundary = _reconstruct_preterminal_attestation_bundle(
        attestation_payload
    )
    preterminal = _validate_preterminal_files(
        root / MANIFEST_MEMBER,
        root / MATRIX_MEMBER,
        expected_ledger_hash=boundary.decision_ledger_receipt_hash,
        expected_result_hash=evidence_by_phase["PRETERMINAL_DECISIONS_SEALED"],
    )
    if (
        attestation_payload.get("artifact_file_sha256")
        != preterminal["artifact_file_sha256"]
        or attestation_payload.get("artifact_file_identity_sha256")
        != preterminal["artifact_file_identity_sha256"]
        or any(
            row.artifact_file_sha256 != preterminal["artifact_file_sha256"]
            or row.artifact_file_identity_sha256
            != preterminal["artifact_file_identity_sha256"]
            or row.validator_runtime_sha256 != _validator_runtime_sha256()
            for row in attestations
        )
        or manifest.get("request_hash") != request.request_hash
        or manifest.get("service_factory_identity_hash")
        != factory.identity.receipt_hash
        or manifest.get("seven_input_contract_hash")
        != config.seven_input_contract_hash
        or manifest.get("source_seal_hash")
        != source_seal.combined_source_sha256
        or manifest.get("source_training_surface_receipt_hash")
        != source_surface.receipt.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v3 preterminal semantic lineage drifted.")

    terminal = _reconstruct_persisted_aggregate_only_terminal_receipt(
        _read_json_object(root / TERMINAL_METRICS_MEMBER)
    )
    final_attestation = _reconstruct_final_aggregate_attestation(
        _read_json_object(root / FINAL_ATTESTATION_MEMBER)
    )
    final_binding = _read_json_object(root / FINAL_BINDING_MEMBER)
    authorized_inputs = build_authorized_seven_input_contract()
    expected_live_binding = {
        "config_contract_hash": config.contract_hash,
        "protocol_hash": config.protocol_hash,
        "seven_input_contract_hash": authorized_inputs.receipt_hash,
        "source_seal_hash": source_seal.combined_source_sha256,
        "source_seal_receipt_hash": source_seal.receipt_hash,
        "source_supervision_contract_hash": (
            source_surface.receipt.contract.contract_hash
        ),
        "source_training_surface_receipt_hash": source_surface.receipt.receipt_hash,
        "source_training_surface_hash": source_surface.surface_hash,
        "preterminal_boundary_receipt_hash": boundary.receipt_hash,
        "preterminal_ledger_receipt_hash": boundary.decision_ledger_receipt_hash,
        "preterminal_attestation_receipt_hashes": [
            row.receipt_hash for row in attestations
        ],
        "terminal_receipt_hash": terminal.receipt_hash,
        "final_attestation_hash": final_attestation.receipt_hash,
        "evaluated_case_count": terminal.evaluated_case_count,
        "exact_p_fallback_count": terminal.exact_p_fallback_count,
    }
    if any(
        final_binding.get(key) != value
        for key, value in expected_live_binding.items()
    ):
        raise ProtocolError("OE-PPUR v3 final aggregate live lineage drifted.")
    lifecycle_evidence = validate_complete_lifecycle_evidence(
        complete_state.get("transitions"),
        inputs_sealed_hash=canonical_hash(launch),
        prediction_seal_hash=prediction.seal_hash,
        preterminal_result_hash=preterminal["result_hash"],
        preterminal_boundary_hash=boundary.receipt_hash,
        terminal_receipt_hash=terminal.receipt_hash,
        final_attestation_hash=final_attestation.receipt_hash,
        final_bundle_receipt_hash=final_bundle.receipt_hash,
    )
    from .run_state import build_run_identity_hash

    if (
        complete_state.get("status") != "COMPLETE"
        or complete_state.get("phase") != "COMPLETE"
        or complete_state.get("config_contract_hash") != config.contract_hash
        or complete_state.get("protocol_hash") != config.protocol_hash
        or complete_state.get("source_seal_hash")
        != source_seal.combined_source_sha256
        or complete_state.get("seven_input_admission_hash") != admission.receipt_hash
        or complete_state.get("authorization_lease_claim_hash") != claim.claim_hash
        or complete_state.get("run_identity_hash") != build_run_identity_hash(admission)
        or complete_state.get("state_hash")
        != canonical_hash(
            {key: value for key, value in complete_state.items() if key != "state_hash"}
        )
        or admission.config_contract_hash != config.contract_hash
        or admission.protocol_hash != config.protocol_hash
        or admission.seven_input_contract_hash
        != authorized_inputs.receipt_hash
        or admission.source_seal_hash != source_seal.combined_source_sha256
        or admission.source_seal_receipt_hash != source_seal.receipt_hash
        or admission.source_training_surface_receipt_hash
        != source_surface.receipt.receipt_hash
        or admission.source_training_surface_hash != source_surface.surface_hash
        or admission.input_location_binding_hash
        != workspace.input_location_binding_hash
        or admission.workspace_input_manifest_sha256
        != workspace.manifest_file_sha256
        or admission.workspace_provenance_receipt_hash != workspace.receipt_hash
        or admission.authorization_amendment_sha256
        != config.authorization_amendment_sha256
        or admission.lifecycle_source_seal_sha256
        != lifecycle_source_seal.lifecycle_source_sha256
        or admission.lifecycle_source_seal_receipt_hash
        != lifecycle_source_seal.receipt_hash
        or claim.payload.get("seven_input_admission_hash") != admission.receipt_hash
        or claim.payload.get("run_identity_hash") != build_run_identity_hash(admission)
        or boundary.seven_input_contract_hash != config.seven_input_contract_hash
        or boundary.source_seal_hash != source_seal.combined_source_sha256
        or boundary.source_training_surface_receipt_hash
        != source_surface.receipt.receipt_hash
        or terminal.boundary_receipt_hash != boundary.receipt_hash
        or terminal.decision_ledger_receipt_hash
        != boundary.decision_ledger_receipt_hash
        or terminal.evaluated_case_count != boundary.case_count
        or terminal.exact_p_fallback_count != boundary.exact_p_fallback_count
        or final_attestation.terminal_receipt_hash != terminal.receipt_hash
        or final_bundle.final_attestation_hash != final_attestation.receipt_hash
    ):
        raise ProtocolError("OE-PPUR v3 complete lifecycle lineage drifted.")

    semantic_body = {
        "schema_version": "oe_ppur_v3_complete_artifact_semantic_validation_v1",
        "config_contract_hash": config.contract_hash,
        "protocol_hash": config.protocol_hash,
        "workspace_provenance_receipt_hash": workspace.receipt_hash,
        "source_seal_hash": source_seal.combined_source_sha256,
        "source_seal_receipt_hash": source_seal.receipt_hash,
        "source_training_surface_receipt_hash": source_surface.receipt.receipt_hash,
        "source_training_surface_hash": source_surface.surface_hash,
        "lifecycle_source_seal_sha256": (
            lifecycle_source_seal.lifecycle_source_sha256
        ),
        "lifecycle_source_seal_receipt_hash": (
            lifecycle_source_seal.receipt_hash
        ),
        "seven_input_admission_hash": admission.receipt_hash,
        "authorization_lease_claim_hash": claim.claim_hash,
        "run_lock_hash": run_lock_hash,
        "launch_receipts_hash": canonical_hash(launch),
        "upstream_receipt_hash": upstream.receipt_hash,
        "test_frame_hash": frame.frame_hash,
        "workstation_receipt_hash": workstation.receipt_hash,
        "resource_capacity_receipt_hash": capacity.receipt_hash,
        "service_factory_identity_hash": factory.identity.receipt_hash,
        "service_preflight_receipt_hash": service_preflight.receipt_hash,
        "source_stream_lock_hash": source_cache.lock_hash,
        "prediction_seal_hash": prediction.seal_hash,
        "prediction_store_hash": prediction.store.store_hash,
        "preterminal_result_hash": preterminal["result_hash"],
        "preterminal_ledger_hash": preterminal["sealed_ledger_receipt_hash"],
        "preterminal_artifact_file_sha256": preterminal["artifact_file_sha256"],
        "preterminal_attestation_hashes": [row.receipt_hash for row in attestations],
        "preterminal_boundary_hash": boundary.receipt_hash,
        "terminal_receipt_hash": terminal.receipt_hash,
        "final_attestation_hash": final_attestation.receipt_hash,
        "final_bundle_receipt_hash": final_bundle.receipt_hash,
        "complete_lifecycle_evidence_hash": lifecycle_evidence.evidence_hash,
        "prepared_state_hash": complete_state["state_hash"],
        "target_labels_reopened": False,
        "raw_labels_persisted": False,
        "cross_run_recovery_allowed": False,
    }
    return _SemanticReopenResult(
        semantic_validation_hash=canonical_hash(semantic_body),
        source_seal_hash=source_seal.combined_source_sha256,
        final_bundle_receipt_hash=final_bundle.receipt_hash,
    )


def _reconstruct_run_admission(
    payload: Mapping[str, object],
) -> SevenInputRunAdmission:
    expected_keys = {
        "schema_version",
        "status",
        "experiment_id",
        "output_artifact_id",
        "direct_input_roles",
        "direct_input_artifact_ids",
        "config_contract_hash",
        "protocol_hash",
        "seven_input_contract_hash",
        "source_seal_hash",
        "source_seal_receipt_hash",
        "source_training_surface_receipt_hash",
        "source_training_surface_hash",
        "input_location_binding_hash",
        "workspace_input_manifest_sha256",
        "workspace_provenance_receipt_hash",
        "authorization_amendment_sha256",
        "lifecycle_source_seal_sha256",
        "lifecycle_source_seal_receipt_hash",
        "artifact_root",
        "scratch_root",
        "source_supervision_materialized",
        "authorization_amendment_issued",
        "execution_authorized",
        "target_labels_opened",
        "mutation_performed",
        "cross_run_recovery_used",
        "receipt_hash",
    }
    if set(payload) != expected_keys:
        raise ProtocolError("OE-PPUR v3 persisted admission schema drifted.")
    admission = SevenInputRunAdmission(
        config_contract_hash=str(payload["config_contract_hash"]),
        protocol_hash=str(payload["protocol_hash"]),
        seven_input_contract_hash=str(payload["seven_input_contract_hash"]),
        source_seal_hash=str(payload["source_seal_hash"]),
        source_seal_receipt_hash=str(payload["source_seal_receipt_hash"]),
        source_training_surface_receipt_hash=str(
            payload["source_training_surface_receipt_hash"]
        ),
        source_training_surface_hash=str(payload["source_training_surface_hash"]),
        input_location_binding_hash=str(payload["input_location_binding_hash"]),
        workspace_input_manifest_sha256=str(
            payload["workspace_input_manifest_sha256"]
        ),
        workspace_provenance_receipt_hash=str(
            payload["workspace_provenance_receipt_hash"]
        ),
        authorization_amendment_sha256=str(
            payload["authorization_amendment_sha256"]
        ),
        lifecycle_source_seal_sha256=str(
            payload["lifecycle_source_seal_sha256"]
        ),
        lifecycle_source_seal_receipt_hash=str(
            payload["lifecycle_source_seal_receipt_hash"]
        ),
        artifact_root=Path(str(payload["artifact_root"])),
        scratch_root=Path(str(payload["scratch_root"])),
        _factory_token=_ADMISSION_TOKEN,
    )
    if admission.to_payload() != dict(payload):
        raise ProtocolError("OE-PPUR v3 persisted admission receipt drifted.")
    return admission


def _reopen_authorization_claim(
    payload: Mapping[str, object],
) -> AuthorizationLeaseClaim:
    claim = AuthorizationLeaseClaim(
        path=Path(str(payload.get("lease_path", ""))),
        payload=dict(payload),
        claim_hash=str(payload.get("claim_hash", "")),
    )
    return validate_authorization_lease(claim)


def _validate_run_lock(
    payload: Mapping[str, object],
    *,
    admission: SevenInputRunAdmission,
    claim: AuthorizationLeaseClaim,
) -> str:
    body = {key: value for key, value in payload.items() if key != "lock_hash"}
    if (
        set(payload)
        != {
            "schema_version",
            "experiment_id",
            "run_identity_hash",
            "authorization_lease_claim_hash",
            "authorization_exhausted",
            "recovery_allowed",
            "lock_hash",
        }
        or payload.get("schema_version") != "oe_ppur_v3_run_lock_v1"
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("run_identity_hash") != claim.payload.get("run_identity_hash")
        or payload.get("authorization_lease_claim_hash") != claim.claim_hash
        or claim.payload.get("seven_input_admission_hash") != admission.receipt_hash
        or payload.get("authorization_exhausted") is not True
        or payload.get("recovery_allowed") is not False
        or payload.get("lock_hash") != canonical_hash(body)
    ):
        raise ProtocolError("OE-PPUR v3 run lock lineage drifted.")
    return str(payload["lock_hash"])


def _reopen_launch_receipts(
    payload: Mapping[str, object],
    *,
    config: object,
    source_seal: object,
    source_surface: object,
    admission: SevenInputRunAdmission,
    upstream_receipt_hash: str,
    frame_hash: str,
):
    expected_keys = {
        "schema_version",
        "seven_input_admission",
        "source_seal",
        "source_training_surface_receipt_hash",
        "source_training_surface_hash",
        "upstream_receipt_hash",
        "test_frame_hash",
        "workstation",
        "resource_capacity",
        "service_factory_identity",
        "service_preflight",
        "target_labels_opened",
    }
    workstation_raw = payload.get("workstation")
    capacity_raw = payload.get("resource_capacity")
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != "oe_ppur_v3_launch_receipts_v1"
        or not isinstance(workstation_raw, Mapping)
        or not isinstance(capacity_raw, Mapping)
    ):
        raise ProtocolError("OE-PPUR v3 launch receipt schema drifted.")
    workstation = validate_workstation_observation(
        {
            "gpu_count": workstation_raw.get("observed_gpu_count"),
            "gpu_names": workstation_raw.get("observed_gpu_names"),
            "cpu_count": workstation_raw.get("observed_cpu_count"),
            "start_method": workstation_raw.get("multiprocessing_start_method"),
        },
        dto_pickle_round_trip_validated=bool(
            workstation_raw.get("dto_pickle_round_trip_validated")
        ),
    )
    capacity = validate_capacity_observation(
        {
            "gpus": capacity_raw.get("gpus"),
            "ram_available_bytes": capacity_raw.get("ram_available_bytes"),
            "artifact_free_bytes": capacity_raw.get("artifact_free_bytes"),
            "scratch_free_bytes": capacity_raw.get("scratch_free_bytes"),
            "artifact_device": capacity_raw.get("artifact_device"),
            "scratch_device": capacity_raw.get("scratch_device"),
        }
    )
    factory = prepare_canonical_scientific_service_factory(
        config,  # type: ignore[arg-type]
        source_seal=source_seal,  # type: ignore[arg-type]
        source_surface=source_surface,  # type: ignore[arg-type]
    )
    service_preflight = factory.build().preflight(
        ServicePreflightRequest(
            seven_input_contract_hash=config.seven_input_contract_hash,  # type: ignore[attr-defined]
            protocol_hash=config.protocol_hash,  # type: ignore[attr-defined]
            source_seal_hash=source_seal.combined_source_sha256,  # type: ignore[attr-defined]
            workstation_receipt_hash=workstation.receipt_hash,
        )
    )
    if (
        payload.get("seven_input_admission") != admission.to_payload()
        or payload.get("source_seal") != source_seal.to_payload()  # type: ignore[attr-defined]
        or payload.get("source_training_surface_receipt_hash")
        != source_surface.receipt.receipt_hash  # type: ignore[attr-defined]
        or payload.get("source_training_surface_hash")
        != source_surface.surface_hash  # type: ignore[attr-defined]
        or payload.get("upstream_receipt_hash") != upstream_receipt_hash
        or payload.get("test_frame_hash") != frame_hash
        or workstation.to_payload() != dict(workstation_raw)
        or capacity.to_payload() != dict(capacity_raw)
        or payload.get("service_factory_identity") != factory.identity.to_payload()
        or payload.get("service_preflight") != service_preflight.to_payload()
        or payload.get("target_labels_opened") is not False
    ):
        raise ProtocolError("OE-PPUR v3 launch receipt lineage drifted.")
    return workstation, capacity, factory, service_preflight


def _reconstruct_preterminal_attestation_bundle(
    payload: Mapping[str, object],
) -> tuple[
    tuple[
        ArtifactOnlyPreterminalAttestationReceipt,
        ArtifactOnlyPreterminalAttestationReceipt,
    ],
    GuardedPreterminalBoundary,
]:
    if (
        set(payload)
        != {
            "schema_version",
            "artifact_file_sha256",
            "artifact_file_identity_sha256",
            "attestations",
            "guarded_boundary",
            "fresh_process_count",
            "target_labels_opened",
        }
        or payload.get("schema_version")
        != "oe_ppur_v3_two_fresh_preterminal_attestations_v1"
        or payload.get("fresh_process_count") != 2
        or payload.get("target_labels_opened") is not False
        or not isinstance(payload.get("attestations"), list)
        or not isinstance(payload.get("guarded_boundary"), Mapping)
    ):
        raise ProtocolError("OE-PPUR v3 persisted preterminal attestation drifted.")
    rows = []
    for raw in payload["attestations"]:  # type: ignore[index]
        if not isinstance(raw, Mapping):
            raise ProtocolError("OE-PPUR v3 persisted attestation row is malformed.")
        receipt = _issue_artifact_only_preterminal_attestation(
            sealed_ledger_receipt_hash=str(raw.get("sealed_ledger_receipt_hash")),
            artifact_file_sha256=str(raw.get("artifact_file_sha256")),
            artifact_file_identity_sha256=str(
                raw.get("artifact_file_identity_sha256")
            ),
            validator_runtime_sha256=str(raw.get("validator_runtime_sha256")),
            process_pid=int(raw.get("process_pid", -1)),
            _validator_token=_ATTESTATION_TOKEN,
        )
        if receipt.to_payload() != dict(raw):
            raise ProtocolError("OE-PPUR v3 persisted attestation receipt drifted.")
        rows.append(receipt)
    attestations = tuple(rows)
    if len(attestations) != 2:
        raise ProtocolError("OE-PPUR v3 persisted attestation count drifted.")
    raw_boundary = payload["guarded_boundary"]
    boundary = seal_guarded_preterminal_boundary(
        seven_input_contract_hash=str(raw_boundary.get("seven_input_contract_hash")),
        source_seal_hash=str(raw_boundary.get("source_seal_hash")),
        source_training_surface_receipt_hash=str(
            raw_boundary.get("source_training_surface_receipt_hash")
        ),
        decision_ledger_receipt_hash=str(
            raw_boundary.get("decision_ledger_receipt_hash")
        ),
        attestations=attestations,
        case_inventory_sha256=str(raw_boundary.get("case_inventory_sha256")),
        case_count=int(raw_boundary.get("case_count", -1)),
        exact_p_fallback_count=int(raw_boundary.get("exact_p_fallback_count", -1)),
    )
    if boundary.to_payload() != dict(raw_boundary):
        raise ProtocolError("OE-PPUR v3 persisted preterminal boundary drifted.")
    return attestations, boundary  # type: ignore[return-value]


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ProtocolError("OE-PPUR v3 complete artifact payload is not canonical JSON.")



__all__: tuple[str, ...] = ()
