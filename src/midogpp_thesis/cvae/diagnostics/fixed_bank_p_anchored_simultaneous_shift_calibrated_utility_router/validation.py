"""Content-first, nonrepairing reconstruction of the complete PSSCUR diagnostic."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from ...runtime.fixed_bank_a1_action_predictions import load_global_prediction_seal
from ...runtime.fixed_bank_a1_prediction_contracts import validate_action_library
from ...runtime.frozen_source_streams import load_frozen_source_streams
from .actions import action_library_by_target
from .bundle import assert_closed_world, validate_content_index
from .constants import (
    COMPOSED_POLICY_IDS,
    EXPECTED_ENVELOPE_CALIBRATION_COUNT,
    EXPECTED_INNER_DONOR_REPLAY_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
)
from .engine import build_preterminal_result
from .evaluation import evaluate_terminal
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
from .persistence import persist_admission, persist_physical_surface, persist_preterminal, persist_terminal
from .physical_runtime import physical_partition_hash, probability_index_rows
from .preflight import load_validated_workstation_preflight
from .probability_surface import build_physical_probability_surface
from .protocol import build_frozen_protocol
from .reports import leakage_report_payload, publication_decision_payload


VALIDATION_SCHEMA = "fixed_bank_psscur_validation_v1"
RECONSTRUCTIVE_MEMBERS = (
    "manifests/protocol_manifest.json",
    "manifests/action_library.json",
    "manifests/physical_surface_seal.json",
    "manifests/outer_plan_seal.json",
    "manifests/policy_menu.json",
    "manifests/decision_barrier.json",
    "manifests/preterminal_aggregate_seal.json",
    "manifests/terminal_evaluation_seal.json",
    "tables/exact_nine_probability_index.json",
    "tables/outer_plans.json",
    "tables/physical_fingerprints.json",
    "tables/support_fold_plans.json",
    "tables/target_local_posterior_models.json",
    "tables/target_local_posterior_predictions.json",
    "tables/route_posterior_ensembles.json",
    "tables/utility_descriptors.json",
    "tables/posterior_utility_predictions.json",
    "tables/donor_utility_rows.json",
    "tables/donor_posterior_utility_predictions.json",
    "tables/envelope_calibrations.json",
    "tables/utility_certificates.json",
    "tables/composed_predictions.json",
    "tables/route_decisions.json",
    "tables/terminal_method_metrics.json",
    "tables/terminal_center_contrasts.json",
    "tables/terminal_case_oracles.json",
    "tables/utility_information_rows.json",
    "tables/utility_information_centers.json",
    "tables/information_gate.json",
    "tables/selection_control.json",
    "reports/diagnostic_summary.json",
    "reports/label_capability_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
)


def validate_p_anchored_simultaneous_shift_calibrated_utility_router_bundle(
    root: str | Path,
    *,
    config: object,
    allow_pending_validation: bool = False,
) -> Mapping[str, object]:
    path = Path(root).resolve()
    protocol = build_frozen_protocol()
    assert_closed_world(path, allow_pending_validation=allow_pending_validation)
    content = validate_content_index(
        path,
        config_contract_hash=str(getattr(config, "contract_hash")),
        protocol_contract_hash=protocol.protocol_hash,
    )
    assert_input_fence(config)
    workspace = validate_active_workspace_binding(config)
    provenance = validate_workspace_provenance(path, config)
    locks = load_validated_locks(config)
    frame = load_label_free_test_frame(config)
    pre_gpu = validate_pre_gpu_firewall(config, frame, locks)
    preflight = load_validated_workstation_preflight(path, runtime=getattr(config, "runtime"))
    source = load_frozen_source_streams(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_generation_lock_hash=str(locks.generation.generation_lock_hash),
    )
    _payload, action_hash = validate_action_library(action_library_by_target())
    prediction = load_global_prediction_seal(
        path,
        expected_config_hash=str(getattr(config, "contract_hash")),
        expected_partition_hash=physical_partition_hash(frame),
        expected_source_lock_hash=source.lock_hash,
        expected_action_library_hash=action_hash,
        expected_target_cache_binding_hash=frame.cache_binding_hash,
    )
    surface = build_physical_probability_surface(prediction.store)
    preterminal = build_preterminal_result(
        surface,
        lambda allowed, role: read_scoped_manifest_labels(
            config, frame, allowed_keys=allowed, role=role
        ),
        use_processes=True,
    )
    terminal = evaluate_terminal(preterminal)
    leakage = leakage_report_payload(
        probability_surface_hash=surface.surface_hash,
        preterminal=preterminal,
        capability_report=terminal.capability_report,
    )
    publication = publication_decision_payload(terminal)
    runtime_summary = _validate_runtime_summary(
        path,
        preflight=preflight,
        runtime=getattr(config, "runtime"),
        source=source,
        prediction=prediction,
    )
    physical = SimpleNamespace(canonical_source_cache=source, prediction=prediction)
    with tempfile.TemporaryDirectory(prefix="psscur-reconstruction-") as temporary:
        reconstructed = Path(temporary)
        for directory in ("manifests", "reports", "tables"):
            (reconstructed / directory).mkdir(parents=True, exist_ok=True)
        persist_admission(
            reconstructed,
            config=config,
            protocol=protocol,
            provenance=provenance,
            frame=frame,
            pre_gpu_firewall=pre_gpu,
        )
        persist_physical_surface(
            reconstructed,
            physical=physical,
            surface=surface,
            probability_index=probability_index_rows(prediction),
        )
        persist_preterminal(reconstructed, preterminal)
        persist_terminal(
            reconstructed,
            terminal=terminal,
            leakage_report=leakage,
            publication_decision=publication,
            runtime_summary=runtime_summary,
        )
        for member in RECONSTRUCTIVE_MEMBERS:
            expected = reconstructed / member
            observed = path / member
            if not observed.is_file() or observed.is_symlink() or observed.read_bytes() != expected.read_bytes():
                raise ProtocolError(f"PSSCUR reconstruction disagrees: {member}.")

    endpoint_fit_count = sum(row.endpoint_model_fit_count for row in preterminal.endpoint_products)
    utility_model_count = 0
    target_posterior_model_count = sum(
        len(rows) for rows in preterminal.target_posterior_models_by_control.values()
    )
    if (
        len(preterminal.plans.outer_plans) != EXPECTED_OUTER_PLAN_COUNT
        or endpoint_fit_count != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
        or utility_model_count != EXPECTED_UTILITY_MODEL_FIT_COUNT
        or target_posterior_model_count != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or len(preterminal.envelope_calibrations) != EXPECTED_ENVELOPE_CALIBRATION_COUNT
        or sum(
            len(row.inner_replays)
            for row in preterminal.envelope_calibrations.values()
        )
        != EXPECTED_INNER_DONOR_REPLAY_COUNT
        or any(
            len(preterminal.composed_predictions_by_policy[policy]) != EXPECTED_OUTER_PLAN_COUNT
            for policy in COMPOSED_POLICY_IDS
        )
        or any(
            len(preterminal.utility_certificates_by_policy[policy])
            != 6 * EXPECTED_OUTER_PLAN_COUNT
            for policy in COMPOSED_POLICY_IDS
        )
        or terminal.capability_report.get("status") != "PASS"
    ):
        raise ProtocolError("PSSCUR reconstructed topology drifted.")
    checks = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "PASS",
        "content_hash": content["content_hash"],
        "config_contract_hash": str(getattr(config, "contract_hash")),
        "protocol_contract_hash": protocol.protocol_hash,
        "workspace_binding": dict(workspace),
        "input_artifact_count": len(provenance),
        "pre_gpu_firewall_status": pre_gpu["status"],
        "workstation_preflight_status": preflight["status"],
        "physical_probability_cell_count": len(prediction.store.cells),
        "outer_route_count": len(preterminal.plans.outer_plans),
        "double_exclusion_state_count": 0,
        "outer_endpoint_model_fit_count": endpoint_fit_count,
        "utility_model_fit_count": utility_model_count,
        "target_posterior_model_fit_count": target_posterior_model_count,
        "route_posterior_ensemble_count": sum(
            len(rows)
            for rows in preterminal.route_posterior_ensembles_by_control.values()
        ),
        "envelope_calibration_count": len(preterminal.envelope_calibrations),
        "inner_donor_replay_count": sum(
            len(row.inner_replays)
            for row in preterminal.envelope_calibrations.values()
        ),
        "donor_utility_row_count": sum(
            len(rows) for rows in preterminal.donor_utility_rows_by_target.values()
        ),
        "utility_descriptor_count": sum(
            len(rows) for rows in preterminal.utility_descriptors_by_center.values()
        ),
        "utility_certificate_count": sum(
            len(rows)
            for rows in preterminal.utility_certificates_by_policy.values()
        ),
        "composed_prediction_count": sum(
            len(rows) for rows in preterminal.composed_predictions_by_policy.values()
        ),
        "terminal_evaluation_hash": terminal.evaluation_hash,
        "all_science_reconstructed_exactly": True,
        "content_index_validated_before_scientific_members": True,
        "nonrepairing_validation": True,
        "closed_world": True,
        "raw_labels_persisted": False,
        "terminal_consumed_test_diagnostic_only": True,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "promotion_eligible": False,
        "may_feed_another_experiment": False,
    }
    if allow_pending_validation:
        return checks
    from .fresh_process_validation import verify_attested_validation

    return verify_attested_validation(
        read_json(path / "reports/validation_report.json"),
        expected_checks=checks,
        persisted_attestation=read_json(path / "reports/fresh_process_attestation.json"),
    )


def _validate_runtime_summary(
    root: Path,
    *,
    preflight: Mapping[str, object],
    runtime: Mapping[str, object],
    source: object,
    prediction: object,
) -> Mapping[str, object]:
    payload = read_json(root / "reports/runtime_summary.json")
    expected = {
        "schema_version": "fixed_bank_psscur_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": getattr(source, "lock_hash"),
        "global_prediction_seal_hash": getattr(prediction, "seal_hash"),
        "source_stream_count": 81,
        "classifier_cell_count": 810,
        "unique_classifier_fit_count": 810,
        "workstation_preflight_hash": canonical_hash(preflight),
        "persistent_generation_worker_count": runtime.get("persistent_generation_worker_count"),
        "classifier_workers": runtime.get("classifier_workers"),
        "route_workers": runtime.get("route_model_workers"),
        "route_worker_blas_threads": runtime.get("classifier_threads_per_worker"),
        "outer_process_blas_threads": 3,
        "target_posterior_process_blas_threads": 1,
        "gpu_generation_completed_before_cpu_phase": True,
        "cuda_visible_devices_during_route_phase": "",
        "double_exclusion_state_count": 0,
        "unused_nested_endpoint_fits_eliminated": True,
        "outer_endpoint_model_fit_count": EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
        "utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
        "target_posterior_model_fit_count": EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
        "envelope_calibration_count": EXPECTED_ENVELOPE_CALIBRATION_COUNT,
        "inner_donor_replay_count": EXPECTED_INNER_DONOR_REPLAY_COUNT,
        "prior_rebinding_additional_endpoint_model_fit_count": 0,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "scratch_root_id": preflight.get("scratch_root_id"),
        "scratch_role": preflight.get("scratch_role"),
        "local_and_canonical_source_lock_identical": True,
        "previous_stage90_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
    }
    if payload != expected:
        raise ProtocolError("PSSCUR runtime summary drifted.")
    return payload


# Descriptive short name used by runner and CLI.
validate_psscur_bundle = validate_p_anchored_simultaneous_shift_calibrated_utility_router_bundle


__all__ = (
    "VALIDATION_SCHEMA",
    "validate_p_anchored_simultaneous_shift_calibrated_utility_router_bundle",
    "validate_psscur_bundle",
)
