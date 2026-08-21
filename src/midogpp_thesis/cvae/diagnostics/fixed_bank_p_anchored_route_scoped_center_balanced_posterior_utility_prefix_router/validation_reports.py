"""Exact report, workstation-state, and fresh-attestation validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .hashing import canonical_hash
from .preflight import load_validated_workstation_preflight
from .reports import leakage_report_payload, publication_decision_payload
from .validation_origin import PhysicalOriginTopology


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}


def validate_scientific_reports(
    root: Path,
    *,
    config: object,
    origin: PhysicalOriginTopology,
    physical: Mapping[str, object],
    capability: Mapping[str, object],
    summary: Mapping[str, object],
    leakage: Mapping[str, object],
    publication: Mapping[str, object],
    runtime: Mapping[str, object],
    require_final: bool,
) -> None:
    preflight = load_validated_workstation_preflight(
        root, runtime=getattr(config, "runtime")
    )
    plan_seal = read_json(root / "manifests/outer_plan_seal.json")
    aggregate = read_json(root / "manifests/preterminal_aggregate_seal.json")
    expected_leakage = leakage_report_payload(
        probability_surface_hash=str(physical["surface_hash"]),
        plan_seal_hash=str(plan_seal["seal_hash"]),
        aggregate_seal_hash=str(aggregate["aggregate_seal_hash"]),
        capability_report=capability,
    )
    if dict(leakage) != expected_leakage:
        raise ProtocolError("CBPUPR leakage report reconstruction drifted.")
    if dict(publication) != publication_decision_payload(summary):
        raise ProtocolError("CBPUPR publication decision reconstruction drifted.")

    configured = getattr(config, "runtime")
    expected_runtime = {
        "schema_version": "fixed_bank_cbpupr_runtime_summary_v1",
        "status": "PASS",
        "source_stream_lock_hash": origin.source_stream_lock_hash,
        "global_prediction_seal_hash": origin.global_prediction_seal_hash,
        "source_stream_count": 81,
        "classifier_cell_count": 810,
        "unique_classifier_fit_count": 810,
        "workstation_preflight_hash": canonical_hash(preflight),
        "persistent_generation_worker_count": 2,
        "classifier_workers": int(configured["classifier_workers"]),
        "route_workers": int(configured["route_model_workers"]),
        "route_worker_blas_threads": int(
            configured["classifier_threads_per_worker"]
        ),
        "outer_process_blas_threads": int(
            configured["classifier_threads_per_worker"]
        ),
        "target_posterior_process_blas_threads": int(
            configured["target_posterior_threads_per_worker"]
        ),
        "gpu_generation_completed_before_cpu_phase": True,
        "cuda_visible_devices_during_route_phase": "",
        "double_exclusion_state_count": int(
            configured["ordered_H_J_pair_count"]
        ),
        "unused_nested_endpoint_fits_eliminated": True,
        "outer_endpoint_model_fit_count": int(
            configured["expected_outer_endpoint_model_fit_count"]
        ),
        "donor_response_model_fit_count": int(
            configured["donor_response_model_fit_count"]
        ),
        "target_posterior_model_fit_count": int(
            configured["expected_target_posterior_model_fit_count"]
        ),
        "pseudo_route_count": int(configured["pseudo_route_count"]),
        "pseudo_posterior_model_fit_count": int(
            configured["expected_pseudo_posterior_model_fit_count"]
        ),
        "total_posterior_model_fit_count": int(
            configured["expected_total_posterior_model_fit_count"]
        ),
        "validation_endpoint_optimizer_refit_count": 0,
        "validation_posterior_optimizer_refit_count": 0,
        "optimizer_fit_correctness_is_content_sealed_trust_boundary": True,
        "prior_rebinding_additional_endpoint_model_fit_count": 0,
        "source_storage_dtype": "float32",
        "probability_storage_dtype": "float32",
        "scientific_reductions_dtype": "float64",
        "scratch_root_id": preflight["scratch_root_id"],
        "scratch_role": preflight["scratch_role"],
        "local_and_canonical_source_lock_identical": True,
        "previous_stage90_scratch_reused": False,
        "recomputed_from_original_six_inputs": True,
    }
    if dict(runtime) != expected_runtime:
        raise ProtocolError("CBPUPR runtime report reconstruction drifted.")
    _validate_run_state(root, require_final=require_final)


def validate_final_attestation(
    root: Path, *, expected_checks: Mapping[str, object]
) -> None:
    attestation = read_json(root / "reports/fresh_process_attestation.json")
    children = attestation.get("child_process_results")
    child_ids = attestation.get("child_process_ids")
    parent_id = attestation.get("parent_process_id")
    if (
        not isinstance(children, list)
        or len(children) != 2
        or not isinstance(child_ids, list)
        or len(child_ids) != 2
        or not all(type(value) is int and value > 0 for value in child_ids)
        or len(set(child_ids)) != 2
        or type(parent_id) is not int
        or parent_id <= 0
        or parent_id in child_ids
    ):
        raise ProtocolError("CBPUPR fresh-process identity topology drifted.")
    for ordinal, (child, child_id) in enumerate(
        zip(children, child_ids, strict=True), start=1
    ):
        expected_result = canonical_hash(
            {"process_id": child_id, "checks": dict(expected_checks)}
        )
        if child != {
            "ordinal": ordinal,
            "process_id": child_id,
            "exit_code": 0,
            "result_hash": expected_result,
        }:
            raise ProtocolError("CBPUPR fresh-process result reconstruction drifted.")
    unhashed = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_hash"
    }
    if (
        unhashed
        != {
            "schema_version": "fixed_bank_cbpupr_fresh_process_attestation_v1",
            "status": "PASS",
            "fresh_python_process_count": 2,
            "independent_fresh_python_processes": True,
            "process_launches_sequential": True,
            "cuda_visible_devices": "",
            "worker_thread_environment": THREAD_ENVIRONMENT,
            "parent_process_id": parent_id,
            "child_process_ids": child_ids,
            "child_process_results": children,
            "reconstructed_checks_exactly_equal": True,
            "reconstructed_checks_hash": canonical_hash(expected_checks),
            "validator_entrypoint": (
                "validate_p_anchored_route_scoped_center_balanced_"
                "posterior_utility_prefix_router_bundle"
            ),
        }
        or attestation.get("attestation_hash") != canonical_hash(unhashed)
    ):
        raise ProtocolError("CBPUPR fresh-process attestation drifted.")
    report_unhashed = {
        "schema_version": "fixed_bank_cbpupr_validation_report_v1",
        "status": "PASS",
        "checks": dict(expected_checks),
        "fresh_process_attestation_hash": attestation["attestation_hash"],
        "formal_claim_authorized": False,
    }
    expected_report = {
        **report_unhashed,
        "validation_report_hash": canonical_hash(report_unhashed),
    }
    if read_json(root / "reports/validation_report.json") != expected_report:
        raise ProtocolError("CBPUPR validation report drifted.")


def _validate_run_state(root: Path, *, require_final: bool) -> None:
    state = read_json(root / "reports/run_state.json")
    timestamp = state.get("updated_at_utc")
    try:
        parsed = datetime.fromisoformat(str(timestamp))
    except ValueError as exc:
        raise ProtocolError("CBPUPR run-state timestamp drifted.") from exc
    expected_status = "COMPLETE" if require_final else "RUNNING"
    expected_phase = (
        "COMPLETE" if require_final else "CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION"
    )
    if (
        set(state)
        != {
            "schema_version",
            "status",
            "phase",
            "error",
            "error_class",
            "updated_at_utc",
            "cross_run_recovery_allowed",
            "terminal_recovery_allowed",
        }
        or state.get("schema_version") != "fixed_bank_cbpupr_run_state_v1"
        or state.get("status") != expected_status
        or state.get("phase") != expected_phase
        or state.get("error") is not None
        or state.get("error_class") is not None
        or parsed.tzinfo is None
        or state.get("cross_run_recovery_allowed") is not False
        or state.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("CBPUPR run-state contract drifted.")


__all__ = (
    "validate_final_attestation",
    "validate_scientific_reports",
)
