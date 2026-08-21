"""Exact workstation admission for the two-GPU then four-worker CPU topology."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.preflight import run_label_free_workstation_preflight as _neutral
from .constants import (
    CPU_WORKERS,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_FINAL_CASE_PREDICTION_COUNT,
    EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT,
    EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY,
    EXPECTED_POLICY_REPLAY_COUNT,
    EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY,
    EXPECTED_PRIMARY_GEOMETRY_DECISION_COUNT,
    EXPECTED_PROJECTED_NO_ENVELOPE_DECISION_COUNT,
    EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT,
    EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT,
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT,
    EXPECTED_TRANSPORT_SCREEN_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    SCRATCH_ROOT,
)
from .scratch import probe_scratch
from .workstation import assert_runtime


SCHEMA = "fixed_bank_pcsi_racr_workstation_preflight_v1"


def run_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    assert_runtime(runtime)
    if (
        runtime.get("resume_policy")
        != "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only"
        or runtime.get("owned_task_checkpoint_replay_allowed") is not False
        or runtime.get("foreign_checkpoint_reuse_forbidden") is not True
        or runtime.get("cross_run_recovery_allowed") is not False
        or runtime.get("terminal_recovery_allowed") is not False
    ):
        raise ProtocolError("PCSI-RACR recovery policy drifted.")
    neutral_runtime = dict(runtime)
    neutral_runtime["resume_policy"] = "hash_validated_atomic_phase_and_task_checkpoints"
    with tempfile.TemporaryDirectory(
        prefix=".pcsi-preflight-", dir=root.parent
    ) as probe:
        result = dict(
            _neutral(
                Path(probe),
                runtime=neutral_runtime,
                expected_scratch_root=SCRATCH_ROOT,
                expected_target_action_identity_count=90,
                expected_target_probability_cell_count=810,
                expected_unique_classifier_fit_count=810,
            )
        )
    result = {
        key: value
        for key, value in result.items()
        if not str(key).casefold().endswith("_path")
    }
    result.pop("scratch_preference", None)
    result.update(
        {
            "schema_version": SCHEMA,
            "resume_policy": runtime["resume_policy"],
            "owned_task_checkpoint_replay_allowed": False,
            "foreign_checkpoint_reuse_forbidden": True,
            "cross_run_recovery_allowed": False,
            "terminal_recovery_allowed": False,
            "outer_route_count": EXPECTED_OUTER_PLAN_COUNT,
            "double_exclusion_pair_count": EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
            "expected_outer_endpoint_model_fit_count": EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
            "expected_utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
            "expected_projected_utility_model_fit_count": (
                EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY
            ),
            "expected_raw_utility_model_fit_count": (
                EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY
            ),
            "expected_legacy_utility_model_fit_count": 0,
            "expected_target_posterior_model_fit_count": (
                EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
            ),
            "expected_policy_replay_count": EXPECTED_POLICY_REPLAY_COUNT,
            "expected_projected_policy_replay_count": (
                EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY
            ),
            "expected_raw_full_action_policy_replay_count": (
                EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY
            ),
            "expected_role_bound_transport_descriptor_count": EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT,
            "expected_numeric_transport_leaf_count": EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT,
            "expected_transport_reference_summary_count": EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT,
            "expected_transport_screen_count": EXPECTED_TRANSPORT_SCREEN_COUNT,
            "expected_primary_geometry_decision_count": EXPECTED_PRIMARY_GEOMETRY_DECISION_COUNT,
            "expected_projected_no_envelope_decision_count": EXPECTED_PROJECTED_NO_ENVELOPE_DECISION_COUNT,
            "expected_final_case_prediction_count": EXPECTED_FINAL_CASE_PREDICTION_COUNT,
            "endpoint_workers": CPU_WORKERS,
            "posterior_utility_replay_workers": CPU_WORKERS,
            "route_model_workers": CPU_WORKERS,
            "route_worker_blas_threads": 3,
            "outer_process_blas_threads": 3,
            "target_posterior_process_blas_threads": 1,
            "utility_process_blas_threads": 1,
            "policy_replay_process_blas_threads": 1,
            "H_J_double_exclusion_enforced": True,
            "prior_rebinding_additional_endpoint_model_fit_count": 0,
            "cuda_disabled_after_prediction_seal": True,
            "transport_identity_level_route_noninterference_required": True,
            "transport_identity_level_route_noninterference_proven": True,
            "transport_authorization_valid": True,
            "transport_protocol_status": "ROUTE_SCOPED_OWN_CASE_NONINTERFERENCE",
            "execution_authorized": True,
            **probe_scratch(root, runtime),
        }
    )
    atomic_json(root / "reports/workstation_preflight.json", result)
    return result


def load_validated_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    payload = read_json(root / "reports/workstation_preflight.json")
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("resume_policy") != runtime.get("resume_policy")
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or payload.get("persistent_gpu_workers") != 2
        or payload.get("classifier_workers") != 4
        or payload.get("blas_threads_per_classifier_worker") != 3
        or payload.get("target_probability_cell_count") != 810
        or payload.get("target_unique_classifier_fit_count") != 810
        or payload.get("outer_route_count") != EXPECTED_OUTER_PLAN_COUNT
        or payload.get("double_exclusion_pair_count")
        != EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT
        or payload.get("expected_outer_endpoint_model_fit_count")
        != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
        or payload.get("expected_utility_model_fit_count")
        != EXPECTED_UTILITY_MODEL_FIT_COUNT
        or payload.get("expected_projected_utility_model_fit_count")
        != EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY
        or payload.get("expected_raw_utility_model_fit_count")
        != EXPECTED_RACR_MODEL_FIT_COUNT_PER_GEOMETRY
        or payload.get("expected_legacy_utility_model_fit_count") != 0
        or payload.get("expected_target_posterior_model_fit_count")
        != EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT
        or payload.get("expected_policy_replay_count")
        != EXPECTED_POLICY_REPLAY_COUNT
        or payload.get("expected_projected_policy_replay_count")
        != EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY
        or payload.get("expected_raw_full_action_policy_replay_count")
        != EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY
        or payload.get("expected_role_bound_transport_descriptor_count")
        != EXPECTED_ROLE_BOUND_TRANSPORT_DESCRIPTOR_COUNT
        or payload.get("expected_numeric_transport_leaf_count")
        != EXPECTED_NUMERIC_TRANSPORT_LEAF_COUNT
        or payload.get("expected_transport_reference_summary_count")
        != EXPECTED_TRANSPORT_REFERENCE_SUMMARY_COUNT
        or payload.get("expected_transport_screen_count")
        != EXPECTED_TRANSPORT_SCREEN_COUNT
        or payload.get("expected_primary_geometry_decision_count")
        != EXPECTED_PRIMARY_GEOMETRY_DECISION_COUNT
        or payload.get("expected_projected_no_envelope_decision_count")
        != EXPECTED_PROJECTED_NO_ENVELOPE_DECISION_COUNT
        or payload.get("expected_final_case_prediction_count")
        != EXPECTED_FINAL_CASE_PREDICTION_COUNT
        or payload.get("endpoint_workers") != CPU_WORKERS
        or payload.get("posterior_utility_replay_workers") != CPU_WORKERS
        or payload.get("outer_process_blas_threads") != 3
        or payload.get("target_posterior_process_blas_threads") != 1
        or payload.get("utility_process_blas_threads") != 1
        or payload.get("policy_replay_process_blas_threads") != 1
        or payload.get("H_J_double_exclusion_enforced") is not True
        or payload.get("transport_identity_level_route_noninterference_required")
        is not True
        or payload.get("transport_identity_level_route_noninterference_proven")
        is not True
        or payload.get("transport_authorization_valid") is not True
        or payload.get("transport_protocol_status")
        != "ROUTE_SCOPED_OWN_CASE_NONINTERFERENCE"
        or payload.get("execution_authorized") is not True
        or payload.get("owned_task_checkpoint_replay_allowed") is not False
        or payload.get("foreign_checkpoint_reuse_forbidden") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("terminal_recovery_allowed") is not False
        or payload.get("scratch_absent_at_launch") is not True
    ):
        raise ProtocolError("PCSI-RACR persisted workstation preflight drifted.")
    return payload


__all__ = ("load_validated_workstation_preflight", "run_workstation_preflight")
