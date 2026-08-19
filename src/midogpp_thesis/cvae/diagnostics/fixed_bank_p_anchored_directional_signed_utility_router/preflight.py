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
    EXPECTED_UTILITY_MODEL_FIT_COUNT,
    EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
    SCRATCH_ROOT,
)
from .scratch import probe_scratch
from .workstation import assert_runtime


SCHEMA = "fixed_bank_pdsur_workstation_preflight_v1"


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
        raise ProtocolError("PDSUR recovery policy drifted.")
    neutral_runtime = dict(runtime)
    neutral_runtime["resume_policy"] = "hash_validated_atomic_phase_and_task_checkpoints"
    with tempfile.TemporaryDirectory(
        prefix=".pdsur-preflight-", dir=root.parent
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
            "double_exclusion_state_count": 0,
            "expected_outer_endpoint_model_fit_count": EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT,
            "expected_utility_model_fit_count": EXPECTED_UTILITY_MODEL_FIT_COUNT,
            "route_model_workers": CPU_WORKERS,
            "route_worker_blas_threads": 3,
            "outer_process_blas_threads": 1,
            "unused_nested_endpoint_fits_eliminated": True,
            "prior_rebinding_additional_endpoint_model_fit_count": 0,
            "cuda_disabled_after_prediction_seal": True,
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
        or payload.get("double_exclusion_state_count") != 0
        or payload.get("expected_outer_endpoint_model_fit_count")
        != EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT
        or payload.get("expected_utility_model_fit_count")
        != EXPECTED_UTILITY_MODEL_FIT_COUNT
        or payload.get("owned_task_checkpoint_replay_allowed") is not False
        or payload.get("foreign_checkpoint_reuse_forbidden") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("terminal_recovery_allowed") is not False
        or payload.get("scratch_absent_at_launch") is not True
    ):
        raise ProtocolError("PDSUR persisted workstation preflight drifted.")
    return payload


__all__ = ("load_validated_workstation_preflight", "run_workstation_preflight")
