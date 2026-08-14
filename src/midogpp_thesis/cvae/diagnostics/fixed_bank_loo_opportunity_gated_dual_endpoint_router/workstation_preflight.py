"""Exact workstation admission for the GPU-then-CPU experiment topology."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .artifact_writers import persist_json
from ...runtime.preflight import run_label_free_workstation_preflight as _preflight
from .experiment_contracts import SCRATCH_ROOT
from .scratch_policy import probe_dedicated_scratch


def run_label_free_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    expected = {
        "resume_policy": "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only",
        "owned_task_checkpoint_replay_allowed": False,
        "foreign_checkpoint_reuse_forbidden": True,
        "cross_run_recovery_allowed": False,
        "terminal_recovery_allowed": False,
        "successful_phase_checkpoint_cleanup_after_validated_global_seal": True,
        "probability_storage_dtype": "float32",
        "confusion_count_dtype": "int64",
    }
    if (
        any(runtime.get(key) != value for key, value in expected.items())
        or tuple(runtime.get("scratch_preference", ()))
        != (SCRATCH_ROOT, "artifact_parent")
    ):
        raise ProtocolError("Dual-endpoint recovery/topology contract drifted.")
    neutral = dict(runtime)
    neutral["resume_policy"] = "hash_validated_atomic_phase_and_task_checkpoints"
    neutral["scratch_preference"] = [SCRATCH_ROOT, "artifact_parent"]
    with tempfile.TemporaryDirectory(
        prefix=".dual-endpoint-preflight-", dir=root.parent
    ) as probe:
        probed = dict(
            _preflight(
                Path(probe),
                runtime=neutral,
                expected_scratch_root=SCRATCH_ROOT,
                expected_target_action_identity_count=90,
                expected_target_probability_cell_count=810,
                expected_unique_classifier_fit_count=810,
            )
        )
    probed = {
        key: value
        for key, value in probed.items()
        if not str(key).casefold().endswith("_path")
    }
    probed.pop("scratch_preference", None)
    probed.update(
        {
            "schema_version": "fixed_bank_dual_endpoint_workstation_preflight_v1",
            "resume_policy": expected["resume_policy"],
            # The literal local path remains a launch-time config contract.  The
            # sealed report carries only a stable role/id so bundle persistence
            # is path-free.
            "scratch_root_id": Path(SCRATCH_ROOT).name,
            "scratch_fallback_role": "artifact_parent",
            "owned_task_checkpoint_replay_allowed": False,
            "task_checkpoints_are_intra_launch_atomicity_only": True,
            "foreign_checkpoint_reuse_forbidden": True,
            "cross_run_recovery_allowed": False,
            "terminal_recovery_allowed": False,
            "successful_phase_checkpoint_cleanup_after_validated_global_seal": True,
            "cuda_disabled_after_prediction_seal": True,
            "probability_storage_dtype": "float32",
            "confusion_count_dtype": "int64",
            "scientific_reductions_dtype": "float64",
            **probe_dedicated_scratch(runtime),
        }
    )
    persist_json(root / "reports/workstation_preflight.json", probed)
    return probed


def load_validated_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    payload = read_json(root / "reports/workstation_preflight.json")
    if (
        payload.get("schema_version")
        != "fixed_bank_dual_endpoint_workstation_preflight_v1"
        or payload.get("status") != "PASS"
        or payload.get("resume_policy") != runtime.get("resume_policy")
        or payload.get("scratch_root_id") != Path(SCRATCH_ROOT).name
        or payload.get("scratch_fallback_role") != "artifact_parent"
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or payload.get("persistent_gpu_workers") != 2
        or payload.get("classifier_workers") != 4
        or payload.get("blas_threads_per_classifier_worker") != 3
        or payload.get("target_probability_cell_count") != 810
        or payload.get("target_action_identity_count") != 90
        or payload.get("target_unique_classifier_fit_count") != 810
        or payload.get("maximum_total_classifier_fit_count") != 810
        or payload.get("probability_storage_dtype") != "float32"
        or payload.get("confusion_count_dtype") != "int64"
        or payload.get("scientific_reductions_dtype") != "float64"
        or payload.get("owned_task_checkpoint_replay_allowed") is not False
        or payload.get("task_checkpoints_are_intra_launch_atomicity_only") is not True
        or payload.get("foreign_checkpoint_reuse_forbidden") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("terminal_recovery_allowed") is not False
        or payload.get(
            "successful_phase_checkpoint_cleanup_after_validated_global_seal"
        )
        is not True
        or payload.get("cuda_disabled_after_prediction_seal") is not True
        or payload.get("dedicated_scratch_absent_at_launch") is not True
        or payload.get("dedicated_scratch_parent_writable") is not True
        or type(payload.get("dedicated_scratch_free_bytes_at_launch")) is not int
        or int(payload["dedicated_scratch_free_bytes_at_launch"]) <= 0
    ):
        raise ProtocolError("Dual-endpoint persisted preflight drifted.")
    return payload


__all__ = (
    "load_validated_workstation_preflight",
    "run_label_free_workstation_preflight",
)
