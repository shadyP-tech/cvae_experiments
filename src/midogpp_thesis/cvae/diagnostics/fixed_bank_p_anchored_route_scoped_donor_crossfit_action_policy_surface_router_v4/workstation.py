"""Exact two-A5000 then four-spawn-worker P-DCAPS v4 boundary."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
from typing import Iterator, Mapping

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, read_json
from ...runtime.preflight import (
    REQUIRED_THREAD_ENVIRONMENT,
    run_label_free_workstation_preflight as _neutral,
)
from .identity import canonical_hash
from .outer_workers import BLAS_ENVIRONMENT_NAMES
from .scratch import probe_scratch
from .worker_dtos import WORKER_DEPTH_ENV


PREFLIGHT_MEMBER = "reports/workstation_preflight.json"
PREFLIGHT_SCHEMA = "pdcaps_v4_workstation_preflight_v1"
CPU_WORKERS = 4
CLASSIFIER_THREADS_PER_WORKER = 3
OUTER_BLAS_THREADS_PER_WORKER = 1
_PARENT_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class WorkstationEstimate:
    outer_centers: int
    pseudo_routes: int
    attempted_action_cells: int
    maximum_crossing_actions: int
    maximum_prefix_cells: int
    action_ridge_fits: int
    policy_ridge_fits: int
    estimated_dense_bytes: int
    estimate_hash: str = field(init=False)

    @property
    def estimate_role(self) -> str:
        return (
            "pessimistic_dense_calibration_record_cap_"
            "excluding_process_and_library_overhead"
        )

    def __post_init__(self) -> None:
        values = (
            self.outer_centers,
            self.pseudo_routes,
            self.attempted_action_cells,
            self.maximum_crossing_actions,
            self.maximum_prefix_cells,
            self.action_ridge_fits,
            self.policy_ridge_fits,
            self.estimated_dense_bytes,
        )
        if any(isinstance(value, bool) or int(value) <= 0 for value in values):
            raise ProtocolError("P-DCAPS v4 workstation estimate drifted.")
        object.__setattr__(
            self,
            "estimate_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_workstation_estimate_v1",
                    "values": values,
                    "estimate_role": self.estimate_role,
                    "measured_peak_memory": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_centers": self.outer_centers,
            "pseudo_routes": self.pseudo_routes,
            "attempted_action_cells": self.attempted_action_cells,
            "maximum_crossing_actions": self.maximum_crossing_actions,
            "maximum_prefix_cells": self.maximum_prefix_cells,
            "action_ridge_fits": self.action_ridge_fits,
            "policy_ridge_fits": self.policy_ridge_fits,
            "estimated_dense_bytes": self.estimated_dense_bytes,
            "estimate_role": self.estimate_role,
            "measured_peak_memory": False,
            "estimate_hash": self.estimate_hash,
        }


def estimate_workstation_surface(
    *,
    case_count: int = 218,
    outer_centers: int = 9,
    action_strata: int = 6,
    feature_count: int = 14,
) -> WorkstationEstimate:
    pseudo_routes = (outer_centers - 1) * case_count
    attempted = pseudo_routes * action_strata
    maximum_prefix = pseudo_routes * (case_count + 1)
    donor_count = outer_centers - 1
    ridge_per_layer = outer_centers * 3 * (
        1 + donor_count + donor_count * (donor_count - 1) // 2
    )
    dense_bytes = (
        attempted * (feature_count * 8 + 3 * 8 + 8)
        + maximum_prefix * (12 * 8 + 3 * 8 + 8)
    )
    return WorkstationEstimate(
        outer_centers,
        pseudo_routes,
        attempted,
        attempted,
        maximum_prefix,
        ridge_per_layer,
        ridge_per_layer,
        dense_bytes,
    )


def assert_runtime(runtime: Mapping[str, object]) -> None:
    """Reject any topology, budget, recovery, or nesting drift."""

    from .config import canonical_runtime_payload

    if dict(runtime) != canonical_runtime_payload():
        raise ProtocolError("P-DCAPS v4 workstation runtime contract drifted.")
    if (
        tuple(runtime.get("generation_devices", ())) != ("cuda:0", "cuda:1")
        or runtime.get("persistent_source_workers") is not True
        or int(runtime.get("persistent_generation_worker_count", -1)) != 2
        or int(runtime.get("source_workers_per_device", -1)) != 1
        or int(runtime.get("generation_workers_per_device", -1)) != 1
        or int(runtime.get("outer_worker_count", -1)) != CPU_WORKERS
        or int(runtime.get("outer_process_workers", -1)) != CPU_WORKERS
        or int(runtime.get("classifier_workers", -1)) != CPU_WORKERS
        or int(runtime.get("classifier_threads_per_worker", -1))
        != CLASSIFIER_THREADS_PER_WORKER
        or int(runtime.get("calibration_threads_per_worker", -1))
        != OUTER_BLAS_THREADS_PER_WORKER
        or runtime.get("outer_task_unit") != "one_complete_outer_center_H"
        or int(runtime.get("outer_task_count", -1)) != 9
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("nested_process_pools_forbidden") is not True
        or runtime.get("worker_DTOs_are_plain_pickle_safe_values") is not True
        or runtime.get(
            "worker_DTOs_forbid_mappingproxy_estimators_handles_and_closures"
        )
        is not True
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("cross_run_recovery_allowed") is not False
        or runtime.get("terminal_recovery_allowed") is not False
        or runtime.get("v1_scratch_or_checkpoint_reuse_forbidden") is not True
        or runtime.get("v2_scratch_or_checkpoint_reuse_forbidden") is not True
        or runtime.get("v3_scratch_or_checkpoint_reuse_forbidden") is not True
        or runtime.get("v1_v2_v3_output_or_run_state_reuse_forbidden") is not True
    ):
        raise ProtocolError("P-DCAPS v4 execution topology drifted.")


def run_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    """Probe neutral hardware, add P-DCAPS fields, then atomically persist once."""

    assert_runtime(runtime)
    estimate = estimate_workstation_surface()
    if estimate.estimated_dense_bytes > int(runtime["maximum_dense_surface_bytes"]):
        raise ProtocolError("P-DCAPS v4 dense surface exceeds the frozen RAM budget.")
    target = Path(root)
    if target.is_symlink() or not target.is_dir():
        raise ProtocolError("P-DCAPS v4 preflight output root is unsafe.")
    # The neutral helper writes its base report.  Keep that mutation confined to
    # a disposable sibling and publish only the extended P-DCAPS payload.
    with tempfile.TemporaryDirectory(
        prefix=".pdcaps-v4-preflight-", dir=target.parent
    ) as probe:
        result = dict(
            _neutral(
                Path(probe),
                runtime=runtime,
                expected_scratch_root=str(runtime["scratch_preference"][0]),
                expected_target_action_identity_count=90,
                expected_target_probability_cell_count=810,
                expected_unique_classifier_fit_count=810,
                expected_resume_policy=str(runtime["resume_policy"]),
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
            "schema_version": PREFLIGHT_SCHEMA,
            "execution_authorized": True,
            "gpu_phase": {
                "device_order": ["cuda:0", "cuda:1"],
                "persistent_worker_count": 2,
                "workers_per_device": 1,
                "completes_before_cpu_phase": True,
            },
            "prediction_phase": {
                "starts_after_persistent_gpu_workers_exit": True,
                "spawn_worker_count": CPU_WORKERS,
                "blas_threads_per_worker": CLASSIFIER_THREADS_PER_WORKER,
                "cuda_hidden": True,
            },
            "cpu_phase": {
                "start_method": "spawn",
                "outer_process_workers": CPU_WORKERS,
                "outer_task_count": 9,
                "outer_task_unit": "one_complete_outer_center_H",
                "blas_threads_per_worker": OUTER_BLAS_THREADS_PER_WORKER,
                "cuda_hidden": True,
                "nested_process_pools": False,
            },
            "outer_process_workers": CPU_WORKERS,
            "outer_process_blas_threads": OUTER_BLAS_THREADS_PER_WORKER,
            "nested_process_pools": False,
            "worker_DTOs_plain_pickle_safe": True,
            "mappingproxy_cross_process_boundary_forbidden": True,
            "phase_disjoint_gpu_and_cpu_pools": True,
            "resume_policy": runtime["resume_policy"],
            "cross_run_recovery_allowed": False,
            "terminal_recovery_allowed": False,
            "v1_scratch_or_checkpoint_used": False,
            "v2_scratch_or_checkpoint_used": False,
            "v3_scratch_or_checkpoint_used": False,
            "v1_v2_v3_output_or_run_state_used": False,
            "unified_worker_depth_environment": WORKER_DEPTH_ENV,
            "pdcaps_surface_estimate": estimate.to_payload(),
            **probe_scratch(target, runtime),
        }
    )
    atomic_json(target / PREFLIGHT_MEMBER, result)
    return MappingProxyType(result)


def load_validated_workstation_preflight(
    root: Path, *, runtime: Mapping[str, object]
) -> Mapping[str, object]:
    assert_runtime(runtime)
    payload = read_json(Path(root) / PREFLIGHT_MEMBER)
    expected_estimate = estimate_workstation_surface().to_payload()
    gpu_phase = payload.get("gpu_phase")
    prediction_phase = payload.get("prediction_phase")
    cpu_phase = payload.get("cpu_phase")
    gpus = payload.get("gpus")
    if (
        payload.get("schema_version") != PREFLIGHT_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("execution_authorized") is not True
        or payload.get("generation_devices") != ["cuda:0", "cuda:1"]
        or payload.get("persistent_gpu_workers") != 2
        or payload.get("classifier_workers") != 4
        or payload.get("blas_threads_per_classifier_worker")
        != CLASSIFIER_THREADS_PER_WORKER
        or payload.get("target_action_identity_count") != 90
        or payload.get("target_probability_cell_count") != 810
        or payload.get("target_unique_classifier_fit_count") != 810
        or payload.get("maximum_total_classifier_fit_count") != 810
        or payload.get("gpu_then_cpu_phase_order") is not True
        or payload.get("parent_cuda_initialized") is not False
        or payload.get("cuda_visible_devices") != "0,1"
        or payload.get("thread_environment") != dict(REQUIRED_THREAD_ENVIRONMENT)
        or int(payload.get("available_cpu_affinity_count", -1))
        < int(runtime["minimum_logical_cpu_count"])
        or int(payload.get("physical_ram_bytes", -1))
        < int(runtime["minimum_physical_ram_bytes"])
        or int(payload.get("disk_free_bytes_at_launch", -1))
        < int(runtime["minimum_artifact_disk_free_bytes"])
        or not isinstance(gpus, list)
        or len(gpus) != 2
        or [row.get("index") for row in gpus if isinstance(row, Mapping)] != [0, 1]
        or any(
            "RTX A5000" not in str(row.get("name"))
            for row in gpus
            if isinstance(row, Mapping)
        )
        or any(
            int(row.get("memory_free_mib", -1))
            < int(runtime["minimum_gpu_free_mib_per_device"])
            for row in gpus
            if isinstance(row, Mapping)
        )
        or not isinstance(gpu_phase, Mapping)
        or gpu_phase.get("device_order") != ["cuda:0", "cuda:1"]
        or gpu_phase.get("persistent_worker_count") != 2
        or gpu_phase.get("completes_before_cpu_phase") is not True
        or not isinstance(prediction_phase, Mapping)
        or prediction_phase.get("starts_after_persistent_gpu_workers_exit")
        is not True
        or prediction_phase.get("spawn_worker_count") != 4
        or prediction_phase.get("blas_threads_per_worker")
        != CLASSIFIER_THREADS_PER_WORKER
        or prediction_phase.get("cuda_hidden") is not True
        or not isinstance(cpu_phase, Mapping)
        or cpu_phase.get("start_method") != "spawn"
        or cpu_phase.get("outer_process_workers") != 4
        or cpu_phase.get("outer_task_count") != 9
        or cpu_phase.get("blas_threads_per_worker")
        != OUTER_BLAS_THREADS_PER_WORKER
        or cpu_phase.get("cuda_hidden") is not True
        or cpu_phase.get("nested_process_pools") is not False
        or payload.get("outer_process_blas_threads")
        != OUTER_BLAS_THREADS_PER_WORKER
        or payload.get("nested_process_pools") is not False
        or payload.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or payload.get("cross_run_recovery_allowed") is not False
        or payload.get("terminal_recovery_allowed") is not False
        or payload.get("scratch_absent_at_launch") is not True
        or payload.get("scratch_recovery_used") is not False
        or int(payload.get("scratch_free_bytes_at_launch", -1))
        < int(runtime["minimum_scratch_disk_free_bytes"])
        or payload.get("v1_scratch_or_checkpoint_used") is not False
        or payload.get("v2_scratch_or_checkpoint_used") is not False
        or payload.get("v3_scratch_or_checkpoint_used") is not False
        or payload.get("v1_v2_v3_output_or_run_state_used") is not False
        or payload.get("unified_worker_depth_environment") != WORKER_DEPTH_ENV
        or payload.get("pdcaps_surface_estimate") != expected_estimate
    ):
        raise ProtocolError("P-DCAPS v4 persisted workstation preflight drifted.")
    return MappingProxyType(payload)


def enter_cuda_free_cpu_phase() -> None:
    """Permanently hide CUDA and cap the orchestration parent to one BLAS thread."""

    global _PARENT_THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - production dependency
        raise ProtocolError("P-DCAPS v4 runtime lacks threadpoolctl.") from exc
    _PARENT_THREADPOOL_LIMITER = threadpool_limits(limits=1)


def assert_cuda_free_cpu_phase() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("P-DCAPS v4 CPU parent still exposes CUDA.")
    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("P-DCAPS v4 parent initialized CUDA.")
    from threadpoolctl import threadpool_info

    pools = tuple(row for row in threadpool_info() if row.get("user_api") == "blas")
    if pools and any(int(row.get("num_threads", -1)) != 1 for row in pools):
        raise ProtocolError("P-DCAPS v4 parent BLAS topology is not one thread.")


@contextmanager
def cpu_phase_environment(
    threads_per_worker: int = OUTER_BLAS_THREADS_PER_WORKER,
) -> Iterator[None]:
    """Prepare one-level spawned H workers; the worker initializer sets depth."""

    if (
        isinstance(threads_per_worker, bool)
        or int(threads_per_worker) != threads_per_worker
        or int(threads_per_worker) != OUTER_BLAS_THREADS_PER_WORKER
        or os.environ.get(WORKER_DEPTH_ENV) is not None
    ):
        raise ProtocolError("P-DCAPS v4 CPU worker environment drifted.")
    names = ("CUDA_VISIBLE_DEVICES", *BLAS_ENVIRONMENT_NAMES)
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for name in BLAS_ENVIRONMENT_NAMES:
            os.environ[name] = str(int(threads_per_worker))
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


__all__ = (
    "CLASSIFIER_THREADS_PER_WORKER",
    "CPU_WORKERS",
    "OUTER_BLAS_THREADS_PER_WORKER",
    "PREFLIGHT_MEMBER",
    "PREFLIGHT_SCHEMA",
    "WorkstationEstimate",
    "assert_cuda_free_cpu_phase",
    "assert_runtime",
    "cpu_phase_environment",
    "enter_cuda_free_cpu_phase",
    "estimate_workstation_surface",
    "load_validated_workstation_preflight",
    "run_workstation_preflight",
)
