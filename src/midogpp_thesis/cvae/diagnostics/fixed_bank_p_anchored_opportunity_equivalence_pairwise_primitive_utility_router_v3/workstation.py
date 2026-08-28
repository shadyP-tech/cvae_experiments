"""Pure workstation plan and observation validation for OE-PPUR v3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import multiprocessing
import os
import subprocess

from ...protocol import ProtocolError
from .hashing import canonical_hash


GPU_WORKER_COUNT = 2
CPU_SPAWN_WORKER_COUNT = 4
BLAS_THREADS_PER_CPU_WORKER = 1
PREDICTION_STORAGE_DTYPE = "<f4"
REDUCTION_DTYPE = "<f8"
MULTIPROCESSING_START_METHOD = "spawn"
CPU_WORKER_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True, slots=True)
class WorkstationPlanReceipt:
    gpu_count: int
    gpu_names: tuple[str, ...]
    cpu_count: int
    start_method: str
    cpu_worker_environment: tuple[tuple[str, str], ...]
    dto_pickle_round_trip_validated: bool
    filesystem_mutation_performed: bool
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.gpu_names)
        environment = tuple((str(key), str(value)) for key, value in self.cpu_worker_environment)
        if (
            type(self.gpu_count) is not int
            or self.gpu_count < GPU_WORKER_COUNT
            or len(names) != self.gpu_count
            or not all("A5000" in value.upper() for value in names[:2])
            or type(self.cpu_count) is not int
            or self.cpu_count < CPU_SPAWN_WORKER_COUNT
            or self.start_method != MULTIPROCESSING_START_METHOD
            or environment != tuple(CPU_WORKER_ENVIRONMENT.items())
            or self.dto_pickle_round_trip_validated is not True
            or self.filesystem_mutation_performed is not False
        ):
            raise ProtocolError("OE-PPUR v3 workstation plan drifted.")
        object.__setattr__(self, "gpu_names", names)
        object.__setattr__(self, "cpu_worker_environment", environment)
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_workstation_receipt_v1",
            "observed_gpu_count": self.gpu_count,
            "observed_gpu_names": list(self.gpu_names),
            "persistent_gpu_worker_count": GPU_WORKER_COUNT,
            "gpu_assignment": ["cuda:0", "cuda:1"],
            "observed_cpu_count": self.cpu_count,
            "spawn_cpu_worker_count": CPU_SPAWN_WORKER_COUNT,
            "multiprocessing_start_method": MULTIPROCESSING_START_METHOD,
            "cpu_worker_environment": dict(self.cpu_worker_environment),
            "cuda_visible_to_cpu_workers": False,
            "blas_threads_per_cpu_worker": BLAS_THREADS_PER_CPU_WORKER,
            "prediction_storage_dtype": PREDICTION_STORAGE_DTYPE,
            "reduction_dtype": REDUCTION_DTYPE,
            "worker_transport": "pickle_primitive_dto_only",
            "dto_pickle_round_trip_validated": True,
            "nested_process_pools_allowed": False,
            "filesystem_mutation_performed": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def workstation_plan_payload() -> dict[str, object]:
    body = {
        "schema_version": "oe_ppur_v3_workstation_plan_v1",
        "persistent_gpu_workers": GPU_WORKER_COUNT,
        "gpu_devices": ["cuda:0", "cuda:1"],
        "spawn_cpu_workers": CPU_SPAWN_WORKER_COUNT,
        "multiprocessing_start_method": MULTIPROCESSING_START_METHOD,
        "cpu_worker_environment": dict(CPU_WORKER_ENVIRONMENT),
        "cuda_visible_to_cpu_workers": False,
        "blas_threads_per_cpu_worker": BLAS_THREADS_PER_CPU_WORKER,
        "prediction_matrix_dtype": PREDICTION_STORAGE_DTYPE,
        "prediction_matrix_memory_order": "C",
        "reduction_dtype": REDUCTION_DTYPE,
        "process_transport": "pickle_primitive_dto_only",
        "nested_process_pools_allowed": False,
        "cross_run_recovery_allowed": False,
    }
    return {**body, "plan_hash": canonical_hash(body)}


def validate_workstation_observation(
    observed: Mapping[str, object],
    *,
    dto_pickle_round_trip_validated: bool,
) -> WorkstationPlanReceipt:
    """Validate caller-independent probe output without changing host state."""

    if not isinstance(observed, Mapping):
        raise ProtocolError("OE-PPUR v3 workstation observation is malformed.")
    names = observed.get("gpu_names")
    if not isinstance(names, (tuple, list)):
        raise ProtocolError("OE-PPUR v3 GPU observation is malformed.")
    try:
        gpu_count = int(observed.get("gpu_count", -1))
        cpu_count = int(observed.get("cpu_count", -1))
    except (TypeError, ValueError) as exc:
        raise ProtocolError("OE-PPUR v3 workstation counts are malformed.") from exc
    return WorkstationPlanReceipt(
        gpu_count=gpu_count,
        gpu_names=tuple(str(value) for value in names),
        cpu_count=cpu_count,
        start_method=str(observed.get("start_method", "")),
        cpu_worker_environment=tuple(CPU_WORKER_ENVIRONMENT.items()),
        dto_pickle_round_trip_validated=dto_pickle_round_trip_validated,
        filesystem_mutation_performed=False,
    )


def cpu_worker_environment() -> dict[str, str]:
    """Return, but never apply, the exact environment for spawn workers."""

    return dict(CPU_WORKER_ENVIRONMENT)


def preflight_workstation() -> WorkstationPlanReceipt:
    """Probe the production host; callers cannot inject workstation facts."""

    # Import lazily so this module remains a leaf during package import.  This
    # is the exact primitive DTO transported by the four spawn workers, rather
    # than a truthy caller assertion about pickle compatibility.
    from .execution.dto import PrimitiveWorkerTask, assert_pickle_round_trip

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("OE-PPUR v3 workstation GPU probe failed.") from exc
    names = tuple(row.strip() for row in completed.stdout.splitlines() if row.strip())
    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise ProtocolError("OE-PPUR v3 workstation CPU topology is unavailable.")
    probe = PrimitiveWorkerTask(
        task_id="oe-ppur-v3-workstation-pickle-probe",
        outer_center_id="0",
        inner_fold_id=0,
        row_start=0,
        row_stop=1,
        source_training_surface_hash="1" * 64,
        candidate_pool_receipt_hash="2" * 64,
        compiled_action_surface_hash="3" * 64,
        random_seed=0,
    )
    dto_round_trip_validated = assert_pickle_round_trip(probe) == probe
    return validate_workstation_observation(
        {
            "gpu_count": len(names),
            "gpu_names": names,
            "cpu_count": cpu_count,
            "start_method": multiprocessing.get_context("spawn").get_start_method(),
        },
        dto_pickle_round_trip_validated=dto_round_trip_validated,
    )


__all__ = (
    "BLAS_THREADS_PER_CPU_WORKER",
    "CPU_SPAWN_WORKER_COUNT",
    "CPU_WORKER_ENVIRONMENT",
    "GPU_WORKER_COUNT",
    "MULTIPROCESSING_START_METHOD",
    "PREDICTION_STORAGE_DTYPE",
    "REDUCTION_DTYPE",
    "WorkstationPlanReceipt",
    "cpu_worker_environment",
    "preflight_workstation",
    "validate_workstation_observation",
    "workstation_plan_payload",
)
