"""Fail-closed workstation topology for the planned OE-PPUR identity."""

from __future__ import annotations

import atexit
from collections.abc import Mapping, MutableMapping
from contextlib import ExitStack
from dataclasses import dataclass
import os

from ..protocol import ProtocolError


BLAS_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
CPU_WORKER_MARKER = "MIDOGPP_OE_PPUR_CPU_OUTER_WORKER"
GPU_WORKER_MARKER = "MIDOGPP_OE_PPUR_PERSISTENT_GPU_WORKER"
_THREADPOOL_SCOPE: ExitStack | None = None
_THREADPOOL_SCOPE_ROLE: str | None = None


@dataclass(frozen=True, slots=True)
class WorkstationPlan:
    profile: str
    gpu_devices: tuple[str, str]
    persistent_gpu_prediction_workers: int
    prediction_store_dtype: str
    prediction_store_mode: str
    reduction_dtype: str
    cpu_outer_workers: int
    blas_threads_per_worker: int
    native_threads_per_worker: int
    multiprocessing_start_method: str
    nested_pools_allowed: bool
    process_transport: tuple[str, ...]
    execution_authorized: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_planned_workstation_v1",
            "profile": self.profile,
            "gpu_devices": list(self.gpu_devices),
            "persistent_gpu_prediction_workers": self.persistent_gpu_prediction_workers,
            "one_persistent_worker_per_physical_gpu": True,
            "prediction_store_dtype": self.prediction_store_dtype,
            "prediction_store_mode": self.prediction_store_mode,
            "prediction_store_read_only_for_cpu_phase": True,
            "reduction_dtype": self.reduction_dtype,
            "cpu_outer_workers": self.cpu_outer_workers,
            "blas_threads_per_worker": self.blas_threads_per_worker,
            "native_threads_per_worker": self.native_threads_per_worker,
            "threadpool_limiter_scope": "worker_process_lifetime",
            "threadpool_info_evidence_required": True,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "top_level_spawn_pool_only": True,
            "nested_pools_allowed": self.nested_pools_allowed,
            "process_transport": list(self.process_transport),
            "mappingproxy_cross_process_allowed": False,
            "estimator_object_cross_process_allowed": False,
            "memmap_object_cross_process_allowed": False,
            "execution_authorized": self.execution_authorized,
            "output_root_resolution_allowed": False,
            "scratch_root_resolution_allowed": False,
            "output_or_scratch_creation_allowed": False,
            "cross_run_recovery_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class NativeThreadpoolRecord:
    """Normalized observable row from ``threadpoolctl.threadpool_info``."""

    user_api: str
    internal_api: str
    prefix: str
    num_threads: int


@dataclass(frozen=True, slots=True)
class ThreadpoolEvidence:
    """Spawn-safe proof that a worker's persistent limiter scope is active."""

    role: str
    limiter_scope_entered: bool
    observation_performed: bool
    loaded_pool_count: int
    loaded_pools: tuple[NativeThreadpoolRecord, ...]
    all_loaded_pools_one_thread: bool


def build_workstation_plan() -> WorkstationPlan:
    """Return the immutable plan without probing or starting hardware."""

    return WorkstationPlan(
        profile="xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb",
        gpu_devices=("cuda:0", "cuda:1"),
        persistent_gpu_prediction_workers=2,
        prediction_store_dtype="float32",
        prediction_store_mode="read_only_memmap",
        reduction_dtype="float64",
        cpu_outer_workers=4,
        blas_threads_per_worker=1,
        native_threads_per_worker=1,
        multiprocessing_start_method="spawn",
        nested_pools_allowed=False,
        process_transport=("paths", "hashes", "tuples", "scalars"),
        execution_authorized=False,
    )


def workstation_payload() -> dict[str, object]:
    return build_workstation_plan().to_payload()


def initialize_cpu_outer_worker(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Hide CUDA and cap native libraries before scientific imports."""

    target = os.environ if environ is None else environ
    if target.get(CPU_WORKER_MARKER) == "1":
        raise ProtocolError("OE-PPUR nested CPU worker initialization refused.")
    if target.get(GPU_WORKER_MARKER) == "1":
        raise ProtocolError("OE-PPUR GPU worker cannot become a CPU worker.")
    target["CUDA_VISIBLE_DEVICES"] = ""
    for name in BLAS_ENVIRONMENT_NAMES:
        target[name] = "1"
    target[CPU_WORKER_MARKER] = "1"
    if environ is None:
        _enter_persistent_threadpool_scope(role="cpu")


def initialize_persistent_gpu_worker(
    device_index: int,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Bind exactly one persistent prediction worker to one physical GPU."""

    target = os.environ if environ is None else environ
    if target.get(CPU_WORKER_MARKER) == "1" or target.get(GPU_WORKER_MARKER) == "1":
        raise ProtocolError("OE-PPUR nested or role-changing GPU worker refused.")
    if int(device_index) not in (0, 1):
        raise ProtocolError("OE-PPUR GPU worker device drifted.")
    target["CUDA_VISIBLE_DEVICES"] = str(int(device_index))
    for name in BLAS_ENVIRONMENT_NAMES:
        target[name] = "1"
    target[GPU_WORKER_MARKER] = "1"
    if environ is None:
        _enter_persistent_threadpool_scope(role="gpu")


def assert_coordinator_process(environ: Mapping[str, str] | None = None) -> None:
    target = os.environ if environ is None else environ
    if target.get(CPU_WORKER_MARKER) == "1" or target.get(GPU_WORKER_MARKER) == "1":
        raise ProtocolError("OE-PPUR nested process pools are forbidden.")


def validate_worker_environment(
    environ: Mapping[str, str], *, role: str
) -> None:
    if any(environ.get(name) != "1" for name in BLAS_ENVIRONMENT_NAMES):
        raise ProtocolError("OE-PPUR worker native-thread environment drifted.")
    if role == "cpu":
        if (
            environ.get(CPU_WORKER_MARKER) != "1"
            or environ.get("CUDA_VISIBLE_DEVICES") != ""
        ):
            raise ProtocolError("OE-PPUR CPU worker environment drifted.")
    elif role == "gpu":
        if (
            environ.get(GPU_WORKER_MARKER) != "1"
            or environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}
        ):
            raise ProtocolError("OE-PPUR GPU worker environment drifted.")
    else:
        raise ProtocolError("OE-PPUR worker role is unknown.")


def capture_threadpool_evidence(*, role: str) -> ThreadpoolEvidence:
    """Observe and fail closed on every currently loaded native threadpool.

    The process initializer enters a persistent ``ExitStack`` and keeps it
    alive for the full worker lifetime.  Environment variables also constrain
    libraries imported after initialization; this observation therefore runs
    after scientific imports at the task boundary.
    """

    validate_worker_environment(os.environ, role=role)
    if _THREADPOOL_SCOPE is None or _THREADPOOL_SCOPE_ROLE != role:
        raise ProtocolError("OE-PPUR worker threadpool limiter scope is not entered.")
    try:
        from threadpoolctl import threadpool_info
    except ImportError as exc:  # pragma: no cover - workstation dependency
        raise ProtocolError("OE-PPUR requires threadpoolctl evidence.") from exc
    records: list[NativeThreadpoolRecord] = []
    for row in threadpool_info():
        user_api = str(row.get("user_api", ""))
        if user_api not in {"blas", "openmp"}:
            continue
        try:
            num_threads = int(row["num_threads"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("OE-PPUR threadpool evidence is malformed.") from exc
        records.append(
            NativeThreadpoolRecord(
                user_api=user_api,
                internal_api=str(row.get("internal_api", "")),
                prefix=str(row.get("prefix", "")),
                num_threads=num_threads,
            )
        )
    ordered = tuple(
        sorted(
            records,
            key=lambda row: (row.user_api, row.internal_api, row.prefix),
        )
    )
    capped = all(row.num_threads == 1 for row in ordered)
    if not capped:
        raise ProtocolError("OE-PPUR worker native threadpool topology drifted.")
    return ThreadpoolEvidence(
        role=role,
        limiter_scope_entered=True,
        observation_performed=True,
        loaded_pool_count=len(ordered),
        loaded_pools=ordered,
        all_loaded_pools_one_thread=True,
    )


def _enter_persistent_threadpool_scope(*, role: str) -> None:
    global _THREADPOOL_SCOPE, _THREADPOOL_SCOPE_ROLE
    if role not in {"cpu", "gpu"}:
        raise ProtocolError("OE-PPUR worker threadpool role is unknown.")
    if _THREADPOOL_SCOPE is not None or _THREADPOOL_SCOPE_ROLE is not None:
        raise ProtocolError("OE-PPUR worker threadpool limiter was already entered.")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - workstation dependency
        raise ProtocolError("OE-PPUR requires threadpoolctl.") from exc
    scope = ExitStack()
    scope.__enter__()
    try:
        scope.enter_context(threadpool_limits(limits=1))
    except Exception:
        scope.close()
        raise
    _THREADPOOL_SCOPE = scope
    _THREADPOOL_SCOPE_ROLE = role
    atexit.register(_close_persistent_threadpool_scope)


def _close_persistent_threadpool_scope() -> None:
    global _THREADPOOL_SCOPE, _THREADPOOL_SCOPE_ROLE
    scope = _THREADPOOL_SCOPE
    _THREADPOOL_SCOPE = None
    _THREADPOOL_SCOPE_ROLE = None
    if scope is not None:
        scope.close()


__all__ = (
    "BLAS_ENVIRONMENT_NAMES",
    "CPU_WORKER_MARKER",
    "GPU_WORKER_MARKER",
    "NativeThreadpoolRecord",
    "ThreadpoolEvidence",
    "WorkstationPlan",
    "assert_coordinator_process",
    "build_workstation_plan",
    "capture_threadpool_evidence",
    "initialize_cpu_outer_worker",
    "initialize_persistent_gpu_worker",
    "validate_worker_environment",
    "workstation_payload",
)
