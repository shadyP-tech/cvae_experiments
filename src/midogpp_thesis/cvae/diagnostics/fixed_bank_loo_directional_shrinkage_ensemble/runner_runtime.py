"""Run locking, CUDA phase boundary, and fail-closed run-state admission."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
from typing import Iterator, Mapping
from typing import Sequence
import multiprocessing as mp

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json
from .bundle import assert_closed_world, relative_files
from .persistence import write_run_state


_ROUTE_SURFACE: object | None = None
_ROUTE_THREADPOOL_LIMITER: object | None = None
_PARENT_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class RouteJobResult:
    plan: object
    counts: tuple[object, ...]
    gains: tuple[object, ...]
    endpoint_decisions: tuple[object, ...]
    control_decisions: tuple[object, ...]


def execute_route_jobs(
    probability_surface: object,
    jobs: Sequence[Mapping[str, object]],
    *,
    workers: int,
    threads_per_worker: int,
) -> tuple[RouteJobResult, ...]:
    """Score/select 218 independent routes in four spawned 3-thread workers."""

    tasks = tuple(dict(job) for job in jobs)
    if workers != 4 or threads_per_worker != 3 or not tasks:
        raise ProtocolError("Directional-shrinkage route worker topology drifted.")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_initialize_route_worker,
        initargs=(probability_surface, threads_per_worker),
    ) as executor:
        results = tuple(executor.map(_execute_route_job, tasks, chunksize=1))
    if len(results) != len(tasks) or any(
        result.plan != tasks[index]["plan"] for index, result in enumerate(results)
    ):
        raise ProtocolError("Directional-shrinkage route worker result order drifted.")
    return results


def _initialize_route_worker(surface: object, threads: int) -> None:
    global _ROUTE_SURFACE, _ROUTE_THREADPOOL_LIMITER
    if threads != 3:
        raise ProtocolError("Directional-shrinkage route worker threads drifted.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "3"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - sklearn runtime dependency
        raise ProtocolError(
            "Directional-shrinkage route worker lacks threadpoolctl."
        ) from exc
    _ROUTE_THREADPOOL_LIMITER = threadpool_limits(limits=threads)
    _ROUTE_SURFACE = surface


def _execute_route_job(job: Mapping[str, object]) -> RouteJobResult:
    if _ROUTE_SURFACE is None or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Directional-shrinkage route worker was not initialized.")
    from threadpoolctl import threadpool_info

    blas = tuple(
        row for row in threadpool_info() if row.get("user_api") == "blas"
    )
    if blas and any(int(row.get("num_threads", -1)) != 3 for row in blas):
        raise ProtocolError(
            "Directional-shrinkage route worker BLAS topology is not three threads."
        )
    from .decisions import (
        select_arm_decisions,
        select_matched_g_decisions,
        select_nested_frequency_committee_control,
        select_raw_directional_loo_control,
    )
    from .scoring import score_case_action_confusions, score_loo_directional_gains

    plan = job["plan"]
    labels = tuple(job["support_labels"])
    priors = tuple(job["donor_priors"])
    counts = score_case_action_confusions(_ROUTE_SURFACE, labels)
    gains = score_loo_directional_gains(counts, plan)
    dcse = select_arm_decisions(
        method_id="DCSE_LOO",
        target_center=str(getattr(plan, "target_center")),
        case_id=str(getattr(plan, "case_id")),
        support_gains=gains,
        donor_priors=priors,
    )
    matched = select_matched_g_decisions(
        target_center=str(getattr(plan, "target_center")),
        case_id=str(getattr(plan, "case_id")),
        donor_priors=priors,
    )
    raw = select_raw_directional_loo_control(
        target_center=str(getattr(plan, "target_center")),
        case_id=str(getattr(plan, "case_id")),
        support_gains=gains,
    )
    frequency = select_nested_frequency_committee_control(
        plan=plan,
        support_counts=counts,
    )
    return RouteJobResult(plan, counts, gains, (*dcse, *matched), (raw, frequency))


def assert_launch_files(root: Path, config: object) -> None:
    required = (
        root / "config.resolved.yaml",
        root / "provenance/input_artifacts.json",
    )
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ProtocolError("Directional-shrinkage launch files are absent or unsafe.")
    if Path(getattr(config, "source_path")).resolve() != required[0].resolve():
        raise ProtocolError("Directional-shrinkage config is not its persisted snapshot.")


def assert_workspace_resolved_paths(config: object, *, root: Path) -> None:
    values = (
        root,
        getattr(config, "artifact_root"),
        getattr(config, "expert_bank_root"),
        getattr(config, "generation_lock_root"),
        getattr(config, "test_cache_root"),
        getattr(config, "test_manifest_path"),
        getattr(config, "test_consumption_ledger_path"),
        getattr(config, "ledger_amendment_path"),
    )
    if any(not Path(value).is_absolute() for value in values) or root.resolve() != Path(
        getattr(config, "artifact_root")
    ).resolve():
        raise ProtocolError("Directional-shrinkage requires workspace-resolved paths.")


def reject_existing_run_state(root: Path) -> None:
    """Recovery is absent: any former run state is foreign, partial, or complete."""

    state = root / "reports/run_state.json"
    if not state.exists():
        return
    if state.is_symlink() or not state.is_file():
        raise ProtocolError("Directional-shrinkage run state is unsafe.")
    payload = read_json(state)
    raise ProtocolError(
        "Directional-shrinkage cross-run recovery is forbidden; "
        f"existing status={payload.get('status')}, phase={payload.get('phase')}."
    )


def enter_cuda_free_cpu_phase() -> None:
    global _PARENT_THREADPOOL_LIMITER
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Parent remains one BLAS thread. Spawned classifier children set 3 each.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - sklearn runtime dependency
        raise ProtocolError(
            "Directional-shrinkage parent lacks threadpoolctl."
        ) from exc
    _PARENT_THREADPOOL_LIMITER = threadpool_limits(limits=1)


def assert_cuda_free_cpu_phase() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Directional-shrinkage CPU phase still exposes CUDA.")
    import sys

    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Directional-shrinkage parent initialized CUDA.")
    from threadpoolctl import threadpool_info

    blas = tuple(row for row in threadpool_info() if row.get("user_api") == "blas")
    if blas and any(int(row.get("num_threads", -1)) != 1 for row in blas):
        raise ProtocolError(
            "Directional-shrinkage parent BLAS topology is not one thread."
        )


def observe(dependencies: object, phase: str) -> None:
    callback = getattr(dependencies, "phase_observer", None)
    if callback is not None:
        callback(phase)


def write_state(
    dependencies: object,
    root: Path,
    *,
    status: str,
    phase: str,
    error: str | None = None,
    error_class: str | None = None,
) -> None:
    writer = getattr(dependencies, "write_state", None) or write_run_state
    writer(
        root,
        status=status,
        phase=phase,
        error=error,
        error_class=error_class,
    )


def assert_no_foreign_or_partial_state(root: Path) -> None:
    assert_closed_world(root, allow_incomplete=True)
    launch = {"config.resolved.yaml", "provenance/input_artifacts.json"}
    present = set(relative_files(root))
    foreign = sorted(present - launch)
    if foreign:
        raise ProtocolError(
            "Directional-shrinkage partial/cross-run state is forbidden; "
            f"pre-existing products={foreign}."
        )


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink():
        raise ProtocolError("Directional-shrinkage run lock is a symlink.")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("Directional-shrinkage diagnostic is already running.") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = (
    "RouteJobResult",
    "assert_cuda_free_cpu_phase",
    "assert_launch_files",
    "assert_no_foreign_or_partial_state",
    "assert_workspace_resolved_paths",
    "enter_cuda_free_cpu_phase",
    "execute_route_jobs",
    "exclusive_run_lock",
    "observe",
    "reject_existing_run_state",
    "write_state",
)
