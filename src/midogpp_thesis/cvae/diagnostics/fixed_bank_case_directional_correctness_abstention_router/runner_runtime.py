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
from .route_numerics import (
    ROUTE_BLAS_THREADS,
    assert_exact_route_blas_topology,
    install_exact_route_blas_topology,
)


_ROUTE_SURFACE: object | None = None
_ROUTE_THREADPOOL_LIMITER: object | None = None
_PARENT_THREADPOOL_LIMITER: object | None = None


@dataclass(frozen=True)
class RouteJobResult:
    plan: object
    support_responses: tuple[object, ...]
    model_fits: tuple[object, ...]
    candidate_scores: tuple[object, ...]
    decisions: tuple[object, ...]
    predictions: tuple[object, ...]


def execute_route_jobs(
    probability_surface: object,
    jobs: Sequence[Mapping[str, object]],
    *,
    workers: int,
    threads_per_worker: int,
) -> tuple[RouteJobResult, ...]:
    """Score/select 218 independent routes in four spawned 3-thread workers."""

    tasks = tuple(dict(job) for job in jobs)
    if (
        workers != 4
        or threads_per_worker != ROUTE_BLAS_THREADS
        or not tasks
    ):
        raise ProtocolError("Case-directional route worker topology drifted.")
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
        raise ProtocolError("Case-directional route worker result order drifted.")
    return results


def _initialize_route_worker(surface: object, threads: int) -> None:
    global _ROUTE_SURFACE, _ROUTE_THREADPOOL_LIMITER
    if threads != ROUTE_BLAS_THREADS:
        raise ProtocolError("Case-directional route worker threads drifted.")
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
    _ROUTE_THREADPOOL_LIMITER = install_exact_route_blas_topology(
        threads=threads
    )
    _ROUTE_SURFACE = surface


def _execute_route_job(job: Mapping[str, object]) -> RouteJobResult:
    if _ROUTE_SURFACE is None or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Case-directional route worker was not initialized.")
    assert_exact_route_blas_topology()
    from .decisions import (
        fit_route_directional_models,
        select_case_directional_abstention_decision,
    )
    from .ensemble import compose_case_predictions
    from .features import permute_route_candidate_feature_blocks
    from .scoring import (
        score_directional_correctness_observations,
        score_permuted_directional_correctness_observations,
        support_class_denominators,
    )

    plan = job["plan"]
    support_labels = tuple(job["support_labels"])
    priors = tuple(job["donor_priors"])
    route_features = tuple(job["route_features"])
    held_features = tuple(
        row for row in route_features if getattr(row, "case_id") == getattr(plan, "case_id")
    )
    observations = score_directional_correctness_observations(
        _ROUTE_SURFACE, support_labels, plan, features=route_features
    )
    denominators = support_class_denominators(
        support_labels, plan, probability_surface_or_rows=_ROUTE_SURFACE
    )
    models = fit_route_directional_models(observations, plan)
    decisions = tuple(
        select_case_directional_abstention_decision(
            method_id=method,
            target_center=getattr(plan, "target_center"),
            case_id=getattr(plan, "case_id"),
            models=models,
            held_features=held_features,
            donor_priors=priors,
            denominators=denominators,
        )
        for method in (
            "CDCA_LOO",
            "G_directional_matched",
            "CDCA_case_proxy_only",
        )
    )
    permuted_features = permute_route_candidate_feature_blocks(
        route_features, plan
    )
    permuted_observations = score_permuted_directional_correctness_observations(
        _ROUTE_SURFACE,
        support_labels,
        plan,
        permuted_features=permuted_features,
    )
    permuted_models = fit_route_directional_models(permuted_observations, plan)
    permuted_held = tuple(
        row
        for row in permuted_features
        if getattr(row, "case_id") == getattr(plan, "case_id")
    )
    permuted_decision = select_case_directional_abstention_decision(
        method_id="CDCA_feature_block_permutation_descriptive",
        target_center=getattr(plan, "target_center"),
        case_id=getattr(plan, "case_id"),
        models=permuted_models,
        held_features=permuted_held,
        donor_priors=priors,
        denominators=denominators,
    )
    all_decisions = (*decisions, permuted_decision)
    predictions = tuple(
        row
        for decision in all_decisions
        for row in compose_case_predictions(_ROUTE_SURFACE, decision)
    )
    score_rows = tuple(
        {
            "method_id": decision.method_id,
            **score.to_payload(),
        }
        for decision in all_decisions
        for direction in (decision.zero_to_one, decision.one_to_zero)
        for score in direction.candidate_scores
    )
    return RouteJobResult(
        plan,
        tuple(observations),
        tuple(models),
        score_rows,
        all_decisions,
        predictions,
    )


def assert_launch_files(root: Path, config: object) -> None:
    required = (
        root / "config.resolved.yaml",
        root / "provenance/input_artifacts.json",
    )
    if any(path.is_symlink() or not path.is_file() for path in required):
        raise ProtocolError("Case-directional launch files are absent or unsafe.")
    if Path(getattr(config, "source_path")).resolve() != required[0].resolve():
        raise ProtocolError("Case-directional config is not its persisted snapshot.")


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
        raise ProtocolError("Case-directional requires workspace-resolved paths.")


def reject_existing_run_state(root: Path) -> None:
    """Reject every state outside the separately admitted exact finalization repair."""

    state = root / "reports/run_state.json"
    if not state.exists():
        return
    if state.is_symlink() or not state.is_file():
        raise ProtocolError("Case-directional run state is unsafe.")
    payload = read_json(state)
    raise ProtocolError(
        "Case-directional cross-run recovery is forbidden; "
        f"existing status={payload.get('status')}, phase={payload.get('phase')}."
    )


def recover_if_possible(
    root: Path, *, config: object, protocol: object
) -> Path | None:
    """Dispatch only the exact registered validator-only finalization recovery."""

    from .recovery import recover_exact_finalization, recovery_capability

    capability = recovery_capability(Path(root))
    if capability is None:
        return None
    return recover_exact_finalization(
        Path(root),
        config=config,
        protocol=protocol,
        capability=capability,
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
            "Case-directional parent lacks threadpoolctl."
        ) from exc
    _PARENT_THREADPOOL_LIMITER = threadpool_limits(limits=1)


def assert_cuda_free_cpu_phase() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("Case-directional CPU phase still exposes CUDA.")
    import sys

    torch_module = sys.modules.get("torch")
    if (
        torch_module is not None
        and getattr(torch_module, "cuda", None) is not None
        and torch_module.cuda.is_initialized()
    ):
        raise ProtocolError("Case-directional parent initialized CUDA.")
    from threadpoolctl import threadpool_info

    blas = tuple(row for row in threadpool_info() if row.get("user_api") == "blas")
    if blas and any(int(row.get("num_threads", -1)) != 1 for row in blas):
        raise ProtocolError(
            "Case-directional parent BLAS topology is not one thread."
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
            "Case-directional partial/cross-run state is forbidden; "
            f"pre-existing products={foreign}."
        )


@contextmanager
def exclusive_run_lock(root: Path) -> Iterator[None]:
    path = root / ".run.lock"
    if path.is_symlink():
        raise ProtocolError("Case-directional run lock is a symlink.")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("Case-directional diagnostic is already running.") from exc
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
    "recover_if_possible",
    "reject_existing_run_state",
    "write_state",
)
