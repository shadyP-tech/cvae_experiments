"""Spawn-only coarse-H execution for the P-DCAPS v4 workstation run."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import multiprocessing
import os
from typing import Iterator

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    POSTERIOR_CONTROL_IDS,
)
from .identity import canonical_hash
from .method_runtime import (
    PreterminalMethodRuntime,
    build_outer_method_runtime,
    build_preterminal_method_runtime,
)
from .outer_workers import (
    BLAS_ENVIRONMENT_NAMES,
    OUTER_WORKER_DEPTH,
    OuterWorkerContext,
    enter_threadpool_limit,
    execute_outer_worker,
    fit_outer_control_pair,
    initialize_outer_worker,
)
from .route_runtime import PseudoResponseRuntime, RouteRuntimeResult
from .worker_dtos import (
    OuterRuntimeRequest,
    OuterRuntimeResult,
    WORKER_DEPTH_ENV,
    WORKER_DTO_KIND,
    assert_pickle_safe,
)


@dataclass(frozen=True)
class OuterRuntimeExecution:
    execution_mode: str
    worker_count: int
    results: tuple[OuterRuntimeResult, ...]
    preterminal: PreterminalMethodRuntime
    science_hash: str = field(init=False)
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        mode = str(self.execution_mode)
        workers = int(self.worker_count)
        results = tuple(self.results)
        if (
            mode not in {"serial", "spawn"}
            or workers <= 0
            or not results
            or tuple(row.request.ordinal for row in results)
            != tuple(range(len(results)))
            or tuple(row.outer_center for row in results)
            != tuple(row.outer_center for row in self.preterminal.outer_results)
        ):
            raise ProtocolError("P-DCAPS v4 outer execution manifest drifted.")
        science_hash = canonical_hash(
            {
                "schema_version": "pdcaps_v4_outer_execution_science_v1",
                "outer_result_hashes": tuple(row.result_hash for row in results),
                "preterminal_runtime_hash": self.preterminal.runtime_hash,
                "output_bundle_hash": self.preterminal.output_hashes.output_bundle_hash,
                "worker_dto_kind": WORKER_DTO_KIND,
                "target_labels_used": False,
            }
        )
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(self, "worker_count", workers)
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "science_hash", science_hash)
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_outer_execution_runtime_v1",
                    "science_hash": science_hash,
                    "execution_mode": mode,
                    "worker_count": workers,
                    "multiprocessing_start_method": (
                        "spawn" if mode == "spawn" else None
                    ),
                    "blas_thread_limit": 1,
                    "cuda_visible_devices": "",
                    "nested_process_pools": False,
                    "worker_dto_kind": WORKER_DTO_KIND,
                }
            ),
        )

    @property
    def identity_admissions(self) -> tuple[object, ...]:
        return tuple(
            row.identity_admission for row in self.preterminal.outer_results
        )

    @property
    def cyclic_admissions(self) -> tuple[object, ...]:
        return tuple(
            row.cyclic_admission for row in self.preterminal.outer_results
        )

    @property
    def admissions_h_major(self) -> tuple[tuple[str, str, object], ...]:
        return tuple(
            item
            for row in self.preterminal.outer_results
            for item in (
                (
                    row.outer_center,
                    POSTERIOR_CONTROL_IDS[0],
                    row.identity_admission,
                ),
                (
                    row.outer_center,
                    POSTERIOR_CONTROL_IDS[1],
                    row.cyclic_admission,
                ),
            )
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v4_outer_execution_runtime_v1",
            "execution_mode": self.execution_mode,
            "worker_count": self.worker_count,
            "multiprocessing_start_method": (
                "spawn" if self.execution_mode == "spawn" else None
            ),
            "blas_thread_limit": 1,
            "cuda_visible_devices": "",
            "nested_process_pools": False,
            "worker_dto_kind": WORKER_DTO_KIND,
            "results": [row.to_payload() for row in self.results],
            "preterminal": self.preterminal.to_payload(),
            "admissions_h_major": [
                {
                    "outer_center": center,
                    "posterior_control_id": control,
                    "admission": admission.to_payload(),
                }
                for center, control, admission in self.admissions_h_major
            ],
            "science_hash": self.science_hash,
            "runtime_hash": self.runtime_hash,
        }


def _observed_outer_centers(route_runtime: RouteRuntimeResult) -> tuple[str, ...]:
    observed = {
        row.route_key.outer_center
        for row in route_runtime.surface_set.identity.routes
    }
    return tuple(center for center in CENTERS if center in observed)


@contextmanager
def _parent_cpu_environment(*, serial: bool) -> Iterator[None]:
    names = ("CUDA_VISIBLE_DEVICES", WORKER_DEPTH_ENV, *BLAS_ENVIRONMENT_NAMES)
    previous = {name: os.environ.get(name) for name in names}
    limiter: object | None = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        for name in BLAS_ENVIRONMENT_NAMES:
            os.environ[name] = "1"
        if serial:
            os.environ[WORKER_DEPTH_ENV] = OUTER_WORKER_DEPTH
            limiter = enter_threadpool_limit(1)
        else:
            os.environ.pop(WORKER_DEPTH_ENV, None)
        yield
    finally:
        if limiter is not None:
            limiter.__exit__(None, None, None)  # type: ignore[attr-defined]
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def execute_outer_runtime(
    route_runtime: RouteRuntimeResult,
    pseudo_responses: PseudoResponseRuntime,
    *,
    use_processes: bool = True,
    max_workers: int = 4,
    minimum_reliability_center_count: int = 6,
    require_complete_center_inventory: bool = True,
) -> OuterRuntimeExecution:
    """Fit one H per task, with identity and cyclic controls sequentially."""

    if use_processes and os.environ.get(WORKER_DEPTH_ENV):
        raise ProtocolError("P-DCAPS v4 forbids nested process pools.")
    if (
        pseudo_responses.route_plan_inventory_hash
        != route_runtime.route_plans.route_plan_inventory_hash
        or pseudo_responses.action_surface_set_seal_hash
        != route_runtime.surface_set.surface_set_seal_hash
    ):
        raise ProtocolError("P-DCAPS v4 outer runtime input lineage drifted.")
    requested_workers = int(max_workers)
    if requested_workers <= 0 or requested_workers > 4:
        raise ProtocolError("P-DCAPS v4 outer worker count drifted.")

    context = OuterWorkerContext(
        route_runtime.surface_set,
        pseudo_responses.responses_by_control,
        int(minimum_reliability_center_count),
        bool(require_complete_center_inventory),
    )
    centers = _observed_outer_centers(route_runtime)
    if centers != route_runtime.expected_inventory.centers:
        raise ProtocolError("P-DCAPS v4 outer center inventory drifted.")
    requests = tuple(
        OuterRuntimeRequest(center, ordinal)
        for ordinal, center in enumerate(centers)
    )
    assert_pickle_safe(context, role="outer worker context")
    assert_pickle_safe(requests, role="outer worker requests")
    worker_count = min(requested_workers, len(requests))
    with _parent_cpu_environment(serial=not use_processes):
        if use_processes:
            spawn_context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=spawn_context,
                initializer=initialize_outer_worker,
                initargs=(context,),
            ) as executor:
                pairs = tuple(
                    executor.map(execute_outer_worker, requests, chunksize=1)
                )
            mode = "spawn"
        else:
            pairs = tuple(
                fit_outer_control_pair(request, context) for request in requests
            )
            mode = "serial"
            worker_count = 1

    pair_by_request = {row.request_hash: row for row in pairs}
    if len(pair_by_request) != len(requests):
        raise ProtocolError("P-DCAPS v4 outer worker result inventory drifted.")
    methods = tuple(
        build_outer_method_runtime(
            surface_set=route_runtime.surface_set,
            identity_result=pair_by_request[request.request_hash].identity_result,
            cyclic_result=pair_by_request[request.request_hash].cyclic_result,
            center_sample_order=route_runtime.sample_order(request.outer_center),
        )
        for request in requests
    )
    results = tuple(
        OuterRuntimeResult(
            request,
            pair_by_request[request.request_hash],
            method,
        )
        for request, method in zip(requests, methods, strict=True)
    )
    preterminal = build_preterminal_method_runtime(
        surface_set=route_runtime.surface_set,
        outer_results=methods,
    )
    return OuterRuntimeExecution(mode, worker_count, results, preterminal)


__all__ = (
    "OuterRuntimeExecution",
    "execute_outer_runtime",
)
