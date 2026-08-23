"""Spawn-only coarse-H execution for authorized P-DCAPS v2.

One task owns one outer center H and fits both posterior controls sequentially.
The immutable global surfaces and pseudo responses are sent once per spawned
worker through its initializer; task requests remain tiny.  Workers return
plain pickle-safe typed outer-science DTOs, never mappings, capabilities, paths,
CUDA objects, executors, or nested worker handles.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..action_surface import ActionResponse
from ..engine import OuterActionPolicyResult, fit_outer_action_policy_surface
from ..surface_set import SealedActionSurfaceSet
from ..target_local_runtime import POSTERIOR_CONTROL_IDS
from .identity import canonical_hash, require_sha256
from .method_runtime import (
    OuterMethodRuntimeResult,
    PreterminalMethodRuntime,
    build_outer_method_runtime,
    build_preterminal_method_runtime,
)
from .route_runtime import PseudoResponseRuntime, RouteRuntimeResult


WORKER_DEPTH_ENV = "MIDOGPP_PDCAPS_V2_OUTER_WORKER_DEPTH"
WORKER_DTO_KIND = "PLAIN_PICKLE_SAFE_SCIENCE_DTOS_NO_COMPACT_OFFSETS"
_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class OuterRuntimeRequest:
    outer_center: str
    ordinal: int
    request_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        ordinal = int(self.ordinal)
        if outer not in CENTERS or ordinal < 0:
            raise ProtocolError("P-DCAPS v2 outer worker request drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(
            self,
            "request_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_outer_runtime_request_v1",
                    "outer_center": outer,
                    "ordinal": ordinal,
                    "controls_in_one_task": POSTERIOR_CONTROL_IDS,
                    "worker_dto_kind": WORKER_DTO_KIND,
                    "target_labels_used": False,
                }
            ),
        )


@dataclass(frozen=True)
class OuterControlPair:
    request_hash: str
    outer_center: str
    identity_result: OuterActionPolicyResult
    cyclic_result: OuterActionPolicyResult
    pair_hash: str = field(init=False)

    def __post_init__(self) -> None:
        request_hash = require_sha256(self.request_hash, "v2 outer request")
        outer = str(self.outer_center)
        if (
            self.identity_result.outer_center != outer
            or self.cyclic_result.outer_center != outer
            or self.identity_result.posterior_control_id
            != POSTERIOR_CONTROL_IDS[0]
            or self.cyclic_result.posterior_control_id != POSTERIOR_CONTROL_IDS[1]
            or self.identity_result.physical_surface_hash
            != self.cyclic_result.physical_surface_hash
            or self.identity_result.action_surface_seal_hash
            == self.cyclic_result.action_surface_seal_hash
        ):
            raise ProtocolError("P-DCAPS v2 paired outer result drifted.")
        object.__setattr__(self, "request_hash", request_hash)
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(
            self,
            "pair_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_outer_control_pair_v1",
                    "request_hash": request_hash,
                    "outer_center": outer,
                    "identity_result_hash": self.identity_result.result_hash,
                    "cyclic_result_hash": self.cyclic_result.result_hash,
                    "controls_fit_sequentially_in_one_h_task": True,
                    "worker_dto_kind": WORKER_DTO_KIND,
                    "target_labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_outer_control_pair_v1",
            "request_hash": self.request_hash,
            "outer_center": self.outer_center,
            "identity_result": self.identity_result.to_payload(),
            "cyclic_result": self.cyclic_result.to_payload(),
            "controls_fit_sequentially_in_one_h_task": True,
            "worker_dto_kind": WORKER_DTO_KIND,
            "target_labels_used": False,
            "pair_hash": self.pair_hash,
        }


@dataclass(frozen=True)
class _OuterWorkerContext:
    surface_set: SealedActionSurfaceSet
    responses_by_control: tuple[tuple[str, tuple[ActionResponse, ...]], ...]
    minimum_reliability_center_count: int
    require_complete_center_inventory: bool
    context_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rows = tuple(
            (str(control), tuple(responses))
            for control, responses in self.responses_by_control
        )
        minimum = int(self.minimum_reliability_center_count)
        if (
            self.surface_set.control_ids != POSTERIOR_CONTROL_IDS
            or tuple(control for control, _responses in rows)
            != POSTERIOR_CONTROL_IDS
            or minimum <= 0
            or any(
                response.key.action_surface_seal_hash
                != self.surface_set.surface(control).action_surface_seal_hash
                for control, responses in rows
                for response in responses
            )
        ):
            raise ProtocolError("P-DCAPS v2 outer worker context drifted.")
        object.__setattr__(self, "responses_by_control", rows)
        object.__setattr__(self, "minimum_reliability_center_count", minimum)
        object.__setattr__(
            self,
            "context_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_outer_worker_context_v1",
                    "surface_set_seal_hash": self.surface_set.surface_set_seal_hash,
                    "response_hashes_by_control": tuple(
                        (
                            control,
                            tuple(response.response_hash for response in responses),
                        )
                        for control, responses in rows
                    ),
                    "minimum_reliability_center_count": minimum,
                    "require_complete_center_inventory": bool(
                        self.require_complete_center_inventory
                    ),
                    "controls_fit_sequentially_in_one_h_task": True,
                    "worker_dto_kind": WORKER_DTO_KIND,
                    "target_labels_used": False,
                }
            ),
        )

    def responses(self, control_id: str, outer_center: str) -> tuple[ActionResponse, ...]:
        rows = dict(self.responses_by_control)[str(control_id)]
        return tuple(
            row
            for row in rows
            if row.key.route_key.outer_center == str(outer_center)
        )


@dataclass(frozen=True)
class OuterRuntimeResult:
    request: OuterRuntimeRequest
    control_pair: OuterControlPair
    methods: OuterMethodRuntimeResult
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.control_pair.request_hash != self.request.request_hash
            or self.control_pair.outer_center != self.request.outer_center
            or self.methods.outer_center != self.request.outer_center
            or self.methods.identity_result.result_hash
            != self.control_pair.identity_result.result_hash
            or self.methods.cyclic_result.result_hash
            != self.control_pair.cyclic_result.result_hash
        ):
            raise ProtocolError("P-DCAPS v2 outer runtime result drifted.")
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_outer_runtime_result_v1",
                    "request_hash": self.request.request_hash,
                    "control_pair_hash": self.control_pair.pair_hash,
                    "method_runtime_hash": self.methods.runtime_hash,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def outer_center(self) -> str:
        return self.request.outer_center

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_outer_runtime_result_v1",
            "request_hash": self.request.request_hash,
            "control_pair": self.control_pair.to_payload(),
            "methods": self.methods.to_payload(),
            "target_labels_used": False,
            "result_hash": self.result_hash,
        }


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
            raise ProtocolError("P-DCAPS v2 outer execution manifest drifted.")
        science_hash = canonical_hash(
            {
                "schema_version": "pdcaps_v2_outer_execution_science_v1",
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
                    "schema_version": "pdcaps_v2_outer_execution_runtime_v1",
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

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_outer_execution_runtime_v1",
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
            "science_hash": self.science_hash,
            "runtime_hash": self.runtime_hash,
        }


_WORKER_CONTEXT: _OuterWorkerContext | None = None
_THREADPOOL_LIMITER: object | None = None


def _set_cpu_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = "1"


def _enter_threadpool_limit() -> object | None:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return None
    limiter = threadpool_limits(limits=1)
    limiter.__enter__()
    return limiter


def _initialize_outer_worker(context: _OuterWorkerContext) -> None:
    global _THREADPOOL_LIMITER, _WORKER_CONTEXT
    _set_cpu_environment()
    os.environ[WORKER_DEPTH_ENV] = "1"
    _THREADPOOL_LIMITER = _enter_threadpool_limit()
    _WORKER_CONTEXT = context


def _fit_outer_control_pair(
    request: OuterRuntimeRequest,
    context: _OuterWorkerContext,
) -> OuterControlPair:
    identity = fit_outer_action_policy_surface(
        context.surface_set.identity,
        context.responses(POSTERIOR_CONTROL_IDS[0], request.outer_center),
        outer_center=request.outer_center,
        minimum_reliability_center_count=(
            context.minimum_reliability_center_count
        ),
        require_complete_center_inventory=(
            context.require_complete_center_inventory
        ),
    )
    cyclic = fit_outer_action_policy_surface(
        context.surface_set.cyclic,
        context.responses(POSTERIOR_CONTROL_IDS[1], request.outer_center),
        outer_center=request.outer_center,
        minimum_reliability_center_count=(
            context.minimum_reliability_center_count
        ),
        require_complete_center_inventory=(
            context.require_complete_center_inventory
        ),
    )
    return OuterControlPair(
        request.request_hash,
        request.outer_center,
        identity,
        cyclic,
    )


def _execute_outer_worker(request: OuterRuntimeRequest) -> OuterControlPair:
    if (
        _WORKER_CONTEXT is None
        or os.environ.get(WORKER_DEPTH_ENV) != "1"
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or any(os.environ.get(name) != "1" for name in _THREAD_ENV_NAMES)
    ):
        raise ProtocolError("P-DCAPS v2 outer worker context is absent.")
    return _fit_outer_control_pair(request, _WORKER_CONTEXT)


def _observed_outer_centers(surface_set: SealedActionSurfaceSet) -> tuple[str, ...]:
    observed = {row.route_key.outer_center for row in surface_set.identity.routes}
    return tuple(center for center in CENTERS if center in observed)


def execute_outer_runtime(
    route_runtime: RouteRuntimeResult,
    pseudo_responses: PseudoResponseRuntime,
    *,
    use_processes: bool = True,
    max_workers: int = 4,
    minimum_reliability_center_count: int = 6,
    require_complete_center_inventory: bool = True,
) -> OuterRuntimeExecution:
    """Fit all H surfaces with one spawned job per H and both controls per job."""

    if use_processes and os.environ.get(WORKER_DEPTH_ENV):
        raise ProtocolError("P-DCAPS v2 forbids nested process pools.")
    if (
        pseudo_responses.route_plan_inventory_hash
        != route_runtime.route_plans.route_plan_inventory_hash
        or pseudo_responses.action_surface_set_seal_hash
        != route_runtime.surface_set.surface_set_seal_hash
    ):
        raise ProtocolError("P-DCAPS v2 outer runtime input lineage drifted.")
    requested_workers = int(max_workers)
    if requested_workers <= 0 or requested_workers > 4:
        raise ProtocolError("P-DCAPS v2 outer worker count drifted.")

    _set_cpu_environment()
    context = _OuterWorkerContext(
        route_runtime.surface_set,
        pseudo_responses.responses_by_control,
        int(minimum_reliability_center_count),
        bool(require_complete_center_inventory),
    )
    centers = _observed_outer_centers(route_runtime.surface_set)
    if centers != route_runtime.expected_inventory.centers:
        raise ProtocolError("P-DCAPS v2 outer center inventory drifted.")
    requests = tuple(
        OuterRuntimeRequest(center, ordinal)
        for ordinal, center in enumerate(centers)
    )
    worker_count = min(requested_workers, len(requests))
    if use_processes:
        spawn_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=spawn_context,
            initializer=_initialize_outer_worker,
            initargs=(context,),
        ) as executor:
            pairs = tuple(
                executor.map(_execute_outer_worker, requests, chunksize=1)
            )
        mode = "spawn"
    else:
        limiter = _enter_threadpool_limit()
        try:
            pairs = tuple(
                _fit_outer_control_pair(request, context) for request in requests
            )
        finally:
            if limiter is not None:
                limiter.__exit__(None, None, None)  # type: ignore[attr-defined]
        mode = "serial"
        worker_count = 1

    pair_by_request = {row.request_hash: row for row in pairs}
    if len(pair_by_request) != len(requests):
        raise ProtocolError("P-DCAPS v2 outer worker result inventory drifted.")
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
    "OuterControlPair",
    "OuterRuntimeExecution",
    "OuterRuntimeRequest",
    "OuterRuntimeResult",
    "WORKER_DEPTH_ENV",
    "WORKER_DTO_KIND",
    "execute_outer_runtime",
)
