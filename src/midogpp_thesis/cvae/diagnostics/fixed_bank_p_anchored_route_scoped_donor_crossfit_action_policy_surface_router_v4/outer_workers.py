"""Top-level spawn worker for one complete P-DCAPS v4 outer center H."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    ActionResponse,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    fit_outer_action_policy_surface,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    SealedActionSurfaceSet,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.target_local_runtime import (
    POSTERIOR_CONTROL_IDS,
)
from .identity import canonical_hash
from .worker_dtos import (
    OuterControlPair,
    OuterRuntimeRequest,
    WORKER_DEPTH_ENV,
    WORKER_DTO_KIND,
    assert_pickle_safe,
)


BLAS_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
OUTER_WORKER_DEPTH = "outer_h"


@dataclass(frozen=True)
class OuterWorkerContext:
    """Immutable global science context sent once to each spawned worker."""

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
            raise ProtocolError("P-DCAPS v4 outer worker context drifted.")
        object.__setattr__(self, "responses_by_control", rows)
        object.__setattr__(self, "minimum_reliability_center_count", minimum)
        object.__setattr__(
            self,
            "context_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v4_outer_worker_context_v1",
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

    def responses(
        self, control_id: str, outer_center: str
    ) -> tuple[ActionResponse, ...]:
        selected = next(
            (
                responses
                for control, responses in self.responses_by_control
                if control == str(control_id)
            ),
            None,
        )
        if selected is None:
            raise ProtocolError("P-DCAPS v4 worker response control is absent.")
        return tuple(
            row
            for row in selected
            if row.key.route_key.outer_center == str(outer_center)
        )


_WORKER_CONTEXT: OuterWorkerContext | None = None
_THREADPOOL_LIMITER: object | None = None


def set_cpu_worker_environment(*, threads: int, depth: str) -> None:
    if isinstance(threads, bool) or int(threads) <= 0 or not str(depth):
        raise ProtocolError("P-DCAPS v4 CPU worker environment drifted.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ[WORKER_DEPTH_ENV] = str(depth)
    for name in BLAS_ENVIRONMENT_NAMES:
        os.environ[name] = str(int(threads))


def enter_threadpool_limit(threads: int) -> object:
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:  # pragma: no cover - workstation dependency
        raise ProtocolError("P-DCAPS v4 runtime lacks threadpoolctl.") from exc
    limiter = threadpool_limits(limits=int(threads))
    limiter.__enter__()
    return limiter


def initialize_outer_worker(context: OuterWorkerContext) -> None:
    """Install one pickle-safe context and enforce a single-thread CPU child."""

    global _THREADPOOL_LIMITER, _WORKER_CONTEXT
    set_cpu_worker_environment(threads=1, depth=OUTER_WORKER_DEPTH)
    _THREADPOOL_LIMITER = enter_threadpool_limit(1)
    _WORKER_CONTEXT = context


def fit_outer_control_pair(
    request: OuterRuntimeRequest,
    context: OuterWorkerContext,
) -> OuterControlPair:
    """Fit identity then cyclic control sequentially in the same H task."""

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
    result = OuterControlPair(
        request.request_hash,
        request.outer_center,
        identity,
        cyclic,
    )
    assert_pickle_safe(result, role="outer worker result")
    return result


def execute_outer_worker(request: OuterRuntimeRequest) -> OuterControlPair:
    if (
        _WORKER_CONTEXT is None
        or os.environ.get(WORKER_DEPTH_ENV) != OUTER_WORKER_DEPTH
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or any(os.environ.get(name) != "1" for name in BLAS_ENVIRONMENT_NAMES)
    ):
        raise ProtocolError("P-DCAPS v4 outer worker context is absent.")
    return fit_outer_control_pair(request, _WORKER_CONTEXT)


__all__ = (
    "BLAS_ENVIRONMENT_NAMES",
    "OUTER_WORKER_DEPTH",
    "OuterWorkerContext",
    "enter_threadpool_limit",
    "execute_outer_worker",
    "fit_outer_control_pair",
    "initialize_outer_worker",
    "set_cpu_worker_environment",
)
