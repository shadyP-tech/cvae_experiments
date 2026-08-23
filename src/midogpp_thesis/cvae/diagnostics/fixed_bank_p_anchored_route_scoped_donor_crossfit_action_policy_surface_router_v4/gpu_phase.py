"""Phase-ordered physical-bank materialization for the v4 workstation run."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Iterator, Mapping

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.physical_contracts import (
    MaterializedPhysicalBank,
)
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.physical_materializer import (
    materialize_physical_bank,
)
from .worker_dtos import WORKER_DEPTH_ENV


GPU_PREDICTION_PHASE_DEPTH = "gpu_then_prediction"


def assert_gpu_prediction_runtime(runtime: Mapping[str, object]) -> None:
    """Validate two persistent GPU workers followed by four 3-thread workers."""

    if (
        tuple(runtime.get("generation_devices", ())) != ("cuda:0", "cuda:1")
        or runtime.get("persistent_source_workers") is not True
        or int(runtime.get("persistent_generation_worker_count", -1)) != 2
        or int(runtime.get("source_workers_per_device", -1)) != 1
        or int(runtime.get("generation_workers_per_device", -1)) != 1
        or int(runtime.get("classifier_workers", -1)) != 4
        or int(runtime.get("classifier_threads_per_worker", -1)) != 3
        or runtime.get("multiprocessing_start_method") != "spawn"
        or runtime.get("gpu_generation_phase_precedes_cpu_phase") is not True
        or runtime.get("phase_disjoint_gpu_and_cpu_pools") is not True
        or runtime.get("nested_process_pools_forbidden") is not True
    ):
        raise ProtocolError("P-DCAPS v4 GPU/prediction topology drifted.")


def _assert_parent_cuda_free() -> None:
    torch_module = sys.modules.get("torch")
    cuda = None if torch_module is None else getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_initialized():
        raise ProtocolError("P-DCAPS v4 parent CUDA context was initialized.")


@contextmanager
def gpu_prediction_phase_environment() -> Iterator[None]:
    """Mark the only pool-bearing preterminal phase with one depth variable."""

    if os.environ.get(WORKER_DEPTH_ENV):
        raise ProtocolError("P-DCAPS v4 forbids nested worker phases.")
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ[WORKER_DEPTH_ENV] = GPU_PREDICTION_PHASE_DEPTH
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        yield
    finally:
        os.environ.pop(WORKER_DEPTH_ENV, None)
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda


def materialize_gpu_phase(
    config: object,
    generation_lock: object,
    frame: object,
    *,
    root: Path,
    prediction_scratch_root: Path,
) -> MaterializedPhysicalBank:
    """Run the exact GPU phase, then the exact CPU prediction phase once."""

    runtime = getattr(config, "runtime", None)
    if not isinstance(runtime, Mapping):
        raise ProtocolError("P-DCAPS v4 config lacks a runtime mapping.")
    assert_gpu_prediction_runtime(runtime)
    _assert_parent_cuda_free()
    with gpu_prediction_phase_environment():
        result = materialize_physical_bank(
            config,
            generation_lock,
            frame,
            root=Path(root),
            prediction_scratch_root=Path(prediction_scratch_root),
        )
    _assert_parent_cuda_free()
    return result


__all__ = (
    "GPU_PREDICTION_PHASE_DEPTH",
    "assert_gpu_prediction_runtime",
    "gpu_prediction_phase_environment",
    "materialize_gpu_phase",
)
