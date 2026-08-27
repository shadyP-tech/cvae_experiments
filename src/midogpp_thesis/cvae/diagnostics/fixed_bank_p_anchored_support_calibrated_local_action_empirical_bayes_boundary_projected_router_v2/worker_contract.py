"""Pre-admission spawn/import contract for the one-shot outer worker."""

from __future__ import annotations

from types import FunctionType
import multiprocessing as mp
import os
import pickle
import queue
from typing import Callable, Mapping

from .hashing import canonical_hash
from .identity import GovernanceError


WORKER_MODULE = (
    "midogpp_thesis.cvae.diagnostics."
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router_v2.scientific_worker"
)
WORKER_NAME = "run_outer_center_science"
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def validate_outer_worker_callback(
    *, spawn_probe: bool = True
) -> tuple[Callable[..., object], Mapping[str, object]]:
    """Resolve and optionally spawn-import the callback before mutation.

    The production runner calls this before input materialization, output-root
    creation, or authorization consumption.  The returned receipt contains no
    labels or fitted state and is safe to persist after admission.
    """

    from .scientific_worker import run_outer_center_science

    callback = run_outer_center_science
    if (
        not isinstance(callback, FunctionType)
        or callback.__closure__ is not None
        or callback.__module__ != WORKER_MODULE
        or callback.__name__ != WORKER_NAME
    ):
        raise GovernanceError("SCALE-BP v2 outer callback identity drifted.")
    try:
        pickle.dumps(callback)
    except Exception as exc:
        raise GovernanceError(
            "SCALE-BP v2 outer callback is not spawn-pickleable."
        ) from exc

    child_payload: dict[str, object] | None = None
    if spawn_probe:
        context = mp.get_context("spawn")
        output = context.Queue(maxsize=1)
        process = context.Process(target=_spawn_import_probe, args=(output,))
        process.start()
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            raise GovernanceError("SCALE-BP v2 outer callback spawn probe timed out.")
        try:
            child_payload = output.get_nowait()
        except queue.Empty as exc:
            raise GovernanceError(
                "SCALE-BP v2 outer callback spawn probe returned no receipt."
            ) from exc
        finally:
            output.close()
            output.join_thread()
        if (
            process.exitcode != 0
            or not isinstance(child_payload, dict)
            or child_payload.get("status") != "PASS"
            or child_payload.get("worker_module") != WORKER_MODULE
            or child_payload.get("worker_name") != WORKER_NAME
            or child_payload.get("cuda_visible_devices") != ""
            or child_payload.get("thread_environment") != _THREAD_ENVIRONMENT
        ):
            raise GovernanceError("SCALE-BP v2 outer callback spawn probe failed.")

    body = {
        "schema_version": "scale_bp_v2_outer_worker_callback_preflight_v1",
        "status": "PASS",
        "worker_module": WORKER_MODULE,
        "worker_name": WORKER_NAME,
        "module_level_function": True,
        "closure_absent": True,
        "pickle_round_trip_available": True,
        "spawn_import_probe_performed": bool(spawn_probe),
        "spawn_import_probe": child_payload,
        "mutation_performed": False,
        "authorization_consumed": False,
    }
    return callback, {**body, "receipt_hash": canonical_hash(body)}


def _spawn_import_probe(output: object) -> None:
    """Spawn child entry; imports no label decoder or runtime state."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name, value in _THREAD_ENVIRONMENT.items():
        os.environ[name] = value
    try:
        from .scientific_worker import run_outer_center_science

        callback = run_outer_center_science
        payload = {
            "status": "PASS",
            "worker_module": callback.__module__,
            "worker_name": callback.__name__,
            "module_level_function": isinstance(callback, FunctionType),
            "closure_absent": callback.__closure__ is None,
            "pickle_round_trip_available": bool(pickle.dumps(callback)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "thread_environment": {
                name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
            },
        }
    except BaseException as exc:  # pragma: no cover - child failure payload
        payload = {
            "status": "FAIL",
            "error_class": type(exc).__name__,
            "error": str(exc),
        }
    # Multiprocessing Queue is intentionally accepted as an opaque primitive
    # transport; no scientific object or label view crosses this boundary.
    output.put(payload)  # type: ignore[attr-defined]


__all__ = (
    "WORKER_MODULE",
    "WORKER_NAME",
    "validate_outer_worker_callback",
)
