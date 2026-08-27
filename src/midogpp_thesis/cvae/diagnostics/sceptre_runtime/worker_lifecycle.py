"""Small, identity-neutral helpers for persistent spawned workers.

This module deliberately knows nothing about datasets, experts, labels,
routing decisions, leases, or artifacts.  Experiment-specific initializers
and task functions remain owned by their experiment package.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


SPAWN_START_METHOD = "spawn"


def single_worker_spawn_executor(
    *,
    initializer: Callable[..., None],
    initargs: tuple[object, ...] = (),
) -> ProcessPoolExecutor:
    """Create one persistent worker with an explicit once-per-process initializer."""

    return ProcessPoolExecutor(
        max_workers=1,
        mp_context=mp.get_context(SPAWN_START_METHOD),
        initializer=initializer,
        initargs=initargs,
    )


__all__ = ("SPAWN_START_METHOD", "single_worker_spawn_executor")
