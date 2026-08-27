"""Canonical spawn-pickleable outer-center callback for SCALE-BP v2.

The public callback intentionally remains in this module because its exact
module/name identity is sealed by the pre-admission worker contract. All
scientific work is delegated to responsibility-scoped modules under
``worker/``; this file is only the stable multiprocessing composition edge.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .execution.dtos import OuterCenterResult, OuterCenterTask
from .worker.contracts import (
    METHOD_IDS,
    ROUTE_CHUNK_SCHEMA,
    TASK_PAYLOAD_SCHEMA,
    WORKER_CAPABILITY_CHUNK_SCHEMA,
)
from .worker.coordinator import coordinate_outer_center_science


def run_outer_center_science(
    task: OuterCenterTask,
    arrays: Mapping[str, np.memmap],
) -> OuterCenterResult:
    """Execute one complete outer H through the sealed worker coordinator."""

    return coordinate_outer_center_science(task, arrays)


__all__ = (
    "METHOD_IDS",
    "ROUTE_CHUNK_SCHEMA",
    "TASK_PAYLOAD_SCHEMA",
    "WORKER_CAPABILITY_CHUNK_SCHEMA",
    "run_outer_center_science",
)
