"""Workstation-safe one-level execution for P-DCAPS."""

from .contracts import (
    ContiguousArray,
    WorkerRequest,
    WorkerResult,
    validate_plain_payload,
)
from .orchestration import ExecutionManifest, OuterWorker, execute_outer_jobs
from .outer_worker import (
    HASH_MANIFEST_OPERATION,
    execute_outer_worker,
    initialize_outer_worker,
)

__all__ = (
    "ContiguousArray",
    "ExecutionManifest",
    "HASH_MANIFEST_OPERATION",
    "OuterWorker",
    "WorkerRequest",
    "WorkerResult",
    "execute_outer_jobs",
    "execute_outer_worker",
    "initialize_outer_worker",
    "validate_plain_payload",
)
