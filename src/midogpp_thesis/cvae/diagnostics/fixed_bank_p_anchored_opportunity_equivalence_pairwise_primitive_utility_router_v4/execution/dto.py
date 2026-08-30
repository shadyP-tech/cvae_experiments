"""Spawn-safe primitive transport objects for OE-PPUR v4 workers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import pickle

from ....protocol import ProtocolError
from ..hashing import require_sha256
from ..identity import CENTERS


@dataclass(frozen=True, slots=True)
class PrimitiveWorkerTask:
    task_id: str
    outer_center_id: str
    inner_fold_id: int
    row_start: int
    row_stop: int
    source_training_surface_hash: str
    candidate_pool_receipt_hash: str
    compiled_action_surface_hash: str
    random_seed: int
    cuda_visible_devices: str = ""
    blas_threads: int = 1
    storage_dtype: str = "<f4"
    reduction_dtype: str = "<f8"

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or self.outer_center_id not in CENTERS
            or type(self.inner_fold_id) is not int
            or self.inner_fold_id < 0
            or type(self.row_start) is not int
            or type(self.row_stop) is not int
            or self.row_start < 0
            or self.row_stop <= self.row_start
            or type(self.random_seed) is not int
            or self.random_seed < 0
            or self.cuda_visible_devices != ""
            or self.blas_threads != 1
            or self.storage_dtype != "<f4"
            or self.reduction_dtype != "<f8"
        ):
            raise ProtocolError("OE-PPUR v4 primitive worker task drifted.")
        for role in (
            "source_training_surface_hash",
            "candidate_pool_receipt_hash",
            "compiled_action_surface_hash",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )


@dataclass(frozen=True, slots=True)
class PrimitiveWorkerResult:
    task_id: str
    outer_center_id: str
    inner_fold_id: int
    worker_pid: int
    model_receipt_hash: str
    source_ordering_receipt_hash: str
    ordered_case_ids: tuple[str, ...]
    ordered_action_ids: tuple[str, ...]
    ordered_scores: tuple[float, ...]
    exact_p_required: bool
    failure_reason: str | None

    def __post_init__(self) -> None:
        case_ids = tuple(str(value) for value in self.ordered_case_ids)
        action_ids = tuple(str(value) for value in self.ordered_action_ids)
        scores = tuple(float(value) for value in self.ordered_scores)
        if (
            not self.task_id
            or self.outer_center_id not in CENTERS
            or type(self.inner_fold_id) is not int
            or self.inner_fold_id < 0
            or type(self.worker_pid) is not int
            or self.worker_pid <= 0
            or not case_ids
            or len(case_ids) != len(action_ids)
            or len(case_ids) != len(scores)
            or len(set(case_ids)) != len(case_ids)
            or any(not value for value in action_ids)
            or any(not math.isfinite(value) for value in scores)
            or type(self.exact_p_required) is not bool
            or (self.exact_p_required and not self.failure_reason)
            or (not self.exact_p_required and self.failure_reason is not None)
        ):
            raise ProtocolError("OE-PPUR v4 primitive worker result drifted.")
        object.__setattr__(self, "model_receipt_hash", require_sha256(self.model_receipt_hash, "model receipt hash"))
        object.__setattr__(self, "source_ordering_receipt_hash", require_sha256(self.source_ordering_receipt_hash, "source ordering receipt hash"))
        object.__setattr__(self, "ordered_case_ids", case_ids)
        object.__setattr__(self, "ordered_action_ids", action_ids)
        object.__setattr__(self, "ordered_scores", scores)


def assert_pickle_round_trip(value: object) -> object:
    """Require an exact spawn-compatible pickle round trip."""

    if type(value) not in {PrimitiveWorkerTask, PrimitiveWorkerResult}:
        raise ProtocolError("OE-PPUR v4 worker transport object is untyped.")
    try:
        rebuilt = pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    except (pickle.PickleError, TypeError, ValueError, AttributeError) as exc:
        raise ProtocolError("OE-PPUR v4 worker DTO is not pickle-safe.") from exc
    if type(rebuilt) is not type(value) or rebuilt != value:
        raise ProtocolError("OE-PPUR v4 worker DTO pickle round trip drifted.")
    return rebuilt


__all__ = (
    "PrimitiveWorkerResult",
    "PrimitiveWorkerTask",
    "assert_pickle_round_trip",
)
