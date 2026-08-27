"""Immutable coordinator-admitted batch receipts for OE-PPUR execution."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from ..hashing import canonical_hash, require_sha256
from ..protocol import ProtocolError
from .dtos import (
    WorkerExecutionDTO,
    WorkerResultDTO,
    assert_pickle_safe_label_free_dto,
)


_BATCH_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ExecutionBatchResult:
    """Coordinator-admitted, deterministically ordered worker transports."""

    role: str
    configured_worker_count: int
    configured_device_indices: tuple[int, ...]
    ordered_task_hashes: tuple[str, ...]
    receipts: tuple[WorkerExecutionDTO, ...]
    worker_roster: tuple[int, ...]
    multiprocessing_start_method: str = "spawn"
    nested_pools_used: bool = False
    labels_opened: bool = False
    filesystem_mutation_count: int = 0
    coordinator_result_bytes_revalidated: bool = True
    _factory_token: InitVar[object] = None
    row_index_sha256: str = field(init=False)
    source_surface_sha256: str = field(init=False)
    result_surface_sha256: str = field(init=False)
    batch_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BATCH_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR execution batch bypassed coordinator admission."
            )
        role = str(self.role)
        task_hashes = tuple(
            require_sha256(value, "execution-batch task hash")
            for value in self.ordered_task_hashes
        )
        receipts = tuple(self.receipts)
        devices = tuple(int(value) for value in self.configured_device_indices)
        roster = tuple(int(value) for value in self.worker_roster)
        expected_workers = (
            2
            if role == "gpu_prediction"
            else 4
            if role == "cpu_outer"
            else 0
        )
        if (
            expected_workers == 0
            or int(self.configured_worker_count) != expected_workers
            or len(roster) != expected_workers
            or len(set(roster)) != expected_workers
            or any(pid <= 0 for pid in roster)
            or not task_hashes
            or len(set(task_hashes)) != len(task_hashes)
            or len(receipts) != len(task_hashes)
            or any(not isinstance(row, WorkerExecutionDTO) for row in receipts)
        ):
            raise ProtocolError("OE-PPUR execution batch topology drifted.")
        row_indices = {row.row_index_sha256 for row in receipts}
        source_surfaces = tuple(row.source_surface_sha256 for row in receipts)
        if (
            any(row.worker_pid not in roster for row in receipts)
            or len(row_indices) != 1
            or tuple(row.task_hash for row in receipts) != task_hashes
            or tuple(row.task_ordinal for row in receipts)
            != tuple(range(len(receipts)))
            or any(row.role != role for row in receipts)
            or self.multiprocessing_start_method != "spawn"
            or bool(self.nested_pools_used)
            or bool(self.labels_opened)
            or int(self.filesystem_mutation_count) != 0
            or not self.coordinator_result_bytes_revalidated
        ):
            raise ProtocolError("OE-PPUR execution batch topology drifted.")
        for row in receipts:
            assert_pickle_safe_label_free_dto(row)
        if role == "gpu_prediction":
            if devices != (0, 1) or any(
                row.physical_device_index != row.task_ordinal % 2
                or row.callback_visible_logical_device != "cuda:0"
                for row in receipts
            ):
                raise ProtocolError("OE-PPUR GPU batch distribution drifted.")
        elif devices or any(
            row.physical_device_index is not None
            or row.callback_visible_logical_device != "cpu"
            for row in receipts
        ):
            raise ProtocolError("OE-PPUR CPU batch declared a GPU context.")
        if role == "cpu_outer" and len(set(source_surfaces)) != 1:
            raise ProtocolError("OE-PPUR CPU batch mixed probability surfaces.")
        row_index = next(iter(row_indices))
        source_surface = (
            source_surfaces[0]
            if role == "cpu_outer"
            else canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_gpu_source_surface_v1",
                    "ordered_task_source_surfaces": source_surfaces,
                }
            )
        )
        result_surface = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_execution_result_surface_v1",
                "role": role,
                "ordered_results": tuple(
                    (
                        row.task_hash,
                        row.worker_result_hash,
                        row.result_hashes,
                        row.row_index_sha256,
                        row.source_surface_sha256,
                    )
                    for row in receipts
                ),
            }
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "ordered_task_hashes", task_hashes)
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "configured_device_indices", devices)
        object.__setattr__(self, "worker_roster", roster)
        object.__setattr__(self, "row_index_sha256", row_index)
        object.__setattr__(self, "source_surface_sha256", source_surface)
        object.__setattr__(self, "result_surface_sha256", result_surface)
        object.__setattr__(self, "batch_hash", canonical_hash(self._payload()))

    @property
    def results(self) -> tuple[WorkerResultDTO, ...]:
        return tuple(row.result for row in self.receipts)

    @property
    def observed_worker_pids(self) -> tuple[int, ...]:
        return self.worker_roster

    @property
    def result_file_hashes(self) -> tuple[str, ...]:
        """Ordered hashes for the exact files claimed by every worker."""

        return tuple(
            digest
            for receipt in self.receipts
            for digest in receipt.result_hashes
        )

    @property
    def verified_input_file_hashes(self) -> tuple[str, ...]:
        """Ordered input hashes revalidated or declared by every worker."""

        return tuple(
            digest
            for receipt in self.receipts
            for digest in receipt.verified_input_content_hashes
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_execution_batch_result_v3",
            "role": self.role,
            "configured_worker_count": self.configured_worker_count,
            "configured_device_indices": self.configured_device_indices,
            "ordered_task_hashes": self.ordered_task_hashes,
            "worker_transport_hashes": tuple(
                row.transport_hash for row in self.receipts
            ),
            "worker_roster": self.worker_roster,
            "multiprocessing_start_method": self.multiprocessing_start_method,
            "nested_pools_used": self.nested_pools_used,
            "labels_opened": self.labels_opened,
            "filesystem_mutation_count": self.filesystem_mutation_count,
            "coordinator_result_bytes_revalidated": (
                self.coordinator_result_bytes_revalidated
            ),
            "row_index_sha256": self.row_index_sha256,
            "source_surface_sha256": self.source_surface_sha256,
            "result_surface_sha256": self.result_surface_sha256,
            "result_file_hashes": self.result_file_hashes,
            "verified_input_file_hashes": self.verified_input_file_hashes,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "batch_hash": self.batch_hash}


def build_execution_batch_result(
    *,
    role: str,
    configured_worker_count: int,
    configured_device_indices: tuple[int, ...],
    ordered_task_hashes: tuple[str, ...],
    receipts: tuple[WorkerExecutionDTO, ...],
    worker_roster: tuple[int, ...],
) -> ExecutionBatchResult:
    """Issue a batch only after the coordinator admits every transport."""

    return ExecutionBatchResult(
        role=role,
        configured_worker_count=configured_worker_count,
        configured_device_indices=configured_device_indices,
        ordered_task_hashes=ordered_task_hashes,
        receipts=receipts,
        worker_roster=worker_roster,
        _factory_token=_BATCH_FACTORY_TOKEN,
    )


__all__ = ("ExecutionBatchResult", "build_execution_batch_result")
