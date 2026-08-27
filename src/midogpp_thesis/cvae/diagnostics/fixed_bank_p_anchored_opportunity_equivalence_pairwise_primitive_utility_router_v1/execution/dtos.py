"""Primitive-only, label-free process DTOs for planned OE-PPUR."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import pickle

from ..fold_scope import FoldScope
from ..hashing import canonical_hash, require_sha256
from ..identity import ACTION_IDS, CENTERS
from ..protocol import ProtocolError
from ..source_fence import assert_not_predecessor_reference


@dataclass(frozen=True, slots=True)
class MemmapSliceDTO:
    path: str
    content_sha256: str
    row_index_sha256: str
    shape: tuple[int, ...]
    byte_offset: int
    byte_length: int
    dtype: str = "float32"
    mode: str = "r"
    dto_hash: str = field(init=False)

    def __post_init__(self) -> None:
        path = assert_not_predecessor_reference(self.path, role="memmap path")
        shape = tuple(int(value) for value in self.shape)
        if not path.startswith("/") or not shape or any(value <= 0 for value in shape):
            raise ProtocolError("OE-PPUR memmap DTO topology drifted.")
        if self.dtype != "float32" or self.mode != "r" or int(self.byte_offset) < 0:
            raise ProtocolError("OE-PPUR memmap DTO is not float32 read-only.")
        expected_bytes = 4
        for value in shape:
            expected_bytes *= value
        if int(self.byte_length) != expected_bytes:
            raise ProtocolError("OE-PPUR memmap DTO byte extent drifted.")
        content = require_sha256(self.content_sha256, "memmap content hash")
        rows = require_sha256(self.row_index_sha256, "memmap row-index hash")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(self, "row_index_sha256", rows)
        object.__setattr__(self, "dto_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_memmap_slice_dto_v1",
            "path": self.path,
            "content_sha256": self.content_sha256,
            "row_index_sha256": self.row_index_sha256,
            "shape": list(self.shape),
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "dtype": self.dtype,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class PredictionTaskDTO:
    task_id: str
    device_index: int
    input_paths: tuple[str, ...]
    input_hashes: tuple[str, ...]
    row_index_sha256: str
    output_path: str
    route_ids: tuple[str, ...]
    source_surface_sha256: str = field(init=False)
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        paths = tuple(
            assert_not_predecessor_reference(value, role="prediction input path")
            for value in self.input_paths
        )
        hashes = tuple(require_sha256(value, "prediction input hash") for value in self.input_hashes)
        row_index = require_sha256(
            self.row_index_sha256,
            "prediction row-index hash",
        )
        output = assert_not_predecessor_reference(self.output_path, role="prediction output path")
        routes = tuple(str(value) for value in self.route_ids)
        if (
            not self.task_id
            or int(self.device_index) not in (0, 1)
            or not paths
            or len(paths) != len(hashes)
            or not output.startswith("/")
            or not routes
            or len(routes) != len(set(routes))
        ):
            raise ProtocolError("OE-PPUR prediction task DTO topology drifted.")
        object.__setattr__(self, "input_paths", paths)
        object.__setattr__(self, "input_hashes", hashes)
        object.__setattr__(self, "row_index_sha256", row_index)
        object.__setattr__(self, "output_path", output)
        object.__setattr__(self, "route_ids", routes)
        source_surface = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_prediction_source_surface_v1",
                "input_hashes": hashes,
                "route_ids": routes,
                "row_index_sha256": row_index,
            }
        )
        object.__setattr__(self, "source_surface_sha256", source_surface)
        object.__setattr__(
            self,
            "task_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_prediction_task_dto_v2",
                    "task_id": self.task_id,
                    "device_index": self.device_index,
                    "input_paths": list(paths),
                    "input_hashes": list(hashes),
                    "row_index_sha256": row_index,
                    "source_surface_sha256": source_surface,
                    "output_path": output,
                    "route_ids": list(routes),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OuterFoldTaskDTO:
    H: str
    J: str
    K: str
    L: str
    d: str
    scope_hash: str
    probability_path: str
    probability_sha256: str
    candidate_probability_surface_sha256: str
    row_index_sha256: str
    action_ids: tuple[str, ...] = ACTION_IDS
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        scope = FoldScope(self.H, self.J, self.K, self.L, self.d)
        if require_sha256(self.scope_hash, "fold-scope hash") != scope.scope_hash:
            raise ProtocolError("OE-PPUR outer task fold-scope hash drifted.")
        path = assert_not_predecessor_reference(self.probability_path, role="probability path")
        probability = require_sha256(self.probability_sha256, "probability hash")
        candidate_surface = require_sha256(
            self.candidate_probability_surface_sha256,
            "candidate probability surface hash",
        )
        rows = require_sha256(self.row_index_sha256, "row-index hash")
        actions = tuple(str(value) for value in self.action_ids)
        if not path.startswith("/") or actions != ACTION_IDS:
            raise ProtocolError("OE-PPUR outer task DTO topology drifted.")
        object.__setattr__(self, "probability_path", path)
        object.__setattr__(self, "probability_sha256", probability)
        object.__setattr__(
            self,
            "candidate_probability_surface_sha256",
            candidate_surface,
        )
        object.__setattr__(self, "row_index_sha256", rows)
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(
            self,
            "task_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_outer_fold_task_dto_v2",
                    "scope_hash": scope.scope_hash,
                    "probability_path": path,
                    "probability_sha256": probability,
                    "candidate_probability_surface_sha256": candidate_surface,
                    "row_index_sha256": rows,
                    "action_ids": list(actions),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkerResultDTO:
    task_hash: str
    result_paths: tuple[str, ...]
    result_hashes: tuple[str, ...]
    row_count: int
    row_index_sha256: str
    source_surface_sha256: str
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        task = require_sha256(self.task_hash, "worker task hash")
        paths = tuple(
            assert_not_predecessor_reference(value, role="worker result path")
            for value in self.result_paths
        )
        hashes = tuple(require_sha256(value, "worker result hash") for value in self.result_hashes)
        row_index = require_sha256(
            self.row_index_sha256,
            "worker-result row-index hash",
        )
        source_surface = require_sha256(
            self.source_surface_sha256,
            "worker-result source-surface hash",
        )
        if not paths or len(paths) != len(hashes) or int(self.row_count) <= 0:
            raise ProtocolError("OE-PPUR worker result DTO topology drifted.")
        object.__setattr__(self, "task_hash", task)
        object.__setattr__(self, "result_paths", paths)
        object.__setattr__(self, "result_hashes", hashes)
        object.__setattr__(self, "row_index_sha256", row_index)
        object.__setattr__(self, "source_surface_sha256", source_surface)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "oe_ppur_v1_worker_result_dto_v2",
                    "task_hash": task,
                    "result_paths": list(paths),
                    "result_hashes": list(hashes),
                    "row_count": self.row_count,
                    "row_index_sha256": row_index,
                    "source_surface_sha256": source_surface,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SealedCallbackDescriptorDTO:
    """Primitive identity for one exact importable top-level callback."""

    callback_role: str
    module_name: str
    member_name: str
    source_path: str
    source_sha256: str
    member_code_sha256: str
    result_evidence_mode: str
    descriptor_hash: str = field(init=False)

    def __post_init__(self) -> None:
        role = str(self.callback_role)
        module = assert_not_predecessor_reference(
            self.module_name,
            role="callback descriptor module",
        )
        member = str(self.member_name)
        source_path = assert_not_predecessor_reference(
            self.source_path,
            role="callback descriptor source path",
        )
        source = require_sha256(self.source_sha256, "callback source hash")
        code = require_sha256(self.member_code_sha256, "callback member-code hash")
        evidence_mode = str(self.result_evidence_mode)
        if (
            role not in {"gpu_prediction", "cpu_outer"}
            or not module
            or not member.isidentifier()
            or "." in member
            or not source_path.startswith("/")
            or evidence_mode not in {"regular_file", "strict_test_fixture"}
        ):
            raise ProtocolError("OE-PPUR callback descriptor topology drifted.")
        object.__setattr__(self, "callback_role", role)
        object.__setattr__(self, "module_name", module)
        object.__setattr__(self, "member_name", member)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "source_sha256", source)
        object.__setattr__(self, "member_code_sha256", code)
        object.__setattr__(self, "result_evidence_mode", evidence_mode)
        object.__setattr__(self, "descriptor_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_sealed_callback_descriptor_v1",
            "callback_role": self.callback_role,
            "module_name": self.module_name,
            "member_name": self.member_name,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "member_code_sha256": self.member_code_sha256,
            "result_evidence_mode": self.result_evidence_mode,
        }


@dataclass(frozen=True, slots=True)
class WorkerExecutionDTO:
    """Flat primitive-only value returned across the spawn result queue."""

    role: str
    task_ordinal: int
    task_hash: str
    callback_descriptor_hash: str
    result_paths: tuple[str, ...]
    result_hashes: tuple[str, ...]
    row_count: int
    row_index_sha256: str
    source_surface_sha256: str
    verified_input_content_hashes: tuple[str, ...]
    input_bytes_revalidated: bool
    worker_result_hash: str
    worker_pid: int
    physical_device_index: int | None
    callback_visible_logical_device: str
    cuda_visible_devices: str
    limiter_scope_entered: bool
    threadpool_observation_performed: bool
    loaded_pool_count: int
    loaded_pools: tuple[tuple[str, str, str, int], ...]
    all_loaded_pools_one_thread: bool
    result_evidence_mode: str
    labels_opened: bool = False
    transport_hash: str = field(init=False)

    def __post_init__(self) -> None:
        role = str(self.role)
        task = require_sha256(self.task_hash, "worker-execution task hash")
        callback = require_sha256(
            self.callback_descriptor_hash,
            "worker callback descriptor hash",
        )
        paths = tuple(
            assert_not_predecessor_reference(value, role="worker-execution result path")
            for value in self.result_paths
        )
        hashes = tuple(
            require_sha256(value, "worker-execution result hash")
            for value in self.result_hashes
        )
        row_index = require_sha256(
            self.row_index_sha256,
            "worker-execution row-index hash",
        )
        source_surface = require_sha256(
            self.source_surface_sha256,
            "worker-execution source-surface hash",
        )
        input_hashes = tuple(
            require_sha256(value, "verified worker input hash")
            for value in self.verified_input_content_hashes
        )
        pools = tuple(
            (str(user), str(internal), str(prefix), int(threads))
            for user, internal, prefix, threads in self.loaded_pools
        )
        worker_result_hash = require_sha256(
            self.worker_result_hash,
            "worker result DTO hash",
        )
        expected_worker_result_hash = canonical_hash(
            {
                "schema_version": "oe_ppur_v1_worker_result_dto_v2",
                "task_hash": task,
                "result_paths": list(paths),
                "result_hashes": list(hashes),
                "row_count": self.row_count,
                "row_index_sha256": row_index,
                "source_surface_sha256": source_surface,
            }
        )
        if (
            role not in {"gpu_prediction", "cpu_outer"}
            or int(self.task_ordinal) < 0
            or not paths
            or len(paths) != len(hashes)
            or any(not path.startswith("/") for path in paths)
            or int(self.row_count) <= 0
            or not input_hashes
            or worker_result_hash != expected_worker_result_hash
            or int(self.worker_pid) <= 0
            or not self.limiter_scope_entered
            or not self.threadpool_observation_performed
            or int(self.loaded_pool_count) != len(pools)
            or any(threads != 1 for _, _, _, threads in pools)
            or not self.all_loaded_pools_one_thread
            or self.result_evidence_mode
            not in {"regular_file", "strict_test_fixture"}
            or (
                self.result_evidence_mode == "regular_file"
                and not bool(self.input_bytes_revalidated)
            )
            or bool(self.labels_opened)
        ):
            raise ProtocolError("OE-PPUR worker-execution transport drifted.")
        if role == "gpu_prediction":
            if (
                self.physical_device_index not in (0, 1)
                or self.cuda_visible_devices != str(self.physical_device_index)
                or self.callback_visible_logical_device != "cuda:0"
            ):
                raise ProtocolError("OE-PPUR GPU physical/logical context drifted.")
        elif (
            self.physical_device_index is not None
            or self.cuda_visible_devices != ""
            or self.callback_visible_logical_device != "cpu"
        ):
            raise ProtocolError("OE-PPUR CPU callback context drifted.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "task_hash", task)
        object.__setattr__(self, "callback_descriptor_hash", callback)
        object.__setattr__(self, "result_paths", paths)
        object.__setattr__(self, "result_hashes", hashes)
        object.__setattr__(self, "row_index_sha256", row_index)
        object.__setattr__(self, "source_surface_sha256", source_surface)
        object.__setattr__(self, "verified_input_content_hashes", input_hashes)
        object.__setattr__(self, "worker_result_hash", worker_result_hash)
        object.__setattr__(self, "loaded_pools", pools)
        object.__setattr__(self, "transport_hash", canonical_hash(self._payload()))

    @property
    def receipt_hash(self) -> str:
        """Compatibility alias for preterminal batch hashing."""

        return self.transport_hash

    @property
    def callback_identity_hash(self) -> str:
        return self.callback_descriptor_hash

    @property
    def result(self) -> WorkerResultDTO:
        """Reconstruct a process-local rich view; it is never transported."""

        return WorkerResultDTO(
            task_hash=self.task_hash,
            result_paths=self.result_paths,
            result_hashes=self.result_hashes,
            row_count=self.row_count,
            row_index_sha256=self.row_index_sha256,
            source_surface_sha256=self.source_surface_sha256,
        )

    @property
    def threadpool_evidence(self):
        """Reconstruct compatibility telemetry outside the spawn boundary."""

        from .workstation import NativeThreadpoolRecord, ThreadpoolEvidence

        records = tuple(
            NativeThreadpoolRecord(user, internal, prefix, threads)
            for user, internal, prefix, threads in self.loaded_pools
        )
        return ThreadpoolEvidence(
            role="gpu" if self.role == "gpu_prediction" else "cpu",
            limiter_scope_entered=self.limiter_scope_entered,
            observation_performed=self.threadpool_observation_performed,
            loaded_pool_count=self.loaded_pool_count,
            loaded_pools=records,
            all_loaded_pools_one_thread=self.all_loaded_pools_one_thread,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v1_worker_execution_transport_v2",
            "role": self.role,
            "task_ordinal": self.task_ordinal,
            "task_hash": self.task_hash,
            "callback_descriptor_hash": self.callback_descriptor_hash,
            "result_paths": self.result_paths,
            "result_hashes": self.result_hashes,
            "row_count": self.row_count,
            "row_index_sha256": self.row_index_sha256,
            "source_surface_sha256": self.source_surface_sha256,
            "verified_input_content_hashes": self.verified_input_content_hashes,
            "input_bytes_revalidated": self.input_bytes_revalidated,
            "worker_result_hash": self.worker_result_hash,
            "worker_pid": self.worker_pid,
            "physical_device_index": self.physical_device_index,
            "callback_visible_logical_device": self.callback_visible_logical_device,
            "cuda_visible_devices": self.cuda_visible_devices,
            "limiter_scope_entered": self.limiter_scope_entered,
            "threadpool_observation_performed": (
                self.threadpool_observation_performed
            ),
            "loaded_pool_count": self.loaded_pool_count,
            "loaded_pools": self.loaded_pools,
            "all_loaded_pools_one_thread": self.all_loaded_pools_one_thread,
            "result_evidence_mode": self.result_evidence_mode,
            "labels_opened": self.labels_opened,
        }


def assert_pickle_safe_label_free_dto(value: object) -> None:
    """Check the DTO schema before a spawn boundary is crossed."""

    allowed_types = (
        MemmapSliceDTO,
        PredictionTaskDTO,
        OuterFoldTaskDTO,
        WorkerResultDTO,
        SealedCallbackDescriptorDTO,
        WorkerExecutionDTO,
    )
    if not isinstance(value, allowed_types) or not is_dataclass(value):
        raise ProtocolError("OE-PPUR process payload is not an approved DTO.")
    for descriptor in fields(value):
        item = getattr(value, descriptor.name)
        if not _is_transport_value(item):
            raise ProtocolError("OE-PPUR process DTO contains a nonprimitive field.")
    try:
        round_tripped = pickle.loads(pickle.dumps(value))
    except Exception as exc:
        raise ProtocolError("OE-PPUR process DTO is not pickle-safe.") from exc
    if type(round_tripped) is not type(value) or round_tripped != value:
        raise ProtocolError("OE-PPUR process DTO pickle round-trip drifted.")


def _is_transport_value(value: object) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, tuple) and all(_is_transport_value(item) for item in value)


__all__ = (
    "MemmapSliceDTO",
    "OuterFoldTaskDTO",
    "PredictionTaskDTO",
    "SealedCallbackDescriptorDTO",
    "WorkerExecutionDTO",
    "WorkerResultDTO",
    "assert_pickle_safe_label_free_dto",
)
