"""Immutable contracts for the OE-PPUR v2 parsed probability matrix."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math
from pathlib import Path

from ....protocol import ProtocolError
from ..hashing import canonical_hash as _canonical_hash
from ..hashing import require_sha256 as _require_sha256
from ..identity import (
    EXPECTED_CASE_COUNT,
    EXPECTED_TERMINAL_CASE_INVENTORY_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_MANIFEST_SHA256,
)


EXPECTED_PROBABILITY_ROW_COUNT = 9_928
EXPECTED_PROBABILITY_COLUMNS = (
    "P_PROTECTED",
    "B::zero_to_one",
    "B::one_to_zero",
    "I::zero_to_one",
    "I::one_to_zero",
    "R::zero_to_one",
    "R::one_to_zero",
)
EXPECTED_PROBABILITY_COLUMN_COUNT = len(EXPECTED_PROBABILITY_COLUMNS)
PROBABILITY_STORAGE_DTYPE = "<f4"
PROBABILITY_STORAGE_BYTE_ORDER = "little"
PROBABILITY_STORAGE_MEMORY_ORDER = "C"
FLOAT32_BYTES = 4
ROW_BYTE_WIDTH = EXPECTED_PROBABILITY_COLUMN_COUNT * FLOAT32_BYTES
_SHARD_RECEIPT_FACTORY_TOKEN = object()
_MATRIX_RECEIPT_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ProbabilityMatrixShardSpec:
    """Primitive metadata for one row-contiguous raw probability shard."""

    path: str
    content_sha256: str
    six_input_admission_hash: str
    row_binding_hash: str
    row_index_sha256: str
    row_alignment_receipt_hash: str
    gpu_prediction_batch_hash: str
    gpu_result_surface_sha256: str
    gpu_worker_result_sha256: str
    row_start: int
    row_stop: int
    declared_shape: tuple[int, int]
    column_ids: tuple[str, ...] = EXPECTED_PROBABILITY_COLUMNS
    dtype: str = PROBABILITY_STORAGE_DTYPE
    byte_order: str = PROBABILITY_STORAGE_BYTE_ORDER
    memory_order: str = PROBABILITY_STORAGE_MEMORY_ORDER

    def __post_init__(self) -> None:
        path = str(self.path)
        columns = tuple(str(value) for value in self.column_ids)
        if not Path(path).is_absolute():
            raise ProtocolError("OE-PPUR v2 probability shard path is not absolute.")
        hashes = {
            name: _require_sha256(getattr(self, name), name.replace("_", " "))
            for name in (
                "content_sha256",
                "six_input_admission_hash",
                "row_binding_hash",
                "row_index_sha256",
                "row_alignment_receipt_hash",
                "gpu_prediction_batch_hash",
                "gpu_result_surface_sha256",
                "gpu_worker_result_sha256",
            )
        }
        if (
            type(self.row_start) is not int
            or type(self.row_stop) is not int
            or self.row_start < 0
            or self.row_stop <= self.row_start
            or self.row_stop > EXPECTED_PROBABILITY_ROW_COUNT
        ):
            raise ProtocolError("OE-PPUR v2 probability shard interval drifted.")
        shape = tuple(self.declared_shape)
        if (
            len(shape) != 2
            or any(type(value) is not int for value in shape)
            or shape
            != (
                self.row_stop - self.row_start,
                EXPECTED_PROBABILITY_COLUMN_COUNT,
            )
        ):
            raise ProtocolError("OE-PPUR v2 probability shard shape drifted.")
        if columns != EXPECTED_PROBABILITY_COLUMNS:
            raise ProtocolError(
                "OE-PPUR v2 probability shard column inventory drifted."
            )
        if (
            self.dtype != PROBABILITY_STORAGE_DTYPE
            or self.byte_order != PROBABILITY_STORAGE_BYTE_ORDER
            or self.memory_order != PROBABILITY_STORAGE_MEMORY_ORDER
        ):
            raise ProtocolError(
                "OE-PPUR v2 probability shard storage contract drifted."
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "declared_shape", shape)
        object.__setattr__(self, "column_ids", columns)
        for name, value in hashes.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ParsedProbabilityMatrixShardReceipt:
    """Content-derived proof for one stable, parsed shard."""

    shard_ordinal: int
    file_sha256: str
    gpu_worker_result_sha256: str
    row_start: int
    row_stop: int
    shape: tuple[int, int]
    column_ids: tuple[str, ...]
    column_content_sha256s: tuple[str, ...]
    dtype: str
    byte_order: str
    memory_order: str
    byte_length: int
    value_count: int
    minimum_probability: float
    maximum_probability: float
    descriptor_read_only: bool
    no_follow_used: bool
    stable_identity_revalidated: bool
    _factory_token: InitVar[object] = None
    shard_receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _SHARD_RECEIPT_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 parsed probability shard bypassed byte admission."
            )
        file_hash = _require_sha256(
            self.file_sha256, "parsed probability shard file hash"
        )
        worker_hash = _require_sha256(
            self.gpu_worker_result_sha256,
            "parsed probability shard worker-result hash",
        )
        column_hashes = tuple(
            _require_sha256(value, "parsed probability shard column hash")
            for value in self.column_content_sha256s
        )
        shape = tuple(self.shape)
        columns = tuple(str(value) for value in self.column_ids)
        row_count = int(self.row_stop) - int(self.row_start)
        if (
            type(self.shard_ordinal) is not int
            or self.shard_ordinal < 0
            or type(self.row_start) is not int
            or type(self.row_stop) is not int
            or self.row_start < 0
            or self.row_stop <= self.row_start
            or self.row_stop > EXPECTED_PROBABILITY_ROW_COUNT
            or shape != (row_count, EXPECTED_PROBABILITY_COLUMN_COUNT)
            or columns != EXPECTED_PROBABILITY_COLUMNS
            or len(column_hashes) != EXPECTED_PROBABILITY_COLUMN_COUNT
            or self.dtype != PROBABILITY_STORAGE_DTYPE
            or self.byte_order != PROBABILITY_STORAGE_BYTE_ORDER
            or self.memory_order != PROBABILITY_STORAGE_MEMORY_ORDER
            or type(self.byte_length) is not int
            or self.byte_length != row_count * ROW_BYTE_WIDTH
            or type(self.value_count) is not int
            or self.value_count
            != row_count * EXPECTED_PROBABILITY_COLUMN_COUNT
            or not math.isfinite(float(self.minimum_probability))
            or not math.isfinite(float(self.maximum_probability))
            or not 0.0 <= float(self.minimum_probability)
            or not float(self.maximum_probability) <= 1.0
            or float(self.minimum_probability) > float(self.maximum_probability)
            or self.descriptor_read_only is not True
            or self.no_follow_used is not True
            or self.stable_identity_revalidated is not True
        ):
            raise ProtocolError(
                "OE-PPUR v2 parsed probability shard receipt drifted."
            )
        object.__setattr__(self, "file_sha256", file_hash)
        object.__setattr__(self, "gpu_worker_result_sha256", worker_hash)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "column_ids", columns)
        object.__setattr__(self, "column_content_sha256s", column_hashes)
        object.__setattr__(
            self, "minimum_probability", float(self.minimum_probability)
        )
        object.__setattr__(
            self, "maximum_probability", float(self.maximum_probability)
        )
        object.__setattr__(
            self, "shard_receipt_hash", _canonical_hash(self._payload())
        )

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_parsed_probability_shard_v2",
            "shard_ordinal": self.shard_ordinal,
            "file_sha256": self.file_sha256,
            "gpu_worker_result_sha256": self.gpu_worker_result_sha256,
            "row_interval": [self.row_start, self.row_stop],
            "shape": list(self.shape),
            "ordered_column_ids": list(self.column_ids),
            "ordered_column_content_sha256s": list(
                self.column_content_sha256s
            ),
            "dtype": self.dtype,
            "byte_order": self.byte_order,
            "memory_order": self.memory_order,
            "byte_length": self.byte_length,
            "value_count": self.value_count,
            "minimum_probability": self.minimum_probability,
            "maximum_probability": self.maximum_probability,
            "descriptor_read_only": self.descriptor_read_only,
            "no_follow_used": self.no_follow_used,
            "stable_identity_revalidated": self.stable_identity_revalidated,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "shard_receipt_hash": self.shard_receipt_hash}


@dataclass(frozen=True, slots=True)
class ParsedProbabilityMatrixScienceReceipt:
    """Guarded proof that the canonical 9,928 x 7 matrix was parsed."""

    six_input_admission_hash: str
    input_binding_hash: str
    row_binding_hash: str
    cache_content_sha256: str
    cache_row_order_sha256: str
    manifest_sha256: str
    case_inventory_sha256: str
    row_index_sha256: str
    row_alignment_receipt_hash: str
    gpu_prediction_batch_hash: str
    gpu_result_surface_sha256: str
    gpu_worker_result_hashes: tuple[str, ...]
    gpu_result_file_hashes: tuple[str, ...]
    shards: tuple[ParsedProbabilityMatrixShardReceipt, ...]
    matrix_content_sha256: str
    column_content_sha256s: tuple[str, ...]
    scientific_values_validated: bool
    _factory_token: InitVar[object] = None
    row_count: int = field(init=False)
    case_count: int = field(init=False)
    shape: tuple[int, int] = field(init=False)
    column_ids: tuple[str, ...] = field(init=False)
    minimum_probability: float = field(init=False)
    maximum_probability: float = field(init=False)
    gpu_to_matrix_binding_sha256: str = field(init=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _MATRIX_RECEIPT_FACTORY_TOKEN:
            raise ProtocolError(
                "OE-PPUR v2 parsed probability matrix bypassed science admission."
            )
        lineage = {
            name: _require_sha256(getattr(self, name), name.replace("_", " "))
            for name in (
                "six_input_admission_hash",
                "input_binding_hash",
                "row_binding_hash",
                "cache_content_sha256",
                "cache_row_order_sha256",
                "manifest_sha256",
                "case_inventory_sha256",
                "row_index_sha256",
                "row_alignment_receipt_hash",
                "gpu_prediction_batch_hash",
                "gpu_result_surface_sha256",
            )
        }
        matrix_hash = _require_sha256(
            self.matrix_content_sha256, "matrix content hash"
        )
        workers = tuple(
            _require_sha256(value, "GPU worker-result hash")
            for value in self.gpu_worker_result_hashes
        )
        files = tuple(
            _require_sha256(value, "GPU result-file hash")
            for value in self.gpu_result_file_hashes
        )
        column_hashes = tuple(
            _require_sha256(value, "matrix column-content hash")
            for value in self.column_content_sha256s
        )
        shards = tuple(self.shards)
        if (
            lineage["cache_content_sha256"]
            != EXPECTED_TEST_CACHE_CONTENT_HASH
            or lineage["cache_row_order_sha256"]
            != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
            or lineage["manifest_sha256"] != EXPECTED_TEST_MANIFEST_SHA256
            or lineage["case_inventory_sha256"]
            != EXPECTED_TERMINAL_CASE_INVENTORY_SHA256
            or lineage["row_index_sha256"]
            != lineage["cache_row_order_sha256"]
            or not workers
            or len(set(workers)) != len(workers)
            or not files
            or not shards
            or len(files) != len(shards)
            or len(column_hashes) != EXPECTED_PROBABILITY_COLUMN_COUNT
            or any(
                not isinstance(row, ParsedProbabilityMatrixShardReceipt)
                for row in shards
            )
            or tuple(row.shard_ordinal for row in shards)
            != tuple(range(len(shards)))
            or tuple(row.file_sha256 for row in shards) != files
            or _ordered_unique(
                tuple(row.gpu_worker_result_sha256 for row in shards)
            )
            != workers
            or self.scientific_values_validated is not True
        ):
            raise ProtocolError(
                "OE-PPUR v2 parsed probability matrix lineage drifted."
            )
        _validate_exact_shard_topology(shards)
        minimum = min(row.minimum_probability for row in shards)
        maximum = max(row.maximum_probability for row in shards)
        binding = _canonical_hash(
            {
                "schema_version": "oe_ppur_v2_gpu_to_probability_matrix_v2",
                **lineage,
                "ordered_gpu_worker_result_hashes": list(workers),
                "ordered_gpu_result_file_hashes": list(files),
                "ordered_parsed_shard_receipt_hashes": [
                    row.shard_receipt_hash for row in shards
                ],
                "matrix_content_sha256": matrix_hash,
                "ordered_column_content_sha256s": list(column_hashes),
                "scientific_values_validated": True,
            }
        )
        for name, value in lineage.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "gpu_worker_result_hashes", workers)
        object.__setattr__(self, "gpu_result_file_hashes", files)
        object.__setattr__(self, "shards", shards)
        object.__setattr__(self, "matrix_content_sha256", matrix_hash)
        object.__setattr__(self, "column_content_sha256s", column_hashes)
        object.__setattr__(self, "row_count", EXPECTED_PROBABILITY_ROW_COUNT)
        object.__setattr__(self, "case_count", EXPECTED_CASE_COUNT)
        object.__setattr__(
            self,
            "shape",
            (EXPECTED_PROBABILITY_ROW_COUNT, EXPECTED_PROBABILITY_COLUMN_COUNT),
        )
        object.__setattr__(self, "column_ids", EXPECTED_PROBABILITY_COLUMNS)
        object.__setattr__(self, "minimum_probability", float(minimum))
        object.__setattr__(self, "maximum_probability", float(maximum))
        object.__setattr__(self, "gpu_to_matrix_binding_sha256", binding)
        object.__setattr__(self, "receipt_hash", _canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_parsed_probability_matrix_science_receipt_v2",
            "six_input_admission_hash": self.six_input_admission_hash,
            "input_binding_hash": self.input_binding_hash,
            "row_binding_hash": self.row_binding_hash,
            "cache_content_sha256": self.cache_content_sha256,
            "cache_row_order_sha256": self.cache_row_order_sha256,
            "manifest_sha256": self.manifest_sha256,
            "case_count": self.case_count,
            "case_inventory_sha256": self.case_inventory_sha256,
            "row_index_sha256": self.row_index_sha256,
            "row_alignment_receipt_hash": self.row_alignment_receipt_hash,
            "gpu_prediction_batch_hash": self.gpu_prediction_batch_hash,
            "gpu_result_surface_sha256": self.gpu_result_surface_sha256,
            "ordered_gpu_worker_result_hashes": list(
                self.gpu_worker_result_hashes
            ),
            "ordered_gpu_result_file_hashes": list(self.gpu_result_file_hashes),
            "ordered_parsed_shard_receipt_hashes": [
                row.shard_receipt_hash for row in self.shards
            ],
            "ordered_shard_intervals": [
                [row.row_start, row.row_stop] for row in self.shards
            ],
            "row_count": self.row_count,
            "shape": list(self.shape),
            "ordered_column_ids": list(self.column_ids),
            "dtype": PROBABILITY_STORAGE_DTYPE,
            "byte_order": PROBABILITY_STORAGE_BYTE_ORDER,
            "memory_order": PROBABILITY_STORAGE_MEMORY_ORDER,
            "minimum_probability": self.minimum_probability,
            "maximum_probability": self.maximum_probability,
            "matrix_content_sha256": self.matrix_content_sha256,
            "ordered_column_content_sha256s": list(
                self.column_content_sha256s
            ),
            "gpu_to_matrix_binding_sha256": self.gpu_to_matrix_binding_sha256,
            "scientific_values_validated": self.scientific_values_validated,
            "labels_present": False,
            "terminal_capability_opened": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def _issue_parsed_probability_shard_receipt(
    **fields: object,
) -> ParsedProbabilityMatrixShardReceipt:
    return ParsedProbabilityMatrixShardReceipt(
        **fields,
        _factory_token=_SHARD_RECEIPT_FACTORY_TOKEN,
    )


def _issue_parsed_probability_matrix_receipt(
    **fields: object,
) -> ParsedProbabilityMatrixScienceReceipt:
    return ParsedProbabilityMatrixScienceReceipt(
        **fields,
        _factory_token=_MATRIX_RECEIPT_FACTORY_TOKEN,
    )


def _rebuild_parsed_probability_shard_receipt(
    receipt: ParsedProbabilityMatrixShardReceipt,
) -> ParsedProbabilityMatrixShardReceipt:
    if not isinstance(receipt, ParsedProbabilityMatrixShardReceipt):
        raise ProtocolError("OE-PPUR v2 parsed probability shard is untyped.")
    return _issue_parsed_probability_shard_receipt(
        shard_ordinal=receipt.shard_ordinal,
        file_sha256=receipt.file_sha256,
        gpu_worker_result_sha256=receipt.gpu_worker_result_sha256,
        row_start=receipt.row_start,
        row_stop=receipt.row_stop,
        shape=receipt.shape,
        column_ids=receipt.column_ids,
        column_content_sha256s=receipt.column_content_sha256s,
        dtype=receipt.dtype,
        byte_order=receipt.byte_order,
        memory_order=receipt.memory_order,
        byte_length=receipt.byte_length,
        value_count=receipt.value_count,
        minimum_probability=receipt.minimum_probability,
        maximum_probability=receipt.maximum_probability,
        descriptor_read_only=receipt.descriptor_read_only,
        no_follow_used=receipt.no_follow_used,
        stable_identity_revalidated=receipt.stable_identity_revalidated,
    )


def _rebuild_parsed_probability_matrix_receipt(
    receipt: ParsedProbabilityMatrixScienceReceipt,
) -> ParsedProbabilityMatrixScienceReceipt:
    if not isinstance(receipt, ParsedProbabilityMatrixScienceReceipt):
        raise ProtocolError("OE-PPUR v2 parsed probability receipt is untyped.")
    return _issue_parsed_probability_matrix_receipt(
        six_input_admission_hash=receipt.six_input_admission_hash,
        input_binding_hash=receipt.input_binding_hash,
        row_binding_hash=receipt.row_binding_hash,
        cache_content_sha256=receipt.cache_content_sha256,
        cache_row_order_sha256=receipt.cache_row_order_sha256,
        manifest_sha256=receipt.manifest_sha256,
        case_inventory_sha256=receipt.case_inventory_sha256,
        row_index_sha256=receipt.row_index_sha256,
        row_alignment_receipt_hash=receipt.row_alignment_receipt_hash,
        gpu_prediction_batch_hash=receipt.gpu_prediction_batch_hash,
        gpu_result_surface_sha256=receipt.gpu_result_surface_sha256,
        gpu_worker_result_hashes=receipt.gpu_worker_result_hashes,
        gpu_result_file_hashes=receipt.gpu_result_file_hashes,
        shards=tuple(
            _rebuild_parsed_probability_shard_receipt(row)
            for row in receipt.shards
        ),
        matrix_content_sha256=receipt.matrix_content_sha256,
        column_content_sha256s=receipt.column_content_sha256s,
        scientific_values_validated=receipt.scientific_values_validated,
    )


def _validate_exact_shard_topology(
    shards: tuple[ParsedProbabilityMatrixShardReceipt, ...],
) -> None:
    cursor = 0
    for shard in shards:
        rebuilt = _rebuild_parsed_probability_shard_receipt(shard)
        if rebuilt != shard or shard.row_start != cursor:
            raise ProtocolError(
                "OE-PPUR v2 parsed probability shard topology drifted."
            )
        cursor = shard.row_stop
    if cursor != EXPECTED_PROBABILITY_ROW_COUNT:
        raise ProtocolError(
            "OE-PPUR v2 parsed matrix row inventory is not exact."
        )


def _ordered_unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = (
    "EXPECTED_PROBABILITY_COLUMN_COUNT",
    "EXPECTED_PROBABILITY_COLUMNS",
    "EXPECTED_PROBABILITY_ROW_COUNT",
    "FLOAT32_BYTES",
    "PROBABILITY_STORAGE_BYTE_ORDER",
    "PROBABILITY_STORAGE_DTYPE",
    "PROBABILITY_STORAGE_MEMORY_ORDER",
    "ParsedProbabilityMatrixScienceReceipt",
    "ParsedProbabilityMatrixShardReceipt",
    "ProbabilityMatrixShardSpec",
    "ROW_BYTE_WIDTH",
)
