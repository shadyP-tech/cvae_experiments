"""Process-local, immutable input cache for HARP v15 classifier workers.

The classifier pool evaluates 81 joint support/target tasks against repeatedly
shared staged source, index, and frame files.
Reopening and rehashing those files per task is both expensive and weaker than
binding every task to their exact validated bytes.
This module owns that worker-local state: every member is hash-verified once,
its path and stat identity are checked on every subsequent use, and read-only
memmaps/source-block hashes are reused only under the same complete binding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import numpy as np

from ....common.hashing import stable_hash
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ...runtime.artifact_io import read_json, sha256_file
from .gpu_surface import (
    EXPECTED_STREAM_COUNT,
    SOURCE_ROWS_PER_CLASS,
    source_block_sha256,
)
from .physical_actions import EXACT_NINE_SEED_PAIRS
from .hash_contracts import require_sha256, require_stable_hash
from .projection_hashing import projection_semantic_hash


CLASSIFIER_THREADS_PER_WORKER = 3


class _SourceOrderedAction(Protocol):
    @property
    def source_order(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class SourceCacheKey:
    """Complete identity of one validated frozen-source array/index pair."""

    semantic_lock_hash: str
    lock_sha256: str
    index_semantic_hash: str
    array_path: str
    array_sha256: str
    index_path: str
    index_sha256: str


@dataclass(frozen=True, slots=True)
class ScopedSourceArray:
    """Capability view that exposes only projection-authorized block offsets."""

    _values: np.ndarray
    allowed_block_ordinals: frozenset[int]
    projection_hash: str

    def __getitem__(self, ordinal: int) -> np.ndarray:
        if type(ordinal) is not int or ordinal not in self.allowed_block_ordinals:
            raise ProtocolError(
                "HARP v15 source projection rejected an excluded block offset."
            )
        return self._values[ordinal]


# A file identity includes ctime as well as mtime.  A same-size in-place write
# can restore mtime, but an unprivileged process cannot restore ctime.
_FileIdentity = tuple[int, int, int, int, int]
_VERIFIED_FILE_CACHE: dict[
    tuple[str, str], tuple[str, _FileIdentity]
] = {}
_PATH_BY_BINDING: dict[tuple[str, str], str] = {}
_BINDING_BY_PATH: dict[tuple[str, str], str] = {}
_SOURCE_ARRAY_CACHE: dict[tuple[str, str, str, str], np.ndarray] = {}
_FRAME_ARRAY_CACHE: dict[tuple[str, str, str, str], np.ndarray] = {}
_SOURCE_BLOCK_HASH_CACHE: dict[tuple[str, str, int], str] = {}
_SOURCE_RECORD_INDEX_CACHE: dict[
    SourceCacheKey,
    tuple[str, Mapping[tuple[str, int, int], Mapping[str, object]]],
] = {}


def initialize_classifier_worker(threads: int) -> None:
    """Bind one spawned CPU worker to the frozen CUDA-blind 4x3 topology."""

    if type(threads) is not int or threads != CLASSIFIER_THREADS_PER_WORKER:
        raise ProtocolError("HARP v15 classifier worker initializer drifted.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    reset_worker_state()


def reset_worker_state() -> None:
    """Drop only process-local read-only views and verification receipts."""

    _VERIFIED_FILE_CACHE.clear()
    _PATH_BY_BINDING.clear()
    _BINDING_BY_PATH.clear()
    _SOURCE_ARRAY_CACHE.clear()
    _FRAME_ARRAY_CACHE.clear()
    _SOURCE_BLOCK_HASH_CACHE.clear()
    _SOURCE_RECORD_INDEX_CACHE.clear()


def load_worker_arrays(
    task: Mapping[str, object],
) -> tuple[ScopedSourceArray, np.ndarray, SourceCacheKey]:
    """Return bound read-only source/frame memmaps for one classifier task."""

    semantic_lock_hash = require_stable_hash(
        task.get("source_stream_lock_hash"), name="source-stream lock hash"
    )
    lock_sha256 = require_sha256(
        task.get("source_stream_lock_sha256"), name="source-stream lock SHA-256"
    )
    index_semantic_hash = require_stable_hash(
        task.get("source_stream_index_hash"), name="source-stream index hash"
    )
    source_sha = require_sha256(
        task.get("source_array_sha256"), name="source-array hash"
    )
    index_sha = require_sha256(
        task.get("source_index_sha256"), name="source-index hash"
    )
    frame_sha = require_sha256(
        task.get("frame_array_sha256"), name="frame-array hash"
    )
    frame_receipt_hash = require_stable_hash(
        task.get("frame_receipt_hash"), name="frame-receipt hash"
    )
    frame_receipt_sha256 = require_sha256(
        task.get("frame_receipt_sha256"), name="frame-receipt SHA-256"
    )
    source_path = _verify_file(
        task.get("source_array_path"),
        expected_sha256=source_sha,
        binding=lock_sha256,
        kind="source-array",
    )
    index_path = _verify_file(
        task.get("source_index_path"),
        expected_sha256=index_sha,
        binding=index_sha,
        kind="source-index",
    )
    frame_path = _verify_file(
        task.get("frame_array_path"),
        expected_sha256=frame_sha,
        binding=frame_receipt_sha256,
        kind="frame-array",
    )
    if task.get("frame_projection_schema") is not None:
        receipt_path = _verify_file(
            task.get("frame_receipt_path"),
            expected_sha256=frame_receipt_sha256,
            binding=frame_receipt_hash,
            kind="frame-receipt",
        )
        receipt = read_json(receipt_path)
        receipt_body = {
            key: value for key, value in receipt.items() if key != "frame_projection_hash"
        }
        schema = task.get("frame_projection_schema")
        common_invalid = (
            schema not in {
                "midogpp_harp_v15_query_frame_projection_v1",
                "midogpp_harp_v15_joint_support_target_projection_v1",
            }
            or receipt.get("schema_version") != task.get("frame_projection_schema")
            or receipt.get("frame_projection_hash") != frame_receipt_hash
            or task.get("frame_projection_hash") != frame_receipt_hash
            or projection_semantic_hash(
                receipt_body, name="frame projection hash"
            )
            != frame_receipt_hash
            or receipt.get("array_sha256") != frame_sha
            or receipt.get("other_center_rows_visible") is not False
            or receipt.get("labels_consumed") is not False
        )
        if common_invalid:
            raise ProtocolError("HARP v15 query frame projection binding drifted.")
        if schema == "midogpp_harp_v15_query_frame_projection_v1":
            if receipt.get("query_center_id") != task.get("query_center_id"):
                raise ProtocolError("HARP v15 query frame projection center drifted.")
        else:
            split = task.get("role_split_offset")
            support_ids = task.get("support_sample_ids")
            target_ids = task.get("target_sample_ids")
            if (
                receipt.get("outer_target_id") != task.get("outer_target_id")
                or receipt.get("support_role") != task.get("support_role")
                or receipt.get("target_role") != task.get("target_role")
                or receipt.get("role_split_offset") != split
                or receipt.get("support_sample_ids") != support_ids
                or receipt.get("target_sample_ids") != target_ids
                or type(split) is not int
                or split <= 0
                or split >= len(task.get("sample_ids", ()))
                or task.get("classifier_fit_reused_across_roles") is not True
            ):
                raise ProtocolError("HARP v15 joint support/target projection drifted.")
    source_key = SourceCacheKey(
        semantic_lock_hash=semantic_lock_hash,
        lock_sha256=lock_sha256,
        index_semantic_hash=index_semantic_hash,
        array_path=str(source_path),
        array_sha256=source_sha,
        index_path=str(index_path),
        index_sha256=index_sha,
    )

    array_cache_key = (
        semantic_lock_hash,
        lock_sha256,
        str(source_path),
        source_sha,
    )
    source = _SOURCE_ARRAY_CACHE.get(array_cache_key)
    if source is None:
        source = _open_memmap(source_path, name="source")
        _assert_current_identity(source_path, kind="source-array")
        _SOURCE_ARRAY_CACHE[array_cache_key] = source
    if (
        source.dtype != np.float32
        or source.shape
        != (EXPECTED_STREAM_COUNT, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
    ):
        raise ProtocolError("HARP v15 source memmap geometry drifted.")

    frame_key = (
        str(frame_path),
        frame_sha,
        frame_receipt_hash,
        frame_receipt_sha256,
    )
    frame = _FRAME_ARRAY_CACHE.get(frame_key)
    if frame is None:
        frame = _open_memmap(frame_path, name="frame")
        _assert_current_identity(frame_path, kind="frame-array")
        _FRAME_ARRAY_CACHE[frame_key] = frame
    if frame.dtype != np.float32 or frame.ndim != 2 or frame.shape[1] != COMMON_OUTPUT_DIM:
        raise ProtocolError("HARP v15 frame memmap geometry drifted.")
    raw_records = task.get("source_records")
    if not isinstance(raw_records, list):
        raise ProtocolError("HARP v15 source projection records are absent.")
    try:
        allowed_ordinals = frozenset(int(row["block_ordinal"]) for row in raw_records)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v15 source projection offsets are malformed.") from exc
    if (
        len(allowed_ordinals) != len(raw_records)
        or any(value < 0 or value >= EXPECTED_STREAM_COUNT for value in allowed_ordinals)
    ):
        raise ProtocolError("HARP v15 source projection offsets drifted.")
    return (
        ScopedSourceArray(
            _values=source,
            allowed_block_ordinals=allowed_ordinals,
            projection_hash=index_semantic_hash,
        ),
        frame,
        source_key,
    )


def load_source_blocks(
    actions: Sequence[_SourceOrderedAction],
    task: Mapping[str, object],
    *,
    source_values: ScopedSourceArray,
    source_key: SourceCacheKey,
) -> dict[str, dict[str, np.ndarray]]:
    """Select and validate source blocks, hashing each block once per child."""

    _assert_current_identity(Path(source_key.array_path), kind="source-array")
    _assert_current_identity(Path(source_key.index_path), kind="source-index")
    records = _validated_source_record_index(task, source_key=source_key)
    source_blocks: dict[str, dict[str, np.ndarray]] = {}
    selected_sources = sorted(
        {value for action in actions for value in action.source_order}
    )
    for source in selected_sources:
        try:
            record = records[
                (source, int(task["training_seed"]), int(task["generation_seed"]))
            ]
        except KeyError as exc:
            raise ProtocolError("HARP v15 selected source record is absent.") from exc
        block_ordinal = int(record["block_ordinal"])
        block = np.asarray(source_values[block_ordinal], dtype=np.float32)
        block_key = (source_key.array_sha256, source_key.array_path, block_ordinal)
        observed_hash = _SOURCE_BLOCK_HASH_CACHE.get(block_key)
        if observed_hash is None:
            observed_hash = source_block_sha256(block)
            _SOURCE_BLOCK_HASH_CACHE[block_key] = observed_hash
        if observed_hash != record["output_sha256"]:
            raise ProtocolError(
                "HARP v15 source-stream bytes drifted in classifier worker."
            )
        source_blocks[source] = {
            "embeddings": block,
            "labels": np.concatenate(
                (
                    np.zeros(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                    np.ones(SOURCE_ROWS_PER_CLASS, dtype=np.int64),
                )
            ),
        }
    return source_blocks


def _verify_file(
    value: object,
    *,
    expected_sha256: str,
    binding: str,
    kind: str,
) -> Path:
    path, before = _checked_plain_file(value, name=kind)
    path_text = str(path)
    binding_key = (kind, binding)
    path_key = (kind, path_text)
    prior_path = _PATH_BY_BINDING.get(binding_key)
    prior_binding = _BINDING_BY_PATH.get(path_key)
    if (prior_path is not None and prior_path != path_text) or (
        prior_binding is not None and prior_binding != binding
    ):
        raise ProtocolError(f"HARP v15 {kind} path/binding drifted.")
    _PATH_BY_BINDING[binding_key] = path_text
    _BINDING_BY_PATH[path_key] = binding

    prior = _VERIFIED_FILE_CACHE.get(path_key)
    if prior is not None:
        prior_sha, prior_identity = prior
        if prior_sha != expected_sha256:
            raise ProtocolError(f"HARP v15 {kind} hash binding drifted.")
        if prior_identity != before:
            raise ProtocolError(
                f"HARP v15 {kind} file identity drifted in classifier worker."
            )
        return path

    observed_sha = sha256_file(path)
    _, after = _checked_plain_file(path_text, name=kind)
    if before != after:
        raise ProtocolError(f"HARP v15 {kind} file changed while it was hashed.")
    if observed_sha != expected_sha256:
        raise ProtocolError(f"HARP v15 {kind} bytes failed their task binding.")
    _VERIFIED_FILE_CACHE[path_key] = (expected_sha256, after)
    return path


def _checked_plain_file(
    value: object, *, name: str
) -> tuple[Path, _FileIdentity]:
    if type(value) is not str:
        raise ProtocolError(f"HARP v15 {name} path is malformed.")
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
        if (
            not path.is_absolute()
            or path != resolved
            or path.is_symlink()
            or not path.is_file()
        ):
            raise ProtocolError(f"HARP v15 {name} path is unsafe.")
        stat = path.stat()
    except ProtocolError:
        raise
    except (OSError, RuntimeError) as exc:
        raise ProtocolError(f"HARP v15 {name} path is unsafe.") from exc
    return path, (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _assert_current_identity(path: Path, *, kind: str) -> None:
    _, identity = _checked_plain_file(str(path), name=kind)
    prior = _VERIFIED_FILE_CACHE.get((kind, str(path)))
    if prior is None or prior[1] != identity:
        raise ProtocolError(f"HARP v15 {kind} file changed while it was opened.")


def _open_memmap(path: Path, *, name: str) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"HARP v15 {name} memmap could not be opened.") from exc


def _validated_source_record_index(
    task: Mapping[str, object], *, source_key: SourceCacheKey
) -> Mapping[tuple[str, int, int], Mapping[str, object]]:
    records = task.get("source_records")
    scoped = task.get("schema_version") == (
        "midogpp_harp_v15_fold_conditioned_classifier_task_v1"
    )
    allowed = tuple(str(value) for value in task.get("allowed_source_ids", ()))
    expected_record_count = (
        len(allowed) * len(EXACT_NINE_SEED_PAIRS)
        if scoped
        else EXPECTED_STREAM_COUNT
    )
    if not isinstance(records, list) or len(records) != expected_record_count:
        raise ProtocolError("HARP v15 source record inventory is malformed.")
    record_hash = canonical_hash(records)
    prior = _SOURCE_RECORD_INDEX_CACHE.get(source_key)
    if prior is not None:
        if prior[0] != record_hash:
            raise ProtocolError("HARP v15 source records drifted within a worker.")
        return prior[1]

    source_index_path = Path(source_key.index_path)
    source_index = read_json(source_index_path)
    _assert_current_identity(source_index_path, kind="source-index")
    if scoped:
        projection_body = {
            key: value
            for key, value in source_index.items()
            if key != "source_record_projection_hash"
        }
        if (
            source_index.get("schema_version")
            != "midogpp_harp_v15_fold_source_record_projection_v1"
            or task.get("source_record_projection_schema")
            != source_index.get("schema_version")
            or source_index.get("source_record_projection_hash")
            != source_key.index_semantic_hash
            or task.get("source_record_projection_hash")
            != source_key.index_semantic_hash
            or source_index.get("source_record_projection_hash")
            != projection_semantic_hash(
                projection_body, name="source-record projection hash"
            )
            or source_index.get("records") != records
            or tuple(source_index.get("allowed_source_ids", ())) != allowed
            or source_index.get("outer_target_id") != task.get("outer_target_id")
            or source_index.get("heldout_center_id")
            != task.get("heldout_center_id")
            or source_index.get("current_query_center_id")
            != task.get("current_query_center_id")
            or source_index.get("excluded_record_offsets_visible") is not False
        ):
            raise ProtocolError(
                "HARP v15 task records differ from the fold projection."
            )
        expected_keys = tuple(
            (center, training_seed, generation_seed)
            for center in allowed
            for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS
        )
    else:
        source_index_unhashed = {
            key: value
            for key, value in source_index.items()
            if key != "source_stream_index_hash"
        }
        if (
            source_index.get("records") != records
            or source_index.get("source_stream_index_hash")
            != source_key.index_semantic_hash
            or stable_hash(source_index_unhashed) != source_key.index_semantic_hash
        ):
            raise ProtocolError(
                "HARP v15 task records differ from the bound source index."
            )
        expected_keys = tuple(
            (center, training_seed, generation_seed)
            for center in CENTERS
            for training_seed, generation_seed in EXACT_NINE_SEED_PAIRS
        )
    expected_fields = {
        "block_ordinal",
        "source_center",
        "training_seed",
        "generation_seed",
        "stream_id",
        "expert_lock_hash",
        "rows_per_class",
        "row_count",
        "feature_dim",
        "output_sha256",
    }
    index: dict[tuple[str, int, int], Mapping[str, object]] = {}
    observed_keys: list[tuple[str, int, int]] = []
    for ordinal, raw in enumerate(records):
        if not isinstance(raw, Mapping) or set(raw) != expected_fields:
            raise ProtocolError("HARP v15 source record schema drifted.")
        if (
            type(raw.get("source_center")) is not str
            or type(raw.get("training_seed")) is not int
            or type(raw.get("generation_seed")) is not int
            or type(raw.get("block_ordinal")) is not int
            or (
                not scoped
                and raw.get("block_ordinal") != ordinal
            )
            or raw.get("rows_per_class") != SOURCE_ROWS_PER_CLASS
            or raw.get("row_count") != 2 * SOURCE_ROWS_PER_CLASS
            or raw.get("feature_dim") != COMMON_OUTPUT_DIM
            or type(raw.get("stream_id")) is not str
            or not raw.get("stream_id")
        ):
            raise ProtocolError("HARP v15 source record geometry drifted.")
        require_stable_hash(raw.get("expert_lock_hash"), name="expert-lock hash")
        require_sha256(raw.get("output_sha256"), name="source-block hash")
        key = (
            str(raw["source_center"]),
            int(raw["training_seed"]),
            int(raw["generation_seed"]),
        )
        observed_keys.append(key)
        index[key] = MappingProxyType(dict(raw))
    if tuple(observed_keys) != expected_keys or len(index) != expected_record_count:
        raise ProtocolError("HARP v15 source record coverage drifted.")
    frozen = MappingProxyType(index)
    _SOURCE_RECORD_INDEX_CACHE[source_key] = (record_hash, frozen)
    return frozen


__all__ = (
    "CLASSIFIER_THREADS_PER_WORKER",
    "SourceCacheKey",
    "ScopedSourceArray",
    "initialize_classifier_worker",
    "load_source_blocks",
    "load_worker_arrays",
    "require_sha256",
    "require_stable_hash",
    "reset_worker_state",
)
