"""Checkpoint and final-array IO for SCEPTRE v5 source generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
import os
from pathlib import Path

import numpy as np

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.artifact_io import (
    atomic_json,
    read_json,
    sha256_array,
    sha256_file,
)

from .source_contracts import SourceGeometry, SourceStreamRecord
from .source_hashing import block_bundle_sha256, canonical_sha256
from .source_planning import task_key


def publish_checkpoint(
    task: Mapping[str, object],
    *,
    blocks: Sequence[np.ndarray],
    geometry: SourceGeometry,
) -> Mapping[str, object]:
    if len(blocks) != len(geometry.generation_seeds):
        raise ProtocolError("SCEPTRE v5 source worker generation-seed coverage drifted.")
    values = np.ascontiguousarray(np.stack(blocks), dtype=np.float32)
    expected_shape = (
        len(geometry.generation_seeds),
        2 * geometry.rows_per_class,
        geometry.feature_dim,
    )
    if (
        values.shape != expected_shape
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ProtocolError("SCEPTRE v5 source worker emitted invalid values.")
    array_path = Path(str(task["checkpoint_array_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    persist_exact_npy(array_path, values)
    records = []
    for ordinal, key in enumerate(task["generation_keys"]):
        block = values[ordinal]
        records.append(
            {
                "generation_seed": int(getattr(key, "generation_seed")),
                "stream_id": str(getattr(key, "stream_id")),
                "expert_lock_hash": str(getattr(key, "expert_lock_hash")),
                "output_sha256": block_bundle_sha256(
                    block, geometry.rows_per_class
                ),
                "array_sha256": sha256_array(block),
            }
        )
    checkpoint_unhashed = {
        "schema_version": "midogpp_sceptre_v5_physical_source_checkpoint_v1",
        "status": "COMPLETE",
        "attempt_id": task["attempt_id"],
        "task_sha256": task["task_sha256"],
        "source_center": task["source_center"],
        "training_seed": task["training_seed"],
        "device": task["device"],
        "array_file_sha256": sha256_file(array_path),
        "array_shape": list(values.shape),
        "array_dtype": str(values.dtype),
        "records": records,
        "target_cache_opened": False,
        "manifest_opened": False,
        "outcomes_available": False,
        "expert_updated": False,
        "float32_outputs": True,
    }
    checkpoint = {
        **checkpoint_unhashed,
        "checkpoint_sha256": canonical_sha256(checkpoint_unhashed),
    }
    persist_exact_json(json_path, checkpoint)
    return checkpoint


def load_checkpoint_if_complete(
    task: Mapping[str, object], *, geometry: SourceGeometry
) -> Mapping[str, object] | None:
    array_path = Path(str(task["checkpoint_array_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    present = (array_path.is_file(), json_path.is_file())
    if any(path.is_symlink() for path in (array_path, json_path)):
        raise ProtocolError("SCEPTRE v5 source checkpoint contains a symlink.")
    if present == (False, False):
        return None
    if present != (True, True):
        raise ProtocolError("SCEPTRE v5 source checkpoint is partial; refusing refit.")
    payload = read_json(json_path)
    unhashed = {
        key: value for key, value in payload.items() if key != "checkpoint_sha256"
    }
    records = payload.get("records")
    values = np.load(array_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        len(geometry.generation_seeds),
        2 * geometry.rows_per_class,
        geometry.feature_dim,
    )
    if (
        payload.get("checkpoint_sha256") != canonical_sha256(unhashed)
        or payload.get("schema_version")
        != "midogpp_sceptre_v5_physical_source_checkpoint_v1"
        or payload.get("status") != "COMPLETE"
        or payload.get("attempt_id") != task["attempt_id"]
        or payload.get("task_sha256") != task["task_sha256"]
        or payload.get("source_center") != task["source_center"]
        or int(payload.get("training_seed", -1)) != task["training_seed"]
        or payload.get("device") != task["device"]
        or payload.get("array_file_sha256") != sha256_file(array_path)
        or payload.get("array_shape") != list(expected_shape)
        or payload.get("array_dtype") != "float32"
        or values.shape != expected_shape
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or not isinstance(records, list)
        or len(records) != len(geometry.generation_seeds)
        or payload.get("target_cache_opened") is not False
        or payload.get("manifest_opened") is not False
        or payload.get("outcomes_available") is not False
        or payload.get("expert_updated") is not False
    ):
        raise ProtocolError("SCEPTRE v5 source checkpoint failed validation.")
    for ordinal, (raw, key) in enumerate(
        zip(records, task["generation_keys"], strict=True)
    ):
        if (
            not isinstance(raw, Mapping)
            or int(raw.get("generation_seed", -1))
            != int(getattr(key, "generation_seed"))
            or raw.get("stream_id") != str(getattr(key, "stream_id"))
            or raw.get("expert_lock_hash") != str(getattr(key, "expert_lock_hash"))
            or raw.get("array_sha256") != sha256_array(values[ordinal])
            or raw.get("output_sha256")
            != block_bundle_sha256(values[ordinal], geometry.rows_per_class)
        ):
            raise ProtocolError("SCEPTRE v5 source checkpoint record drifted.")
    return payload


def publish_source_array(
    path: Path,
    *,
    tasks: Sequence[Mapping[str, object]],
    completed: Mapping[tuple[str, int], Mapping[str, object]],
    geometry: SourceGeometry,
) -> tuple[SourceStreamRecord, ...]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    records: list[SourceStreamRecord] = []
    cursor = 0
    try:
        target = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=geometry.array_shape,
        )
        for task in tasks:
            checkpoint = completed[task_key(task)]
            values = np.load(
                Path(str(task["checkpoint_array_path"])),
                mmap_mode="r",
                allow_pickle=False,
            )
            for seed_ordinal, raw in enumerate(checkpoint["records"]):
                target[cursor] = values[seed_ordinal]
                records.append(
                    SourceStreamRecord(
                        block_ordinal=cursor,
                        source_center=str(task["source_center"]),
                        training_seed=int(task["training_seed"]),
                        generation_seed=int(raw["generation_seed"]),
                        stream_id=str(raw["stream_id"]),
                        expert_lock_hash=str(raw["expert_lock_hash"]),
                        rows_per_class=geometry.rows_per_class,
                        feature_dim=geometry.feature_dim,
                        output_sha256=str(raw["output_sha256"]),
                        array_sha256=str(raw["array_sha256"]),
                    )
                )
                cursor += 1
        target.flush()
        del target
        if cursor != geometry.stream_count:
            raise ProtocolError("SCEPTRE v5 source final array coverage drifted.")
        if path.exists():
            raise ProtocolError("SCEPTRE v5 source final array appeared during publication.")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return tuple(records)


def persist_exact_npy(path: Path, values: np.ndarray) -> None:
    if path.is_symlink():
        raise ProtocolError("SCEPTRE v5 source checkpoint array is a symlink.")
    if path.exists():
        if not path.is_file():
            raise ProtocolError("SCEPTRE v5 source checkpoint array is unsafe.")
        observed = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            observed.shape != values.shape
            or observed.dtype != values.dtype
            or sha256_array(observed) != sha256_array(values)
        ):
            raise ProtocolError(
                "SCEPTRE v5 source checkpoint differs; refusing regeneration."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def persist_exact_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise ProtocolError("SCEPTRE v5 source JSON member is a symlink.")
    if path.exists():
        if not path.is_file() or read_json(path) != dict(payload):
            raise ProtocolError("SCEPTRE v5 source JSON differs; refusing overwrite.")
        return
    atomic_json(path, payload)


def validate_checkpoint_directory(directory: Path, geometry: SourceGeometry) -> None:
    if not directory.exists():
        if directory.is_symlink():
            raise ProtocolError("SCEPTRE v5 source checkpoint root is a dangling symlink.")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("SCEPTRE v5 source checkpoint root is unsafe.")
    expected = {
        f"source_{source}_train_{seed}.{suffix}"
        for source, seed in product(geometry.centers, geometry.training_seeds)
        for suffix in ("json", "npy")
    }
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected:
            raise ProtocolError("SCEPTRE v5 source checkpoint tree has an unknown member.")


__all__ = (
    "load_checkpoint_if_complete",
    "persist_exact_json",
    "persist_exact_npy",
    "publish_checkpoint",
    "publish_source_array",
    "validate_checkpoint_directory",
)
