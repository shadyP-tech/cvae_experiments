"""Scratch and plain worker-task planning for fixed-bank A1 predictions."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ...common.hashing import stable_hash
from ..expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
    GENERATION_SEEDS,
    TRAINING_SEEDS,
)
from ..generation.contracts import COMMON_OUTPUT_DIM
from ..protocol import ProtocolError
from .artifact_io import atomic_json, atomic_npy, read_json, sha256_array, sha256_file
from .fixed_bank_a1_prediction_contracts import (
    CHECKPOINT_DIRECTORY,
    EXPECTED_TASK_COUNT,
    PredictionConfig,
)
from .frozen_source_streams import FrozenSourceStreamCache


def write_target_scratch(
    root: Path,
    frame: object,
    partition_hash: str,
    binding: str,
) -> Mapping[str, object]:
    directory = root / CHECKPOINT_DIRECTORY
    _plain_directory(directory)
    manifest_path = directory / "target_scratch.json"
    array_path = directory / "target_embeddings.npy"
    if manifest_path.is_symlink() or array_path.is_symlink():
        raise ProtocolError("Fixed-bank A1 target scratch member is a symlink.")
    if manifest_path.is_file() or array_path.is_file():
        if manifest_path.is_file() and not array_path.is_file():
            raise ProtocolError("Fixed-bank A1 target scratch is partial.")
        if manifest_path.is_file():
            payload = read_json(manifest_path)
            validate_target_scratch(payload, partition_hash, binding)
            return payload
    rows: list[object] = []
    offsets: dict[str, dict[str, object]] = {}
    row_ids: dict[str, list[str]] = {}
    case_ids: dict[str, list[str]] = {}
    cursor = 0
    for target in CENTERS:
        target_rows = tuple(getattr(frame, "rows_by_center")[target])
        identifiers = [_row_identity(row) for row in target_rows]
        cases = [str(getattr(row, "case_id")) for row in target_rows]
        offsets[target] = {
            "start": cursor,
            "stop": cursor + len(target_rows),
            "row_identity_hash": stable_hash(identifiers),
        }
        rows.extend(target_rows)
        row_ids[target] = identifiers
        case_ids[target] = cases
        cursor += len(target_rows)
    embeddings = np.ascontiguousarray(
        getattr(frame, "embeddings_for")(rows), dtype=np.float32
    )
    if (
        embeddings.shape != (cursor, COMMON_OUTPUT_DIM)
        or not np.isfinite(embeddings).all()
    ):
        raise ProtocolError("Fixed-bank A1 target scratch geometry drifted.")
    _persist_or_validate_target_array(array_path, embeddings)
    for target in CENTERS:
        offset = offsets[target]
        offset["target_slice_sha256"] = sha256_array(
            embeddings[int(offset["start"]) : int(offset["stop"])]
        )
    unhashed = {
        "schema_version": "fixed_bank_a1_target_scratch_v1",
        "array_path": str(array_path.resolve()),
        "array_sha256": sha256_file(array_path),
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "partition_hash": partition_hash,
        "target_cache_binding_hash": binding,
        "offsets": offsets,
        "row_ids_by_center": row_ids,
        "case_ids_by_center": case_ids,
        "labels_stored": False,
    }
    payload = {**unhashed, "scratch_hash": stable_hash(unhashed)}
    atomic_json(manifest_path, payload)
    return payload


def _persist_or_validate_target_array(path: Path, expected: np.ndarray) -> None:
    """Preserve an exact array-only crash predecessor; reject any drift."""

    if path.is_symlink():
        raise ProtocolError("Fixed-bank A1 target scratch array is a symlink.")
    if path.exists():
        if not path.is_file():
            raise ProtocolError("Fixed-bank A1 target scratch array is not a file.")
        try:
            observed = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError(
                "Existing fixed-bank A1 target scratch array is unreadable; refusing repair."
            ) from exc
        if (
            observed.dtype != expected.dtype
            or observed.shape != expected.shape
            or sha256_array(observed) != sha256_array(expected)
        ):
            raise ProtocolError(
                "Existing fixed-bank A1 target scratch array differs; refusing repair."
            )
        return
    atomic_npy(path, expected)


def validate_target_scratch(
    payload: Mapping[str, object], partition_hash: str, binding: str
) -> None:
    path = Path(str(payload.get("array_path", "")))
    if (
        not path.is_file()
        or path.is_symlink()
        or payload.get("scratch_hash")
        != stable_hash(
            {key: value for key, value in payload.items() if key != "scratch_hash"}
        )
        or payload.get("partition_hash") != partition_hash
        or payload.get("target_cache_binding_hash") != binding
        or payload.get("array_sha256") != sha256_file(path)
        or payload.get("labels_stored") is not False
    ):
        raise ProtocolError("Fixed-bank A1 target scratch validation failed.")


def build_prediction_tasks(
    config: PredictionConfig,
    source: FrozenSourceStreamCache,
    scratch: Mapping[str, object],
    library: Mapping[str, Sequence[Mapping[str, object]]],
    library_hash: str,
    partition_hash: str,
    root: Path,
) -> tuple[dict[str, object], ...]:
    classifier = getattr(config, "classifier")
    classifier_payload = (
        classifier.to_payload()
        if hasattr(classifier, "to_payload")
        else dict(classifier)
    )
    records = [record.to_payload() for record in source.records]
    source_array_sha256 = sha256_file(source.source_array_path)
    task_root = root / CHECKPOINT_DIRECTORY / "tasks"
    _plain_directory(task_root)
    tasks: list[dict[str, object]] = []
    for target, training, generation in product(
        CENTERS, TRAINING_SEEDS, GENERATION_SEEDS
    ):
        offset = scratch["offsets"][target]
        task_id = f"target_{target}_train_{training}_generation_{generation}"
        unhashed = {
            "schema_version": "fixed_bank_a1_prediction_task_v1",
            "task_id": task_id,
            "config_contract_hash": config.contract_hash,
            "source_stream_lock_hash": source.lock_hash,
            "partition_hash": partition_hash,
            "action_library_hash": library_hash,
            "target_center": target,
            "training_seed": training,
            "generation_seed": generation,
            "candidate_sources": [value for value in CENTERS if value != target],
            "source_array_path": str(source.source_array_path.resolve()),
            "source_array_sha256": source_array_sha256,
            "source_index_rows": records,
            "source_index_rows_hash": stable_hash(records),
            "target_array_path": str(scratch["array_path"]),
            "target_array_sha256": str(scratch["array_sha256"]),
            "target_start": int(offset["start"]),
            "target_stop": int(offset["stop"]),
            "target_row_identity_hash": str(offset["row_identity_hash"]),
            "target_slice_sha256": str(offset["target_slice_sha256"]),
            "actions": [dict(value) for value in library[target]],
            "classifier": classifier_payload,
            "threads_per_fit": int(
                config.runtime["classifier_threads_per_worker"]
            ),
            "labels_available": False,
            "target_expert_available": False,
        }
        tasks.append(
            {
                **unhashed,
                "task_hash": stable_hash(unhashed),
                "checkpoint_json_path": str(task_root / f"{task_id}.json"),
                "checkpoint_npz_path": str(task_root / f"{task_id}.npz"),
            }
        )
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ProtocolError("Fixed-bank A1 task coverage drifted.")
    return tuple(tasks)


def _plain_directory(path: Path) -> None:
    if path.is_symlink():
        raise ProtocolError(f"Fixed-bank A1 scratch directory is a symlink: {path}.")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise ProtocolError(f"Fixed-bank A1 scratch directory is unsafe: {path}.")


def _row_identity(row: object) -> str:
    value = getattr(row, "evaluation_row_id", None)
    if value is None:
        value = getattr(row, "sample_id", None)
    if value is None or not str(value):
        raise ProtocolError("Fixed-bank A1 target row identity is absent.")
    return str(value)


__all__ = (
    "build_prediction_tasks",
    "validate_target_scratch",
    "write_target_scratch",
)
