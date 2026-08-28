"""Plain-DTO planning for the exact 324 held-pair prediction tasks."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np

from ....protocol import ProtocolError
from ....runtime.artifact_io import (
    atomic_json,
    atomic_npy,
    read_json,
    sha256_array,
    sha256_file,
)
from ....runtime.frozen_source_streams import FrozenSourceStreamCache
from ..hashing import canonical_hash
from ..identity import CENTERS
from .held_actions import (
    GENERATION_SEEDS,
    TRAINING_SEEDS,
    actions_for_held_pair,
    canonical_held_action_library,
    held_candidate_sources,
)
from .resume import (
    fsync_directory,
    fsync_file,
    prepare_exact_checkpoint_directory,
    validate_exact_directory_tree,
)
from .runtime import SourceProductionRuntimeConfig
from .source_frame import LabelFreeSourceFrame


def write_label_free_source_scratch(
    root: Path,
    frame: LabelFreeSourceFrame,
) -> Mapping[str, object]:
    if not isinstance(frame, LabelFreeSourceFrame):
        raise ProtocolError("OE-PPUR v3 source scratch requires a typed frame.")
    path = Path(root)
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise ProtocolError("OE-PPUR v3 source evaluation scratch is unsafe.")
    if path.exists():
        validate_exact_directory_tree(
            path,
            expected_directories=("arrays", "manifests"),
            expected_files=(
                "arrays/source_eval_by_center.npy",
                "manifests/source_eval_index.json",
            ),
        )
    else:
        path.mkdir(parents=True, mode=0o750, exist_ok=False)
        fsync_directory(path.parent)
    array_path = path / "arrays/source_eval_by_center.npy"
    blocks = [frame.embeddings_for_center(center) for center in CENTERS]
    matrix = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float32)
    if array_path.exists():
        if array_path.is_symlink() or not array_path.is_file():
            raise ProtocolError("OE-PPUR v3 source evaluation array is unsafe.")
        try:
            observed_matrix = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ProtocolError(
                "OE-PPUR v3 source evaluation array is unreadable."
            ) from exc
        if (
            observed_matrix.shape != matrix.shape
            or observed_matrix.dtype != np.float32
            or sha256_array(observed_matrix) != sha256_array(matrix)
        ):
            raise ProtocolError("OE-PPUR v3 resumed source evaluation array drifted.")
    else:
        atomic_npy(array_path, matrix)
    fsync_file(array_path)
    fsync_directory(array_path.parent)
    cursor = 0
    views = []
    for center, block in zip(CENTERS, blocks, strict=True):
        rows = frame.rows_by_center[center]
        stop = cursor + len(rows)
        values = matrix[cursor:stop]
        views.append(
            {
                "center": center,
                "start": cursor,
                "stop": stop,
                "source_row_ids": [row.source_row_id for row in rows],
                "source_cache_row_indices": [row.source_cache_row_index for row in rows],
                "row_identity_hash": canonical_hash(tuple(row.source_row_id for row in rows)),
                "slice_sha256": sha256_array(values),
            }
        )
        cursor = stop
    payload = {
        "schema_version": "oe_ppur_v3_source_evaluation_scratch_v1",
        "source_frame_hash": frame.frame_hash,
        "array_path": str(array_path),
        "array_sha256": sha256_file(array_path),
        "array_shape": list(matrix.shape),
        "array_dtype": str(matrix.dtype),
        "views": views,
        "labels_present": False,
        "target_rows_present": False,
    }
    payload["scratch_hash"] = canonical_hash(payload)
    index_path = path / "manifests/source_eval_index.json"
    if index_path.exists():
        if index_path.is_symlink() or not index_path.is_file() or read_json(index_path) != payload:
            raise ProtocolError("OE-PPUR v3 resumed source evaluation index drifted.")
    else:
        atomic_json(index_path, payload)
    fsync_file(index_path)
    fsync_directory(index_path.parent)
    fsync_directory(path)
    return payload


def build_held_prediction_tasks(
    config: SourceProductionRuntimeConfig,
    source: FrozenSourceStreamCache,
    evaluation_scratch: Mapping[str, object],
    *,
    checkpoint_root: Path,
) -> tuple[dict[str, object], ...]:
    if (
        not isinstance(config, SourceProductionRuntimeConfig)
        or type(source) is not FrozenSourceStreamCache
        or evaluation_scratch.get("labels_present") is not False
        or evaluation_scratch.get("target_rows_present") is not False
        or evaluation_scratch.get("source_frame_hash") is None
    ):
        raise ProtocolError("OE-PPUR v3 held task planner input drifted.")
    root = Path(checkpoint_root)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ProtocolError("OE-PPUR v3 held prediction checkpoint root is unsafe.")
    views = evaluation_scratch.get("views")
    if not isinstance(views, list) or tuple(str(row.get("center")) for row in views if isinstance(row, Mapping)) != CENTERS:
        raise ProtocolError("OE-PPUR v3 source evaluation views drifted.")
    view_by_center = {str(row["center"]): dict(row) for row in views if isinstance(row, Mapping)}
    record_by_key = {record.key: record for record in source.records}
    library = canonical_held_action_library()
    tasks: list[dict[str, object]] = []
    ordinal = 0
    for first_index, first in enumerate(CENTERS):
        for second in CENTERS[first_index + 1 :]:
            pair = (first, second)
            candidates = held_candidate_sources(pair)
            actions = [action.to_payload() for action in actions_for_held_pair(*pair)]
            for training_seed, generation_seed in product(TRAINING_SEEDS, GENERATION_SEEDS):
                records = [
                    record_by_key[(source_center, training_seed, generation_seed)].to_payload()
                    for source_center in candidates
                ]
                task_id = f"held_{first}_{second}_train_{training_seed}_gen_{generation_seed}"
                body = {
                    "schema_version": "oe_ppur_v3_held_prediction_task_v1",
                    "task_ordinal": ordinal,
                    "task_id": task_id,
                    "excluded_centers": list(pair),
                    "training_seed": training_seed,
                    "generation_seed": generation_seed,
                    "candidate_sources": list(candidates),
                    "actions": actions,
                    "held_action_library_sha256": library.library_hash,
                    "held_mass_policy_receipt_sha256": library.mass_policy.receipt_hash,
                    "classifier": config.classifier.to_payload(),
                    "threads_per_fit": 1,
                    "source_array_path": str(source.source_array_path),
                    "source_array_sha256": str(source.lock_payload["source_array_sha256"]),
                    "source_stream_lock_hash": source.lock_hash,
                    "source_records": records,
                    "source_records_hash": canonical_hash(records),
                    "evaluation_array_path": str(evaluation_scratch["array_path"]),
                    "evaluation_array_sha256": str(evaluation_scratch["array_sha256"]),
                    "source_frame_hash": str(evaluation_scratch["source_frame_hash"]),
                    "evaluation_views": [view_by_center[first], view_by_center[second]],
                    "labels_available": False,
                    "target_rows_present": False,
                    "scaler_fit_used_sample_weight": False,
                    "sample_weight_scope": "logistic_regression_fit_only",
                    "checkpoint_npz_path": str(root / f"{task_id}.npz"),
                    "checkpoint_json_path": str(root / f"{task_id}.json"),
                }
                body["task_hash"] = canonical_hash(
                    {key: value for key, value in body.items() if not key.startswith("checkpoint_")}
                )
                tasks.append(body)
                ordinal += 1
    if len(tasks) != 324 or tuple(task["task_ordinal"] for task in tasks) != tuple(range(324)):
        raise ProtocolError("OE-PPUR v3 held task inventory drifted.")
    prepare_exact_checkpoint_directory(
        root,
        expected_members=tuple(
            Path(str(task[role])).name
            for task in tasks
            for role in ("checkpoint_npz_path", "checkpoint_json_path")
        ),
    )
    return tuple(tasks)


__all__ = ("build_held_prediction_tasks", "write_label_free_source_scratch")
