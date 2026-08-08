"""Canonical consolidation of exact-tail coarse prediction checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import ExactTailUtilitySurfaceConfig
from .contracts import (
    DevelopmentPartition,
    action_library_for,
    expected_prediction_keys,
    row_identity_hash,
)
from .prediction_checkpoint_store import atomic_json, atomic_save_npz, sha256_file
from .prediction_contracts import (
    PREDICTION_ARRAY_MEMBER,
    PREDICTION_INDEX_MEMBER,
    PREDICTION_INDEX_SCHEMA,
    CoarsePredictionRecord,
    ConsolidatedPredictionArtifacts,
)
from .runtime import coarse_prediction_tasks
from .scoring import array_sha256
from .seals import PredictionCellSeal


def consolidate_prediction_records(
    config: ExactTailUtilitySurfaceConfig,
    partitions: Mapping[str, DevelopmentPartition],
    records: Sequence[CoarsePredictionRecord],
    *,
    root: Path,
) -> ConsolidatedPredictionArtifacts:
    by_task = {record.task.key: record for record in records}
    planned = coarse_prediction_tasks()
    if set(by_task) != {task.key for task in planned}:
        raise ProtocolError("Exact-tail consolidation task coverage drifted.")
    flat_predictions: list[np.ndarray] = []
    flat_probabilities: list[np.ndarray] = []
    offsets = [0]
    cells: list[PredictionCellSeal] = []
    prediction_mapping: dict[tuple[str, str, str, int, int], np.ndarray] = {}
    index_cells: list[dict[str, object]] = []
    for task in planned:
        record = by_task[task.key]
        with np.load(record.checkpoint_relative_path, allow_pickle=False) as payload:
            checkpoint_predictions = np.asarray(payload["predictions"])
            checkpoint_probabilities = np.asarray(payload["probabilities"])
        actions = action_library_for(
            outer_target=task.outer_target, pseudo_query=task.pseudo_query
        )
        evaluation_hash = row_identity_hash(
            partitions[task.pseudo_query].evaluation_rows
        )
        for index, action in enumerate(actions):
            pred = np.ascontiguousarray(checkpoint_predictions[index], dtype=np.uint8)
            prob = np.ascontiguousarray(
                checkpoint_probabilities[index], dtype=np.float32
            )
            key = (
                task.outer_target,
                task.pseudo_query,
                action.action_id,
                task.training_seed,
                task.generation_seed,
            )
            start = offsets[-1]
            stop = start + len(pred)
            offsets.append(stop)
            flat_predictions.append(pred)
            flat_probabilities.append(prob)
            prediction_mapping[key] = pred
            cell = PredictionCellSeal(
                outer_target=task.outer_target,
                pseudo_query=task.pseudo_query,
                action_id=action.action_id,
                training_seed=task.training_seed,
                generation_seed=task.generation_seed,
                action_hash=action.action_hash,
                evaluation_row_identity_hash=evaluation_hash,
                prediction_sha256=array_sha256(pred),
                probability_sha256=array_sha256(prob),
                composition_sha256=record.action_composition_sha256[
                    action.action_id
                ],
                classifier_config_hash=config.classifier.config_hash,
            )
            cells.append(cell)
            index_cells.append(
                {
                    "cell_ordinal": len(index_cells),
                    **cell.to_payload(),
                    "array_start": start,
                    "array_stop": stop,
                    "scaler_state_hash": record.action_scaler_state_hash[
                        action.action_id
                    ],
                    "labels_stored": False,
                }
            )
    if tuple(cell.key for cell in cells) != expected_prediction_keys():
        raise ProtocolError("Exact-tail consolidated cell order drifted.")
    arrays_path = root / PREDICTION_ARRAY_MEMBER
    atomic_save_npz(
        arrays_path,
        predictions=np.concatenate(flat_predictions).astype(np.uint8, copy=False),
        probabilities=np.concatenate(flat_probabilities).astype(np.float32, copy=False),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    arrays_sha = sha256_file(arrays_path)
    index_unhashed = {
        "schema_version": PREDICTION_INDEX_SCHEMA,
        "array_member": PREDICTION_ARRAY_MEMBER,
        "array_file_sha256": arrays_sha,
        "allowed_array_keys": ["predictions", "probabilities", "offsets"],
        "prediction_dtype": "uint8",
        "probability_dtype": "float32",
        "offset_dtype": "int64",
        "cell_count": len(index_cells),
        "cells": index_cells,
        "labels_stored": False,
        "all_predictions_materialized_before_development_labels": True,
    }
    index_payload = {
        **index_unhashed,
        "prediction_index_hash": stable_hash(index_unhashed),
    }
    index_path = root / PREDICTION_INDEX_MEMBER
    atomic_json(index_path, index_payload)
    return ConsolidatedPredictionArtifacts(
        predictions_by_key=prediction_mapping,
        cells=tuple(cells),
        prediction_index_path=index_path,
        prediction_arrays_path=arrays_path,
        prediction_index_sha256=sha256_file(index_path),
        prediction_arrays_sha256=arrays_sha,
    )


__all__ = ("consolidate_prediction_records",)
