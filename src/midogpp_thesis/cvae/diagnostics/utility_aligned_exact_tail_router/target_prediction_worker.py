"""Spawn-safe target classifier worker with composition-fit deduplication."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....common.hashing import stable_hash
from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .artifact_io import atomic_json, atomic_npz, sha256_file
from .config import CLASSIFIER
from .contracts import expected_target_action_ids
from .source_cache_contracts import SOURCE_ROWS_PER_CLASS
from .target_prediction_contracts import array_sha256


def target_prediction_task(task: Mapping[str, object]) -> Mapping[str, object]:
    target = str(task["target_center"])
    training_seed = int(task["training_seed"])
    generation_seed = int(task["generation_seed"])
    candidates = tuple(str(value) for value in task["candidate_sources"])
    if (
        task.get("labels_available") is not False
        or task.get("target_support_labels_used") is not False
        or task.get("target_evaluation_used_for_route") is not False
        or task.get("seed_selection_performed") is not False
        or task.get("policy_authorized") is not False
    ):
        raise ProtocolError("Target task escaped its pre-label diagnostic boundary.")
    source_array = np.load(
        Path(str(task["source_array_path"])), mmap_mode="r", allow_pickle=False
    )
    if (
        source_array.shape != (81, 2 * SOURCE_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
        or source_array.dtype != np.float32
    ):
        raise ProtocolError("Target worker source cache geometry drifted.")
    source_index = {
        (
            str(row["source_center"]),
            int(row["training_seed"]),
            int(row["generation_seed"]),
        ): row
        for row in task["source_index_rows"]
    }
    blocks: dict[str, np.ndarray] = {}
    stream_ids: dict[str, str] = {}
    for source in candidates:
        try:
            record = source_index[(source, training_seed, generation_seed)]
        except KeyError as exc:
            raise ProtocolError("Target worker source stream is absent.") from exc
        ordinal = int(record["block_ordinal"])
        blocks[source] = source_array[ordinal]
        stream_ids[source] = str(record["stream_id"])
    evaluation_all = np.load(
        Path(str(task["evaluation_array_path"])), mmap_mode="r", allow_pickle=False
    )
    offset = task["evaluation_offset"]
    if not isinstance(offset, Mapping):
        raise ProtocolError("Target task evaluation offset is malformed.")
    start, stop = int(offset["start"]), int(offset["stop"])
    if (
        evaluation_all.ndim != 2
        or evaluation_all.shape[1] != COMMON_OUTPUT_DIM
        or not 0 <= start < stop <= len(evaluation_all)
    ):
        raise ProtocolError("Target evaluation scratch geometry drifted.")
    evaluation = np.ascontiguousarray(evaluation_all[start:stop], dtype=np.float32)
    spec = _classifier(task["classifier"])
    actions = tuple(task["actions"])
    if tuple(str(action["action_id"]) for action in actions) != expected_target_action_ids(
        target
    ):
        raise ProtocolError("Target worker action order drifted.")
    fitted_by_composition: dict[str, Mapping[str, object]] = {}
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    action_rows: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("Target fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for action in actions:
            counts = _counts(action, candidates=candidates)
            composition_hash = stable_hash(
                {
                    "schema_version": "midogpp_utility_aligned_stage90_target_composition_v1",
                    "counts_by_class": counts,
                    "source_stream_ids": stream_ids,
                    "prefix_rows_per_class": SOURCE_ROWS_PER_CLASS,
                }
            )
            fitted = fitted_by_composition.get(composition_hash)
            aliased = fitted is not None
            if fitted is None:
                train_x, train_y = _compose(blocks, counts)
                result = fit_logistic_classifier(
                    train_x, train_y, evaluation, spec=spec
                )
                pred = np.asarray(result.predictions, dtype=np.uint8)
                all_prob = np.asarray(result.probabilities, dtype=np.float64)
                if (
                    tuple(int(value) for value in result.classes) != (0, 1)
                    or pred.shape != (len(evaluation),)
                    or all_prob.shape != (len(evaluation), 2)
                    or not np.isin(pred, (0, 1)).all()
                    or not np.isfinite(all_prob).all()
                    or not np.allclose(all_prob.sum(axis=1), 1.0, rtol=0.0, atol=1e-7)
                    or not result.converged
                    or result.classifier_config_hash != spec.config_hash
                ):
                    raise ProtocolError("Target classifier fit drifted.")
                fitted = {
                    "predictions": pred,
                    "probabilities": all_prob[:, 1].astype(np.float32, copy=False),
                    "scaler_state_hash": str(result.scaler_state_hash),
                }
                fitted_by_composition[composition_hash] = fitted
            pred = np.asarray(fitted["predictions"], dtype=np.uint8)
            prob = np.asarray(fitted["probabilities"], dtype=np.float32)
            predictions.append(pred)
            probabilities.append(prob)
            action_rows.append(
                {
                    "action_id": str(action["action_id"]),
                    "action_hash": str(action["action_hash"]),
                    "prediction_sha256": array_sha256(pred),
                    "probability_sha256": array_sha256(prob),
                    "composition_sha256": composition_hash,
                    "scaler_state_hash": str(fitted["scaler_state_hash"]),
                    "aliased_fit": aliased,
                }
            )
    array_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    atomic_npz(
        array_path,
        predictions=np.ascontiguousarray(np.stack(predictions), dtype=np.uint8),
        probabilities=np.ascontiguousarray(np.stack(probabilities), dtype=np.float32),
    )
    unhashed = {
        "schema_version": "midogpp_utility_aligned_stage90_target_checkpoint_v1",
        "task_id": str(task["task_id"]),
        "task_hash": str(task["task_hash"]),
        "target_center": target,
        "training_seed": training_seed,
        "generation_seed": generation_seed,
        "evaluation_row_count": len(evaluation),
        "evaluation_row_identity_hash": str(offset["row_identity_hash"]),
        "array_file_sha256": sha256_file(array_path),
        "actions": action_rows,
        "unique_classifier_fit_count": len(fitted_by_composition),
        "labels_used": False,
    }
    payload = {**unhashed, "checkpoint_hash": stable_hash(unhashed)}
    atomic_json(json_path, payload)
    return payload


def _compose(
    blocks: Mapping[str, np.ndarray],
    counts: Mapping[int, Mapping[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label in (0, 1):
        for source, count in counts[label].items():
            if source not in blocks or not 0 < count <= SOURCE_ROWS_PER_CLASS:
                raise ProtocolError("Target composition source/count drifted.")
            start = label * SOURCE_ROWS_PER_CLASS
            arrays.append(np.asarray(blocks[source][start : start + count], dtype=np.float32))
            labels.append(np.full(count, label, dtype=np.uint8))
    embeddings = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    if embeddings.ndim != 2 or embeddings.shape[1] != COMMON_OUTPUT_DIM:
        raise ProtocolError("Target composed embeddings drifted.")
    return embeddings, truth


def _counts(
    action: Mapping[str, object],
    *,
    candidates: Sequence[str],
) -> dict[int, dict[str, int]]:
    raw = action.get("final_counts_by_class")
    if not isinstance(raw, Mapping):
        raise ProtocolError("Target action lacks final counts.")
    result: dict[int, dict[str, int]] = {}
    for label in (0, 1):
        values = raw.get(str(label), raw.get(label))
        if not isinstance(values, Mapping):
            raise ProtocolError("Target action class counts are malformed.")
        counts = {str(source): int(count) for source, count in values.items()}
        if tuple(counts) != tuple(candidates) or sum(counts.values()) not in {1024, 1152}:
            raise ProtocolError("Target action final geometry drifted.")
        result[label] = counts
    return result


def _classifier(raw: object) -> ClassifierSpec:
    if not isinstance(raw, Mapping):
        raise ProtocolError("Target classifier payload is malformed.")
    try:
        spec = ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None if raw["class_weight"] is None else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Target classifier payload is malformed.") from exc
    if spec != CLASSIFIER:
        raise ProtocolError("Target classifier identity drifted.")
    return spec


__all__ = ("target_prediction_task",)
