"""Spawn-safe CPU worker for one atomic base-plus-seven-tail task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ....real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    fit_logistic_classifier,
)
from ...generation.contracts import COMMON_OUTPUT_DIM
from ...protocol import ProtocolError
from .development_prediction_contracts import (
    BLAS_THREADS_PER_WORKER,
    CoarseDevelopmentTask,
    ExactTailDevelopmentAction,
    PredictionCheckpointRecord,
    PredictionWorkerInput,
    SourceSlice,
    action_library_for,
)
from .development_prediction_store import array_sha256, write_prediction_checkpoint
from .source_cache_contracts import (
    EXPECTED_SOURCE_STREAM_COUNT,
    SOURCE_ROWS_PER_CLASS,
)


def prediction_worker(item: PredictionWorkerInput) -> PredictionCheckpointRecord:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("Stage-90 exact-tail fitting requires threadpoolctl.") from exc
    if (
        item.task.candidate_sources != tuple(
            source.source_center for source in item.source_slices
        )
        or item.threads_per_fit != BLAS_THREADS_PER_WORKER
    ):
        raise ProtocolError("Stage-90 prediction worker source order drifted.")
    source_cache = np.load(
        Path(item.source_array_path), mmap_mode="r", allow_pickle=False
    )
    if source_cache.shape != (
        EXPECTED_SOURCE_STREAM_COUNT,
        2 * SOURCE_ROWS_PER_CLASS,
        COMMON_OUTPUT_DIM,
    ) or source_cache.dtype != np.float32:
        raise ProtocolError("Stage-90 prediction source memmap drifted.")
    source_arrays = {
        source.source_center: source_cache[source.block_ordinal]
        for source in item.source_slices
    }
    evaluation = np.load(
        Path(item.evaluation_array_path), mmap_mode="r", allow_pickle=False
    )
    if (
        evaluation.ndim != 2
        or evaluation.shape[1] != COMMON_OUTPUT_DIM
        or len(evaluation) != len(item.evaluation_row_ids)
        or evaluation.dtype != np.float32
        or not np.isfinite(evaluation).all()
    ):
        raise ProtocolError("Stage-90 evaluation memmap drifted.")
    classifier = classifier_from_payload(item.classifier_payload)
    actions = action_library_for(
        outer_target=item.task.outer_target, query_center=item.task.query_center
    )
    if tuple(action.action_id for action in actions) != item.task.action_ids:
        raise ProtocolError("Stage-90 prediction action menu drifted.")

    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    prediction_hashes: dict[str, str] = {}
    probability_hashes: dict[str, str] = {}
    composition_hashes: dict[str, str] = {}
    scaler_hashes: dict[str, str] = {}
    with threadpool_limits(limits=item.threads_per_fit):
        for action in actions:
            train_embeddings, train_labels = compose_exact_tail_action(
                source_arrays, action
            )
            fitted = fit_logistic_classifier(
                train_embeddings,
                train_labels,
                evaluation,
                spec=classifier,
            )
            prediction = np.asarray(fitted.predictions, dtype=np.uint8)
            probability_matrix = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                fitted.classes != (0, 1)
                or prediction.shape != (len(evaluation),)
                or probability_matrix.shape != (len(evaluation), 2)
                or not np.isin(prediction, (0, 1)).all()
                or not np.isfinite(probability_matrix).all()
                or not np.allclose(
                    probability_matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-7
                )
                or not fitted.converged
                or fitted.classifier_config_hash != classifier.config_hash
                or not fitted.scaler_state_hash
            ):
                raise ProtocolError("Stage-90 classifier fit drifted.")
            probability = probability_matrix[:, 1].astype(np.float32, copy=False)
            prediction_hashes[action.action_id] = array_sha256(prediction)
            probability_hashes[action.action_id] = array_sha256(probability)
            composition_hashes[action.action_id] = composition_sha256(
                action, item.source_slices
            )
            scaler_hashes[action.action_id] = str(fitted.scaler_state_hash)
            predictions.append(prediction)
            probabilities.append(probability)
    return write_prediction_checkpoint(
        item,
        predictions=np.ascontiguousarray(np.stack(predictions), dtype=np.uint8),
        probabilities=np.ascontiguousarray(np.stack(probabilities), dtype=np.float32),
        classifier_config_hash=classifier.config_hash,
        action_prediction_sha256=prediction_hashes,
        action_probability_sha256=probability_hashes,
        action_composition_sha256=composition_hashes,
        action_scaler_state_hash=scaler_hashes,
    )


def compose_exact_tail_action(
    sources: Mapping[str, np.ndarray], action: ExactTailDevelopmentAction
) -> tuple[np.ndarray, np.ndarray]:
    """Compose 7x144 base rows plus an optional 126-row selected tail."""

    if tuple(sources) != action.source_order:
        raise ProtocolError("Stage-90 exact-tail composition source order drifted.")
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    feature_dim: int | None = None
    for label in (0, 1):
        for source in action.source_order:
            values = np.asarray(sources[source])
            count = int(action.counts_per_class[source])
            if (
                values.ndim != 2
                or values.shape[0] != 2 * SOURCE_ROWS_PER_CLASS
                or count not in {144, 270}
            ):
                raise ProtocolError("Stage-90 source prefix geometry drifted.")
            feature_dim = values.shape[1] if feature_dim is None else feature_dim
            if values.shape[1] != feature_dim:
                raise ProtocolError("Stage-90 source feature dimensions drifted.")
            start = label * SOURCE_ROWS_PER_CLASS
            rows.append(np.asarray(values[start : start + count], dtype=np.float32))
            labels.append(np.full(count, label, dtype=np.uint8))
    embeddings = np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    if (
        embeddings.shape != (2 * action.total_per_class, int(feature_dim or 0))
        or truth.shape != (2 * action.total_per_class,)
        or not np.isfinite(embeddings).all()
    ):
        raise ProtocolError("Stage-90 exact-tail composition drifted.")
    return embeddings, truth


def classifier_from_payload(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        spec = ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=None
            if raw["class_weight"] is None
            else str(raw["class_weight"]),
            random_state=int(raw["random_state"]),
            l1_ratio=None if raw["l1_ratio"] is None else float(raw["l1_ratio"]),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Stage-90 classifier payload is malformed.") from exc
    if spec.to_payload() != dict(raw):
        raise ProtocolError("Stage-90 classifier payload round-trip drifted.")
    return spec


def composition_sha256(
    action: ExactTailDevelopmentAction, sources: Sequence[SourceSlice]
) -> str:
    payload = {
        "schema_version": "midogpp_stage90_utility_aligned_composition_binding_v1",
        "action": action.to_payload(),
        "source_streams": [
            {
                "source_center": source.source_center,
                "block_ordinal": source.block_ordinal,
                "stream_id": source.stream_id,
                "expert_lock_hash": source.expert_lock_hash,
                "output_sha256": source.output_sha256,
            }
            for source in sources
        ],
        "class_row_order": "class_0_then_class_1",
        "prefix_reuse": True,
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


__all__ = (
    "classifier_from_payload",
    "compose_exact_tail_action",
    "composition_sha256",
    "prediction_worker",
)
