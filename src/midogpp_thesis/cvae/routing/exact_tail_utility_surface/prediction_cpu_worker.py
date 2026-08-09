"""CPU classifier work for one exact-tail coarse prediction task."""

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
from .config import CLASSIFIER
from .contracts import SOURCE_PREFIX_ROWS_PER_CLASS, action_library_for
from .prediction_checkpoint_store import write_checkpoint
from .prediction_contracts import (
    CoarsePredictionRecord,
    PredictionWorkerInput,
)
from .runtime import CLASSIFIER_THREADS_PER_WORKER
from .scoring import array_sha256
from .source_contracts import SourceBlockRecord


def prediction_worker(item: PredictionWorkerInput) -> CoarsePredictionRecord:
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover - workstation dependency
        raise RuntimeError("Exact-tail fitting requires threadpoolctl.") from exc
    spec = classifier_from_payload(item.classifier_payload)
    eval_embeddings = np.load(
        item.evaluation_array_path, mmap_mode="r", allow_pickle=False
    )
    if not item.support_array_path:
        raise ProtocolError("Exact-tail support memmap path is absent.")
    support_embeddings = np.load(
        item.support_array_path, mmap_mode="r", allow_pickle=False
    )
    if (
        eval_embeddings.ndim != 2
        or eval_embeddings.shape[1] != COMMON_OUTPUT_DIM
        or len(eval_embeddings) == 0
        or not np.isfinite(eval_embeddings).all()
    ):
        raise ProtocolError("Exact-tail evaluation memmap geometry drifted.")
    if (
        support_embeddings.ndim != 2
        or support_embeddings.shape[1] != COMMON_OUTPUT_DIM
        or len(support_embeddings) == 0
        or support_embeddings.dtype != np.float32
        or not np.isfinite(support_embeddings).all()
    ):
        raise ProtocolError("Exact-tail support memmap geometry drifted.")
    prediction_embeddings = np.ascontiguousarray(
        np.concatenate((eval_embeddings, support_embeddings), axis=0),
        dtype=np.float32,
    )
    source_arrays: dict[str, np.ndarray] = {}
    for record in item.source_records:
        path = safe_source_member(Path(item.cache_root), record.relative_path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (2 * SOURCE_PREFIX_ROWS_PER_CLASS, COMMON_OUTPUT_DIM)
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ProtocolError("Exact-tail source memmap geometry drifted.")
        source_arrays[record.source_center] = array
    if tuple(source_arrays) != item.task.candidate_sources:
        raise ProtocolError("Exact-tail task source order drifted.")

    actions = action_library_for(
        outer_target=item.task.outer_target, pseudo_query=item.task.pseudo_query
    )
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    support_probabilities: list[np.ndarray] = []
    compositions: dict[str, str] = {}
    scaler_hashes: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    probability_hashes: dict[str, str] = {}
    support_probability_hashes: dict[str, str] = {}
    with threadpool_limits(limits=CLASSIFIER_THREADS_PER_WORKER):
        for action in actions:
            train_embeddings, train_labels = compose_action(
                source_arrays, action.counts_per_class
            )
            fitted = fit_logistic_classifier(
                train_embeddings,
                train_labels,
                prediction_embeddings,
                spec=spec,
            )
            all_predictions = np.asarray(fitted.predictions)
            all_probabilities = np.asarray(fitted.probabilities, dtype=np.float64)
            if (
                tuple(int(value) for value in fitted.classes) != (0, 1)
                or all_predictions.shape != (len(prediction_embeddings),)
                or all_probabilities.shape != (len(prediction_embeddings), 2)
                or not np.isin(all_predictions, (0, 1)).all()
                or not np.isfinite(all_probabilities).all()
                or not np.allclose(
                    all_probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-7
                )
                or not fitted.converged
                or fitted.classifier_config_hash != spec.config_hash
                or not fitted.scaler_state_hash
            ):
                raise ProtocolError("Exact-tail classifier fit drifted.")
            pred = all_predictions[: len(eval_embeddings)].astype(
                np.uint8, copy=False
            )
            prob = all_probabilities[: len(eval_embeddings), 1].astype(
                np.float32, copy=False
            )
            support_prob = all_probabilities[len(eval_embeddings) :, 1].astype(
                np.float32, copy=False
            )
            prediction_hashes[action.action_id] = array_sha256(pred)
            probability_hashes[action.action_id] = array_sha256(prob)
            support_probability_hashes[action.action_id] = array_sha256(
                support_prob
            )
            predictions.append(pred)
            probabilities.append(prob)
            support_probabilities.append(support_prob)
            compositions[action.action_id] = composition_sha256(
                action.to_payload(), item.source_records
            )
            scaler_hashes[action.action_id] = str(fitted.scaler_state_hash)

    array_payload = np.ascontiguousarray(np.stack(predictions), dtype=np.uint8)
    probability_payload = np.ascontiguousarray(
        np.stack(probabilities), dtype=np.float32
    )
    support_probability_payload = np.ascontiguousarray(
        np.stack(support_probabilities), dtype=np.float32
    )
    return write_checkpoint(
        item,
        classifier_config_hash=spec.config_hash,
        predictions=array_payload,
        probabilities=probability_payload,
        support_probabilities=support_probability_payload,
        action_prediction_sha256=prediction_hashes,
        action_probability_sha256=probability_hashes,
        action_support_probability_sha256=support_probability_hashes,
        action_composition_sha256=compositions,
        action_scaler_state_hash=scaler_hashes,
        evaluation_row_count=len(eval_embeddings),
        support_row_count=len(support_embeddings),
    )


def compose_action(
    sources: Mapping[str, np.ndarray], counts: Mapping[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    class_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for label in (0, 1):
        for source, count in counts.items():
            if source not in sources or count not in {144, 270}:
                raise ProtocolError("Exact-tail composition count drifted.")
            start = label * SOURCE_PREFIX_ROWS_PER_CLASS
            stop = start + int(count)
            class_rows.append(np.asarray(sources[source][start:stop], dtype=np.float32))
            labels.append(np.full(int(count), label, dtype=np.uint8))
    embeddings = np.ascontiguousarray(np.concatenate(class_rows), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != COMMON_OUTPUT_DIM
        or len(embeddings) != 2 * sum(counts.values())
        or not np.isfinite(embeddings).all()
    ):
        raise ProtocolError("Exact-tail composed training geometry drifted.")
    return embeddings, truth


def composition_sha256(
    action_payload: Mapping[str, object], records: Sequence[SourceBlockRecord]
) -> str:
    return canonical_sha256(
        {
            "schema_version": "midogpp_exact_tail_composition_binding_v1",
            "action": dict(action_payload),
            "source_streams": [
                {
                    "source_center": record.source_center,
                    "stream_id": record.stream_id,
                    "file_sha256": record.file_sha256,
                    "output_sha256": record.output_sha256,
                }
                for record in records
            ],
            "class_row_order": "class_0_then_class_1",
            "prefix_reuse": True,
        }
    )


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
        raise ProtocolError("Exact-tail classifier payload is malformed.") from exc
    if spec.to_payload() != CLASSIFIER.to_payload():
        raise ProtocolError("Exact-tail classifier payload drifted.")
    return spec


def safe_source_member(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError("Exact-tail source member path is unsafe.")
    resolved = (root.resolve() / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError("Exact-tail source member escaped cache root.") from exc
    return resolved


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "canonical_sha256",
    "classifier_from_payload",
    "compose_action",
    "composition_sha256",
    "prediction_worker",
    "safe_source_member",
)
