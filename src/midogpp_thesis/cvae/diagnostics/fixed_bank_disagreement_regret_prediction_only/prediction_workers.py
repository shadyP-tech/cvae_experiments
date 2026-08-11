"""Pure task workers and classifier math for prediction-only execution."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import warnings

import numpy as np

from ....real_features.classifier_reference.classifiers import ClassifierSpec
from ...protocol import ProtocolError
from ...runtime.artifact_io import (
    atomic_json,
    atomic_npz,
    sha256_array,
    sha256_file,
)
from ...runtime.frozen_source_streams import EXPECTED_STREAM_COUNT, source_block_sha256
from .constants import (
    FEATURE_DIM,
    PHYSICAL_ACTION_COUNT_PER_TARGET,
    PREDICTION_BATCH_ROWS,
    SOURCE_ROWS_PER_CLASS,
)
from .hashing import canonical_hash
from .prediction_contracts import classifier_parameter_sha256
from .prediction_plans import validate_source_task, validate_test_task


def source_prediction_task(task: Mapping[str, object]) -> None:
    validate_source_task(task)
    candidates = tuple(str(value) for value in task["candidate_sources"])
    blocks, source_values = load_source_task_arrays(task, candidates=candidates)
    classifier = classifier_from_payload(task["classifier"])
    actions = task["actions"]
    probability_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] = []
    intercept_rows: list[float] = []
    metadata_rows: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Prediction-only fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for raw in actions:
            if not isinstance(raw, Mapping):
                raise ProtocolError("Prediction-only source action is malformed.")
            train_x, train_y, weights, composition_hash = compose_action(
                blocks, raw, candidates
            )
            fitted = fit_action_classifier(
                train_x,
                train_y,
                spec=classifier,
                sample_weight=(None if np.all(weights == 1.0) else weights),
            )
            probabilities = predict_probability_batched(
                source_values,
                fitted["mean"],
                fitted["scale"],
                fitted["coefficient"],
                float(fitted["intercept"]),
                batch_rows=int(task["prediction_batch_rows"]),
            )
            parameter_hash = classifier_parameter_sha256(
                fitted["mean"],
                fitted["scale"],
                fitted["coefficient"],
                float(fitted["intercept"]),
            )
            probability_hash = sha256_array(probabilities)
            prediction_hash = sha256_array(
                (probabilities >= np.float32(0.5)).astype(np.uint8)
            )
            fit_unhashed = {
                "schema_version": "midogpp_prediction_only_action_classifier_fit_v1",
                "task_hash": task["task_hash"],
                "action_id": raw["action_id"],
                "action_hash": raw["action_hash"],
                "composition_hash": composition_hash,
                "classifier_config_hash": classifier.config_hash,
                "scaler_state_hash": fitted["scaler_state_hash"],
                "parameter_sha256": parameter_hash,
                "n_iter": list(fitted["n_iter"]),
                "converged": fitted["converged"],
                "source_probability_sha256": probability_hash,
                "source_prediction_sha256": prediction_hash,
                "source_labels_available": False,
                "test_cache_admitted": False,
                "sample_weight_scope": "logistic_regression_fit_only",
                "scaler_fit_used_sample_weight": False,
            }
            probability_rows.append(probabilities)
            mean_rows.append(np.asarray(fitted["mean"], dtype=np.float64))
            scale_rows.append(np.asarray(fitted["scale"], dtype=np.float64))
            coefficient_rows.append(
                np.asarray(fitted["coefficient"], dtype=np.float64)
            )
            intercept_rows.append(float(fitted["intercept"]))
            metadata_rows.append(
                {
                    "action_id": str(raw["action_id"]),
                    "action_hash": str(raw["action_hash"]),
                    "composition_hash": composition_hash,
                    "scaler_state_hash": str(fitted["scaler_state_hash"]),
                    "parameter_sha256": parameter_hash,
                    "fit_provenance_hash": canonical_hash(fit_unhashed),
                    "classifier_config_hash": classifier.config_hash,
                    "n_iter": list(fitted["n_iter"]),
                    "converged": bool(fitted["converged"]),
                    "probability_sha256": probability_hash,
                    "predictions_sha256": prediction_hash,
                }
            )
    arrays = {
        "source_probabilities": np.ascontiguousarray(
            np.stack(probability_rows), dtype=np.float32
        ),
        "scaler_mean": np.ascontiguousarray(np.stack(mean_rows), dtype=np.float64),
        "scaler_scale": np.ascontiguousarray(np.stack(scale_rows), dtype=np.float64),
        "coefficients": np.ascontiguousarray(
            np.stack(coefficient_rows), dtype=np.float64
        ),
        "intercepts": np.ascontiguousarray(intercept_rows, dtype=np.float64),
    }
    npz_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    atomic_npz(npz_path, **arrays)
    checkpoint_unhashed = {
        "schema_version": "midogpp_prediction_only_source_fit_checkpoint_v1",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "target_center": task["target_center"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "source_row_identity_hash": task["source_row_identity_hash"],
        "array_sha256": sha256_file(npz_path),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
        "actions": metadata_rows,
        "physical_fit_count": PHYSICAL_ACTION_COUNT_PER_TARGET,
        "source_labels_available": False,
        "test_cache_admitted": False,
    }
    atomic_json(
        json_path,
        {
            **checkpoint_unhashed,
            "checkpoint_hash": canonical_hash(checkpoint_unhashed),
        },
    )


def test_prediction_task(task: Mapping[str, object]) -> None:
    validate_test_task(task)
    values = np.load(
        Path(str(task["test_array_path"])), mmap_mode="r", allow_pickle=False
    )
    start, stop = int(task["test_start"]), int(task["test_stop"])
    evaluation = values[start:stop]
    if (
        evaluation.shape != (stop - start, FEATURE_DIM)
        or evaluation.dtype != np.float32
        or sha256_array(evaluation) != task["test_slice_sha256"]
    ):
        raise ProtocolError("Prediction-only test task slice drifted.")
    mean = np.load(
        Path(str(task["scaler_mean_path"])), mmap_mode="r", allow_pickle=False
    )
    scale = np.load(
        Path(str(task["scaler_scale_path"])), mmap_mode="r", allow_pickle=False
    )
    coefficient = np.load(
        Path(str(task["coefficient_path"])), mmap_mode="r", allow_pickle=False
    )
    intercept = np.load(
        Path(str(task["intercept_path"])), mmap_mode="r", allow_pickle=False
    )
    probabilities: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    for raw, ordinal in zip(
        task["actions"], task["classifier_cell_ordinals"], strict=True
    ):
        index = int(ordinal)
        parameter_hash = classifier_parameter_sha256(
            mean[index], scale[index], coefficient[index], float(intercept[index])
        )
        positive = predict_probability_batched(
            evaluation,
            mean[index],
            scale[index],
            coefficient[index],
            float(intercept[index]),
            batch_rows=int(task["prediction_batch_rows"]),
        )
        probabilities.append(positive)
        metadata.append(
            {
                "action_id": str(raw["action_id"]),
                "action_hash": str(raw["action_hash"]),
                "classifier_cell_ordinal": index,
                "classifier_parameter_sha256": parameter_hash,
                "probability_sha256": sha256_array(positive),
                "predictions_sha256": sha256_array(
                    (positive >= np.float32(0.5)).astype(np.uint8)
                ),
            }
        )
    matrix = np.ascontiguousarray(np.stack(probabilities), dtype=np.float32)
    npz_path = Path(str(task["checkpoint_npz_path"]))
    json_path = Path(str(task["checkpoint_json_path"]))
    atomic_npz(npz_path, test_probabilities=matrix)
    unhashed = {
        "schema_version": "midogpp_prediction_only_test_inference_checkpoint_v1",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "target_center": task["target_center"],
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "test_row_identity_hash": task["test_row_identity_hash"],
        "array_sha256": sha256_file(npz_path),
        "array_shape": list(matrix.shape),
        "array_dtype": str(matrix.dtype),
        "actions": metadata,
        "classifier_fit_count": 0,
        "labels_available": False,
        "target_scoring_permitted": False,
    }
    atomic_json(json_path, {**unhashed, "checkpoint_hash": canonical_hash(unhashed)})


def load_source_task_arrays(
    task: Mapping[str, object], *, candidates: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    raw_index = task.get("generated_index_rows")
    if not isinstance(raw_index, list) or task.get(
        "generated_index_rows_hash"
    ) != canonical_hash(raw_index):
        raise ProtocolError("Prediction-only generated source index drifted.")
    index: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for raw in raw_index:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Prediction-only generated source row is malformed.")
        key = (
            str(raw.get("source_center", "")),
            int(raw.get("training_seed", -1)),
            int(raw.get("generation_seed", -1)),
        )
        index[key] = raw
    if len(index) != EXPECTED_STREAM_COUNT:
        raise ProtocolError("Prediction-only generated source coverage drifted.")
    generated = np.load(
        Path(str(task["generated_array_path"])), mmap_mode="r", allow_pickle=False
    )
    if generated.shape != (
        EXPECTED_STREAM_COUNT,
        2 * SOURCE_ROWS_PER_CLASS,
        FEATURE_DIM,
    ):
        raise ProtocolError("Prediction-only generated source geometry drifted.")
    training, generation = int(task["training_seed"]), int(task["generation_seed"])
    blocks: dict[str, np.ndarray] = {}
    for source in candidates:
        record = index[(source, training, generation)]
        block = generated[int(record["block_ordinal"])]
        if source_block_sha256(block) != record.get("output_sha256"):
            raise ProtocolError("Prediction-only generated source bytes drifted.")
        blocks[source] = block
    source_values = np.load(
        Path(str(task["source_array_path"])), mmap_mode="r", allow_pickle=False
    )
    if (
        list(source_values.shape) != list(task["source_array_shape"])
        or source_values.dtype != np.float32
        or sha256_file(Path(str(task["source_array_path"])))
        != task["source_array_file_sha256"]
    ):
        raise ProtocolError("Prediction-only source evaluation scratch drifted.")
    return blocks, source_values


def compose_action(
    blocks: Mapping[str, np.ndarray],
    action: Mapping[str, object],
    candidates: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    raw_counts = action.get("counts_by_class")
    raw_weights = action.get("sample_weight_by_source")
    if not isinstance(raw_counts, Mapping) or not isinstance(raw_weights, Mapping):
        raise ProtocolError("Prediction-only action composition is incomplete.")
    weights_by_source = {
        str(source): float(value) for source, value in raw_weights.items()
    }
    if tuple(weights_by_source) != candidates or not all(
        np.isfinite(value) and value > 0.0 for value in weights_by_source.values()
    ):
        raise ProtocolError("Prediction-only action weights drifted.")
    arrays: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    canonical_counts: dict[str, dict[str, int]] = {}
    for label in (0, 1):
        raw = raw_counts.get(str(label), raw_counts.get(label))
        if not isinstance(raw, Mapping):
            raise ProtocolError("Prediction-only action class counts are absent.")
        counts = {str(source): int(value) for source, value in raw.items()}
        if tuple(counts) != candidates:
            raise ProtocolError("Prediction-only action source order drifted.")
        canonical_counts[str(label)] = counts
        for source, count in counts.items():
            if count <= 0 or count > SOURCE_ROWS_PER_CLASS:
                raise ProtocolError("Prediction-only source prefix exceeds capacity.")
            start = label * SOURCE_ROWS_PER_CLASS
            arrays.append(
                np.asarray(blocks[source][start : start + count], dtype=np.float32)
            )
            labels.append(np.full(count, label, dtype=np.uint8))
            weights.append(
                np.full(count, weights_by_source[source], dtype=np.float64)
            )
    embeddings = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
    truth = np.ascontiguousarray(np.concatenate(labels), dtype=np.uint8)
    sample_weight = np.ascontiguousarray(np.concatenate(weights), dtype=np.float64)
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != FEATURE_DIM
        or not np.isfinite(embeddings).all()
        or not np.isfinite(sample_weight).all()
    ):
        raise ProtocolError("Prediction-only composed fit surface drifted.")
    composition = {
        "counts_by_class": canonical_counts,
        "sample_weight_by_source": weights_by_source,
        "action_hash": action["action_hash"],
        "sample_weight_scope": "logistic_regression_fit_only",
        "scaler_fit_used_sample_weight": False,
    }
    return embeddings, truth, sample_weight, canonical_hash(composition)


def fit_action_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    spec: ClassifierSpec,
    sample_weight: np.ndarray | None,
) -> Mapping[str, object]:
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Prediction-only fitting requires scikit-learn.") from exc
    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != FEATURE_DIM or set(y.tolist()) != {0, 1}:
        raise ProtocolError("Prediction-only classifier fit data drifted.")
    scaler = StandardScaler(copy=True)
    scaled = scaler.fit_transform(x)
    classifier = LogisticRegression(**spec.to_sklearn_kwargs())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if sample_weight is None:
            classifier.fit(scaled, y)
        else:
            classifier.fit(
                scaled, y, sample_weight=np.asarray(sample_weight, dtype=np.float64)
            )
    convergence = [
        warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
    ]
    classes = tuple(int(value) for value in classifier.classes_.tolist())
    n_iter = tuple(int(value) for value in classifier.n_iter_.tolist())
    converged = not convergence and all(value < spec.max_iter for value in n_iter)
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    coefficient = np.asarray(classifier.coef_[0], dtype=np.float64)
    intercept = float(classifier.intercept_[0])
    scaler_state_hash = canonical_hash(
        {
            "mean_sha256": sha256_array(mean),
            "var_sha256": sha256_array(np.asarray(scaler.var_, dtype=np.float64)),
            "scale_sha256": sha256_array(scale),
            "n_features_in": int(scaler.n_features_in_),
            "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).item()),
        }
    )
    if (
        classes != (0, 1)
        or not converged
        or mean.shape != (FEATURE_DIM,)
        or scale.shape != (FEATURE_DIM,)
        or coefficient.shape != (FEATURE_DIM,)
        or not all(np.isfinite(value).all() for value in (mean, scale, coefficient))
        or not np.isfinite(intercept)
        or np.any(scale <= 0.0)
    ):
        raise ProtocolError("Prediction-only classifier fit drifted.")
    return MappingProxyType(
        {
            "mean": mean,
            "scale": scale,
            "coefficient": coefficient,
            "intercept": intercept,
            "scaler_state_hash": scaler_state_hash,
            "n_iter": n_iter,
            "converged": True,
        }
    )


def predict_probability_batched(
    embeddings: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    *,
    batch_rows: int,
) -> np.ndarray:
    values = np.asarray(embeddings)
    if (
        values.ndim != 2
        or values.shape[1] != FEATURE_DIM
        or batch_rows != PREDICTION_BATCH_ROWS
    ):
        raise ProtocolError("Prediction-only inference batch geometry drifted.")
    output = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), batch_rows):
        stop = min(len(values), start + batch_rows)
        chunk = np.asarray(values[start:stop], dtype=np.float64)
        chunk -= np.asarray(mean, dtype=np.float64)
        chunk /= np.asarray(scale, dtype=np.float64)
        logits = chunk @ np.asarray(coefficient, dtype=np.float64) + float(intercept)
        probabilities = np.empty_like(logits, dtype=np.float64)
        positive = logits >= 0.0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_value = np.exp(logits[~positive])
        probabilities[~positive] = exp_value / (1.0 + exp_value)
        output[start:stop] = probabilities.astype(np.float32)
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise ProtocolError(
            "Prediction-only frozen classifier emitted invalid probabilities."
        )
    return output


def classifier_from_payload(raw: Mapping[str, object]) -> ClassifierSpec:
    try:
        return ClassifierSpec(
            family=str(raw["family"]),
            C=float(raw["C"]),
            penalty=str(raw["penalty"]),
            solver=str(raw["solver"]),
            max_iter=int(raw["max_iter"]),
            class_weight=(
                None if raw["class_weight"] is None else str(raw["class_weight"])
            ),
            random_state=int(raw["random_state"]),
            l1_ratio=(None if raw["l1_ratio"] is None else float(raw["l1_ratio"])),
            threshold_policy=str(raw["threshold_policy"]),
            scaler_fit=str(raw["scaler_fit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("Prediction-only classifier payload is malformed.") from exc


__all__ = (
    "classifier_from_payload",
    "compose_action",
    "fit_action_classifier",
    "load_source_task_arrays",
    "predict_probability_batched",
    "source_prediction_task",
    "test_prediction_task",
)
