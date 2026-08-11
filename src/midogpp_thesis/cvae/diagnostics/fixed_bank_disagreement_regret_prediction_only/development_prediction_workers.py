"""Worker math for unordered-pair strict source-OOF fits."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...runtime.artifact_io import atomic_json, atomic_npz, sha256_array, sha256_file
from .constants import FEATURE_DIM
from .development_actions import DEVELOPMENT_ACTION_COUNT_PER_TASK
from .development_prediction_plans import validate_development_source_task
from .hashing import canonical_hash
from .prediction_contracts import classifier_parameter_sha256
from .prediction_workers import (
    classifier_from_payload,
    compose_action,
    fit_action_classifier,
    load_source_task_arrays,
    predict_probability_batched,
)


def development_source_prediction_task(task: Mapping[str, object]) -> None:
    validate_development_source_task(task)
    candidates = tuple(str(value) for value in task["candidate_sources"])
    blocks, source_values = load_source_task_arrays(task, candidates=candidates)
    classifier = classifier_from_payload(task["classifier"])
    views = task["evaluation_views"]
    actions = task["actions"]
    probability_rows: list[list[np.ndarray]] = [[], []]
    mean_rows: list[np.ndarray] = []
    scale_rows: list[np.ndarray] = []
    coefficient_rows: list[np.ndarray] = []
    intercept_rows: list[float] = []
    metadata_rows: list[dict[str, object]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Strict source-OOF fitting requires threadpoolctl.") from exc
    with threadpool_limits(limits=int(task["threads_per_fit"])):
        for ordinal, raw in enumerate(actions):
            if not isinstance(raw, Mapping):
                raise ProtocolError("Strict source-OOF action is malformed.")
            train_x, train_y, weights, base_composition_hash = compose_action(
                blocks, raw, candidates
            )
            composition_hash = canonical_hash(
                {
                    "schema_version": "midogpp_strict_source_oof_composition_v1",
                    "base_composition_hash": base_composition_hash,
                    "excluded_pair": list(task["excluded_pair"]),
                    "candidate_sources": list(candidates),
                    "logistic_mass_normalization": raw[
                        "logistic_mass_normalization"
                    ],
                    "sample_weight_scope": "logistic_regression_fit_only",
                    "scaler_fit_used_sample_weight": False,
                    "query_excluded": True,
                    "outer_target_excluded": True,
                }
            )
            fitted = fit_action_classifier(
                train_x,
                train_y,
                spec=classifier,
                sample_weight=weights,
            )
            parameter_hash = classifier_parameter_sha256(
                fitted["mean"],
                fitted["scale"],
                fitted["coefficient"],
                float(fitted["intercept"]),
            )
            logical_rows: list[dict[str, object]] = []
            for view_ordinal, view in enumerate(views):
                if not isinstance(view, Mapping):
                    raise ProtocolError("Strict source-OOF view is malformed.")
                start, stop = int(view["start"]), int(view["stop"])
                evaluation = source_values[start:stop]
                if (
                    evaluation.shape != (int(view["row_count"]), FEATURE_DIM)
                    or sha256_array(evaluation) != view["embedding_slice_sha256"]
                ):
                    raise ProtocolError("Strict source-OOF evaluation slice drifted.")
                probabilities = predict_probability_batched(
                    evaluation,
                    fitted["mean"],
                    fitted["scale"],
                    fitted["coefficient"],
                    float(fitted["intercept"]),
                    batch_rows=int(task["prediction_batch_rows"]),
                )
                probability_rows[view_ordinal].append(probabilities)
                logical_rows.append(
                    {
                        "outer_target": str(view["outer_target"]),
                        "query_center": str(view["query_center"]),
                        "orientation_hash": str(view["orientation_hashes"][ordinal]),
                        "row_identity_hash": str(view["row_identity_hash"]),
                        "probability_sha256": sha256_array(probabilities),
                        "predictions_sha256": sha256_array(
                            (probabilities >= np.float32(0.5)).astype(np.uint8)
                        ),
                    }
                )
            fit_unhashed = {
                "schema_version": "midogpp_strict_source_oof_classifier_fit_v1",
                "task_hash": task["task_hash"],
                "excluded_pair": list(task["excluded_pair"]),
                "action_id": raw["action_id"],
                "action_hash": raw["action_hash"],
                "composition_hash": composition_hash,
                "classifier_config_hash": classifier.config_hash,
                "scaler_state_hash": fitted["scaler_state_hash"],
                "parameter_sha256": parameter_hash,
                "n_iter": list(fitted["n_iter"]),
                "converged": fitted["converged"],
                "logical_predictions": logical_rows,
                "physical_fit_reused_for_two_orientations": True,
                "source_labels_available": False,
                "test_cache_admitted": False,
                "sample_weight_scope": "logistic_regression_fit_only",
                "scaler_fit_used_sample_weight": False,
            }
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
                    "logical_predictions": logical_rows,
                }
            )
    arrays = {
        "source_probabilities_view_0": np.ascontiguousarray(
            np.stack(probability_rows[0]), dtype=np.float32
        ),
        "source_probabilities_view_1": np.ascontiguousarray(
            np.stack(probability_rows[1]), dtype=np.float32
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
    unhashed = {
        "schema_version": "midogpp_strict_source_oof_checkpoint_v1",
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "excluded_pair": list(task["excluded_pair"]),
        "training_seed": task["training_seed"],
        "generation_seed": task["generation_seed"],
        "array_sha256": sha256_file(npz_path),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "actions": metadata_rows,
        "physical_fit_count": DEVELOPMENT_ACTION_COUNT_PER_TASK,
        "logical_prediction_count": 2 * DEVELOPMENT_ACTION_COUNT_PER_TASK,
        "source_labels_available": False,
        "test_cache_admitted": False,
    }
    atomic_json(json_path, {**unhashed, "checkpoint_hash": canonical_hash(unhashed)})


__all__ = ("development_source_prediction_task",)
