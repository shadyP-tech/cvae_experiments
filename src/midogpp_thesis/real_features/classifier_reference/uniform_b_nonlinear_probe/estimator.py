"""Exact nested-LODO Nyström-RBF computation with pair-level reuse."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from midogpp_thesis.common.hashing import stable_hash

from ..protocol import ProtocolError
from .config import Candidate, NonlinearProbeConfig
from .statistics import binary_metrics


def run_source_inner_selection(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    *,
    config: NonlinearProbeConfig,
    class_weights: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    pairs = tuple(combinations(config.heldout_centers, 2))
    if len(pairs) != 36:
        raise ProtocolError("Expected exactly 36 unordered excluded-center pairs.")
    outputs = Parallel(
        n_jobs=config.runtime.pair_jobs,
        backend="loky",
        max_nbytes="10M",
        mmap_mode="r",
    )(
        delayed(_score_pair)(
            pair,
            x,
            y,
            centers,
            sample_ids,
            config,
            class_weights,
        )
        for pair in pairs
    )
    cells = sorted(
        [row for output in outputs for row in output[0]],
        key=lambda row: (
            str(row["outer_center"]),
            str(row["inner_center"]),
            str(row["candidate_id"]),
        ),
    )
    audits = sorted(
        [row for output in outputs for row in output[1]],
        key=lambda row: (
            str(row["fit_key"]),
            float(row["width_multiplier"]),
            int(row["n_components"]),
        ),
    )
    return cells, audits


def _score_pair(
    pair: tuple[str, str],
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    config: NonlinearProbeConfig,
    class_weights: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    with threadpool_limits(limits=config.runtime.threads_per_job):
        excluded = set(pair)
        train_mask = ~np.isin(centers, list(excluded))
        eval_masks = {center: centers == center for center in pair}
        train_centers = tuple(
            center for center in config.heldout_centers if center not in excluded
        )
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train_mask]).astype(np.float32, copy=False)
        eval_x = {
            center: scaler.transform(x[mask]).astype(np.float32, copy=False)
            for center, mask in eval_masks.items()
        }
        median = median_distance_fit(
            train_x,
            sample_ids[train_mask],
            seed=config.gamma_sample_seed,
            cap=config.gamma_sample_cap,
            fit_key="pair:" + ",".join(pair),
        )
        fit_row_hash = _string_hash(sorted(sample_ids[train_mask].tolist()))
        scaler_hash = _array_state_hash((scaler.mean_, scaler.scale_))
        cells: list[dict[str, object]] = []
        audits: list[dict[str, object]] = []
        for width in config.width_multipliers:
            gamma = effective_gamma(float(width), float(median["median_distance"]))
            for n_components in config.components:
                transformer = Nystroem(
                    kernel="rbf",
                    gamma=gamma,
                    n_components=n_components,
                    random_state=config.primary_landmark_seed,
                    n_jobs=1,
                )
                train_z = transformer.fit_transform(train_x)
                eval_z = {
                    center: transformer.transform(values)
                    for center, values in eval_x.items()
                }
                audits.append(
                    kernel_audit_row(
                        role="source_inner_pair",
                        fit_key=",".join(pair),
                        train_centers=train_centers,
                        fit_row_hash=fit_row_hash,
                        scaler_hash=scaler_hash,
                        median=median,
                        width_multiplier=width,
                        gamma=gamma,
                        n_components=n_components,
                        landmark_seed=config.primary_landmark_seed,
                        transformer=transformer,
                        train_sample_ids=sample_ids[train_mask],
                    )
                )
                for c_value in config.logistic_cs:
                    candidate = Candidate(width, n_components, c_value)
                    for outer, inner in (pair, pair[::-1]):
                        model = fit_logistic(
                            train_z,
                            y[train_mask],
                            c_value=c_value,
                            class_weight=class_weights[outer],
                            max_iter=config.classifier_max_iter,
                        )
                        pred = model.predict(eval_z[inner]).astype(np.int8)
                        metrics = binary_metrics(y[eval_masks[inner]], pred)
                        cells.append(
                            {
                                "schema_version": "midogpp_uniform_b_nonlinear_selector_cell_v1",
                                "outer_center": outer,
                                "inner_center": inner,
                                "excluded_pair": json.dumps(list(pair)),
                                "train_centers": json.dumps(list(train_centers)),
                                "n_train": int(np.sum(train_mask)),
                                "n_eval": int(np.sum(eval_masks[inner])),
                                "fit_row_hash": fit_row_hash,
                                "eval_row_hash": _string_hash(
                                    sorted(sample_ids[eval_masks[inner]].tolist())
                                ),
                                "candidate_id": candidate.candidate_id,
                                **candidate.to_payload(),
                                "inherited_outer_class_weight": _class_weight_name(
                                    class_weights[outer]
                                ),
                                "landmark_seed": config.primary_landmark_seed,
                                "gamma_sample_seed": config.gamma_sample_seed,
                                **metrics,
                                "selection_used_outer_labels": False,
                                "fit_used_outer_center": False,
                            }
                        )
        return cells, audits


def fit_outer_models(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    case_ids: np.ndarray,
    *,
    config: NonlinearProbeConfig,
    selected: Mapping[str, Mapping[str, object]],
    class_weights: Mapping[str, object],
    baseline_predictions: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    outputs = Parallel(
        n_jobs=config.runtime.pair_jobs,
        backend="loky",
        max_nbytes="10M",
        mmap_mode="r",
    )(
        delayed(_fit_outer)(
            outer,
            x,
            y,
            centers,
            sample_ids,
            case_ids,
            config,
            selected[outer],
            class_weights[outer],
            baseline_predictions,
        )
        for outer in config.heldout_centers
    )
    return sorted(outputs, key=lambda row: config.heldout_centers.index(row["outer_center"]))


def _fit_outer(
    outer: str,
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    sample_ids: np.ndarray,
    case_ids: np.ndarray,
    config: NonlinearProbeConfig,
    selection: Mapping[str, object],
    class_weight: object,
    baseline_predictions: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    with threadpool_limits(limits=config.runtime.threads_per_job):
        candidate = selection["candidate"]
        if not isinstance(candidate, Candidate):
            raise ProtocolError("Selected nonlinear candidate is malformed.")
        train_mask = centers != outer
        target_mask = centers == outer
        train_centers = tuple(
            center for center in config.heldout_centers if center != outer
        )
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train_mask]).astype(np.float32, copy=False)
        target_x = scaler.transform(x[target_mask]).astype(np.float32, copy=False)
        median = median_distance_fit(
            train_x,
            sample_ids[train_mask],
            seed=config.gamma_sample_seed,
            cap=config.gamma_sample_cap,
            fit_key="outer:" + outer,
        )
        gamma = effective_gamma(candidate.width_multiplier, median["median_distance"])
        fit_row_hash = _string_hash(sorted(sample_ids[train_mask].tolist()))
        eval_row_hash = _string_hash(sorted(sample_ids[target_mask].tolist()))
        scaler_hash = _array_state_hash((scaler.mean_, scaler.scale_))
        class_centroids = {
            label: np.mean(train_x[y[train_mask] == label], axis=0)
            for label in (0, 1)
        }
        target_distances = {
            label: np.mean((target_x - centroid) ** 2, axis=1)
            for label, centroid in class_centroids.items()
        }
        centroid_pred = (
            target_distances[1] < target_distances[0]
        ).astype(np.int8)
        target_ids = sample_ids[target_mask]
        target_cases = case_ids[target_mask]
        target_y = y[target_mask]
        seed_outputs = []
        for landmark_seed in (
            config.primary_landmark_seed,
            *config.stability_landmark_seeds,
        ):
            transformer = Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=candidate.n_components,
                random_state=landmark_seed,
                n_jobs=1,
            )
            train_z = transformer.fit_transform(train_x)
            target_z = transformer.transform(target_x)
            model = fit_logistic(
                train_z,
                y[train_mask],
                c_value=candidate.logistic_c,
                class_weight=class_weight,
                max_iter=config.classifier_max_iter,
            )
            probability = model.predict_proba(target_z)[:, 1]
            prediction = model.predict(target_z).astype(np.int8)
            metrics = binary_metrics(target_y, prediction)
            predictions = []
            for index, sample_id in enumerate(target_ids):
                baseline = baseline_predictions[str(sample_id)]
                baseline_wrong = int(baseline["y_pred"]) != int(target_y[index])
                predictions.append(
                    {
                        "schema_version": "midogpp_uniform_b_nonlinear_prediction_v1",
                        "outer_center": outer,
                        "sample_id": str(sample_id),
                        "case_id": str(target_cases[index]),
                        "center": outer,
                        "y_true": int(target_y[index]),
                        "y_pred": int(prediction[index]),
                        "prob_pos": float(probability[index]),
                        "landmark_seed": landmark_seed,
                        "candidate_id": candidate.candidate_id,
                        "eval_row_hash": eval_row_hash,
                        "baseline_y_pred": int(baseline["y_pred"]),
                        "baseline_prob_pos": float(baseline["prob_pos"]),
                        "baseline_wrong": baseline_wrong,
                        "baseline_confidence": max(
                            float(baseline["prob_pos"]),
                            1.0 - float(baseline["prob_pos"]),
                        ),
                        "centroid_prediction": int(centroid_pred[index]),
                        "centroid_true_closer": int(centroid_pred[index])
                        == int(target_y[index]),
                        "centroid_distance_negative": float(target_distances[0][index]),
                        "centroid_distance_positive": float(target_distances[1][index]),
                        "target_labels_used_for_scoring_only": True,
                        "selection_used_target_labels": False,
                        "fit_used_target_center": False,
                    }
                )
            seed_outputs.append(
                {
                    "landmark_seed": landmark_seed,
                    "metrics": metrics,
                    "predictions": predictions,
                    "audit": kernel_audit_row(
                        role="outer_final",
                        fit_key=outer,
                        train_centers=train_centers,
                        fit_row_hash=fit_row_hash,
                        scaler_hash=scaler_hash,
                        median=median,
                        width_multiplier=candidate.width_multiplier,
                        gamma=gamma,
                        n_components=candidate.n_components,
                        landmark_seed=landmark_seed,
                        transformer=transformer,
                        train_sample_ids=sample_ids[train_mask],
                    ),
                    "n_iter": int(np.max(model.n_iter_)),
                }
            )
        return {
            "outer_center": outer,
            "candidate": candidate,
            "class_weight": class_weight,
            "selection": dict(selection),
            "train_centers": train_centers,
            "n_train": int(np.sum(train_mask)),
            "n_eval": int(np.sum(target_mask)),
            "fit_row_hash": fit_row_hash,
            "eval_row_hash": eval_row_hash,
            "seed_outputs": seed_outputs,
        }


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    c_value: float,
    class_weight: object,
    max_iter: int,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        class_weight=class_weight,
        max_iter=int(max_iter),
        random_state=23,
    )
    model.fit(x, y)
    if int(np.max(model.n_iter_)) >= max_iter:
        raise ProtocolError("Nyström logistic classifier did not converge.")
    return model


def median_distance_fit(
    train_x: np.ndarray,
    train_sample_ids: np.ndarray,
    *,
    seed: int,
    cap: int,
    fit_key: str,
) -> dict[str, object]:
    n_sample = min(int(cap), len(train_x))
    effective_seed = _namespaced_seed("gamma_sample", fit_key, seed)
    rng = np.random.default_rng(effective_seed)
    indices = np.sort(rng.choice(len(train_x), size=n_sample, replace=False))
    squared = pdist(train_x[indices].astype(np.float64), metric="sqeuclidean")
    nonzero = squared[squared > 0.0]
    if nonzero.size == 0:
        raise ProtocolError("Median-distance estimate has no nonzero pairs.")
    median_squared = float(np.median(nonzero))
    return {
        "gamma_sample_seed": int(seed),
        "gamma_sample_effective_seed": int(effective_seed),
        "gamma_sample_count": int(n_sample),
        "gamma_sample_row_hash": _string_hash(
            sorted(train_sample_ids[indices].tolist())
        ),
        "median_squared_distance": median_squared,
        "median_distance": float(np.sqrt(median_squared)),
        "nonzero_pair_count": int(nonzero.size),
    }


def effective_gamma(width_multiplier: float, median_distance: float) -> float:
    sigma = float(width_multiplier) * float(median_distance)
    if sigma <= 0.0:
        raise ProtocolError("Nyström RBF sigma must be positive.")
    return 1.0 / (2.0 * sigma * sigma)


def kernel_audit_row(
    *,
    role: str,
    fit_key: str,
    train_centers: Sequence[str],
    fit_row_hash: str,
    scaler_hash: str,
    median: Mapping[str, object],
    width_multiplier: float,
    gamma: float,
    n_components: int,
    landmark_seed: int,
    transformer: Nystroem,
    train_sample_ids: np.ndarray,
) -> dict[str, object]:
    component_indices = np.asarray(transformer.component_indices_, dtype=np.int64)
    landmark_ids = train_sample_ids[component_indices]
    return {
        "schema_version": "midogpp_uniform_b_nonlinear_kernel_fit_audit_v1",
        "fit_role": role,
        "fit_key": fit_key,
        "train_centers": json.dumps(list(train_centers)),
        "fit_row_hash": fit_row_hash,
        "scaler_state_hash": scaler_hash,
        **dict(median),
        "width_multiplier": float(width_multiplier),
        "sigma": float(width_multiplier) * float(median["median_distance"]),
        "effective_gamma": float(gamma),
        "n_components": int(n_components),
        "landmark_seed": int(landmark_seed),
        "landmark_row_hash": _string_hash(sorted(landmark_ids.tolist())),
        "nystroem_state_hash": _array_state_hash(
            (
                component_indices,
                np.asarray(transformer.components_),
                np.asarray(transformer.normalization_),
            )
        ),
        "gpu_used": False,
    }


def _class_weight_name(value: object) -> str:
    return "none" if value is None else str(value)


def _namespaced_seed(namespace: str, key: str, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{namespace}|{key}|{int(base_seed)}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _string_hash(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _array_state_hash(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()
