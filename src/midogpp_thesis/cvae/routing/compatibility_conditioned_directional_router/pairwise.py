"""Candidate-aware hurdle and pairwise ridge with nested source-center LODO."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    ALPHA_GRID,
    ENDPOINTS,
    ActionKind,
    ActionPrediction,
    CandidateFeatureVector,
    EndpointEffects,
    FoldLoss,
    HurdlePairwiseModel,
    SourceActionObservation,
    SourceOOFPrediction,
)
from .hashing import canonical_hash


_KINDS = (ActionKind.U.value, ActionKind.HXE.value)
_DIRECTIONS = ("ALL_MARGINS", "D01", "D10")
_EPSILON = 1e-12


def action_key(feature: CandidateFeatureVector) -> str:
    """Generalized uncertainty key, intentionally independent of source id."""

    if feature.action_kind is ActionKind.U:
        return f"{ActionKind.U.value}:{feature.direction.value}"
    return f"{ActionKind.HXE.value}:{feature.direction.value}"


def _canonical_rows(
    observations: Sequence[SourceActionObservation], *, outer_target_id: str
) -> tuple[SourceActionObservation, ...]:
    rows = tuple(
        sorted(
            tuple(observations),
            key=lambda row: (
                row.feature.query_center_id,
                row.feature.case_id,
                row.feature.action_id,
            ),
        )
    )
    if not rows or any(not isinstance(row, SourceActionObservation) for row in rows):
        raise ProtocolError("Pairwise fitting requires typed source-development observations.")
    h = str(outer_target_id)
    keys = tuple(
        (row.feature.query_center_id, row.feature.case_id, row.feature.action_id) for row in rows
    )
    schemas = {row.feature.feature_names for row in rows}
    all_centers = {row.candidate_pool.all_center_ids for row in rows}
    bank_hashes = {row.candidate_pool.bank_lock_hash for row in rows}
    if (
        len(set(keys)) != len(keys)
        or len(schemas) != 1
        or len(all_centers) != 1
        or len(bank_hashes) != 1
        or any(
            row.feature.outer_target_id != h
            or row.candidate_pool.outer_target_id != h
            or row.feature.query_center_id == h
            or (
                row.feature.action_kind is ActionKind.HXE
                and row.feature.candidate_source_id
                not in row.candidate_pool.candidate_center_ids
            )
            for row in rows
        )
    ):
        raise ProtocolError("Pairwise surface crossed H, schema, bank, or candidate roles.")
    query_ids = tuple(sorted({row.feature.query_center_id for row in rows}))
    if len(query_ids) < 3:
        raise ProtocolError("Nested source-center LODO requires at least three source queries.")
    grouped: dict[tuple[str, str], list[SourceActionObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.feature.query_center_id, row.feature.case_id)].append(row)
    if any(len({item.feature.action_id for item in group}) != len(group) for group in grouped.values()):
        raise ProtocolError("Pairwise source cases contain duplicate action identities.")
    return rows


def _row_weights(rows: Sequence[SourceActionObservation]) -> np.ndarray:
    grouped_cases: dict[str, set[str]] = defaultdict(set)
    action_count: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        q = row.feature.query_center_id
        case = row.feature.case_id
        grouped_cases[q].add(case)
        action_count[(q, case)] += 1
    center_count = len(grouped_cases)
    values = tuple(
        1.0
        / (
            center_count
            * len(grouped_cases[row.feature.query_center_id])
            * action_count[(row.feature.query_center_id, row.feature.case_id)]
        )
        for row in rows
    )
    return np.asarray(values, dtype=np.float64)


def _normalization(
    rows: Sequence[SourceActionObservation],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray([row.feature.feature_values for row in rows], dtype=np.float64)
    weights = _row_weights(rows)
    weights /= np.sum(weights, dtype=np.float64)
    mean = np.sum(weights[:, None] * matrix, axis=0, dtype=np.float64)
    variance = np.sum(weights[:, None] * (matrix - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    return mean, scale


def _design_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    output: list[str] = []
    output.extend(f"kind_intercept::{kind}" for kind in _KINDS)
    output.extend(f"global_feature::{name}" for name in feature_names)
    output.extend(
        f"kind_feature::{kind}::{name}" for kind in _KINDS for name in feature_names
    )
    output.extend(
        f"direction_feature::{direction}::{name}"
        for direction in _DIRECTIONS
        for name in feature_names
    )
    return tuple(output)


def _feature_vector(
    feature: CandidateFeatureVector,
    *,
    feature_names: tuple[str, ...],
    mean: np.ndarray,
    scale: np.ndarray,
    design_names: tuple[str, ...],
) -> np.ndarray:
    if feature.feature_names != feature_names:
        raise ProtocolError("Pairwise query schema drifted from its source model.")
    kind = feature.action_kind.value
    direction = feature.direction.value
    if kind not in _KINDS or direction not in _DIRECTIONS:
        raise ProtocolError("Pairwise query has an unsupported kind or direction.")
    standardized = (np.asarray(feature.feature_values, dtype=np.float64) - mean) / scale
    index = {name: ordinal for ordinal, name in enumerate(design_names)}
    vector = np.zeros(len(design_names), dtype=np.float64)
    vector[index[f"kind_intercept::{kind}"]] = 1.0
    for name, value in zip(feature_names, standardized, strict=True):
        vector[index[f"global_feature::{name}"]] = value
        vector[index[f"kind_feature::{kind}::{name}"]] = value
        vector[index[f"direction_feature::{direction}::{name}"]] = value
    return vector


def _matrix(
    rows: Sequence[SourceActionObservation],
    *,
    feature_names: tuple[str, ...],
    mean: np.ndarray,
    scale: np.ndarray,
    design_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [
            _feature_vector(
                row.feature,
                feature_names=feature_names,
                mean=mean,
                scale=scale,
                design_names=design_names,
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def _ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    normal = matrix.T @ (weights[:, None] * matrix)
    normal.flat[:: normal.shape[0] + 1] += float(alpha)
    rhs = matrix.T @ (weights * response)
    try:
        result = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        result = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    if not np.isfinite(result).all():
        raise ProtocolError("Candidate-aware ridge produced non-finite coefficients.")
    return result


def _logistic_ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    identity = np.eye(matrix.shape[1], dtype=np.float64)
    for _ in range(32):
        linear = np.clip(matrix @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        curvature = np.maximum(probability * (1.0 - probability), 1e-6)
        adjusted = linear + (response - probability) / curvature
        combined = weights * curvature
        normal = matrix.T @ (combined[:, None] * matrix) + float(alpha) * identity
        rhs = matrix.T @ (combined * adjusted)
        try:
            updated = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(normal, rhs, rcond=None)[0]
        if not np.isfinite(updated).all():
            raise ProtocolError("Hurdle logistic ridge produced non-finite coefficients.")
        if float(np.max(np.abs(updated - coefficients))) <= 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def _contrasts(
    rows: Sequence[SourceActionObservation], matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ordinal, row in enumerate(rows):
        grouped[(row.feature.query_center_id, row.feature.case_id)].append(ordinal)
    query_cases: dict[str, set[str]] = defaultdict(set)
    for query, case in grouped:
        query_cases[query].add(case)
    center_count = len(query_cases)
    vectors: list[np.ndarray] = []
    responses: list[float] = []
    weights: list[float] = []
    zero = np.zeros(matrix.shape[1], dtype=np.float64)
    for (query, case), indices in sorted(grouped.items()):
        # B is an exact zero-score, zero-gain member of every case.
        members = [(-1, zero, 0.0)] + [
            (index, matrix[index], rows[index].effects.bacc_gain) for index in indices
        ]
        pair_count = len(members) * (len(members) - 1) // 2
        case_weight = 1.0 / (center_count * len(query_cases[query]) * pair_count)
        for left, right in combinations(members, 2):
            vectors.append(left[1] - right[1])
            responses.append(left[2] - right[2])
            weights.append(case_weight)
    if not vectors:
        raise ProtocolError("Pairwise fitting has no within-case contrasts.")
    return (
        np.asarray(vectors, dtype=np.float64),
        np.asarray(responses, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def _fit_heads(
    rows: tuple[SourceActionObservation, ...],
    *,
    alpha: float,
    fit_endpoints: bool = True,
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    tuple[tuple[str, tuple[float, ...]], ...],
]:
    feature_names = rows[0].feature.feature_names
    mean, scale = _normalization(rows)
    names = _design_names(feature_names)
    matrix = _matrix(
        rows,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        design_names=names,
    )
    weights = _row_weights(rows)
    hurdle_response = np.asarray(
        [float(row.effects.bacc_gain > 0.0) for row in rows], dtype=np.float64
    )
    hurdle = _logistic_ridge(matrix, hurdle_response, weights, alpha=alpha)
    contrast_matrix, contrast_response, contrast_weights = _contrasts(rows, matrix)
    pairwise = _ridge(
        contrast_matrix, contrast_response, contrast_weights, alpha=alpha
    )
    endpoint_coefficients: list[tuple[str, tuple[float, ...]]] = []
    if fit_endpoints:
        for endpoint in ENDPOINTS:
            response = np.asarray([getattr(row.effects, endpoint) for row in rows], dtype=np.float64)
            coefficients = _ridge(matrix, response, weights, alpha=alpha)
            endpoint_coefficients.append(
                (endpoint, tuple(float(value) for value in coefficients))
            )
    return (
        feature_names,
        mean,
        scale,
        names,
        hurdle,
        pairwise,
        tuple(endpoint_coefficients),
    )


def _prepare_fold(
    training: tuple[SourceActionObservation, ...],
    held: tuple[SourceActionObservation, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Stage alpha-invariant fold arrays once for workstation efficiency."""

    feature_names = training[0].feature.feature_names
    mean, scale = _normalization(training)
    names = _design_names(feature_names)
    training_matrix = _matrix(
        training,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        design_names=names,
    )
    held_matrix = _matrix(
        held,
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        design_names=names,
    )
    training_weights = _row_weights(training)
    held_weights = _row_weights(held)
    training_hurdle = np.asarray(
        [float(row.effects.bacc_gain > 0.0) for row in training], dtype=np.float64
    )
    held_hurdle = np.asarray(
        [float(row.effects.bacc_gain > 0.0) for row in held], dtype=np.float64
    )
    training_contrasts, training_response, training_contrast_weights = _contrasts(
        training, training_matrix
    )
    held_contrasts, held_response, held_contrast_weights = _contrasts(held, held_matrix)
    # These arrays are staged once to avoid rebuilding the variable per-case
    # action inventory for every alpha.
    return (
        training_matrix,
        training_weights,
        training_hurdle,
        held_matrix,
        held_weights,
        held_hurdle,
        training_contrasts,
        training_response,
        training_contrast_weights,
        held_contrasts,
        held_response,
        held_contrast_weights,
    )


def _prepared_losses(
    prepared: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    alpha: float,
) -> tuple[float, float]:
    (
        training_matrix,
        training_weights,
        training_hurdle,
        held_matrix,
        held_weights,
        held_hurdle,
        training_contrasts,
        training_response,
        training_contrast_weights,
        held_contrasts,
        held_response,
        held_contrast_weights,
    ) = prepared
    hurdle = _logistic_ridge(
        training_matrix, training_hurdle, training_weights, alpha=alpha
    )
    pairwise = _ridge(
        training_contrasts,
        training_response,
        training_contrast_weights,
        alpha=alpha,
    )
    linear = np.clip(held_matrix @ hurdle, -30.0, 30.0)
    probability = np.clip(1.0 / (1.0 + np.exp(-linear)), _EPSILON, 1.0 - _EPSILON)
    hurdle_loss = float(
        np.sum(
            held_weights
            * -(
                held_hurdle * np.log(probability)
                + (1.0 - held_hurdle) * np.log(1.0 - probability)
            )
        )
        / np.sum(held_weights)
    )
    residual = held_response - held_contrasts @ pairwise
    pairwise_loss = float(
        np.sum(held_contrast_weights * residual * residual)
        / np.sum(held_contrast_weights)
    )
    return hurdle_loss, pairwise_loss


def _training_without_center(
    rows: tuple[SourceActionObservation, ...], center: str
) -> tuple[SourceActionObservation, ...]:
    return tuple(
        row
        for row in rows
        if row.feature.query_center_id != center
        and row.feature.candidate_source_id != center
    )


def fit_hurdle_pairwise_model(
    observations: Sequence[SourceActionObservation],
    *,
    outer_target_id: str,
    alpha_grid: Sequence[float] = ALPHA_GRID,
) -> HurdlePairwiseModel:
    """Tune on nested source-center LODO, then refit on exact C-minus-H rows.

    For held source center ``K``, every training row with query ``K`` *or*
    candidate ``K`` is removed before normalization and fitting.  Validation is
    the query-K surface, whose own candidate pool already excludes K.
    """

    rows = _canonical_rows(observations, outer_target_id=outer_target_id)
    grid = tuple(sorted({float(value) for value in alpha_grid}))
    if not grid or any(not math.isfinite(value) or value <= 0.0 for value in grid):
        raise ProtocolError("Pairwise alpha grid must be finite, positive, and nonempty.")
    queries = tuple(sorted({row.feature.query_center_id for row in rows}))
    fold_losses: list[FoldLoss] = []
    for held_center in queries:
        training = _training_without_center(rows, held_center)
        held = tuple(row for row in rows if row.feature.query_center_id == held_center)
        training_queries = {row.feature.query_center_id for row in training}
        if (
            not held
            or len(training_queries) < 2
            or any(row.feature.candidate_source_id == held_center for row in training)
        ):
            raise ProtocolError("Nested source-center LODO fold is empty or role-incomplete.")
        prepared = _prepare_fold(training, held)
        for alpha in grid:
            hurdle_loss, pairwise_loss = _prepared_losses(prepared, alpha=alpha)
            fold_losses.append(
                FoldLoss(
                    held_center_id=held_center,
                    alpha=alpha,
                    hurdle_log_loss=hurdle_loss,
                    pairwise_mse=pairwise_loss,
                )
            )
    summaries: list[tuple[float, float, float]] = []
    for alpha in grid:
        values = tuple(
            row.hurdle_log_loss + row.pairwise_mse
            for row in fold_losses
            if row.alpha == alpha
        )
        if len(values) != len(queries):
            raise ProtocolError("Nested alpha loss surface is incomplete.")
        summaries.append((alpha, max(values), sum(values) / len(values)))
    selected_alpha = min(summaries, key=lambda row: (row[1], row[2], row[0]))[0]
    feature_names, mean, scale, names, hurdle, pairwise, endpoints = _fit_heads(
        rows, alpha=selected_alpha
    )
    return HurdlePairwiseModel(
        outer_target_id=str(outer_target_id),
        feature_names=feature_names,
        normalization_mean=tuple(float(value) for value in mean),
        normalization_scale=tuple(float(value) for value in scale),
        design_names=names,
        hurdle_coefficients=tuple(float(value) for value in hurdle),
        pairwise_coefficients=tuple(float(value) for value in pairwise),
        endpoint_coefficients=endpoints,
        selected_alpha=selected_alpha,
        alpha_grid=grid,
        fold_losses=tuple(
            sorted(fold_losses, key=lambda row: (row.held_center_id, row.alpha))
        ),
        training_query_ids=queries,
        training_candidate_ids=tuple(
            sorted(
                {
                    row.feature.candidate_source_id
                    for row in rows
                    if row.feature.candidate_source_id is not None
                }
            )
        ),
        training_case_count=len(
            {(row.feature.query_center_id, row.feature.case_id) for row in rows}
        ),
        training_row_hash=canonical_hash(
            tuple(row.source_response_hash for row in rows)
        ),
    )


def predict_action(
    model: HurdlePairwiseModel, feature: CandidateFeatureVector
) -> ActionPrediction:
    """Predict opportunity, latent BACC rank, and endpoint means label-free."""

    if not isinstance(model, HurdlePairwiseModel) or not isinstance(
        feature, CandidateFeatureVector
    ):
        raise ProtocolError("Pairwise prediction requires typed model and feature.")
    if feature.outer_target_id != model.outer_target_id:
        raise ProtocolError("Pairwise target feature belongs to another outer H.")
    vector = _feature_vector(
        feature,
        feature_names=model.feature_names,
        mean=np.asarray(model.normalization_mean, dtype=np.float64),
        scale=np.asarray(model.normalization_scale, dtype=np.float64),
        design_names=model.design_names,
    )
    linear = float(np.dot(vector, np.asarray(model.hurdle_coefficients, dtype=np.float64)))
    opportunity = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, linear))))
    ranking = float(np.dot(vector, np.asarray(model.pairwise_coefficients, dtype=np.float64)))
    effects = EndpointEffects(
        **{
            endpoint: float(
                np.dot(
                    vector,
                    np.asarray(model.endpoint_coefficients_for(endpoint), dtype=np.float64),
                )
            )
            for endpoint in ENDPOINTS
        }
    )
    return ActionPrediction(
        feature=feature,
        opportunity_probability=opportunity,
        ranking_score=ranking,
        predicted_effects=effects,
        model_hash=model.model_hash,
    )


def crossfit_source_predictions(
    observations: Sequence[SourceActionObservation],
    *,
    model: HurdlePairwiseModel,
) -> tuple[SourceOOFPrediction, ...]:
    """Return strict role-complete source-query-OOF predictions.

    The full model's nested-selected alpha is frozen; each held-query model is
    refit after removing the held center from both query and candidate roles.
    """

    rows = _canonical_rows(observations, outer_target_id=model.outer_target_id)
    if model.training_row_hash != canonical_hash(tuple(row.source_response_hash for row in rows)):
        raise ProtocolError("Crossfit surface drifted from the fitted full model.")
    output: list[SourceOOFPrediction] = []
    for held_center in sorted({row.feature.query_center_id for row in rows}):
        training = _training_without_center(rows, held_center)
        held = tuple(row for row in rows if row.feature.query_center_id == held_center)
        feature_names, mean, scale, names, hurdle, pairwise, endpoints = _fit_heads(
            training, alpha=model.selected_alpha
        )
        endpoint_map = dict(endpoints)
        training_queries = tuple(sorted({row.feature.query_center_id for row in training}))
        training_candidates = tuple(
            sorted(
                {
                    row.feature.candidate_source_id
                    for row in training
                    if row.feature.candidate_source_id is not None
                }
            )
        )
        fold_hash = canonical_hash(
            {
                "schema_version": "compatibility_directional_oof_fold_v1",
                "outer_target_H": model.outer_target_id,
                "held_center": held_center,
                "training_row_hashes": tuple(row.source_response_hash for row in training),
                "training_queries": training_queries,
                "training_candidates": training_candidates,
                "selected_alpha": model.selected_alpha,
                "feature_names": feature_names,
                "mean": tuple(float(value) for value in mean),
                "scale": tuple(float(value) for value in scale),
                "design_names": names,
                "hurdle": tuple(float(value) for value in hurdle),
                "pairwise": tuple(float(value) for value in pairwise),
                "endpoints": endpoints,
                "target_labels_used": False,
            }
        )
        for row in held:
            vector = _feature_vector(
                row.feature,
                feature_names=feature_names,
                mean=mean,
                scale=scale,
                design_names=names,
            )
            linear = float(np.dot(vector, hurdle))
            opportunity = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, linear))))
            prediction = ActionPrediction(
                feature=row.feature,
                opportunity_probability=opportunity,
                ranking_score=float(np.dot(vector, pairwise)),
                predicted_effects=EndpointEffects(
                    **{
                        endpoint: float(np.dot(vector, np.asarray(endpoint_map[endpoint])))
                        for endpoint in ENDPOINTS
                    }
                ),
                model_hash=fold_hash,
            )
            output.append(
                SourceOOFPrediction(
                    held_center_id=held_center,
                    prediction=prediction,
                    observed=row.effects,
                    fold_training_query_ids=training_queries,
                    fold_training_candidate_ids=training_candidates,
                    fold_hash=fold_hash,
                )
            )
    return tuple(
        sorted(
            output,
            key=lambda row: (
                row.held_center_id,
                row.prediction.feature.case_id,
                row.prediction.feature.action_id,
            ),
        )
    )


__all__ = (
    "action_key",
    "crossfit_source_predictions",
    "fit_hurdle_pairwise_model",
    "predict_action",
)
