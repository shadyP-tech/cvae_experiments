"""Action-specific antisymmetric feature design for pairwise ridge ranking."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import ActionQuery, ActionUtilityObservation, P_ACTION_ID
from .pairwise_contrasts import (
    PairwiseContrast,
    action_row_weights,
    center_case_balanced_contrast_weights,
)


def normalization(
    rows: Sequence[ActionUtilityObservation],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    weights = action_row_weights(rows)
    total = float(np.sum(weights, dtype=np.float64))
    mean = np.sum(weights[:, None] * matrix, axis=0, dtype=np.float64) / total
    variance = np.sum(weights[:, None] * (matrix - mean) ** 2, axis=0, dtype=np.float64) / total
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    return mean, scale


def design_names(
    feature_names: Sequence[str], action_schema: Sequence[tuple[str, str, str]]
) -> tuple[str, ...]:
    actions = tuple(action for action, _, _ in action_schema)
    families = tuple(sorted({family for _, family, _ in action_schema}))
    directions = tuple(sorted({direction for _, _, direction in action_schema}))
    output: list[str] = []
    output.extend(f"action_intercept::{action}" for action in actions)
    output.extend(
        f"action_feature::{action}::{feature}"
        for action in actions
        for feature in feature_names
    )
    output.extend(
        f"family_feature::{family}::{feature}"
        for family in families
        for feature in feature_names
    )
    output.extend(
        f"direction_feature::{direction}::{feature}"
        for direction in directions
        for feature in feature_names
    )
    return tuple(output)


def feature_vector(
    query: ActionQuery,
    *,
    feature_names: tuple[str, ...],
    mean: np.ndarray,
    scale: np.ndarray,
    action_schema: tuple[tuple[str, str, str], ...],
    design_names: tuple[str, ...],
) -> np.ndarray:
    if query.feature_names != feature_names:
        raise ProtocolError("Pairwise query feature schema drifted from its model.")
    if query.action_id == P_ACTION_ID:
        return np.zeros(len(design_names), dtype=np.float64)
    schema_by_action = {
        action: (family, direction) for action, family, direction in action_schema
    }
    if query.action_id not in schema_by_action:
        raise ProtocolError(f"Pairwise query action was absent from source training: {query.action_id}")
    if schema_by_action[query.action_id] != (query.family, query.direction):
        raise ProtocolError("Pairwise query action family/direction drifted.")
    standardized = (np.asarray(query.feature_values, dtype=np.float64) - mean) / scale
    index = {name: ordinal for ordinal, name in enumerate(design_names)}
    vector = np.zeros(len(design_names), dtype=np.float64)
    vector[index[f"action_intercept::{query.action_id}"]] = 1.0
    for feature, value in zip(feature_names, standardized, strict=True):
        vector[index[f"action_feature::{query.action_id}::{feature}"]] = value
        vector[index[f"family_feature::{query.family}::{feature}"]] = value
        vector[index[f"direction_feature::{query.direction}::{feature}"]] = value
    return vector


def contrast_matrix(
    contrasts: Sequence[PairwiseContrast],
    *,
    feature_names: tuple[str, ...],
    mean: np.ndarray,
    scale: np.ndarray,
    action_schema: tuple[tuple[str, str, str], ...],
    design_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = tuple(contrasts)
    matrix = np.asarray(
        [
            feature_vector(
                row.left,
                feature_names=feature_names,
                mean=mean,
                scale=scale,
                action_schema=action_schema,
                design_names=design_names,
            )
            - feature_vector(
                row.right,
                feature_names=feature_names,
                mean=mean,
                scale=scale,
                action_schema=action_schema,
                design_names=design_names,
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    response = np.asarray([row.realized_contrast for row in rows], dtype=np.float64)
    weights = center_case_balanced_contrast_weights(rows)
    return matrix, response, weights


__all__ = ("contrast_matrix", "design_names", "feature_vector", "normalization")
