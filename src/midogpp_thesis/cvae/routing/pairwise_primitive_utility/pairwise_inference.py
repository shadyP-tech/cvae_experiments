"""Deterministic antisymmetric inference for a fitted pairwise ranker."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import ActionQuery, P_ACTION_ID, PairwisePrediction, PairwiseRankerModel
from .pairwise_features import feature_vector


def predict_action_score(model: PairwiseRankerModel, query: ActionQuery) -> float:
    """Predict a latent source utility score; protected P is exactly zero."""

    if query.action_id == P_ACTION_ID:
        return 0.0
    vector = feature_vector(
        query,
        feature_names=model.feature_names,
        mean=np.asarray(model.feature_mean, dtype=np.float64),
        scale=np.asarray(model.feature_scale, dtype=np.float64),
        action_schema=model.action_schema,
        design_names=model.design_names,
    )
    result = float(vector @ np.asarray(model.coefficients, dtype=np.float64))
    if not math.isfinite(result):
        raise ProtocolError("Pairwise action score is non-finite.")
    return result


def predict_pairwise_contrast(
    model: PairwiseRankerModel, left: ActionQuery, right: ActionQuery
) -> PairwisePrediction:
    """Return score(left)-score(right), guaranteeing exact antisymmetry."""

    if left.action_id == right.action_id:
        raise ProtocolError("Pairwise contrast requires distinct actions.")
    mean = predict_action_score(model, left) - predict_action_score(model, right)
    return PairwisePrediction(
        left_action_id=left.action_id,
        right_action_id=right.action_id,
        mean_contrast=mean,
        model_hash=model.model_hash,
    )


def rank_action_queries(
    model: PairwiseRankerModel, queries: Sequence[ActionQuery]
) -> tuple[tuple[str, float], ...]:
    """Rank unique active queries with deterministic P as an included anchor."""

    rows = tuple(queries)
    if len({row.action_id for row in rows}) != len(rows):
        raise ProtocolError("Pairwise inference queries contain duplicate actions.")
    if P_ACTION_ID not in {row.action_id for row in rows}:
        rows = (*rows, ActionQuery.p_anchor(model.feature_names))
    scores = tuple((row.action_id, predict_action_score(model, row)) for row in rows)
    return tuple(sorted(scores, key=lambda row: (-row[1], row[0])))


__all__ = ("predict_action_score", "predict_pairwise_contrast", "rank_action_queries")
