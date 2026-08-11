"""Fixed hierarchical design and regret-weighted pair construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    FEATURE_NAMES,
    CaseActionFeatureRow,
    CaseActionResponseRow,
)
from .runtime import assert_dense_fit_within_budget


SHARED_L2_PENALTY = 1.0
ACTION_L2_PENALTY = 4.0
PAIR_TOLERANCE = 1.0e-15


@dataclass(frozen=True)
class DesignEncoder:
    """Known-bank partial-pooling encoder with U fixed at the origin."""

    action_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def __post_init__(self) -> None:
        if not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise ProtocolError("Design actions must be unique and nonempty.")
        if self.feature_names != FEATURE_NAMES:
            raise ProtocolError("Hierarchical design feature schema drifted.")
        if self.feature_mean.shape != (len(FEATURE_NAMES),) or self.feature_scale.shape != (
            len(FEATURE_NAMES),
        ):
            raise ProtocolError("Hierarchical design standardization dimension drifted.")
        if (
            not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_scale).all()
            or np.any(self.feature_scale <= 0.0)
        ):
            raise ProtocolError("Hierarchical design standardization must be finite.")
        self.feature_mean.setflags(write=False)
        self.feature_scale.setflags(write=False)

    @property
    def dimension(self) -> int:
        feature_count = len(self.feature_names)
        action_count = len(self.action_ids)
        return feature_count + action_count + action_count * feature_count

    @property
    def penalty_diagonal(self) -> np.ndarray:
        feature_count = len(self.feature_names)
        action_count = len(self.action_ids)
        result = np.asarray(
            [SHARED_L2_PENALTY] * feature_count
            + [ACTION_L2_PENALTY] * action_count
            + [ACTION_L2_PENALTY] * action_count * feature_count,
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    def encode(self, row: CaseActionFeatureRow) -> np.ndarray:
        if row.action_id not in self.action_ids:
            raise ProtocolError("Feature action is absent from the fitted known-bank design.")
        standardized = (
            np.asarray(row.values, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        action_index = self.action_ids.index(row.action_id)
        feature_count = len(self.feature_names)
        action_count = len(self.action_ids)
        result = np.zeros(self.dimension, dtype=np.float64)
        result[:feature_count] = standardized
        result[feature_count + action_index] = 1.0
        interaction_start = feature_count + action_count + action_index * feature_count
        result[interaction_start : interaction_start + feature_count] = standardized
        return result

    def encode_control(self) -> np.ndarray:
        """U is the exact zero-gain reference and therefore the design origin."""

        return np.zeros(self.dimension, dtype=np.float64)


@dataclass(frozen=True)
class PairwiseTrainingDesign:
    encoder: DesignEncoder
    values: np.ndarray
    outcomes: np.ndarray
    weights: np.ndarray
    query_ids: tuple[str, ...]
    informative_query_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        row_count, dimension = self.values.shape
        if dimension != self.encoder.dimension or self.outcomes.shape != (row_count,):
            raise ProtocolError("Pairwise training design dimensions drifted.")
        if self.weights.shape != (row_count,) or len(self.query_ids) != row_count:
            raise ProtocolError("Pairwise weights/query identities do not align.")
        if row_count <= 0 or not all(
            np.isfinite(value).all() for value in (self.values, self.outcomes, self.weights)
        ):
            raise ProtocolError("Pairwise training design must be finite and nonempty.")
        if np.any(self.weights <= 0.0) or not np.isclose(
            self.weights.sum(), 1.0, rtol=1.0e-12, atol=1.0e-12
        ):
            raise ProtocolError("Pairwise weights must be positive and sum to one.")
        for value in (self.values, self.outcomes, self.weights):
            value.setflags(write=False)


def fit_design_encoder(
    rows: Sequence[CaseActionFeatureRow],
    *,
    action_ids: Sequence[str],
) -> DesignEncoder:
    selected = tuple(rows)
    actions = tuple(sorted(str(action) for action in action_ids))
    if not selected or len(actions) != len(set(actions)):
        raise ProtocolError("Design fitting requires rows and unique known-bank actions.")
    matrix = np.asarray([row.values for row in selected], dtype=np.float64)
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = matrix.std(axis=0, ddof=0, dtype=np.float64)
    scale[scale <= np.sqrt(np.finfo(np.float64).eps)] = 1.0
    return DesignEncoder(
        action_ids=actions,
        feature_names=FEATURE_NAMES,
        feature_mean=mean,
        feature_scale=scale,
    )


def build_pairwise_training_design(
    feature_rows: Sequence[CaseActionFeatureRow],
    response_rows: Sequence[CaseActionResponseRow],
    *,
    legal_query_ids: Sequence[str],
    action_ids: Sequence[str],
    control_action_id: str,
) -> PairwiseTrainingDesign:
    """Build query-equal, magnitude-weighted preference observations."""

    legal = tuple(sorted(set(str(query) for query in legal_query_ids)))
    if len(legal) < 3:
        raise ProtocolError("Pairwise fitting requires at least three legal donor queries.")
    features = tuple(row for row in feature_rows if row.query_id in set(legal))
    responses = tuple(row for row in response_rows if row.query_id in set(legal))
    feature_by_key = {row.row_key: row for row in features}
    response_by_key = {row.row_key: row for row in responses}
    if len(feature_by_key) != len(features) or len(response_by_key) != len(responses):
        raise ProtocolError("Pairwise fitting inputs contain duplicate case-action rows.")
    if set(feature_by_key) != set(response_by_key):
        raise ProtocolError("Pairwise feature and exact-response rows are misaligned.")
    if {row.query_id for row in responses} != set(legal):
        raise ProtocolError("Every legal donor query must contribute exact responses.")
    grouped: dict[
        tuple[str, str], list[tuple[CaseActionFeatureRow, float, float]]
    ] = defaultdict(list)
    for key in sorted(feature_by_key):
        feature = feature_by_key[key]
        response = response_by_key[key]
        grouped[(feature.query_id, feature.case_id)].append(
            (
                feature,
                response.exact_bacc_gain_vs_control,
                response.exact_regret_from_case_best,
            )
        )

    pair_count_by_query: dict[str, int] = defaultdict(int)
    magnitude_by_query: dict[str, float] = defaultdict(float)
    for (query, _case_id), block in sorted(grouped.items()):
        case_best = max((gain for _feature, gain, _regret in block), default=0.0)
        case_best = max(case_best, 0.0)
        actions = sorted(
            (
                *((feature.action_id, gain, regret) for feature, gain, regret in block),
                (str(control_action_id), 0.0, case_best),
            ),
            key=lambda row: row[0],
        )
        if len({action for action, _gain, _regret in actions}) != len(actions):
            raise ProtocolError("Control identity collides with a modeled action.")
        for left, right in combinations(actions, 2):
            gain_difference = float(left[1] - right[1])
            regret_difference = float(left[2] - right[2])
            if not math.isclose(
                gain_difference,
                -regret_difference,
                rel_tol=1.0e-10,
                abs_tol=1.0e-14,
            ):
                raise ProtocolError("Exact gain and case-best regret ordering drifted.")
            if abs(regret_difference) <= PAIR_TOLERANCE:
                continue
            pair_count_by_query[query] += 1
            magnitude_by_query[query] += abs(regret_difference)
    informative = tuple(
        sorted(query for query, count in pair_count_by_query.items() if count)
    )
    if len(informative) < 3:
        raise ProtocolError("Exact utility produced fewer than three informative donor queries.")
    pair_count = sum(pair_count_by_query[query] for query in informative)
    action_tuple = tuple(sorted(str(action) for action in action_ids))
    feature_count = len(FEATURE_NAMES)
    design_dimension = (
        feature_count + len(action_tuple) + len(action_tuple) * feature_count
    )
    assert_dense_fit_within_budget(
        pair_count=pair_count,
        design_dimension=design_dimension,
        encoded_row_count=len(features),
    )
    encoder = fit_design_encoder(features, action_ids=action_tuple)
    encoded_by_key = {
        row.row_key: encoder.encode(row)
        for row in features
    }
    control = encoder.encode_control()
    values = np.empty((pair_count, encoder.dimension), dtype=np.float64)
    outcomes = np.empty(pair_count, dtype=np.float64)
    weights = np.empty(pair_count, dtype=np.float64)
    pair_queries: list[str] = []
    output_index = 0
    informative_set = set(informative)
    for (query, _case_id), block in sorted(grouped.items()):
        if query not in informative_set:
            continue
        case_best = max((gain for _feature, gain, _regret in block), default=0.0)
        case_best = max(case_best, 0.0)
        actions = sorted(
            (
                *(
                    (
                        feature.action_id,
                        encoded_by_key[feature.row_key],
                        gain,
                        regret,
                    )
                    for feature, gain, regret in block
                ),
                (str(control_action_id), control, 0.0, case_best),
            ),
            key=lambda row: row[0],
        )
        for left, right in combinations(actions, 2):
            regret_difference = float(left[3] - right[3])
            if abs(regret_difference) <= PAIR_TOLERANCE:
                continue
            magnitude_sum = magnitude_by_query[query]
            if magnitude_sum <= 0.0:
                raise ProtocolError("An informative query has zero pairwise magnitude.")
            values[output_index] = left[1] - right[1]
            outcomes[output_index] = 1.0 if regret_difference < 0.0 else 0.0
            weights[output_index] = (
                abs(regret_difference) / magnitude_sum / len(informative)
            )
            pair_queries.append(query)
            output_index += 1
    if output_index != pair_count:
        raise ProtocolError("Pre-counted pairwise design cardinality drifted.")
    return PairwiseTrainingDesign(
        encoder=encoder,
        values=values,
        outcomes=outcomes,
        weights=weights,
        query_ids=tuple(pair_queries),
        informative_query_ids=informative,
    )


__all__ = (
    "ACTION_L2_PENALTY",
    "PAIR_TOLERANCE",
    "SHARED_L2_PENALTY",
    "DesignEncoder",
    "PairwiseTrainingDesign",
    "build_pairwise_training_design",
    "fit_design_encoder",
)
