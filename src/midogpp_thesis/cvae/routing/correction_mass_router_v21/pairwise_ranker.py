"""Case-weighted relative donor comparisons and ranking."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import (
    CompositeKind,
    Direction,
    LabelFreeAction,
    LabelFreeCaseMenu,
    SupportActionOutcome,
    SupportCaseClassProfile,
    canonical_text,
)
from .hashing import canonical_hash

from .ranker_numerics import _DIRECTIONS, _case_weights, _solve_logistic_ridge
from .ranker_features import FittedFeatureTransform


@dataclass(frozen=True, slots=True)
class PairwiseComparison:
    center_id: str
    case_id: str
    direction: Direction
    left_arm_id: str
    right_arm_id: str
    feature_difference: tuple[float, ...]
    preference: float
    case_weight: float
    pair_count_in_case: int
    comparison_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ProtocolError("HARP v21 pairwise comparison direction is malformed.")
        values = tuple(float(value) for value in self.feature_difference)
        preference = float(self.preference)
        if (
            not values
            or any(not math.isfinite(value) for value in values)
            or preference not in (0.0, 0.5, 1.0)
            or not math.isfinite(self.case_weight)
            or self.case_weight <= 0.0
            or type(self.pair_count_in_case) is not int
            or self.pair_count_in_case < 1
        ):
            raise ProtocolError("HARP v21 pairwise comparison is malformed.")
        object.__setattr__(self, "feature_difference", values)
        object.__setattr__(self, "preference", preference)
        object.__setattr__(
            self,
            "comparison_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_grouped_pairwise_comparison_v21",
                    "center_id": self.center_id,
                    "case_id": self.case_id,
                    "direction": self.direction.value,
                    "left_arm_id": self.left_arm_id,
                    "right_arm_id": self.right_arm_id,
                    "feature_difference": values,
                    "preference": preference,
                    "case_weight": self.case_weight,
                    "pair_count_in_case": self.pair_count_in_case,
                }
            ),
        )


def build_pairwise_comparisons(
    menus: Sequence[LabelFreeCaseMenu],
    outcomes: Sequence[SupportActionOutcome],
    *,
    transform: FittedFeatureTransform,
) -> tuple[PairwiseComparison, ...]:
    menu_rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    outcome_by_action = {row.action.action_hash: row for row in outcomes}
    keys = tuple((row.center_id, row.case_id) for row in menu_rows)
    case_weights = _case_weights(keys)
    output: list[PairwiseComparison] = []
    for menu in menu_rows:
        groups: dict[Direction, tuple[LabelFreeAction, ...]] = {
            direction: menu.actions_for(direction) for direction in _DIRECTIONS
        }
        nonempty = tuple(direction for direction, actions in groups.items() if actions)
        if not nonempty:
            continue
        all_specs: list[tuple[Direction, str, str, np.ndarray, float]] = []
        for direction in nonempty:
            members: list[tuple[str, np.ndarray, float]] = [("B", np.zeros(len(transform.pairwise_design_names), dtype=np.float64), 0.0)]
            for action in groups[direction]:
                try:
                    outcome = outcome_by_action[action.action_hash]
                except KeyError as exc:
                    raise ProtocolError("HARP v21 pairwise source outcome coverage is incomplete.") from exc
                members.append((action.arm_id, transform.action_vector(action), outcome.bacc_gain))
            members.sort(key=lambda row: row[0])
            for left, right in combinations(members, 2):
                preference = 1.0 if left[2] > right[2] else 0.0 if left[2] < right[2] else 0.5
                all_specs.append((direction, left[0], right[0], left[1] - right[1], preference))
        pair_count = len(all_specs)
        if pair_count < 1:
            continue
        weight = case_weights[(menu.center_id, menu.case_id)] / pair_count
        output.extend(
            PairwiseComparison(
                center_id=menu.center_id,
                case_id=menu.case_id,
                direction=direction,
                left_arm_id=left_id,
                right_arm_id=right_id,
                feature_difference=tuple(float(value) for value in difference),
                preference=preference,
                case_weight=weight,
                pair_count_in_case=pair_count,
            )
            for direction, left_id, right_id, difference, preference in all_specs
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PairwiseRanker:
    coefficients: tuple[float, ...]
    ridge_alpha: float
    design_names: tuple[str, ...]
    transform_hash: str
    unique_comparison_count: int
    training_case_keys: tuple[tuple[str, str], ...]
    ranker_hash: str = field(init=False)

    def __post_init__(self) -> None:
        coefficients = tuple(float(value) for value in self.coefficients)
        if (
            not coefficients
            or len(coefficients) != len(self.design_names)
            or any(not math.isfinite(value) for value in coefficients)
            or self.ridge_alpha <= 0.0
            or self.unique_comparison_count < 0
        ):
            raise ProtocolError("HARP v21 pairwise ranker is malformed.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "ranker_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_grouped_bt_ranker_v21",
                    "coefficients": coefficients,
                    "ridge_alpha": self.ridge_alpha,
                    "design_names": self.design_names,
                    "transform_hash": self.transform_hash,
                    "unique_comparison_count": self.unique_comparison_count,
                    "training_case_keys": self.training_case_keys,
                    "duplicate_pairs_change_fit": False,
                    "deterministic_ties": "descending_score_then_arm_id_then_donor_id",
                }
            ),
        )

    def score(self, action: LabelFreeAction, transform: FittedFeatureTransform) -> float:
        return float(transform.action_vector(action) @ np.asarray(self.coefficients, dtype=np.float64))

    def public_payload(self) -> dict[str, object]:
        return {
            "coefficients": list(self.coefficients),
            "ridge_alpha": self.ridge_alpha,
            "design_names": list(self.design_names),
            "transform_hash": self.transform_hash,
            "unique_comparison_count": self.unique_comparison_count,
            "training_case_keys": [list(value) for value in self.training_case_keys],
            "ranker_hash": self.ranker_hash,
            "duplicate_pairs_change_fit": False,
            "deterministic_ties": "descending_score_then_arm_id_then_donor_id",
        }


def _deduplicate_comparisons(
    comparisons: Sequence[PairwiseComparison],
) -> tuple[PairwiseComparison, ...]:
    by_identity: dict[tuple[str, str, str, str, str], PairwiseComparison] = {}
    for row in comparisons:
        if not isinstance(row, PairwiseComparison):
            raise ProtocolError("HARP v21 pairwise fitting received an untyped row.")
        key = (row.center_id, row.case_id, row.direction.value, row.left_arm_id, row.right_arm_id)
        previous = by_identity.get(key)
        if previous is not None and previous.comparison_hash != row.comparison_hash:
            raise ProtocolError("HARP v21 duplicated pair identity has conflicting contents.")
        by_identity[key] = row
    return tuple(by_identity[key] for key in sorted(by_identity))


def fit_pairwise_ranker(
    comparisons: Sequence[PairwiseComparison],
    *,
    alpha: float,
    transform: FittedFeatureTransform,
) -> PairwiseRanker:
    rows = _deduplicate_comparisons(comparisons)
    if not math.isfinite(alpha) or alpha != 1.0:
        raise ProtocolError("HARP v21 pairwise ranker inputs are malformed.")
    if not rows:
        return PairwiseRanker(tuple(0.0 for _ in transform.pairwise_design_names),float(alpha),
                              transform.pairwise_design_names,transform.transform_hash,0,
                              transform.training_case_keys)
    matrix = np.asarray([row.feature_difference for row in rows], dtype=np.float64)
    response = np.asarray([row.preference for row in rows], dtype=np.float64)
    weights = np.asarray([row.case_weight for row in rows], dtype=np.float64)
    coefficients = _solve_logistic_ridge(
        matrix,
        response,
        weights,
        alpha=float(alpha),
        penalize_intercept=True,
    )
    return PairwiseRanker(
        coefficients=tuple(float(value) for value in coefficients),
        ridge_alpha=float(alpha),
        design_names=transform.pairwise_design_names,
        transform_hash=transform.transform_hash,
        unique_comparison_count=len(rows),
        training_case_keys=transform.training_case_keys,
    )
