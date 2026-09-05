"""Pooled direction-first opportunity and grouped Bradley--Terry models."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Mapping, Sequence

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


_DIRECTIONS = (Direction.D01, Direction.D10)
_OPPORTUNITY_HEADS = (*_DIRECTIONS, Direction.FULL)
_SOLVER_ITERATIONS = 64
_SOLVER_TOLERANCE = 1.0e-10


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _solve_ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    penalize_intercept: bool,
) -> np.ndarray:
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    normal = matrix.T @ (weights[:, None] * matrix) + penalty
    right = matrix.T @ (weights * response)
    try:
        coefficients = np.linalg.solve(normal, right)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, right, rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise ProtocolError("HARP v17 ridge fit produced non-finite coefficients.")
    return coefficients


def _solve_logistic_ridge(
    matrix: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    penalize_intercept: bool,
) -> np.ndarray:
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)
    prevalence = float(np.dot(weights, response) / np.sum(weights, dtype=np.float64))
    prevalence = min(max(prevalence, 1.0e-6), 1.0 - 1.0e-6)
    if not penalize_intercept:
        coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * float(alpha)
    if not penalize_intercept:
        penalty[0, 0] = 0.0
    for _ in range(_SOLVER_ITERATIONS):
        probability = np.clip(_sigmoid(matrix @ coefficients), 1.0e-6, 1.0 - 1.0e-6)
        curvature = weights * probability * (1.0 - probability)
        gradient = matrix.T @ (weights * (response - probability)) - penalty @ coefficients
        hessian = matrix.T @ (curvature[:, None] * matrix) + penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficients += step
        if float(np.max(np.abs(step))) <= _SOLVER_TOLERANCE:
            break
    if not np.isfinite(coefficients).all():
        raise ProtocolError("HARP v17 logistic ridge produced non-finite coefficients.")
    return coefficients


def _case_weights(keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], float]:
    unique = tuple(sorted(set(keys)))
    centers = tuple(sorted({center for center, _ in unique}))
    cases_by_center = Counter(center for center, _ in unique)
    if not centers or any(cases_by_center[center] < 1 for center in centers):
        raise ProtocolError("HARP v17 equal-center case weights are undefined.")
    return {
        key: 1.0 / (len(centers) * cases_by_center[key[0]])
        for key in unique
    }


@dataclass(frozen=True, slots=True)
class FittedFeatureTransform:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    donor_ids: tuple[str, ...]
    training_case_keys: tuple[tuple[str, str], ...]
    transform_hash: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(canonical_text(value, name="model feature name") for value in self.feature_names)
        means = tuple(float(value) for value in self.means)
        scales = tuple(float(value) for value in self.scales)
        donors = tuple(sorted(canonical_text(value, name="model donor id") for value in self.donor_ids))
        keys = tuple(sorted((canonical_text(c, name="training center"), canonical_text(k, name="training case")) for c, k in self.training_case_keys))
        if (
            not names
            or len(names) != len(set(names))
            or len(names) != len(means)
            or len(names) != len(scales)
            or any(not math.isfinite(value) for value in (*means, *scales))
            or any(value <= 0.0 for value in scales)
            or len(donors) != len(set(donors))
            or not keys
            or len(keys) != len(set(keys))
        ):
            raise ProtocolError("HARP v17 fitted feature transform is malformed.")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "donor_ids", donors)
        object.__setattr__(self, "training_case_keys", keys)
        object.__setattr__(
            self,
            "transform_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_feature_transform_v17",
                    "feature_names": names,
                    "means": means,
                    "scales": scales,
                    "donor_ids": donors,
                    "training_case_keys": keys,
                    "center_identity_is_not_a_feature": True,
                    "center_case_equal_weighted": True,
                }
            ),
        )

    @property
    def opportunity_design_names(self) -> tuple[str, ...]:
        return ("intercept", *(f"mean::{name}" for name in self.feature_names), "log_action_count")

    @property
    def pairwise_design_names(self) -> tuple[str, ...]:
        return (
            *(f"D01::{name}" for name in self.feature_names),
            *(f"D10::{name}" for name in self.feature_names),
            *(f"donor::{donor}" for donor in self.donor_ids),
        )

    def numeric(self, action: LabelFreeAction) -> np.ndarray:
        values = dict(zip(action.feature_names, action.feature_values, strict=True))
        if any(name not in values for name in self.feature_names):
            raise ProtocolError("HARP v17 action feature schema drifted from its fitted fold.")
        return (
            np.asarray([values[name] for name in self.feature_names], dtype=np.float64)
            - np.asarray(self.means, dtype=np.float64)
        ) / np.asarray(self.scales, dtype=np.float64)

    def opportunity_vector(self, menu: LabelFreeCaseMenu, direction: Direction) -> np.ndarray:
        if direction not in _OPPORTUNITY_HEADS:
            raise ProtocolError("HARP v17 opportunity direction is malformed.")
        # The registered exact-U control is always represented, including when
        # it is byte-equal to B. Directional no-ops remain excluded.
        actions = menu.actions_for(
            direction,
            active_only=direction is not Direction.FULL,
        )
        if not actions:
            # A physical zero-frontier direction remains a deterministic all-zero
            # aggregate.  It cannot later satisfy a K arm.
            mean = np.zeros(len(self.feature_names), dtype=np.float64)
        else:
            mean = np.mean(
                np.asarray([self.numeric(action) for action in actions], dtype=np.float64),
                axis=0,
                dtype=np.float64,
            )
        return np.asarray([1.0, *mean.tolist(), math.log1p(len(actions))], dtype=np.float64)

    def action_vector(self, action: LabelFreeAction) -> np.ndarray:
        if action.direction not in _DIRECTIONS or action.donor_id not in self.donor_ids:
            raise ProtocolError("HARP v17 pairwise action has an unseen direction/donor.")
        numeric = self.numeric(action)
        zeros = np.zeros(len(numeric), dtype=np.float64)
        donor = np.asarray([float(action.donor_id == value) for value in self.donor_ids], dtype=np.float64)
        blocks = (numeric, zeros) if action.direction is Direction.D01 else (zeros, numeric)
        return np.concatenate((*blocks, donor)).astype(np.float64, copy=False)

    def public_payload(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "donor_ids": list(self.donor_ids),
            "training_case_keys": [list(value) for value in self.training_case_keys],
            "opportunity_design_names": list(self.opportunity_design_names),
            "pairwise_design_names": list(self.pairwise_design_names),
            "transform_hash": self.transform_hash,
            "center_identity_is_not_a_feature": True,
            "center_case_equal_weighted": True,
        }


def fit_feature_transform(
    menus: Sequence[LabelFreeCaseMenu], *, maximum_numeric_features: int
) -> FittedFeatureTransform:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    actions = tuple(action for menu in rows for action in menu.actions if action.direction in _DIRECTIONS)
    if not rows or not actions or type(maximum_numeric_features) is not int or maximum_numeric_features < 1:
        raise ProtocolError("HARP v17 feature fitting requires directional source actions.")
    schemas = {action.feature_names for action in actions}
    if len(schemas) != 1:
        raise ProtocolError("HARP v17 source action feature schema is not singular.")
    schema = next(iter(schemas))
    names = tuple(schema[:maximum_numeric_features])
    positions = {name: index for index, name in enumerate(schema)}
    keys = tuple((row.center_id, row.case_id) for row in rows)
    case_weight = _case_weights(keys)
    action_counts = Counter((action.center_id, action.case_id) for action in actions)
    weights = np.asarray(
        [case_weight[(action.center_id, action.case_id)] / action_counts[(action.center_id, action.case_id)] for action in actions],
        dtype=np.float64,
    )
    matrix = np.asarray(
        [[action.feature_values[positions[name]] for name in names] for action in actions],
        dtype=np.float64,
    )
    weight_sum = float(np.sum(weights, dtype=np.float64))
    means = np.sum(weights[:, None] * matrix, axis=0, dtype=np.float64) / weight_sum
    variance = np.sum(weights[:, None] * (matrix - means) ** 2, axis=0, dtype=np.float64) / weight_sum
    scales = np.sqrt(np.maximum(variance, 0.0))
    scales[scales <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    donors = tuple(sorted({action.donor_id for action in actions if action.donor_id is not None}))
    return FittedFeatureTransform(
        feature_names=names,
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        donor_ids=donors,
        training_case_keys=tuple(sorted(set(keys))),
    )


@dataclass(frozen=True, slots=True)
class DirectionOpportunityHead:
    direction: Direction
    logistic_coefficients: tuple[float, ...]
    gain_coefficients: tuple[float, ...]
    ridge_alpha: float
    training_case_keys: tuple[tuple[str, str], ...]
    transform_hash: str
    head_hash: str = field(init=False)

    def __post_init__(self) -> None:
        logistic = tuple(float(value) for value in self.logistic_coefficients)
        gain = tuple(float(value) for value in self.gain_coefficients)
        if (
            self.direction not in _OPPORTUNITY_HEADS
            or not logistic
            or len(logistic) != len(gain)
            or any(not math.isfinite(value) for value in (*logistic, *gain))
            or not math.isfinite(self.ridge_alpha)
            or self.ridge_alpha <= 0.0
        ):
            raise ProtocolError("HARP v17 direction opportunity head is malformed.")
        object.__setattr__(self, "logistic_coefficients", logistic)
        object.__setattr__(self, "gain_coefficients", gain)
        object.__setattr__(
            self,
            "head_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_action_opportunity_head_v17",
                    "direction": self.direction.value,
                    "logistic_coefficients": logistic,
                    "gain_coefficients": gain,
                    "ridge_alpha": self.ridge_alpha,
                    "training_case_keys": self.training_case_keys,
                    "transform_hash": self.transform_hash,
                    "independent_direction_head": True,
                    "outcome_target": (
                        "exact_u_positive_bacc_and_nonnegative_gain"
                        if self.direction is Direction.FULL
                        else "directional_error_opportunity_and_best_nonnegative_gain"
                    ),
                }
            ),
        )

    def predict(self, vector: np.ndarray) -> tuple[float, float]:
        logistic = np.asarray(self.logistic_coefficients, dtype=np.float64)
        gain = np.asarray(self.gain_coefficients, dtype=np.float64)
        if vector.shape != logistic.shape:
            raise ProtocolError("HARP v17 opportunity prediction shape drifted.")
        probability = float(_sigmoid(np.asarray([float(vector @ logistic)], dtype=np.float64))[0])
        predicted_gain = float(vector @ gain)
        return probability, predicted_gain

    def public_payload(self) -> dict[str, object]:
        return {
            "direction": self.direction.value,
            "logistic_coefficients": list(self.logistic_coefficients),
            "gain_coefficients": list(self.gain_coefficients),
            "ridge_alpha": self.ridge_alpha,
            "training_case_keys": [list(value) for value in self.training_case_keys],
            "transform_hash": self.transform_hash,
            "head_hash": self.head_hash,
            "independent_direction_head": True,
            "outcome_target": (
                "exact_u_positive_bacc_and_nonnegative_gain"
                if self.direction is Direction.FULL
                else "directional_error_opportunity_and_best_nonnegative_gain"
            ),
        }


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
            raise ProtocolError("HARP v17 pairwise comparison direction is malformed.")
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
            raise ProtocolError("HARP v17 pairwise comparison is malformed.")
        object.__setattr__(self, "feature_difference", values)
        object.__setattr__(self, "preference", preference)
        object.__setattr__(
            self,
            "comparison_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_grouped_pairwise_comparison_v17",
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
                    raise ProtocolError("HARP v17 pairwise source outcome coverage is incomplete.") from exc
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
    if not output:
        raise ProtocolError("HARP v17 pairwise fitting has no eligible comparisons.")
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
            or self.unique_comparison_count < 1
        ):
            raise ProtocolError("HARP v17 pairwise ranker is malformed.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "ranker_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_grouped_bt_ranker_v17",
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
            raise ProtocolError("HARP v17 pairwise fitting received an untyped row.")
        key = (row.center_id, row.case_id, row.direction.value, row.left_arm_id, row.right_arm_id)
        previous = by_identity.get(key)
        if previous is not None and previous.comparison_hash != row.comparison_hash:
            raise ProtocolError("HARP v17 duplicated pair identity has conflicting contents.")
        by_identity[key] = row
    return tuple(by_identity[key] for key in sorted(by_identity))


def fit_pairwise_ranker(
    comparisons: Sequence[PairwiseComparison],
    *,
    alpha: float,
    transform: FittedFeatureTransform,
) -> PairwiseRanker:
    rows = _deduplicate_comparisons(comparisons)
    if not rows or not math.isfinite(alpha) or alpha <= 0.0:
        raise ProtocolError("HARP v17 pairwise ranker inputs are malformed.")
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


def _fit_opportunity_heads(
    menus: Sequence[LabelFreeCaseMenu],
    profiles: Sequence[SupportCaseClassProfile],
    outcomes: Sequence[SupportActionOutcome],
    *,
    alpha: float,
    transform: FittedFeatureTransform,
) -> tuple[
    DirectionOpportunityHead,
    DirectionOpportunityHead,
    DirectionOpportunityHead,
]:
    profile_by_key = {(row.center_id, row.case_id): row for row in profiles}
    outcome_by_key: dict[
        tuple[str, str, Direction], list[SupportActionOutcome]
    ] = defaultdict(list)
    for row in outcomes:
        outcome_by_key[(row.action.center_id, row.action.case_id, row.action.direction)].append(row)
    keys = tuple((row.center_id, row.case_id) for row in menus)
    weights_by_key = _case_weights(keys)
    heads: list[DirectionOpportunityHead] = []
    for direction in _OPPORTUNITY_HEADS:
        matrix: list[np.ndarray] = []
        opportunity: list[float] = []
        gain: list[float] = []
        weights: list[float] = []
        for menu in menus:
            key = (menu.center_id, menu.case_id)
            try:
                profile = profile_by_key[key]
            except KeyError as exc:
                raise ProtocolError("HARP v17 opportunity profiles are incomplete.") from exc
            matrix.append(transform.opportunity_vector(menu, direction))
            candidates = outcome_by_key.get((menu.center_id, menu.case_id, direction), ())
            if direction is Direction.FULL:
                if (
                    len(candidates) != 1
                    or candidates[0].action.arm_id != menu.full_action.arm_id
                ):
                    raise ProtocolError(
                        "HARP v17 exact-U opportunity outcome coverage is incomplete."
                    )
                exact_u_gain = candidates[0].bacc_gain
                opportunity.append(float(exact_u_gain > 0.0))
                gain.append(max(0.0, exact_u_gain))
            else:
                opportunity.append(float(profile.has_opportunity(direction)))
                gain.append(max((0.0, *(row.bacc_gain for row in candidates))))
            weights.append(weights_by_key[key])
        design = np.asarray(matrix, dtype=np.float64)
        weight_array = np.asarray(weights, dtype=np.float64)
        logistic = _solve_logistic_ridge(
            design,
            np.asarray(opportunity, dtype=np.float64),
            weight_array,
            alpha=alpha,
            penalize_intercept=False,
        )
        ridge = _solve_ridge(
            design,
            np.asarray(gain, dtype=np.float64),
            weight_array,
            alpha=alpha,
            penalize_intercept=False,
        )
        heads.append(
            DirectionOpportunityHead(
                direction=direction,
                logistic_coefficients=tuple(float(value) for value in logistic),
                gain_coefficients=tuple(float(value) for value in ridge),
                ridge_alpha=float(alpha),
                training_case_keys=transform.training_case_keys,
                transform_hash=transform.transform_hash,
            )
        )
    return (heads[0], heads[1], heads[2])


@dataclass(frozen=True, slots=True)
class CaseModelPrediction:
    menu_hash: str
    d01_ranked_action_ids: tuple[str, ...]
    d10_ranked_action_ids: tuple[str, ...]
    d01_opportunity_probability: float
    d10_opportunity_probability: float
    u_full_opportunity_probability: float
    d01_predicted_gain: float
    d10_predicted_gain: float
    u_full_predicted_gain: float
    directional_route_score: float
    u_full_route_score: float
    route_score: float
    model_hash: str
    prediction_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.d01_opportunity_probability,
            self.d10_opportunity_probability,
            self.u_full_opportunity_probability,
            self.d01_predicted_gain,
            self.d10_predicted_gain,
            self.u_full_predicted_gain,
            self.directional_route_score,
            self.u_full_route_score,
            self.route_score,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.0 <= values[0] <= 1.0
            or not 0.0 <= values[1] <= 1.0
            or not 0.0 <= values[2] <= 1.0
            or any(value < 0.0 for value in values[6:])
            or values[8] != max(values[6], values[7])
        ):
            raise ProtocolError("HARP v17 case model prediction is malformed.")
        object.__setattr__(
            self,
            "prediction_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_case_prediction_v17",
                    "menu_hash": self.menu_hash,
                    "d01_ranked_action_ids": self.d01_ranked_action_ids,
                    "d10_ranked_action_ids": self.d10_ranked_action_ids,
                    "d01_opportunity_probability": values[0],
                    "d10_opportunity_probability": values[1],
                    "u_full_opportunity_probability": values[2],
                    "d01_predicted_gain": values[3],
                    "d10_predicted_gain": values[4],
                    "u_full_predicted_gain": values[5],
                    "directional_route_score": values[6],
                    "u_full_route_score": values[7],
                    "route_score": values[8],
                    "model_hash": self.model_hash,
                    "labels_consumed": False,
                    "selected_action_family_route_score": True,
                    "u_full_score_source": "explicit_exact_u_outcome_head",
                }
            ),
        )

    def route_score_for(self, kind: CompositeKind) -> float:
        """Return the score fitted for the requested action family."""

        if kind is CompositeKind.B:
            return 0.0
        if kind is CompositeKind.U_FULL:
            return self.u_full_route_score
        if kind is CompositeKind.SOFT_TOPK:
            return self.directional_route_score
        raise ProtocolError("HARP v17 requested action family is malformed.")

    def public_payload(self) -> dict[str, object]:
        return {
            "menu_hash": self.menu_hash,
            "d01_ranked_action_ids": list(self.d01_ranked_action_ids),
            "d10_ranked_action_ids": list(self.d10_ranked_action_ids),
            "d01_opportunity_probability": self.d01_opportunity_probability,
            "d10_opportunity_probability": self.d10_opportunity_probability,
            "u_full_opportunity_probability": self.u_full_opportunity_probability,
            "d01_predicted_gain": self.d01_predicted_gain,
            "d10_predicted_gain": self.d10_predicted_gain,
            "u_full_predicted_gain": self.u_full_predicted_gain,
            "directional_route_score": self.directional_route_score,
            "u_full_route_score": self.u_full_route_score,
            "route_score": self.route_score,
            "model_hash": self.model_hash,
            "prediction_hash": self.prediction_hash,
            "labels_consumed": False,
            "selected_action_family_route_score": True,
            "u_full_score_source": "explicit_exact_u_outcome_head",
        }


@dataclass(frozen=True, slots=True)
class PooledScienceModel:
    transform: FittedFeatureTransform
    d01_opportunity: DirectionOpportunityHead
    d10_opportunity: DirectionOpportunityHead
    u_full_opportunity: DirectionOpportunityHead
    ranker: PairwiseRanker
    opportunity_alpha: float
    ranker_alpha: float
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.d01_opportunity.direction is not Direction.D01
            or self.d10_opportunity.direction is not Direction.D10
            or self.u_full_opportunity.direction is not Direction.FULL
            or self.d01_opportunity.transform_hash != self.transform.transform_hash
            or self.d10_opportunity.transform_hash != self.transform.transform_hash
            or self.u_full_opportunity.transform_hash != self.transform.transform_hash
            or self.ranker.transform_hash != self.transform.transform_hash
        ):
            raise ProtocolError("HARP v17 pooled science model components drifted.")
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_pairwise_science_model_v17",
                    "transform_hash": self.transform.transform_hash,
                    "d01_head_hash": self.d01_opportunity.head_hash,
                    "d10_head_hash": self.d10_opportunity.head_hash,
                    "u_full_head_hash": self.u_full_opportunity.head_hash,
                    "ranker_hash": self.ranker.ranker_hash,
                    "opportunity_alpha": self.opportunity_alpha,
                    "ranker_alpha": self.ranker_alpha,
                    "pooled_known_center_fit": True,
                    "target_identity_is_not_a_feature": True,
                    "selected_action_family_route_score": True,
                    "u_full_score_source": "explicit_exact_u_outcome_head",
                }
            ),
        )

    @property
    def training_case_keys(self) -> tuple[tuple[str, str], ...]:
        return self.transform.training_case_keys

    def predict_menu(self, menu: LabelFreeCaseMenu) -> CaseModelPrediction:
        predictions: dict[Direction, tuple[float, float]] = {}
        rankings: dict[Direction, tuple[str, ...]] = {}
        for direction, head in (
            (Direction.D01, self.d01_opportunity),
            (Direction.D10, self.d10_opportunity),
        ):
            vector = self.transform.opportunity_vector(menu, direction)
            predictions[direction] = head.predict(vector)
            actions = menu.actions_for(direction)
            ranked = tuple(
                row.arm_id
                for row in sorted(
                    actions,
                    key=lambda row: (
                        -self.ranker.score(row, self.transform),
                        row.arm_id,
                        row.donor_id or "",
                    ),
                )
            )
            rankings[direction] = ranked
        p01, g01 = predictions[Direction.D01]
        p10, g10 = predictions[Direction.D10]
        u_probability, u_gain = self.u_full_opportunity.predict(
            self.transform.opportunity_vector(menu, Direction.FULL)
        )
        directional_score = max(p01 * max(g01, 0.0), p10 * max(g10, 0.0))
        u_full_score = u_probability * max(u_gain, 0.0)
        score = max(directional_score, u_full_score)
        return CaseModelPrediction(
            menu_hash=menu.menu_hash,
            d01_ranked_action_ids=rankings[Direction.D01],
            d10_ranked_action_ids=rankings[Direction.D10],
            d01_opportunity_probability=p01,
            d10_opportunity_probability=p10,
            u_full_opportunity_probability=u_probability,
            d01_predicted_gain=g01,
            d10_predicted_gain=g10,
            u_full_predicted_gain=u_gain,
            directional_route_score=directional_score,
            u_full_route_score=u_full_score,
            route_score=score,
            model_hash=self.model_hash,
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "transform": self.transform.public_payload(),
            "d01_opportunity": self.d01_opportunity.public_payload(),
            "d10_opportunity": self.d10_opportunity.public_payload(),
            "u_full_opportunity": self.u_full_opportunity.public_payload(),
            "ranker": self.ranker.public_payload(),
            "opportunity_alpha": self.opportunity_alpha,
            "ranker_alpha": self.ranker_alpha,
            "model_hash": self.model_hash,
            "pooled_known_center_fit": True,
            "target_identity_is_not_a_feature": True,
            "selected_action_family_route_score": True,
            "u_full_score_source": "explicit_exact_u_outcome_head",
        }


def fit_pooled_science_model(
    menus: Sequence[LabelFreeCaseMenu],
    profiles: Sequence[SupportCaseClassProfile],
    outcomes: Sequence[SupportActionOutcome],
    *,
    opportunity_alpha: float,
    ranker_alpha: float,
    maximum_numeric_features: int,
) -> PooledScienceModel:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    transform = fit_feature_transform(rows, maximum_numeric_features=maximum_numeric_features)
    heads = _fit_opportunity_heads(
        rows,
        profiles,
        outcomes,
        alpha=float(opportunity_alpha),
        transform=transform,
    )
    comparisons = build_pairwise_comparisons(rows, outcomes, transform=transform)
    ranker = fit_pairwise_ranker(comparisons, alpha=float(ranker_alpha), transform=transform)
    return PooledScienceModel(
        transform=transform,
        d01_opportunity=heads[0],
        d10_opportunity=heads[1],
        u_full_opportunity=heads[2],
        ranker=ranker,
        opportunity_alpha=float(opportunity_alpha),
        ranker_alpha=float(ranker_alpha),
    )


def component_validation_losses(
    model: PooledScienceModel,
    menus: Sequence[LabelFreeCaseMenu],
    profiles: Sequence[SupportCaseClassProfile],
    outcomes: Sequence[SupportActionOutcome],
) -> tuple[float, float]:
    """Return equal-center opportunity and BT log losses on held cases."""

    profile_by_key = {(row.center_id, row.case_id): row for row in profiles}
    outcome_by_key: dict[
        tuple[str, str, Direction], list[SupportActionOutcome]
    ] = defaultdict(list)
    for row in outcomes:
        outcome_by_key[(row.action.center_id, row.action.case_id, row.action.direction)].append(row)
    keys = tuple((row.center_id, row.case_id) for row in menus)
    case_weights = _case_weights(keys)
    opportunity_loss = 0.0
    for menu in menus:
        key = (menu.center_id, menu.case_id)
        profile = profile_by_key[key]
        for direction, head in (
            (Direction.D01, model.d01_opportunity),
            (Direction.D10, model.d10_opportunity),
            (Direction.FULL, model.u_full_opportunity),
        ):
            probability, predicted_gain = head.predict(
                model.transform.opportunity_vector(menu, direction)
            )
            candidates = outcome_by_key.get((*key, direction), ())
            if direction is Direction.FULL:
                if (
                    len(candidates) != 1
                    or candidates[0].action.arm_id != menu.full_action.arm_id
                ):
                    raise ProtocolError(
                        "HARP v17 held exact-U outcome coverage is incomplete."
                    )
                best_gain = max(0.0, candidates[0].bacc_gain)
                target = float(candidates[0].bacc_gain > 0.0)
            else:
                target = float(profile.has_opportunity(direction))
                best_gain = max((0.0, *(row.bacc_gain for row in candidates)))
            probability = min(max(probability, 1.0e-6), 1.0 - 1.0e-6)
            opportunity_loss += (1.0 / 3.0) * case_weights[key] * (
                -(target * math.log(probability) + (1.0 - target) * math.log1p(-probability))
                + (predicted_gain - best_gain) ** 2
            )
    comparisons = build_pairwise_comparisons(menus, outcomes, transform=model.transform)
    coefficients = np.asarray(model.ranker.coefficients, dtype=np.float64)
    pairwise_loss = 0.0
    for row in comparisons:
        probability = float(_sigmoid(np.asarray([np.dot(row.feature_difference, coefficients)], dtype=np.float64))[0])
        probability = min(max(probability, 1.0e-6), 1.0 - 1.0e-6)
        pairwise_loss += row.case_weight * (
            -(row.preference * math.log(probability) + (1.0 - row.preference) * math.log1p(-probability))
        )
    return float(opportunity_loss), float(pairwise_loss)


__all__ = (
    "CaseModelPrediction",
    "DirectionOpportunityHead",
    "FittedFeatureTransform",
    "PairwiseComparison",
    "PairwiseRanker",
    "PooledScienceModel",
    "build_pairwise_comparisons",
    "component_validation_losses",
    "fit_feature_transform",
    "fit_pairwise_ranker",
    "fit_pooled_science_model",
)
