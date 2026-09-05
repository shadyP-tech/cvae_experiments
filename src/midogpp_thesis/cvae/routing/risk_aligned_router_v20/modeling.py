"""Fold-local label-free transforms and grouped donor proposal ranker."""

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


_DIRECTIONS = (Direction.D01, Direction.D10)
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
        raise ProtocolError("HARP v20 ridge fit produced non-finite coefficients.")
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
        raise ProtocolError("HARP v20 logistic ridge produced non-finite coefficients.")
    return coefficients


def _case_weights(keys: Sequence[tuple[str, str]]) -> dict[tuple[str, str], float]:
    unique = tuple(sorted(set(keys)))
    centers = tuple(sorted({center for center, _ in unique}))
    cases_by_center = Counter(center for center, _ in unique)
    if not centers or any(cases_by_center[center] < 1 for center in centers):
        raise ProtocolError("HARP v20 equal-center case weights are undefined.")
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
            raise ProtocolError("HARP v20 fitted feature transform is malformed.")
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
                    "schema_version": "pooled_pairwise_feature_transform_v20",
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
    def pairwise_design_names(self) -> tuple[str, ...]:
        return (
            *(f"D01::{name}" for name in self.feature_names),
            *(f"D10::{name}" for name in self.feature_names),
            *(f"donor::{donor}" for donor in self.donor_ids),
        )

    def numeric(self, action: LabelFreeAction) -> np.ndarray:
        values = dict(zip(action.feature_names, action.feature_values, strict=True))
        if any(name not in values for name in self.feature_names):
            raise ProtocolError("HARP v20 action feature schema drifted from its fitted fold.")
        return (
            np.asarray([values[name] for name in self.feature_names], dtype=np.float64)
            - np.asarray(self.means, dtype=np.float64)
        ) / np.asarray(self.scales, dtype=np.float64)

    def action_vector(self, action: LabelFreeAction) -> np.ndarray:
        if action.direction not in _DIRECTIONS:
            raise ProtocolError("HARP v20 pairwise action has an unseen direction/donor.")
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
            "pairwise_design_names": list(self.pairwise_design_names),
            "transform_hash": self.transform_hash,
            "center_identity_is_not_a_feature": True,
            "center_case_equal_weighted": True,
        }


def fit_feature_transform(
    menus: Sequence[LabelFreeCaseMenu], *, maximum_numeric_features: int
) -> FittedFeatureTransform:
    rows = tuple(sorted(menus, key=lambda row: (row.center_id, row.case_id)))
    actions = tuple(action for menu in rows for action in menu.actions)
    if not rows or not actions or type(maximum_numeric_features) is not int or maximum_numeric_features < 1:
        raise ProtocolError("HARP v20 feature fitting requires directional source actions.")
    schemas = {action.feature_names for action in actions}
    if len(schemas) != 1:
        raise ProtocolError("HARP v20 source action feature schema is not singular.")
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
            raise ProtocolError("HARP v20 pairwise comparison direction is malformed.")
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
            raise ProtocolError("HARP v20 pairwise comparison is malformed.")
        object.__setattr__(self, "feature_difference", values)
        object.__setattr__(self, "preference", preference)
        object.__setattr__(
            self,
            "comparison_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_grouped_pairwise_comparison_v20",
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
                    raise ProtocolError("HARP v20 pairwise source outcome coverage is incomplete.") from exc
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
            raise ProtocolError("HARP v20 pairwise ranker is malformed.")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "ranker_hash",
            canonical_hash(
                {
                    "schema_version": "pooled_grouped_bt_ranker_v20",
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
            raise ProtocolError("HARP v20 pairwise fitting received an untyped row.")
        key = (row.center_id, row.case_id, row.direction.value, row.left_arm_id, row.right_arm_id)
        previous = by_identity.get(key)
        if previous is not None and previous.comparison_hash != row.comparison_hash:
            raise ProtocolError("HARP v20 duplicated pair identity has conflicting contents.")
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
        raise ProtocolError("HARP v20 pairwise ranker inputs are malformed.")
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



@dataclass(frozen=True, slots=True)
class CaseModelPrediction:
    menu_hash: str
    d01_ranked_action_ids: tuple[str, ...]
    d10_ranked_action_ids: tuple[str, ...]
    model_hash: str

    def public_payload(self) -> dict[str, object]:
        return {"menu_hash":self.menu_hash,"d01_ranked_action_ids":list(self.d01_ranked_action_ids),
                "d10_ranked_action_ids":list(self.d10_ranked_action_ids),"model_hash":self.model_hash,
                "proposal_only":True,"labels_consumed":False}


@dataclass(frozen=True, slots=True)
class ProposalModel:
    transform: FittedFeatureTransform
    ranker: PairwiseRanker
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.transform.transform_hash != self.ranker.transform_hash
            or self.transform.training_case_keys != self.ranker.training_case_keys):
            raise ProtocolError("HARP v20 proposal ranker/transform role binding drifted.")
        object.__setattr__(self,"model_hash",canonical_hash({
            "schema_version":"case_conditional_proposal_model_v20",
            "transform_hash":self.transform.transform_hash,"ranker_hash":self.ranker.ranker_hash,
            "opportunity_heads_used":False}))

    @property
    def training_case_keys(self) -> tuple[tuple[str,str], ...]:
        return self.transform.training_case_keys

    def predict_menu(self, menu: LabelFreeCaseMenu) -> CaseModelPrediction:
        if (menu.center_id,menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v20 honest proposal prediction includes a fitted case.")
        coefficients = np.asarray(self.ranker.coefficients,dtype=np.float64)
        rankings = []
        for direction in _DIRECTIONS:
            rows = menu.actions_for(direction)
            if not rows:
                rankings.append(())
                continue
            matrix = np.stack([self.transform.action_vector(a) for a in rows])
            scores = matrix @ coefficients
            order = sorted(range(len(rows)),key=lambda i:(-float(scores[i]),rows[i].arm_id,rows[i].donor_id))
            rankings.append(tuple(rows[i].arm_id for i in order))
        return CaseModelPrediction(menu.menu_hash,rankings[0],rankings[1],self.model_hash)

    def public_payload(self) -> dict[str, object]:
        return {"schema_version":"case_conditional_proposal_model_v20",
                "transform":self.transform.public_payload(),"ranker":self.ranker.public_payload(),
                "model_hash":self.model_hash,"training_case_keys":[list(k) for k in self.training_case_keys],
                "opportunity_heads_used":False}


def fit_proposal_model(menus: Sequence[LabelFreeCaseMenu], profiles: Sequence[SupportCaseClassProfile],
                       outcomes: Sequence[SupportActionOutcome], *, maximum_numeric_features: int = 20
                       ) -> ProposalModel:
    rows = tuple(menus)
    keys = tuple(sorted((m.center_id,m.case_id) for m in rows))
    if (keys != tuple(sorted((p.center_id,p.case_id) for p in profiles))
        or len(keys)!=len(set(keys)) or not keys):
        raise ProtocolError("HARP v20 proposal fitting role inventories differ.")
    menu_by_key = {(m.center_id,m.case_id):m for m in rows}
    expected = {a.action_hash for m in rows for a in m.actions}
    if ({o.action.action_hash for o in outcomes} != expected
        or len(outcomes)!=len(expected)
        or any(o.menu_hash != menu_by_key[(o.action.center_id,o.action.case_id)].menu_hash for o in outcomes)):
        raise ProtocolError("HARP v20 primitive outcomes drifted from fitted menu inventory.")
    from .aligned_metrics import ClassSupportNormalizer
    from dataclasses import replace
    norm = ClassSupportNormalizer.fit(profiles)
    if any(o.class_0_gain is None and o.class_1_gain is None for o in outcomes):
        raise ProtocolError("HARP v20 primitive training requires raw classwise recall deltas.")
    normalized = tuple(replace(o,bacc_gain=norm.contribution(o.action.center_id,o.class_0_gain,o.class_1_gain),
                               normalization_hash=norm.normalization_hash) for o in outcomes)
    transform = fit_feature_transform(rows,maximum_numeric_features=maximum_numeric_features)
    comparisons = build_pairwise_comparisons(rows,normalized,transform=transform)
    return ProposalModel(transform,fit_pairwise_ranker(comparisons,alpha=1.0,transform=transform))


# Public compatibility names identify the successor's proposal layer only.
PooledScienceModel = ProposalModel
fit_pooled_science_model = fit_proposal_model

__all__ = ("FittedFeatureTransform","PairwiseComparison","PairwiseRanker","ProposalModel",
           "PooledScienceModel","CaseModelPrediction","fit_feature_transform",
           "build_pairwise_comparisons","fit_pairwise_ranker","fit_proposal_model",
           "fit_pooled_science_model")
