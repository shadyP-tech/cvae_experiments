"""Center/case-balanced pairwise residual ranker with virtual baseline B."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .contracts import SourceActionOutcome
from .effective_menu import EffectiveMenu
from .hashing import canonical_hash
from .residuals import ResidualActionFeatures, residualize_menu


@dataclass(frozen=True, slots=True)
class ScaleOnlyTransform:
    names: tuple[str, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.names
            or len(self.names) != len(self.scale)
            or len(set(self.names)) != len(self.names)
            or any(not math.isfinite(value) or value <= 0.0 for value in self.scale)
        ):
            raise ProtocolError("HARP v10 scale-only transform is malformed.")

    def apply(self, values: Sequence[float]) -> np.ndarray:
        vector = np.asarray(tuple(values), dtype=np.float64)
        if vector.shape != (len(self.names),) or not np.all(np.isfinite(vector)):
            raise ProtocolError("HARP v10 rank feature vector is malformed.")
        return vector / np.asarray(self.scale, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PairwiseRanker:
    outer_target_id: str
    training_center_ids: tuple[str, ...]
    training_candidate_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    transform: ScaleOnlyTransform
    coefficients: tuple[float, ...]
    budget_width: int
    allocation_width: int
    pairwise_alpha: float
    residual_alpha: float
    pairwise_tie_tolerance: float
    training_pair_count: int
    ranker_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(sorted(self.training_center_ids))
        candidates = tuple(sorted(self.training_candidate_ids))
        excluded = tuple(sorted(self.excluded_center_ids))
        pairwise_alpha = float(self.pairwise_alpha)
        residual_alpha = float(self.residual_alpha)
        tolerance = float(self.pairwise_tie_tolerance)
        if (
            self.outer_target_id not in excluded
            or set(training) & set(excluded)
            or set(candidates) & set(excluded)
            or not training
            or len(self.coefficients) != len(self.transform.names)
            or self.budget_width + self.allocation_width + 5 != len(self.coefficients)
            or any(not math.isfinite(value) for value in self.coefficients)
            or not math.isfinite(pairwise_alpha)
            or not math.isfinite(residual_alpha)
            or pairwise_alpha <= 0.0
            or residual_alpha <= 0.0
            or not math.isfinite(tolerance)
            or tolerance < 0.0
            or self.training_pair_count < 1
        ):
            raise ProtocolError("HARP v10 pairwise ranker roles or parameters are malformed.")
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(self, "training_candidate_ids", candidates)
        object.__setattr__(self, "excluded_center_ids", excluded)
        object.__setattr__(self, "pairwise_alpha", pairwise_alpha)
        object.__setattr__(self, "residual_alpha", residual_alpha)
        object.__setattr__(self, "pairwise_tie_tolerance", tolerance)
        object.__setattr__(
            self,
            "ranker_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_pairwise_residual_ranker_v10",
                    "outer_target_id": self.outer_target_id,
                    "training_center_ids": training,
                    "training_candidate_ids": candidates,
                    "excluded_center_ids": excluded,
                    "transform": self.transform,
                    "coefficients": self.coefficients,
                    "budget_width": self.budget_width,
                    "allocation_width": self.allocation_width,
                    "pairwise_alpha": pairwise_alpha,
                    "residual_alpha": residual_alpha,
                    "pairwise_tie_tolerance": tolerance,
                    "training_pair_count": self.training_pair_count,
                    "virtual_B_zero_effect": True,
                    "center_case_pair_equal_weighting": True,
                    "target_evaluation_labels_used": False,
                }
            ),
        )

    def contributions(self, features: ResidualActionFeatures) -> tuple[float, float]:
        if features.names != self.transform.names:
            raise ProtocolError("HARP v10 rank feature schema drifted.")
        vector = self.transform.apply(features.values)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        budget_stop = self.budget_width
        allocation_stop = budget_stop + self.allocation_width
        # U and direction indicators are budget terms. HXE/reference terms are
        # allocation terms. Their sum is exactly the scalar pairwise score.
        budget_indices = tuple(range(budget_stop)) + (allocation_stop, allocation_stop + 2, allocation_stop + 3)
        allocation_indices = tuple(range(budget_stop, allocation_stop)) + (
            allocation_stop + 1,
            allocation_stop + 4,
        )
        budget = float(np.dot(coefficients[list(budget_indices)], vector[list(budget_indices)]))
        allocation = float(
            np.dot(coefficients[list(allocation_indices)], vector[list(allocation_indices)])
        )
        return budget, allocation

    def public_payload(self) -> dict[str, object]:
        return {
            "ranker_hash": self.ranker_hash,
            "outer_target_id": self.outer_target_id,
            "training_center_ids": list(self.training_center_ids),
            "training_candidate_ids": list(self.training_candidate_ids),
            "excluded_center_ids": list(self.excluded_center_ids),
            "feature_names": list(self.transform.names),
            "feature_scale": list(self.transform.scale),
            "coefficients": list(self.coefficients),
            "budget_width": self.budget_width,
            "allocation_width": self.allocation_width,
            "pairwise_alpha": self.pairwise_alpha,
            "residual_alpha": self.residual_alpha,
            "pairwise_tie_tolerance": self.pairwise_tie_tolerance,
            "training_pair_count": self.training_pair_count,
        }


@dataclass(frozen=True, slots=True)
class _TrainingCase:
    menu: EffectiveMenu
    outcomes: tuple[SourceActionOutcome, ...]


def fit_pairwise_ranker(
    cases: Sequence[_TrainingCase],
    *,
    excluded_center_ids: Sequence[str],
    pairwise_alpha: float,
    residual_alpha: float,
    pairwise_tie_tolerance: float,
) -> PairwiseRanker:
    typed = tuple(cases)
    if not typed:
        raise ProtocolError("HARP v10 pairwise fit has no source cases.")
    outer = typed[0].menu.outer_target_id
    excluded = tuple(sorted(set(str(value) for value in excluded_center_ids)))
    if outer not in excluded:
        raise ProtocolError("HARP v10 pairwise fit did not exclude outer H.")
    centers = tuple(sorted({case.menu.query_center_id for case in typed}))
    if not centers or set(centers) & set(excluded):
        raise ProtocolError("HARP v10 pairwise fit crossed an excluded query center.")
    residual_by_case: dict[tuple[str, str], tuple[ResidualActionFeatures, ...]] = {}
    all_vectors: list[np.ndarray] = []
    vector_weights: list[float] = []
    cases_per_center = {center: sum(case.menu.query_center_id == center for case in typed) for center in centers}
    for case in typed:
        residuals = residualize_menu(case.menu)
        residual_by_case[(case.menu.query_center_id, case.menu.case_id)] = residuals
        denom = len(centers) * cases_per_center[case.menu.query_center_id] * max(len(residuals) + 1, 1)
        all_vectors.append(np.zeros(len(residuals[0].values) if residuals else len(case.menu.feature_names) * 2 + 5))
        vector_weights.append(1.0 / denom)
        for row in residuals:
            all_vectors.append(np.asarray(row.values, dtype=np.float64))
            vector_weights.append(1.0 / denom)
    matrix = np.asarray(all_vectors, dtype=np.float64)
    weights = np.asarray(vector_weights, dtype=np.float64)
    rms = np.sqrt(
        np.sum(weights[:, None] * matrix * matrix, axis=0, dtype=np.float64)
        / np.sum(weights, dtype=np.float64)
    )
    rms[rms <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    names = residualize_menu(typed[0].menu)[0].names if typed[0].menu.actions else tuple(
        [*(f"budget__{name}" for name in typed[0].menu.feature_names), *(f"allocation__{name}" for name in typed[0].menu.feature_names), "kind__U", "kind__HXE", "direction__D01", "direction__D10", "allocation__has_uniform_reference"]
    )
    transform = ScaleOnlyTransform(names, tuple(float(value) for value in rms))
    pair_x: list[np.ndarray] = []
    pair_y: list[float] = []
    pair_w: list[float] = []
    for case in typed:
        residuals = residual_by_case[(case.menu.query_center_id, case.menu.case_id)]
        by_id = {row.action.action_id: row for row in case.outcomes}
        items: list[tuple[str, np.ndarray, float]] = [("B", np.zeros(len(names)), 0.0)]
        for residual in residuals:
            outcome = by_id.get(residual.action.action_id)
            if outcome is None:
                raise ProtocolError("HARP v10 pairwise outcome is absent from its sealed menu.")
            items.append(
                (residual.action.action_id, transform.apply(residual.values), outcome.bacc_gain)
            )
        raw_pairs: list[tuple[np.ndarray, float]] = []
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                delta = items[left][2] - items[right][2]
                if abs(delta) <= pairwise_tie_tolerance:
                    continue
                raw_pairs.append((items[left][1] - items[right][1], delta))
        if not raw_pairs:
            continue
        weight = 1.0 / (
            len(centers) * cases_per_center[case.menu.query_center_id] * len(raw_pairs)
        )
        for vector, response in raw_pairs:
            pair_x.append(vector)
            pair_y.append(response)
            pair_w.append(weight)
    if not pair_x:
        raise ProtocolError("HARP v10 pairwise surface has no non-tied contrasts.")
    x = np.asarray(pair_x, dtype=np.float64)
    y = np.asarray(pair_y, dtype=np.float64)
    w = np.asarray(pair_w, dtype=np.float64)
    normalized = w * (len(w) / np.sum(w, dtype=np.float64))
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(pairwise_alpha)
    allocation_start = len(typed[0].menu.feature_names)
    allocation_stop = allocation_start * 2
    allocation_indices = tuple(range(allocation_start, allocation_stop)) + (
        allocation_stop + 1,
        allocation_stop + 4,
    )
    penalty[list(allocation_indices), list(allocation_indices)] = float(residual_alpha)
    normal = x.T @ (normalized[:, None] * x) + penalty
    rhs = x.T @ (normalized * y)
    coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    candidates = tuple(
        sorted(
            {
                row.action.candidate_source_id
                for case in typed
                for row in case.outcomes
                if row.action.candidate_source_id is not None
            }
        )
    )
    if set(candidates) & set(excluded):
        raise ProtocolError("HARP v10 pairwise fit retained an excluded candidate center.")
    return PairwiseRanker(
        outer_target_id=outer,
        training_center_ids=centers,
        training_candidate_ids=candidates,
        excluded_center_ids=excluded,
        transform=transform,
        coefficients=tuple(float(value) for value in coefficients),
        budget_width=len(typed[0].menu.feature_names),
        allocation_width=len(typed[0].menu.feature_names),
        pairwise_alpha=float(pairwise_alpha),
        residual_alpha=float(residual_alpha),
        pairwise_tie_tolerance=float(pairwise_tie_tolerance),
        training_pair_count=len(pair_x),
    )


__all__ = ("PairwiseRanker", "ScaleOnlyTransform", "fit_pairwise_ranker")
