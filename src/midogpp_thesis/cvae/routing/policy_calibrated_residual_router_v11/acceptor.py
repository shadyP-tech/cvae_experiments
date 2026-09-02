"""Cross-fitted selected-action acceptor for HARP v11."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256


SELECTION_FEATURE_NAMES = (
    "selected__pairwise_score",
    "selected__budget_gain",
    "selected__allocation_gain",
    "selected__rank_margin",
    "menu__score_mean",
    "menu__score_std",
    "menu__score_max_abs",
    "menu__physical_action_count",
    "selected__is_U",
    "selected__is_HXE",
    "selected__direction_D01",
    "selected__direction_D10",
)


@dataclass(frozen=True, slots=True)
class SelectedActionObservation:
    outer_target_id: str
    query_center_id: str
    case_id: str
    selected_action_id: str
    feature_values: tuple[float, ...]
    bacc_gain: float
    brier_delta: float
    log_delta: float
    selection_excluded_center_ids: tuple[str, ...]
    selection_ranker_hash: str
    record_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.feature_values)
        excluded = tuple(sorted(self.selection_excluded_center_ids))
        if (
            len(values) != len(SELECTION_FEATURE_NAMES)
            or any(not math.isfinite(value) for value in values)
            or self.outer_target_id not in excluded
            or self.query_center_id not in excluded
            or len(set(excluded)) != len(excluded)
            or self.selected_action_id == ""
        ):
            raise ProtocolError("HARP v11 selected-action observation leaked or is malformed.")
        for value in (self.bacc_gain, self.brier_delta, self.log_delta):
            if not math.isfinite(float(value)):
                raise ProtocolError("HARP v11 selected-action endpoint is non-finite.")
        ranker_hash = require_sha256(self.selection_ranker_hash, name="selection ranker hash")
        object.__setattr__(self, "feature_values", values)
        object.__setattr__(self, "selection_excluded_center_ids", excluded)
        object.__setattr__(self, "selection_ranker_hash", ranker_hash)
        object.__setattr__(
            self,
            "record_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_selected_action_observation_v11",
                    "outer_target_id": self.outer_target_id,
                    "query_center_id": self.query_center_id,
                    "case_id": self.case_id,
                    "selected_action_id": self.selected_action_id,
                    "feature_values": values,
                    "bacc_gain": self.bacc_gain,
                    "brier_delta": self.brier_delta,
                    "log_delta": self.log_delta,
                    "selection_excluded_center_ids": excluded,
                    "selection_ranker_hash": ranker_hash,
                    "selection_outcome_disjoint": True,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class Standardizer:
    names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            self.names != SELECTION_FEATURE_NAMES
            or len(self.mean) != len(self.names)
            or len(self.scale) != len(self.names)
            or any(not math.isfinite(value) for value in (*self.mean, *self.scale))
            or any(value <= 0.0 for value in self.scale)
        ):
            raise ProtocolError("HARP v11 acceptor standardizer is malformed.")

    def apply(self, values: Sequence[float]) -> np.ndarray:
        vector = np.asarray(tuple(values), dtype=np.float64)
        return (vector - np.asarray(self.mean)) / np.asarray(self.scale)


@dataclass(frozen=True, slots=True)
class LinearHead:
    intercept: float
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(SELECTION_FEATURE_NAMES) or any(
            not math.isfinite(value) for value in (self.intercept, *self.coefficients)
        ):
            raise ProtocolError("HARP v11 acceptor head is malformed.")

    def linear(self, vector: np.ndarray) -> float:
        return float(self.intercept + np.dot(np.asarray(self.coefficients), vector))

    def probability(self, vector: np.ndarray) -> float:
        value = max(-30.0, min(30.0, self.linear(vector)))
        return float(1.0 / (1.0 + math.exp(-value)))


@dataclass(frozen=True, slots=True)
class SelectedActionAcceptor:
    outer_target_id: str
    training_center_ids: tuple[str, ...]
    excluded_center_ids: tuple[str, ...]
    standardizer: Standardizer
    beneficial_head: LinearHead
    harm_head: LinearHead
    gain_head: LinearHead
    brier_head: LinearHead
    log_head: LinearHead
    ridge_alpha: float
    max_brier_delta: float
    max_log_delta: float
    training_record_hashes: tuple[str, ...]
    acceptor_hash: str = field(init=False)

    def __post_init__(self) -> None:
        training = tuple(sorted(self.training_center_ids))
        excluded = tuple(sorted(self.excluded_center_ids))
        if (
            self.outer_target_id not in excluded
            or set(training) & set(excluded)
            or not training
            or not self.training_record_hashes
            or self.ridge_alpha <= 0.0
            or not all(math.isfinite(value) for value in (self.ridge_alpha, self.max_brier_delta, self.max_log_delta))
        ):
            raise ProtocolError("HARP v11 selected-action acceptor roles are malformed.")
        object.__setattr__(self, "training_center_ids", training)
        object.__setattr__(self, "excluded_center_ids", excluded)
        object.__setattr__(
            self,
            "acceptor_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_selected_action_acceptor_v11",
                    "outer_target_id": self.outer_target_id,
                    "training_center_ids": training,
                    "excluded_center_ids": excluded,
                    "standardizer": self.standardizer,
                    "beneficial_head": self.beneficial_head,
                    "harm_head": self.harm_head,
                    "gain_head": self.gain_head,
                    "brier_head": self.brier_head,
                    "log_head": self.log_head,
                    "ridge_alpha": self.ridge_alpha,
                    "max_brier_delta": self.max_brier_delta,
                    "max_log_delta": self.max_log_delta,
                    "training_record_hashes": self.training_record_hashes,
                    "selected_actions_generated_strictly_oof": True,
                    "target_evaluation_labels_used": False,
                }
            ),
        )

    def predict(self, features: Sequence[float]) -> tuple[float, float, float, float, float]:
        vector = self.standardizer.apply(features)
        return (
            self.beneficial_head.probability(vector),
            self.harm_head.probability(vector),
            self.gain_head.linear(vector),
            self.brier_head.linear(vector),
            self.log_head.linear(vector),
        )

    def public_payload(self) -> dict[str, object]:
        def head(value: LinearHead) -> dict[str, object]:
            return {"intercept": value.intercept, "coefficients": list(value.coefficients)}

        return {
            "acceptor_hash": self.acceptor_hash,
            "outer_target_id": self.outer_target_id,
            "training_center_ids": list(self.training_center_ids),
            "excluded_center_ids": list(self.excluded_center_ids),
            "feature_names": list(self.standardizer.names),
            "feature_mean": list(self.standardizer.mean),
            "feature_scale": list(self.standardizer.scale),
            "beneficial_head": head(self.beneficial_head),
            "harm_head": head(self.harm_head),
            "gain_head": head(self.gain_head),
            "brier_head": head(self.brier_head),
            "log_head": head(self.log_head),
            "ridge_alpha": self.ridge_alpha,
            "max_brier_delta": self.max_brier_delta,
            "max_log_delta": self.max_log_delta,
            "training_record_hashes": list(self.training_record_hashes),
        }


def selected_action_features(
    *,
    selected_score: float,
    budget_gain: float,
    allocation_gain: float,
    rank_margin: float,
    all_scores: Sequence[float],
    action_kind: str,
    direction: str,
) -> tuple[float, ...]:
    values = np.asarray(tuple(all_scores), dtype=np.float64)
    if values.size == 0:
        values = np.asarray((0.0,), dtype=np.float64)
    output = (
        float(selected_score),
        float(budget_gain),
        float(allocation_gain),
        float(rank_margin),
        float(np.mean(values, dtype=np.float64)),
        float(np.std(values, dtype=np.float64)),
        float(np.max(np.abs(values))),
        float(len(all_scores)),
        float(action_kind == "U"),
        float(action_kind == "HXE"),
        float(direction == "D01"),
        float(direction == "D10"),
    )
    if any(not math.isfinite(value) for value in output):
        raise ProtocolError("HARP v11 selected-action feature vector is non-finite.")
    return output


def _weights(rows: Sequence[SelectedActionObservation]) -> np.ndarray:
    cases_per_center: dict[str, int] = defaultdict(int)
    for row in rows:
        cases_per_center[row.query_center_id] += 1
    centers = len(cases_per_center)
    return np.asarray(
        [1.0 / (centers * cases_per_center[row.query_center_id]) for row in rows],
        dtype=np.float64,
    )


def _standardize(matrix: np.ndarray, weights: np.ndarray) -> tuple[Standardizer, np.ndarray]:
    normalized = weights / np.sum(weights, dtype=np.float64)
    mean = np.sum(normalized[:, None] * matrix, axis=0, dtype=np.float64)
    variance = np.sum(normalized[:, None] * (matrix - mean) ** 2, axis=0, dtype=np.float64)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= math.sqrt(np.finfo(np.float64).eps)] = 1.0
    standardizer = Standardizer(
        names=SELECTION_FEATURE_NAMES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
    )
    return standardizer, (matrix - mean) / scale


def _ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> LinearHead:
    design = np.column_stack((np.ones(x.shape[0]), x))
    normalized = w * (len(w) / np.sum(w, dtype=np.float64))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.lstsq(
        design.T @ (normalized[:, None] * design) + penalty,
        design.T @ (normalized * y),
        rcond=None,
    )[0]
    return LinearHead(float(coefficients[0]), tuple(float(value) for value in coefficients[1:]))


def _logistic(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float, max_iterations: int
) -> LinearHead:
    normalized = w * (len(w) / np.sum(w, dtype=np.float64))
    if np.all(y == y[0]):
        positive = float(np.sum(normalized * y, dtype=np.float64))
        probability = (positive + 0.5) / (float(np.sum(normalized)) + 1.0)
        return LinearHead(
            math.log(probability / (1.0 - probability)),
            tuple(0.0 for _ in range(x.shape[1])),
        )
    design = np.column_stack((np.ones(x.shape[0]), x))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        curvature = np.maximum(probability * (1.0 - probability), 1e-6)
        combined = normalized * curvature
        adjusted = linear + (y - probability) / curvature
        updated = np.linalg.lstsq(
            design.T @ (combined[:, None] * design) + penalty,
            design.T @ (combined * adjusted),
            rcond=None,
        )[0]
        if float(np.max(np.abs(updated - coefficients))) <= 1e-10:
            coefficients = updated
            break
        coefficients = updated
    return LinearHead(float(coefficients[0]), tuple(float(value) for value in coefficients[1:]))


def fit_selected_action_acceptor(
    observations: Sequence[SelectedActionObservation],
    *,
    excluded_center_ids: Sequence[str],
    ridge_alpha: float,
    max_brier_delta: float,
    max_log_delta: float,
    max_irls_iterations: int,
) -> SelectedActionAcceptor:
    rows = tuple(observations)
    if not rows:
        raise ProtocolError("HARP v11 acceptor has no cross-fitted selected actions.")
    outer = rows[0].outer_target_id
    if any(row.outer_target_id != outer for row in rows):
        raise ProtocolError("HARP v11 acceptor crossed outer targets.")
    excluded = tuple(sorted(set(str(value) for value in excluded_center_ids)))
    centers = tuple(sorted({row.query_center_id for row in rows}))
    if outer not in excluded or set(centers) & set(excluded):
        raise ProtocolError("HARP v11 acceptor retained an excluded center.")
    # Every selection was generated by a ranker that excluded its own query
    # center.  Outcomes are attached only after that selection is frozen.
    if any(row.query_center_id not in row.selection_excluded_center_ids for row in rows):
        raise ProtocolError("HARP v11 acceptor selection/outcome surfaces are not disjoint.")
    matrix = np.asarray([row.feature_values for row in rows], dtype=np.float64)
    weights = _weights(rows)
    standardizer, x = _standardize(matrix, weights)
    gain = np.asarray([row.bacc_gain for row in rows], dtype=np.float64)
    brier = np.asarray([row.brier_delta for row in rows], dtype=np.float64)
    log_delta = np.asarray([row.log_delta for row in rows], dtype=np.float64)
    beneficial = np.asarray(
        [
            row.bacc_gain > 0.0
            and row.brier_delta <= max_brier_delta
            and row.log_delta <= max_log_delta
            for row in rows
        ],
        dtype=np.float64,
    )
    harm = np.asarray([row.bacc_gain < 0.0 for row in rows], dtype=np.float64)
    return SelectedActionAcceptor(
        outer_target_id=outer,
        training_center_ids=centers,
        excluded_center_ids=excluded,
        standardizer=standardizer,
        beneficial_head=_logistic(x, beneficial, weights, ridge_alpha, max_irls_iterations),
        harm_head=_logistic(x, harm, weights, ridge_alpha, max_irls_iterations),
        gain_head=_ridge(x, gain, weights, ridge_alpha),
        brier_head=_ridge(x, brier, weights, ridge_alpha),
        log_head=_ridge(x, log_delta, weights, ridge_alpha),
        ridge_alpha=float(ridge_alpha),
        max_brier_delta=float(max_brier_delta),
        max_log_delta=float(max_log_delta),
        training_record_hashes=tuple(sorted(row.record_hash for row in rows)),
    )


__all__ = (
    "SELECTION_FEATURE_NAMES",
    "SelectedActionAcceptor",
    "SelectedActionObservation",
    "fit_selected_action_acceptor",
    "selected_action_features",
)
