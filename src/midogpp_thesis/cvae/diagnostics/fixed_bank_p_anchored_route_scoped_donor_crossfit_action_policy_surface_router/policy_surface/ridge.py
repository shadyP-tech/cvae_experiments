"""Deterministic center/route/prefix-weighted policy ridge calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..identity import METRICS, RIDGE_ALPHA, canonical_hash
from .contracts import PolicyObservation, PrefixCell
from .descriptors import POLICY_FEATURE_NAMES, PolicyDescriptor, descriptor_for_metric


@dataclass(frozen=True)
class PolicyRidgeModel:
    metric: str
    alpha: float
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    intercept: float
    coefficients: tuple[float, ...]
    model_hash: str = field(init=False)

    def __post_init__(self) -> None:
        metric = str(self.metric)
        names = tuple(str(value) for value in self.feature_names)
        means = tuple(float(value) for value in self.feature_means)
        scales = tuple(float(value) for value in self.feature_scales)
        coefficients = tuple(float(value) for value in self.coefficients)
        values = np.asarray(
            (*means, *scales, self.intercept, *coefficients), dtype=np.float64
        )
        if (
            metric not in METRICS
            or float(self.alpha) != RIDGE_ALPHA
            or names != POLICY_FEATURE_NAMES
            or not (len(means) == len(scales) == len(coefficients) == len(names))
            or not np.isfinite(values).all()
            or any(value <= 0.0 for value in scales)
        ):
            raise ProtocolError("P-DCAPS policy ridge contract drifted.")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_means", means)
        object.__setattr__(self, "feature_scales", scales)
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "model_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_ridge_model_v1",
                    "metric": metric,
                    "alpha": self.alpha,
                    "feature_names": names,
                    "feature_means": means,
                    "feature_scales": scales,
                    "intercept": self.intercept,
                    "coefficients": coefficients,
                    "fit_only_standardization": True,
                    "hyperparameter_selection_used": False,
                }
            ),
        )

    def predict_descriptor(self, descriptor: PolicyDescriptor) -> float:
        if (
            descriptor.metric != self.metric
            or descriptor.feature_names != self.feature_names
        ):
            raise ProtocolError("P-DCAPS policy ridge descriptor lineage drifted.")
        values = descriptor.as_array()
        standardized = (
            values - np.asarray(self.feature_means, dtype=np.float64)
        ) / np.asarray(self.feature_scales, dtype=np.float64)
        result = self.intercept + float(
            standardized @ np.asarray(self.coefficients, dtype=np.float64)
        )
        if not np.isfinite(result):
            raise ProtocolError("P-DCAPS policy ridge prediction is nonfinite.")
        return float(result)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_ridge_model_v1",
            "metric": self.metric,
            "alpha": self.alpha,
            "feature_names": list(self.feature_names),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "fit_only_standardization": True,
            "hyperparameter_selection_used": False,
            "model_hash": self.model_hash,
        }


@dataclass(frozen=True)
class PolicyCalibration:
    outer_center: str
    scored_center: str | None
    excluded_centers: tuple[str, ...]
    supported_centers: tuple[str, ...]
    models: tuple[PolicyRidgeModel, ...]
    observation_hashes: tuple[str, ...]
    observation_weights: tuple[tuple[str, float], ...]
    additional_excluded_centers: tuple[str, ...] = ()
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = tuple(sorted({str(value) for value in self.excluded_centers}))
        supported = tuple(sorted({str(value) for value in self.supported_centers}))
        models = tuple(self.models)
        hashes = tuple(str(value) for value in self.observation_hashes)
        weights = tuple((str(key), float(value)) for key, value in self.observation_weights)
        additional = tuple(
            sorted({str(value) for value in self.additional_excluded_centers})
        )
        expected_excluded = {str(self.outer_center)}
        if self.scored_center is not None:
            expected_excluded.add(str(self.scored_center))
        expected_excluded.update(additional)
        values = np.asarray(tuple(value for _, value in weights), dtype=np.float64)
        if (
            set(excluded) != expected_excluded
            or set(excluded).intersection(supported)
            or tuple(model.metric for model in models) != METRICS
            or tuple(key for key, _ in weights) != hashes
            or len(set(hashes)) != len(hashes)
            or not hashes
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
            or abs(float(values.sum()) - 1.0) > 1.0e-12
        ):
            raise ProtocolError("P-DCAPS policy calibration topology drifted.")
        object.__setattr__(self, "outer_center", str(self.outer_center))
        object.__setattr__(
            self,
            "scored_center",
            None if self.scored_center is None else str(self.scored_center),
        )
        object.__setattr__(self, "excluded_centers", excluded)
        object.__setattr__(self, "supported_centers", supported)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "observation_hashes", hashes)
        object.__setattr__(self, "observation_weights", weights)
        object.__setattr__(self, "additional_excluded_centers", additional)
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_calibration_v1",
                    "outer_center": self.outer_center,
                    "scored_center": self.scored_center,
                    "excluded_centers": excluded,
                    "supported_centers": supported,
                    "model_hashes": tuple(model.model_hash for model in models),
                    "observation_hashes": hashes,
                    "observation_weights": weights,
                    "additional_excluded_centers": additional,
                    "hierarchical_weighting": "equal_center_then_route_then_prefix",
                }
            ),
        )

    def predict(self, cell: PrefixCell) -> FavorableUtility:
        # Exact P is a protected physical baseline, never a regression output.
        if cell.k == 0:
            return FavorableUtility.zeros()
        return FavorableUtility.from_array(
            tuple(
                model.predict_descriptor(descriptor_for_metric(cell, model.metric))
                for model in self.models
            )
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_calibration_v1",
            "outer_center": self.outer_center,
            "scored_center": self.scored_center,
            "excluded_centers": list(self.excluded_centers),
            "supported_centers": list(self.supported_centers),
            "models": [model.to_payload() for model in self.models],
            "observation_hashes": list(self.observation_hashes),
            "observation_weights": [list(row) for row in self.observation_weights],
            "additional_excluded_centers": list(
                self.additional_excluded_centers
            ),
            "hierarchical_weighting": "equal_center_then_route_then_prefix",
            "calibration_hash": self.calibration_hash,
        }


def equal_center_route_prefix_weights(
    observations: Sequence[PolicyObservation],
) -> np.ndarray:
    """Return weights with unit mass split center -> route -> prefix."""

    rows = tuple(observations)
    if not rows or len({row.observation_hash for row in rows}) != len(rows):
        raise ProtocolError("P-DCAPS policy observations are empty or duplicated.")
    centers = tuple(sorted({row.center for row in rows}))
    route_groups = {
        center: tuple(sorted({row.route_hash for row in rows if row.center == center}))
        for center in centers
    }
    weights: list[float] = []
    for row in rows:
        routes = route_groups[row.center]
        prefix_count = sum(
            candidate.center == row.center and candidate.route_hash == row.route_hash
            for candidate in rows
        )
        weights.append(1.0 / len(centers) / len(routes) / prefix_count)
    result = np.ascontiguousarray(weights, dtype=np.float64)
    if not np.isfinite(result).all() or abs(float(result.sum()) - 1.0) > 1.0e-12:
        raise ProtocolError("P-DCAPS hierarchical policy weights drifted.")
    result.setflags(write=False)
    return result


def fit_policy_calibration(
    observations: Sequence[PolicyObservation],
    *,
    outer_center: str,
    scored_center: str | None = None,
    additional_excluded_centers: Sequence[str] = (),
) -> PolicyCalibration:
    """Fit a final H model or a nested leave-J model.

    Every supplied observation must be a pseudo response whose action surface
    already excluded H and its own scored center.  Rows from J are filtered
    before fit when ``scored_center`` is provided.
    """

    all_rows = tuple(observations)
    outer = str(outer_center)
    scored = None if scored_center is None else str(scored_center)
    additional = tuple(
        sorted({str(value) for value in additional_excluded_centers})
    )
    if not all_rows or any(
        row.cell.provenance.surface_role != "pseudo"
        or row.cell.provenance.outer_center != outer
        or set(row.cell.provenance.excluded_centers)
        != {outer, row.center}
        for row in all_rows
    ):
        raise ProtocolError("P-DCAPS policy fit received invalid H/J responses.")
    excluded = {outer}
    if scored is not None:
        if scored == outer or scored not in {row.center for row in all_rows}:
            raise ProtocolError("P-DCAPS nested policy scored center drifted.")
        excluded.add(scored)
    if (
        len(additional) != len(tuple(additional_excluded_centers))
        or outer in additional
        or scored in additional
        or any(center not in {row.center for row in all_rows} for center in additional)
    ):
        raise ProtocolError("P-DCAPS policy additional exclusion drifted.")
    excluded.update(additional)
    rows = tuple(row for row in all_rows if row.center not in excluded)
    if not rows or any(row.center in excluded for row in rows):
        raise ProtocolError("P-DCAPS policy fit did not exclude H/J.")
    weights = equal_center_route_prefix_weights(rows)
    models = tuple(
        _fit_metric(rows, weights=weights, metric=metric) for metric in METRICS
    )
    return PolicyCalibration(
        outer,
        scored,
        tuple(sorted(excluded)),
        tuple(sorted({row.center for row in rows})),
        models,
        tuple(row.observation_hash for row in rows),
        tuple(
            (row.observation_hash, float(weight))
            for row, weight in zip(rows, weights, strict=True)
        ),
        additional,
    )


def _fit_metric(
    observations: tuple[PolicyObservation, ...],
    *,
    weights: np.ndarray,
    metric: str,
) -> PolicyRidgeModel:
    descriptors = tuple(descriptor_for_metric(row.cell, metric) for row in observations)
    x = np.ascontiguousarray(
        np.stack([row.as_array() for row in descriptors], axis=0), dtype=np.float64
    )
    metric_index = METRICS.index(metric)
    y = np.ascontiguousarray(
        [
            row.cell.realized_utility.as_tuple()[metric_index]  # type: ignore[union-attr]
            for row in observations
        ],
        dtype=np.float64,
    )
    if x.shape != (len(observations), len(POLICY_FEATURE_NAMES)):
        raise ProtocolError("P-DCAPS policy ridge matrix drifted.")
    total_weight = float(weights.sum())
    means = (weights[:, None] * x).sum(axis=0) / total_weight
    variances = (weights[:, None] * np.square(x - means)).sum(axis=0) / total_weight
    scales = np.sqrt(np.maximum(variances, 0.0))
    scales[scales <= np.finfo(np.float64).eps] = 1.0
    standardized = (x - means) / scales
    design = np.column_stack((np.ones(len(observations), dtype=np.float64), standardized))
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_y = y * sqrt_weights
    penalty = np.diag((0.0, *(RIDGE_ALPHA for _ in POLICY_FEATURE_NAMES)))
    normal = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_y
    try:
        solution = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    if not np.isfinite(solution).all():
        raise ProtocolError("P-DCAPS policy ridge fit is nonfinite.")
    return PolicyRidgeModel(
        metric,
        RIDGE_ALPHA,
        POLICY_FEATURE_NAMES,
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
        float(solution[0]),
        tuple(float(value) for value in solution[1:]),
    )


__all__ = (
    "PolicyCalibration",
    "PolicyRidgeModel",
    "equal_center_route_prefix_weights",
    "fit_policy_calibration",
)
