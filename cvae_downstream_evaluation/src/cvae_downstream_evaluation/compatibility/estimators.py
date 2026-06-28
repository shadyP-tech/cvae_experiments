"""Source-inner downstream utility estimators."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..protocol import ProtocolError
from ..features import deployable_feature_columns


MODEL_SCHEMA_VERSION = "source_inner_linear_utility_estimator_v1"


@dataclass(frozen=True)
class MeanUtilityEstimator:
    """A simple source-inner-only fallback estimator for smoke tests."""

    mean_utility: float

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, object]], *, label: str = "source_inner_heldout_bacc") -> "MeanUtilityEstimator":
        if not rows:
            raise ProtocolError("Cannot fit estimator with no source-inner rows.")
        values = [float(row[label]) for row in rows if label in row]
        if not values:
            raise ProtocolError(f"No source-inner label column {label!r} found.")
        return cls(mean_utility=sum(values) / float(len(values)))

    def predict_one(self, features: Mapping[str, object]) -> float:
        _ = features
        return float(self.mean_utility)


@dataclass(frozen=True)
class LinearUtilityEstimator:
    """Small ridge-linear estimator with source-inner-only supervision."""

    feature_columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    label: str
    ridge_lambda: float
    schema_version: str = MODEL_SCHEMA_VERSION

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, object]],
        *,
        feature_columns: Sequence[str],
        label: str = "source_inner_heldout_bacc",
        ridge_lambda: float = 1e-6,
    ) -> "LinearUtilityEstimator":
        if not rows:
            raise ProtocolError("Cannot fit linear estimator with no source-inner rows.")
        columns = deployable_feature_columns(feature_columns)
        if not columns:
            raise ProtocolError("Linear estimator requires at least one deployable feature column.")
        x_raw = [[_as_float(row.get(column, 0.0), column) for column in columns] for row in rows]
        y = [_as_float(row[label], label) for row in rows if label in row]
        if len(y) != len(rows):
            raise ProtocolError(f"Every source-inner row must contain label column {label!r}.")
        means = tuple(sum(values) / float(len(values)) for values in zip(*x_raw))
        scales = tuple(_std(values) or 1.0 for values in zip(*x_raw))
        x = [
            [(value - means[idx]) / scales[idx] for idx, value in enumerate(row)]
            for row in x_raw
        ]
        design = [[1.0] + row for row in x]
        beta = _solve_ridge(design, y, ridge_lambda=float(ridge_lambda))
        return cls(
            feature_columns=columns,
            coefficients=tuple(float(v) for v in beta[1:]),
            intercept=float(beta[0]),
            feature_means=means,
            feature_scales=scales,
            label=label,
            ridge_lambda=float(ridge_lambda),
        )

    def predict_one(self, features: Mapping[str, object]) -> float:
        total = float(self.intercept)
        for idx, column in enumerate(self.feature_columns):
            value = _as_float(features.get(column, 0.0), column)
            scaled = (value - self.feature_means[idx]) / self.feature_scales[idx]
            total += self.coefficients[idx] * scaled
        return float(total)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_columns": list(self.feature_columns),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "label": self.label,
            "ridge_lambda": self.ridge_lambda,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LinearUtilityEstimator":
        if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise ProtocolError(
                f"Unexpected estimator schema_version={payload.get('schema_version')!r}"
            )
        return cls(
            feature_columns=tuple(str(v) for v in payload.get("feature_columns", ())),
            coefficients=tuple(float(v) for v in payload.get("coefficients", ())),
            intercept=float(payload.get("intercept", 0.0)),
            feature_means=tuple(float(v) for v in payload.get("feature_means", ())),
            feature_scales=tuple(float(v) for v in payload.get("feature_scales", ())),
            label=str(payload.get("label", "source_inner_heldout_bacc")),
            ridge_lambda=float(payload.get("ridge_lambda", 0.0)),
        )


def save_estimator(path: Path, estimator: LinearUtilityEstimator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimator.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_estimator(path: Path) -> LinearUtilityEstimator:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Malformed estimator JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Estimator JSON must be an object: {path}")
    return LinearUtilityEstimator.from_payload(payload)


def predict_rows(
    estimator: LinearUtilityEstimator,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        copied = dict(row)
        copied["predicted_primary_utility"] = estimator.predict_one(row)
        out.append(copied)
    return out


def _solve_ridge(design: Sequence[Sequence[float]], y: Sequence[float], *, ridge_lambda: float) -> list[float]:
    n_cols = len(design[0])
    xtx = [[0.0 for _ in range(n_cols)] for _ in range(n_cols)]
    xty = [0.0 for _ in range(n_cols)]
    for row, target in zip(design, y):
        for i in range(n_cols):
            xty[i] += row[i] * target
            for j in range(n_cols):
                xtx[i][j] += row[i] * row[j]
    for idx in range(1, n_cols):
        xtx[idx][idx] += float(ridge_lambda)
    return _gaussian_solve(xtx, xty)


def _gaussian_solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    a = [list(row) + [float(vector[idx])] for idx, row in enumerate(matrix)]
    n = len(a)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ProtocolError("Linear estimator design matrix is singular; increase ridge_lambda.")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        pivot_value = a[col][col]
        a[col] = [value / pivot_value for value in a[col]]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            if factor == 0.0:
                continue
            a[row] = [value - factor * a[col][idx] for idx, value in enumerate(a[row])]
    return [a[row][-1] for row in range(n)]


def _as_float(value: object, column: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Feature/label column {column!r} must be numeric; got {value!r}") from exc


def _std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / float(len(vals))
    return (sum((value - mean) ** 2 for value in vals) / float(len(vals) - 1)) ** 0.5
