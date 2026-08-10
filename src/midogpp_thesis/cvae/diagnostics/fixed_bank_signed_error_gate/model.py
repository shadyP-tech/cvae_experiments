"""Strict outer-center/nested-query ridge fits for a single signed correction."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from ..fixed_bank_hierarchical_residual_stacker.core_hashing import canonical_hash
from ..fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    MIDOGPP_CENTERS,
    candidate_sources,
)
from .constants import (
    FEATURE_NAMES,
    GLOBAL_FEATURE_INDICES,
    MAX_ABSOLUTE_CORRECTION_LOGIT,
    MIN_NESTED_MODELS,
    RIDGE_ALPHA_GRID,
    STANDARDIZATION_SCALE_FLOOR,
    UNCERTAINTY_Z,
)
from .contracts import (
    CorrectionRow,
    GradientTargetRow,
    SignedFeatureRow,
    SignedGateModel,
    Standardization,
)


@dataclass(frozen=True)
class NestedSignedGateModel:
    heldout_query_center: str
    model: SignedGateModel


@dataclass(frozen=True)
class SignedGateFit:
    final_model: SignedGateModel
    nested_models: tuple[NestedSignedGateModel, ...]
    validation_mse_by_alpha: tuple[tuple[float, float], ...]
    fit_hash: str = field(init=False)

    def __post_init__(self) -> None:
        nested = tuple(self.nested_models)
        path = tuple(
            (float(alpha), float(mse)) for alpha, mse in self.validation_mse_by_alpha
        )
        expected_queries = tuple(
            center
            for center in MIDOGPP_CENTERS
            if center != self.final_model.target_center
        )
        if (
            tuple(value.heldout_query_center for value in nested) != expected_queries
            or tuple(alpha for alpha, _mse in path) != RIDGE_ALPHA_GRID
            or any(not math.isfinite(mse) or mse < 0.0 for _alpha, mse in path)
            or any(
                value.model.target_center != self.final_model.target_center
                or value.model.family != self.final_model.family
                or value.model.ridge_alpha != self.final_model.ridge_alpha
                or set(value.model.donor_centers)
                != set(expected_queries).difference((value.heldout_query_center,))
                for value in nested
            )
        ):
            raise ProtocolError("Signed-gate fit path or nested provenance drifted.")
        object.__setattr__(self, "nested_models", nested)
        object.__setattr__(self, "validation_mse_by_alpha", path)
        object.__setattr__(
            self,
            "fit_hash",
            canonical_hash(
                {
                    "schema_version": "fixed_bank_signed_error_fit_v1",
                    "final_model_hash": self.final_model.model_hash,
                    "nested_models": [
                        {
                            "heldout_query_center": value.heldout_query_center,
                            "model_hash": value.model.model_hash,
                        }
                        for value in nested
                    ],
                    "validation_mse_by_alpha": [list(value) for value in path],
                    "alpha_tie_break": "larger_alpha",
                }
            ),
        )


def fit_signed_gate(
    features: Sequence[SignedFeatureRow],
    gradients: Sequence[GradientTargetRow],
    *,
    target_center: str,
    family: str,
    nested_training_features: Mapping[str, Sequence[SignedFeatureRow]],
    alpha_grid: Sequence[float] = RIDGE_ALPHA_GRID,
) -> SignedGateFit:
    """Fit without any target-center labels; choose alpha by donor-center OOF."""

    if target_center not in MIDOGPP_CENTERS or family not in ("G", "R", "P"):
        raise ProtocolError("Unknown signed-gate target center or model family.")
    feature_by_key = {row.sample_key: row for row in features}
    gradient_by_key = {row.sample_key: row for row in gradients}
    if len(feature_by_key) != len(tuple(features)) or len(gradient_by_key) != len(
        tuple(gradients)
    ):
        raise ProtocolError("Signed-gate fit inputs contain duplicate sample keys.")
    donor_centers = tuple(center for center in MIDOGPP_CENTERS if center != target_center)
    if (
        {key[0] for key in gradient_by_key} != set(donor_centers)
        or any(key[0] == target_center for key in gradient_by_key)
        or not set(gradient_by_key).issubset(feature_by_key)
    ):
        raise ProtocolError("Signed-gate gradients violate strict outer-center exclusion.")
    if any(
        row.context_excluded_centers != (target_center,)
        for row in feature_by_key.values()
    ):
        raise ProtocolError("Final signed-gate features lack the exact outer-H context.")
    nested_by_query = {
        str(query): {row.sample_key: row for row in rows}
        for query, rows in nested_training_features.items()
    }
    if set(nested_by_query) != set(donor_centers):
        raise ProtocolError("Signed-gate nested feature contexts are incomplete.")
    for query, rows in nested_by_query.items():
        expected_exclusions = tuple(sorted((target_center, query)))
        if (
            len(rows) != len(tuple(nested_training_features[query]))
            or not set(gradient_by_key).issubset(rows)
            or any(row.context_excluded_centers != expected_exclusions for row in rows.values())
        ):
            raise ProtocolError("Nested signed-gate features violate H/q candidate exclusion.")
    grid = tuple(float(value) for value in alpha_grid)
    if grid != RIDGE_ALPHA_GRID:
        raise ProtocolError("Signed-gate alpha selection left the frozen grid.")
    validation_scores: list[tuple[float, float]] = []
    for alpha in grid:
        squared_errors: list[float] = []
        for heldout_query in donor_centers:
            training_centers = tuple(
                center for center in donor_centers if center != heldout_query
            )
            nested = _fit_model(
                nested_by_query[heldout_query],
                gradient_by_key,
                target_center=target_center,
                family=family,
                ridge_alpha=alpha,
                donor_centers=training_centers,
                nested_model_hashes=(),
            )
            for key, gradient in gradient_by_key.items():
                if key[0] == heldout_query:
                    prediction = predict_one(
                        nested, nested_by_query[heldout_query][key]
                    )
                    squared_errors.append(
                        (prediction - gradient.negative_log_loss_gradient) ** 2
                    )
        if not squared_errors:
            raise ProtocolError("Nested signed-gate alpha selection has no validation rows.")
        validation_scores.append((alpha, math.fsum(squared_errors) / len(squared_errors)))
    selected_alpha = min(validation_scores, key=lambda item: (item[1], -item[0]))[0]
    nested_models = tuple(
        NestedSignedGateModel(
            heldout_query,
            _fit_model(
                nested_by_query[heldout_query],
                gradient_by_key,
                target_center=target_center,
                family=family,
                ridge_alpha=selected_alpha,
                donor_centers=tuple(
                    center for center in donor_centers if center != heldout_query
                ),
                nested_model_hashes=(),
            ),
        )
        for heldout_query in donor_centers
    )
    final = _fit_model(
        feature_by_key,
        gradient_by_key,
        target_center=target_center,
        family=family,
        ridge_alpha=selected_alpha,
        donor_centers=donor_centers,
        nested_model_hashes=tuple(value.model.model_hash for value in nested_models),
    )
    return SignedGateFit(final, nested_models, tuple(validation_scores))


def _fit_model(
    features: dict[tuple[str, str, str], SignedFeatureRow],
    gradients: dict[tuple[str, str, str], GradientTargetRow],
    *,
    target_center: str,
    family: str,
    ridge_alpha: float,
    donor_centers: tuple[str, ...],
    nested_model_hashes: tuple[str, ...],
) -> SignedGateModel:
    keys = tuple(sorted(key for key in features if key[0] in donor_centers))
    if not keys:
        raise ProtocolError("Signed-gate model has no legal donor rows.")
    raw = np.asarray([features[key].values for key in keys], dtype=np.float64)
    target = np.asarray(
        [gradients[key].negative_log_loss_gradient for key in keys], dtype=np.float64
    )
    standardization = _fit_standardization(raw)
    design = _standardize(raw, standardization)
    if family == "G":
        disabled = tuple(
            index
            for index in range(len(FEATURE_NAMES))
            if index not in GLOBAL_FEATURE_INDICES
        )
        design[:, disabled] = 0.0
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_alpha)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs, rcond=1.0e-12) @ rhs
    if family == "G":
        for index in range(len(FEATURE_NAMES)):
            if index not in GLOBAL_FEATURE_INDICES:
                coefficients[index] = 0.0
    return SignedGateModel(
        target_center,
        family,
        ridge_alpha,
        tuple(float(value) for value in coefficients),
        standardization,
        donor_centers,
        nested_model_hashes,
    )


def _fit_standardization(raw: np.ndarray) -> Standardization:
    means = raw[:, 1:].mean(axis=0)
    scales = raw[:, 1:].std(axis=0, ddof=0)
    scales = np.where(scales >= STANDARDIZATION_SCALE_FLOOR, scales, 1.0)
    return Standardization(
        tuple(float(value) for value in means),
        tuple(float(value) for value in scales),
    )


def _standardize(raw: np.ndarray, standardization: Standardization) -> np.ndarray:
    result = raw.copy()
    result[:, 1:] = (
        result[:, 1:] - np.asarray(standardization.means, dtype=np.float64)
    ) / np.asarray(standardization.scales, dtype=np.float64)
    result[:, 0] = 1.0
    return result


def predict_one(model: SignedGateModel, feature: SignedFeatureRow) -> float:
    vector = np.asarray([feature.values], dtype=np.float64)
    design = _standardize(vector, model.standardization)[0]
    if model.family == "G":
        for index in range(len(FEATURE_NAMES)):
            if index not in GLOBAL_FEATURE_INDICES:
                design[index] = 0.0
    return float(design @ np.asarray(model.coefficients, dtype=np.float64))


def predict_corrections(
    fit: SignedGateFit,
    features: Sequence[SignedFeatureRow],
    *,
    nested_prediction_features: Mapping[str, Sequence[SignedFeatureRow]],
) -> tuple[CorrectionRow, ...]:
    """Seal raw and directional-uncertainty-gated corrections separately."""

    if len(fit.nested_models) < MIN_NESTED_MODELS:
        raise ProtocolError("Signed correction uncertainty lacks nested models.")
    target = fit.final_model.target_center
    final_exclusions = (target,)
    feature_rows = tuple(features)
    feature_lookup = _prediction_feature_lookup(
        feature_rows,
        expected_exclusions=final_exclusions,
        surface_name="final",
    )
    nested_items = tuple(nested_prediction_features.items())
    if any(type(query) is not str for query, _rows in nested_items):
        raise ProtocolError("Signed correction nested query identities are malformed.")
    expected_queries = {
        value.heldout_query_center for value in fit.nested_models
    }
    if {query for query, _rows in nested_items} != expected_queries:
        raise ProtocolError("Signed correction lacks nested prediction contexts.")
    nested_feature_lookup = {
        query: _prediction_feature_lookup(
            tuple(rows),
            expected_exclusions=tuple(sorted((target, query))),
            surface_name=f"nested H={target},q={query}",
        )
        for query, rows in nested_items
    }
    target_keys = {
        key for key in feature_lookup if key[0] == target
    }
    if (
        not target_keys
        or any(
            {key for key in rows if key[0] == target} != target_keys
            for rows in nested_feature_lookup.values()
        )
    ):
        raise ProtocolError("Signed correction lacks nested prediction contexts.")
    output: list[CorrectionRow] = []
    for row in sorted(feature_rows):
        if row.target_center != target:
            continue
        raw = max(
            -MAX_ABSOLUTE_CORRECTION_LOGIT,
            min(MAX_ABSOLUTE_CORRECTION_LOGIT, predict_one(fit.final_model, row)),
        )
        nested = tuple(
            max(
                -MAX_ABSOLUTE_CORRECTION_LOGIT,
                min(
                    MAX_ABSOLUTE_CORRECTION_LOGIT,
                    predict_one(
                        value.model,
                        nested_feature_lookup[value.heldout_query_center][
                            row.sample_key
                        ],
                    ),
                ),
            )
            for value in fit.nested_models
        )
        nested_mean = math.fsum(nested) / len(nested)
        standard_error = math.sqrt(
            math.fsum((value - nested_mean) ** 2 for value in nested)
            / max(len(nested) - 1, 1)
        ) / math.sqrt(len(nested))
        same_direction = raw != 0.0 and all(value * raw > 0.0 for value in nested)
        admitted = same_direction and abs(raw) - UNCERTAINTY_Z * standard_error > 0.0
        output.append(
            CorrectionRow(
                row.target_center,
                row.case_id,
                row.sample_id,
                fit.final_model.family,
                raw,
                standard_error,
                raw if admitted else 0.0,
                admitted,
            )
        )
    if not output:
        raise ProtocolError("Signed-gate prediction has no target-center samples.")
    return tuple(output)


def _prediction_feature_lookup(
    rows: tuple[SignedFeatureRow, ...],
    *,
    expected_exclusions: tuple[str, ...],
    surface_name: str,
) -> dict[tuple[str, str, str], SignedFeatureRow]:
    lookup = {row.sample_key: row for row in rows}
    if len(lookup) != len(rows):
        raise ProtocolError(
            f"Signed correction {surface_name} features contain duplicate sample rows."
        )
    for row in rows:
        expected_candidates = tuple(
            source
            for source in candidate_sources(row.target_center)
            if source not in expected_exclusions
        )
        if row.context_excluded_centers != expected_exclusions:
            raise ProtocolError(
                f"Signed correction {surface_name} feature exclusions drifted."
            )
        if row.candidate_source_ids != expected_candidates:
            raise ProtocolError(
                f"Signed correction {surface_name} candidate sources drifted."
            )
    return lookup


def correction_surface_hash(
    rows: Sequence[CorrectionRow], *, surface: str = "combined"
) -> str:
    if surface not in ("raw", "safe", "combined"):
        raise ProtocolError("Unknown signed-correction seal surface.")
    values = []
    for row in sorted(rows):
        payload = row.to_payload()
        if surface == "raw":
            payload = {
                key: value
                for key, value in payload.items()
                if key not in ("safe_correction", "uncertainty_admitted", "correction_hash")
            }
        elif surface == "safe":
            payload = {
                key: value
                for key, value in payload.items()
                if key not in ("raw_correction", "correction_hash")
            }
        values.append(payload)
    return canonical_hash(
        {
            "schema_version": "fixed_bank_signed_error_correction_surface_v1",
            "surface": surface,
            "rows": values,
            "raw_and_safe_separately_sealed": surface in ("raw", "safe"),
        }
    )


__all__ = (
    "SignedGateFit",
    "NestedSignedGateModel",
    "correction_surface_hash",
    "fit_signed_gate",
    "predict_corrections",
    "predict_one",
)
