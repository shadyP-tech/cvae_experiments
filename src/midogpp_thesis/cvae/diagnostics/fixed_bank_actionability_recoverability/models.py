"""Fixed-alpha ridge G/R/P models with strict H/q/e exclusions."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CASE_ACTION_FEATURE_NAMES,
    GEOMETRY_IDS,
    MIDOGPP_CENTERS,
    RIDGE_ALPHA,
    STANDARDIZATION_SCALE_FLOOR,
    candidate_sources,
)
from .contracts import (
    ActionScoreRow,
    CaseActionFeatureRow,
    RidgeActionModel,
    UtilityTargetRow,
)
from .features import matched_blocked_feature_permutation


_SCALE_FLOOR = STANDARDIZATION_SCALE_FLOOR


def _legal_training_centers(
    *, outer_target_center: str, selected_source: str, heldout_donor_center: str | None
) -> tuple[str, ...]:
    excluded = {outer_target_center, selected_source}
    if heldout_donor_center is not None:
        excluded.add(heldout_donor_center)
    return tuple(center for center in MIDOGPP_CENTERS if center not in excluded)


def _standardization(design: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.zeros(design.shape[1], dtype=np.float64)
    scales = np.ones(design.shape[1], dtype=np.float64)
    if design.shape[1] > 1:
        means[1:] = design[:, 1:].mean(axis=0)
        observed = design[:, 1:].std(axis=0, ddof=0)
        scales[1:] = np.where(observed >= _SCALE_FLOOR, observed, 1.0)
    standardized = (design - means) / scales
    standardized[:, 0] = 1.0
    return standardized, means, scales


def _fit_one(
    feature_by_key: dict[tuple[str, str, str, str], CaseActionFeatureRow],
    target_by_key: dict[tuple[str, str, str, str], UtilityTargetRow],
    *,
    outer_target_center: str,
    heldout_donor_center: str | None,
    geometry_id: str,
    selected_source: str,
    family: str,
    ridge_alpha: float,
) -> RidgeActionModel:
    training_centers = _legal_training_centers(
        outer_target_center=outer_target_center,
        selected_source=selected_source,
        heldout_donor_center=heldout_donor_center,
    )
    keys = tuple(
        sorted(
            key
            for key in target_by_key
            if key[0] in training_centers
            and key[2] == geometry_id
            and key[3] == selected_source
        )
    )
    if not keys or {key[0] for key in keys} != set(training_centers):
        raise ProtocolError("Ridge fit lacks utility evidence from every legal donor query.")
    if any(key not in feature_by_key for key in keys):
        raise ProtocolError("Ridge response rows are not aligned to case-action features.")
    response_kinds = {target_by_key[key].response_kind for key in keys}
    if len(response_kinds) != 1:
        raise ProtocolError("One ridge model cannot mix response definitions.")
    if family == "G":
        feature_names = ("intercept",)
        raw = np.ones((len(keys), 1), dtype=np.float64)
    else:
        feature_names = CASE_ACTION_FEATURE_NAMES
        raw = np.asarray([feature_by_key[key].values for key in keys], dtype=np.float64)
    target = np.asarray([target_by_key[key].response for key in keys], dtype=np.float64)
    design, means, scales = _standardization(raw)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ target
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs, rcond=1.0e-12) @ rhs
    return RidgeActionModel(
        outer_target_center=outer_target_center,
        heldout_donor_center=heldout_donor_center,
        geometry_id=geometry_id,
        selected_source=selected_source,
        family=family,
        ridge_alpha=ridge_alpha,
        feature_names=tuple(feature_names),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        training_query_centers=training_centers,
        response_kind=next(iter(response_kinds)),
    )


def fit_fixed_alpha_ridge_models(
    features: Sequence[CaseActionFeatureRow],
    targets: Sequence[UtilityTargetRow],
    *,
    outer_target_center: str,
    geometry_id: str,
    family: str,
    heldout_donor_center: str | None = None,
    ridge_alpha: float = RIDGE_ALPHA,
) -> tuple[RidgeActionModel, ...]:
    """Fit legal candidate models without H, nested q, or candidate e.

    ``ridge_alpha`` is accepted for configuration replay but must equal the
    single frozen value.  There is intentionally no grid or support tuning.
    """

    outer = str(outer_target_center)
    heldout = None if heldout_donor_center is None else str(heldout_donor_center)
    if outer not in MIDOGPP_CENTERS or geometry_id not in GEOMETRY_IDS or family not in ("G", "R", "P"):
        raise ProtocolError("Ridge family fit has an invalid outer/geometry/family.")
    alpha = float(ridge_alpha)
    if not math.isfinite(alpha) or not math.isclose(alpha, RIDGE_ALPHA, rel_tol=0.0, abs_tol=0.0):
        raise ProtocolError("Actionability ridge alpha is fixed at 1.0; no grid is legal.")
    if heldout is not None and heldout not in candidate_sources(outer):
        raise ProtocolError("Nested donor q must be a non-target MIDOG++ center.")
    raw_features = tuple(features)
    raw_targets = tuple(targets)
    if not raw_features or not raw_targets:
        raise ProtocolError("Ridge fitting inputs must be non-empty.")
    if any(not isinstance(row, CaseActionFeatureRow) for row in raw_features) or any(
        not isinstance(row, UtilityTargetRow) for row in raw_targets
    ):
        raise ProtocolError("Ridge fitting requires typed feature/target rows.")
    if family == "P":
        exclusions = (outer,) if heldout is None else (outer, heldout)
        model_features = matched_blocked_feature_permutation(
            raw_features, excluded_candidate_centers=exclusions
        )
    else:
        model_features = raw_features
    feature_by_key = {row.row_key: row for row in model_features}
    target_by_key = {row.row_key: row for row in raw_targets}
    if len(feature_by_key) != len(model_features) or len(target_by_key) != len(raw_targets):
        raise ProtocolError("Ridge fitting inputs contain duplicate case-action keys.")
    model_sources = tuple(
        source
        for source in candidate_sources(outer)
        if source != heldout
    )
    return tuple(
        _fit_one(
            feature_by_key,
            target_by_key,
            outer_target_center=outer,
            heldout_donor_center=heldout,
            geometry_id=geometry_id,
            selected_source=source,
            family=family,
            ridge_alpha=alpha,
        )
        for source in model_sources
    )


def fit_all_model_families(
    features: Sequence[CaseActionFeatureRow],
    targets: Sequence[UtilityTargetRow],
    *,
    outer_target_center: str,
    geometry_id: str,
    heldout_donor_center: str | None = None,
) -> tuple[RidgeActionModel, ...]:
    """Convenience fit for the matched G/R/P family set."""

    return tuple(
        model
        for family in ("G", "R", "P")
        for model in fit_fixed_alpha_ridge_models(
            features,
            targets,
            outer_target_center=outer_target_center,
            geometry_id=geometry_id,
            family=family,
            heldout_donor_center=heldout_donor_center,
        )
    )


def _predict_one(model: RidgeActionModel, row: CaseActionFeatureRow) -> float:
    raw = np.ones(1, dtype=np.float64) if model.family == "G" else np.asarray(row.values, dtype=np.float64)
    means = np.asarray(model.means, dtype=np.float64)
    scales = np.asarray(model.scales, dtype=np.float64)
    design = (raw - means) / scales
    design[0] = 1.0
    return float(design @ np.asarray(model.coefficients, dtype=np.float64))


def predict_action_scores(
    models: Sequence[RidgeActionModel],
    features: Sequence[CaseActionFeatureRow],
) -> tuple[ActionScoreRow, ...]:
    """Predict all candidate gains for the models' strictly held-out target H."""

    fitted = tuple(models)
    if not fitted:
        raise ProtocolError("Action scoring requires a complete model family.")
    contexts = {
        (
            model.outer_target_center,
            model.heldout_donor_center,
            model.geometry_id,
            model.family,
        )
        for model in fitted
    }
    if len(contexts) != 1:
        raise ProtocolError("Action scoring cannot mix model contexts.")
    outer, heldout, geometry, family = next(iter(contexts))
    if heldout is not None:
        raise ProtocolError("Nested donor-q models cannot score the terminal target surface.")
    model_by_source = {model.selected_source: model for model in fitted}
    if tuple(model_by_source) != candidate_sources(outer) or len(model_by_source) != len(fitted):
        raise ProtocolError("Action scoring requires exactly eight source models in canonical order.")
    raw_features = tuple(features)
    if family == "P":
        scoring_features = matched_blocked_feature_permutation(
            raw_features, excluded_candidate_centers=(outer,)
        )
    else:
        scoring_features = raw_features
    selected = tuple(
        row
        for row in scoring_features
        if row.query_center == outer and row.geometry_id == geometry
    )
    by_key = {row.row_key: row for row in selected}
    cases = tuple(sorted({row.case_id for row in selected}))
    if not cases:
        raise ProtocolError("No target-H feature rows are available for action scoring.")
    output: list[ActionScoreRow] = []
    for case_id in cases:
        for source in candidate_sources(outer):
            key = (outer, case_id, geometry, source)
            if key not in by_key:
                raise ProtocolError("Target-H feature surface lacks a candidate action.")
            model = model_by_source[source]
            output.append(
                ActionScoreRow(
                    target_center=outer,
                    case_id=case_id,
                    geometry_id=geometry,
                    selected_source=source,
                    family=family,
                    predicted_gain=_predict_one(model, by_key[key]),
                    model_hash=model.model_hash,
                )
            )
    return tuple(output)


__all__ = (
    "ActionScoreRow",
    "RidgeActionModel",
    "UtilityTargetRow",
    "fit_all_model_families",
    "fit_fixed_alpha_ridge_models",
    "predict_action_scores",
)
