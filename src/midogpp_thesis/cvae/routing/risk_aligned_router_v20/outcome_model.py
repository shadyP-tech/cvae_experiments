"""Shared risk-adjusted and damage estimates for actual hard-changing actions.

The three-category probabilities are coherent. Conditional magnitudes are
nonnegative and pooled across action families. Their score is an estimate,
never an action-level confidence bound; policy admission remains independent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from ...protocol import ProtocolError
from .aligned_metrics import ClassSupportNormalizer
from .contracts import CompositeKind, LabelFreeCaseMenu, SoftTopKComposite, decode_probability_hex
from .estimators import DEFAULT_RIDGE_ALPHA, fit_mean_ridge, fit_softmax_ridge, predict_softmax, standardize_design
from .features import CompositeFeatureScope, composite_descriptor, descriptor_names, fit_composite_feature_scope
from .hashing import canonical_hash
from .outcome_targets import category_target, prepare_candidate_population
from .truth import CompositeOutcome

_descriptor = composite_descriptor


def _descriptor_matrix(menu, composites, transform, patch_probability):
    base = np.asarray(decode_probability_hex(menu.baseline_probability_hex), dtype=float)
    cache = {}
    return np.stack([_descriptor(menu, c, transform, baseline_array=base, numeric_cache=cache, patch_probability=patch_probability) for c in composites])


@dataclass(frozen=True, slots=True)
class ActionOutcomePrediction:
    composite_hash: str
    predicted_gain: float
    predicted_harm: float
    predicted_brier_delta: float
    predicted_logloss_delta: float
    safe_positive_probability: float
    predicted_class_0_gain: float
    predicted_class_1_gain: float
    approximate_gain_lower_score: float  # Compatibility only: safe score, NOT a bound.
    risk_adjusted_score: float = 0.0
    remaining_probability: float = 0.0
    safe_gain_magnitude: float = 0.0
    harm_gain_magnitude: float = 0.0

    def public_payload(self):
        return {**{name: getattr(self, name) for name in self.__dataclass_fields__},
                "per_action_safety_guarantee": False, "lower_score_is_model_based": True,
                "approximate_gain_lower_score_deprecated": True,
                "risk_adjusted_score_is_confidence_bound": False}


@dataclass(frozen=True, slots=True)
class ActionOutcomeModel:
    transform: CompositeFeatureScope
    patch_model: object
    design_names: tuple[str, ...]
    descriptor_means: tuple[float, ...]
    descriptor_scales: tuple[float, ...]
    continuous_coefficients: tuple[tuple[float, ...], ...]
    category_coefficients: tuple[tuple[float, ...], ...]
    magnitude_coefficients: tuple[tuple[float, ...], ...]
    training_composite_hashes: tuple[str, ...]
    training_outcome_hashes: tuple[str, ...]
    row_weights: tuple[float, ...]
    normalizer: ClassSupportNormalizer
    gain_magnitude_bound: float
    population_exclusions: tuple[int, int, int]
    residual_rmse: float
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA
    model_hash: str = field(init=False)

    def __post_init__(self):
        dimension = len(self.design_names)
        coefficients = (*self.continuous_coefficients, *self.category_coefficients, *self.magnitude_coefficients)
        if (dimension < 1 or len(self.descriptor_means) != dimension-1 or len(self.descriptor_scales) != dimension-1
            or len(self.continuous_coefficients) != 5 or len(self.category_coefficients) != 3
            or len(self.magnitude_coefficients) != 2 or any(len(c) != dimension for c in coefficients)
            or any(not math.isfinite(v) for c in coefficients for v in c)
            or any(s <= 0 or not math.isfinite(s) for s in self.descriptor_scales)
            or any(not math.isfinite(v) for v in self.descriptor_means)
            or not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0
            or self.gain_magnitude_bound <= 0 or not math.isfinite(self.gain_magnitude_bound)
            or len(self.training_composite_hashes) != len(self.row_weights)
            or len(self.training_outcome_hashes) != len(self.row_weights)
            or any(w <= 0 or not math.isfinite(w) for w in self.row_weights)
            or (self.row_weights and abs(sum(self.row_weights)-1.) > 1e-10)
            or not math.isfinite(self.residual_rmse) or self.residual_rmse < 0
            or self.normalizer.case_keys != self.training_case_keys
            or self.patch_model.training_case_keys != self.training_case_keys):
            raise ProtocolError("HARP v20 risk-adjusted action model is malformed.")
        object.__setattr__(self, "model_hash", canonical_hash(self._payload()))

    @property
    def training_case_keys(self):
        return self.transform.training_case_keys

    @property
    def empty_population(self):
        return not self.training_composite_hashes

    @property
    def normalization_payload(self):
        return self.normalizer.public_payload()

    def _payload(self):
        return {"schema_version": "harp_v20_risk_aligned_action_model",
                "patch_evidence_model": self.patch_model.public_payload(),
                "transform": self.transform.public_payload(), "design_names": self.design_names,
                "descriptor_means": self.descriptor_means, "descriptor_scales": self.descriptor_scales,
                "continuous_coefficients": self.continuous_coefficients,
                "continuous_targets": ["signed_aligned_gain", "brier_delta", "logloss_delta", "class_0_recall_delta", "class_1_recall_delta"],
                "category_coefficients": self.category_coefficients, "category_names": ["SAFE_POSITIVE", "HARM", "OTHER"],
                "magnitude_coefficients": self.magnitude_coefficients,
                "training_composite_hashes": self.training_composite_hashes,
                "training_outcome_hashes": self.training_outcome_hashes, "row_weights": self.row_weights,
                "normalization": self.normalization_payload, "gain_magnitude_bound": self.gain_magnitude_bound,
                "excluded_baseline_no_hard_change_duplicate_counts": self.population_exclusions,
                "empty_population": self.empty_population, "residual_rmse": self.residual_rmse,
                "residual_rmse_used_for_routing": False, "ridge_alpha": self.ridge_alpha,
                "regularization_objective": "weighted_mean_loss_plus_half_alpha_squared_nonintercept_norm",
                "one_weight_per_participating_case_shared_across_candidates": True,
                "prediction_changing_candidates_only": True, "negative_examples_retained": True,
                "safe_positive_event": "gain>0 and Brier_delta<=0 and logloss_delta<=0",
                "selection_formula": "aligned_gain_minus_scale_times_weighted_risk_excess",
                "category_magnitudes_are_auxiliary_not_selection_objective": True,
                "per_action_safety_guarantee": False, "risk_adjusted_score_is_confidence_bound": False}

    def public_payload(self):
        return {**self._payload(), "model_hash": self.model_hash}

    def predict_composites(self, menu: LabelFreeCaseMenu, composites: Sequence[SoftTopKComposite]):
        rows = tuple(composites)
        if (menu.center_id, menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v20 honest action prediction includes a fitted case.")
        if not rows:
            return ()
        matrix = _descriptor_matrix(menu, rows, self.transform, self.patch_model.predict(menu))
        matrix[:, 1:] = (matrix[:, 1:] - np.asarray(self.descriptor_means)) / np.asarray(self.descriptor_scales)
        continuous = matrix @ np.asarray(self.continuous_coefficients).T
        probabilities = predict_softmax(matrix, self.category_coefficients)
        magnitudes = np.clip(matrix @ np.asarray(self.magnitude_coefficients).T, 0., self.gain_magnitude_bound)
        result = []
        for i, composite in enumerate(rows):
            if composite.kind is CompositeKind.B:
                result.append(ActionOutcomePrediction(composite.composite_hash, 0., 0., 0., 0., 0., 0., 0., 0., remaining_probability=1.))
                continue
            if self.empty_population:
                result.append(ActionOutcomePrediction(composite.composite_hash, 0., float(composite.prediction_changed),
                    1., -math.log(1e-6), 0., 0., 0., 0., remaining_probability=float(not composite.prediction_changed)))
                continue
            gain, brier, logloss, g0, g1 = map(float, continuous[i])
            safe, harm, other = map(float, probabilities[i])
            safe_magnitude, harm_magnitude = map(float, magnitudes[i])
            score = gain  # The proposer applies the selected risk penalties before ranking.
            if not composite.prediction_changed:
                # BACC algebra is exact even though proper losses may change.
                gain = g0 = g1 = score = safe = harm = safe_magnitude = harm_magnitude = 0.
                other = 1.
            result.append(ActionOutcomePrediction(composite.composite_hash, gain, harm, brier, logloss,
                          safe, g0, g1, score, score, other, safe_magnitude, harm_magnitude))
        return tuple(result)


def fit_action_outcome_model(menus, composites, outcomes: Sequence[CompositeOutcome], *,
                             maximum_numeric_features=20, ridge_alpha=DEFAULT_RIDGE_ALPHA,
                             normalization_profiles=None, patch_oof=None, patch_model=None):
    menu_rows = tuple(sorted(menus, key=lambda m:(m.center_id,m.case_id)))
    population = prepare_candidate_population(menu_rows, composites, outcomes,
                                              normalization_profiles=normalization_profiles)
    transform = fit_composite_feature_scope(menu_rows, maximum_numeric_features=maximum_numeric_features)
    if patch_oof is None or patch_model is None or set(patch_oof) != {(m.center_id,m.case_id) for m in menu_rows}:
        raise ProtocolError('HARP v20 action training requires complete held-out patch evidence.')
    from .patch_evidence import HeldPatchEvidence
    if patch_model.training_case_keys != tuple((m.center_id,m.case_id) for m in menu_rows):
        raise ProtocolError('HARP v20 final patch evidence fitting scope drifted.')
    for menu in menu_rows:
        evidence = patch_oof[(menu.center_id,menu.case_id)]
        if (not isinstance(evidence,HeldPatchEvidence) or evidence.menu_hash != menu.menu_hash
            or evidence.case_key != (menu.center_id,menu.case_id)
            or not set(evidence.training_case_keys).issubset(patch_model.training_case_keys)):
            raise ProtocolError('HARP v20 action features lack exact held-case patch lineage.')
    names = descriptor_names(transform)
    rows, truth = population.composites, population.outcomes
    coefficients = np.zeros((5, len(names)))
    categorical = np.zeros((3, len(names)))
    magnitudes = np.zeros((2, len(names)))
    means, scales, rmse = np.zeros(len(names)-1), np.ones(len(names)-1), 0.
    if rows:
        by_case = {(m.center_id, m.case_id): m for m in menu_rows}
        groups = {}
        for index, c in enumerate(rows):
            groups.setdefault((c.center_id, c.case_id), []).append(index)
        raw = np.empty((len(rows), len(names)))
        for key, indices in groups.items():
            raw[indices] = _descriptor_matrix(by_case[key], tuple(rows[i] for i in indices), transform, patch_oof[key])
        weights = np.asarray(population.row_weights)
        matrix, means, scales = standardize_design(raw, weights)
        targets = np.asarray([[o.bacc_gain, o.brier_delta, o.log_loss_delta,
                               o.class_0_gain or 0., o.class_1_gain or 0.] for o in truth])
        for index in range(5):
            head_weights = weights.copy()
            if index in (3, 4):
                head_weights *= np.asarray([getattr(o, f"class_{index-3}_gain") is not None for o in truth])
            if head_weights.sum() > 0:
                coefficients[index] = fit_mean_ridge(matrix, targets[:, index], head_weights, alpha=ridge_alpha)
        categories = np.asarray([category_target(o) for o in truth])
        categorical = fit_softmax_ridge(matrix, categories, weights, alpha=ridge_alpha)
        for index, sign in ((0, 1), (1, -1)):
            mask = categories == index
            if np.any(mask):
                magnitudes[index] = fit_mean_ridge(matrix[mask], sign * targets[mask, 0], weights[mask], alpha=ridge_alpha)
        rmse = math.sqrt(float(weights @ ((targets[:, 0] - matrix @ coefficients[0]) ** 2)))
    return ActionOutcomeModel(transform, patch_model, names, tuple(map(float, means)), tuple(map(float, scales)),
        tuple(tuple(map(float, c)) for c in coefficients), tuple(tuple(map(float, c)) for c in categorical),
        tuple(tuple(map(float, c)) for c in magnitudes), tuple(c.composite_hash for c in rows),
        tuple(o.outcome_hash for o in truth), population.row_weights, population.normalizer,
        population.gain_magnitude_bound, (population.excluded_baseline_count, population.excluded_no_hard_change_count,
                                        population.excluded_duplicate_count), rmse, ridge_alpha)


__all__ = ("ActionOutcomePrediction", "ActionOutcomeModel", "fit_action_outcome_model")
