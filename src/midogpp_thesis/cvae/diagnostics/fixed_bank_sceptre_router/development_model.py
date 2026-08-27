"""Low-capacity pairwise utility ranker with strict nested center deletion.

The model is deliberately small: candidate-family intercepts, shared evidence
effects, and candidate-by-evidence interactions.  It learns only pairwise BACC
contrasts after the exact nine seed cells have been averaged.  Hyperparameters
are selected by complete nested LODO and all reported fold results are
descriptive because the architecture was designed on this consumed surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError

from .development_surface import AggregatedUtility, OuterDevelopmentView
from .evidence_contracts import EvidenceFeatureRow
from .hashing import canonical_hash


@dataclass(frozen=True, slots=True)
class PairwiseUtilityModel:
    outer_target: str
    candidate_centers: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    alpha: float
    training_query_centers: tuple[str, ...]
    parent_exclusion_receipt_hash: str
    evidence_receipt_hash: str
    training_receipt_hash: str

    def predict(self, row: EvidenceFeatureRow) -> float:
        if row.query_center != self.outer_target:
            raise ProtocolError("SCEPTRE prediction row is not for the outer target.")
        if row.candidate_center not in self.candidate_centers:
            raise ProtocolError("SCEPTRE prediction candidate is outside C minus H.")
        if row.feature_names != self.feature_names:
            raise ProtocolError("SCEPTRE prediction feature schema drifted.")
        normalized = _normalize(row.values, self.feature_means, self.feature_scales)
        phi = _design_vector(
            row.candidate_center,
            normalized,
            self.candidate_centers,
        )
        return float(np.dot(phi, np.asarray(self.coefficients, dtype=np.float64)))

    def rank_target(
        self,
        rows: Iterable[EvidenceFeatureRow],
    ) -> tuple[tuple[str, float], ...]:
        by_candidate: dict[str, EvidenceFeatureRow] = {}
        for row in rows:
            if row.candidate_center in by_candidate:
                raise ProtocolError("SCEPTRE target evidence contains a duplicate candidate.")
            by_candidate[row.candidate_center] = row
        if set(by_candidate) != set(self.candidate_centers):
            raise ProtocolError("SCEPTRE target evidence does not cover exact C minus H.")
        scored = tuple(
            (candidate, self.predict(by_candidate[candidate]))
            for candidate in self.candidate_centers
        )
        return tuple(sorted(scored, key=lambda item: (-item[1], item[0])))


@dataclass(frozen=True, slots=True)
class NestedLodoFold:
    held_center: str
    alpha: float
    training_query_centers: tuple[str, ...]
    training_candidate_centers: tuple[str, ...]
    validation_candidate_count: int
    selected_candidate_set: tuple[str, ...]
    bacc_regret: float
    training_transform_receipt_hash: str
    validation_transform_receipt_hash: str
    strict_candidate_deletion: bool = True
    strict_query_deletion: bool = True


@dataclass(frozen=True, slots=True)
class AlphaAssessment:
    alpha: float
    folds: tuple[NestedLodoFold, ...]
    mean_center_regret: float
    worst_center_regret: float


@dataclass(frozen=True, slots=True)
class NestedLodoFit:
    outer_target: str
    selected_alpha: float
    assessments: tuple[AlphaAssessment, ...]
    final_model: PairwiseUtilityModel
    descriptive_only: bool
    adaptive_surface: bool
    outer_evidence_receipt_hash: str
    receipt_hash: str


def fit_nested_lodo_pairwise_ranker(
    view: OuterDevelopmentView,
    evidence_bundle: object,
    *,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
) -> NestedLodoFit:
    """Tune and refit after strict H and nested-K q/e exclusion."""

    if not isinstance(view, OuterDevelopmentView):
        raise ProtocolError("SCEPTRE development view type drifted.")
    alpha_grid = tuple(sorted({float(alpha) for alpha in alphas}))
    if not alpha_grid or any(not math.isfinite(alpha) or alpha <= 0 for alpha in alpha_grid):
        raise ProtocolError("SCEPTRE ridge alpha grid is invalid.")

    from .evidence_builder import build_nested_lodo_evidence

    evidence = _outer_evidence(view, evidence_bundle)
    utility_by_key = {
        (row.query_center, row.candidate_center): row for row in view.aggregate_rows
    }
    evidence_by_key = {row.key: row for row in evidence}
    if set(utility_by_key) != set(evidence_by_key):
        raise ProtocolError("SCEPTRE evidence and aggregated utility grids differ.")

    assessments: list[AlphaAssessment] = []
    for alpha in alpha_grid:
        folds: list[NestedLodoFold] = []
        for held in view.query_centers:
            nested_train, nested_validation = build_nested_lodo_evidence(
                evidence_bundle, held_center=held
            )
            candidate_universe = tuple(
                center
                for center in CENTERS
                if center not in {view.outer_target, held}
            )
            train_keys = tuple(
                key
                for key in sorted(utility_by_key)
                if key[0] != held and key[1] != held
            )
            validation_keys = tuple(
                key
                for key in sorted(utility_by_key)
                if key[0] == held and key[1] != held
            )
            if any(view.outer_target in key or held in (key[0], key[1]) for key in train_keys):
                raise ProtocolError("SCEPTRE nested deletion failed before fitting.")
            if set(key[1] for key in validation_keys) != set(candidate_universe):
                raise ProtocolError("SCEPTRE nested validation candidate grid drifted.")
            model = _fit_model(
                outer_target=held,
                utility_rows=tuple(utility_by_key[key] for key in train_keys),
                evidence_rows=nested_train.rows,
                candidate_centers=candidate_universe,
                alpha=alpha,
                parent_receipt_hash=view.exclusion_receipt_hash,
                evidence_receipt_hash=nested_train.receipt.receipt_hash,
            )
            validation = nested_validation.rows
            if {row.key for row in validation} != set(validation_keys):
                raise ProtocolError("SCEPTRE nested validation evidence keys drifted.")
            ranked = model.rank_target(validation)
            best_score = ranked[0][1]
            winners = tuple(
                candidate for candidate, score in ranked if score == best_score
            )
            best_observed = max(utility_by_key[key].mean_bacc for key in validation_keys)
            # Ties are assessed conservatively because they would fall back in
            # the executable selector rather than being resolved by center ID.
            selected_observed = min(
                utility_by_key[(held, candidate)].mean_bacc for candidate in winners
            )
            folds.append(
                NestedLodoFold(
                    held_center=held,
                    alpha=alpha,
                    training_query_centers=tuple(
                        center
                        for center in view.query_centers
                        if center != held
                    ),
                    training_candidate_centers=candidate_universe,
                    validation_candidate_count=len(validation_keys),
                    selected_candidate_set=winners,
                    bacc_regret=float(best_observed - selected_observed),
                    training_transform_receipt_hash=(
                        nested_train.receipt.receipt_hash
                    ),
                    validation_transform_receipt_hash=(
                        nested_validation.receipt.receipt_hash
                    ),
                )
            )
        regrets = tuple(fold.bacc_regret for fold in folds)
        assessments.append(
            AlphaAssessment(
                alpha=alpha,
                folds=tuple(folds),
                mean_center_regret=float(np.mean(regrets, dtype=np.float64)),
                worst_center_regret=max(regrets),
            )
        )

    # Prefer stronger regularization on an exact descriptive tie.
    selected = min(
        assessments,
        key=lambda item: (
            item.mean_center_regret,
            item.worst_center_regret,
            -item.alpha,
        ),
    )
    final_candidates = tuple(center for center in CENTERS if center != view.outer_target)
    final_model = _fit_model(
        outer_target=view.outer_target,
        utility_rows=view.aggregate_rows,
        evidence_rows=evidence,
        candidate_centers=final_candidates,
        alpha=selected.alpha,
        parent_receipt_hash=view.exclusion_receipt_hash,
        evidence_receipt_hash=evidence_bundle.receipt.receipt_hash,
    )
    receipt_body = {
        "schema_version": "sceptre_nested_lodo_fit_v1",
        "outer_target": view.outer_target,
        "outer_exclusion_receipt_hash": view.exclusion_receipt_hash,
        "outer_evidence_transform_receipt_hash": (
            evidence_bundle.receipt.receipt_hash
        ),
        "alpha_grid": list(alpha_grid),
        "selected_alpha": selected.alpha,
        "folds": [
            {
                "held_center": fold.held_center,
                "bacc_regret": fold.bacc_regret,
                "selected_candidate_set": list(fold.selected_candidate_set),
                "q_and_e_deleted_before_fit": True,
                "training_transform_receipt_hash": (
                    fold.training_transform_receipt_hash
                ),
                "validation_transform_receipt_hash": (
                    fold.validation_transform_receipt_hash
                ),
            }
            for fold in selected.folds
        ],
        "seed_cells_are_nuisance_replications": True,
        "adaptive_surface": True,
        "descriptive_only": True,
        "model_receipt_hash": final_model.training_receipt_hash,
    }
    return NestedLodoFit(
        outer_target=view.outer_target,
        selected_alpha=selected.alpha,
        assessments=tuple(assessments),
        final_model=final_model,
        descriptive_only=True,
        adaptive_surface=True,
        outer_evidence_receipt_hash=evidence_bundle.receipt.receipt_hash,
        receipt_hash=canonical_hash(receipt_body),
    )


def _outer_evidence(
    view: OuterDevelopmentView,
    evidence_bundle: object,
) -> tuple[EvidenceFeatureRow, ...]:
    """Delete all q==H/e==H evidence before any schema or value transform."""

    from .evidence_builder import EvidenceFeatureBundle, FEATURE_NAMES

    if not isinstance(evidence_bundle, EvidenceFeatureBundle):
        raise ProtocolError("SCEPTRE fit requires a receipt-bound evidence bundle.")
    if (
        evidence_bundle.receipt.role != "OUTER_DEVELOPMENT"
        or evidence_bundle.receipt.target_center != view.outer_target
        or evidence_bundle.receipt.feature_names != FEATURE_NAMES
    ):
        raise ProtocolError("SCEPTRE outer evidence receipt does not match H.")

    filtered = tuple(
        sorted(
            (
                row
                for row in evidence_bundle.rows
                if row.query_center != view.outer_target
                and row.candidate_center != view.outer_target
            ),
            key=lambda row: row.key,
        )
    )
    if len({row.key for row in filtered}) != len(filtered):
        raise ProtocolError("SCEPTRE outer evidence contains duplicate q/e rows.")
    if not filtered:
        raise ProtocolError("SCEPTRE outer evidence is empty.")
    schema = filtered[0].feature_names
    if any(row.feature_names != schema for row in filtered):
        raise ProtocolError("SCEPTRE evidence schemas differ across rows.")
    return filtered


def _fit_model(
    *,
    outer_target: str,
    utility_rows: Sequence[AggregatedUtility],
    evidence_rows: Sequence[EvidenceFeatureRow],
    candidate_centers: tuple[str, ...],
    alpha: float,
    parent_receipt_hash: str,
    evidence_receipt_hash: str,
) -> PairwiseUtilityModel:
    utility_by_key = {
        (row.query_center, row.candidate_center): row for row in utility_rows
    }
    evidence_by_key = {row.key: row for row in evidence_rows}
    if set(utility_by_key) != set(evidence_by_key):
        raise ProtocolError("SCEPTRE fit utility/evidence keys differ.")
    if not utility_rows or not evidence_rows:
        raise ProtocolError("SCEPTRE pairwise fit surface is empty.")
    feature_names = evidence_rows[0].feature_names
    matrix = np.asarray([row.values for row in evidence_rows], dtype=np.float64)
    means = np.mean(matrix, axis=0, dtype=np.float64)
    scales = np.std(matrix, axis=0, dtype=np.float64)
    scales[scales == 0.0] = 1.0

    query_centers = tuple(sorted({row.query_center for row in utility_rows}))
    pair_rows: list[np.ndarray] = []
    responses: list[float] = []
    weights: list[float] = []
    for query in query_centers:
        candidates = tuple(
            candidate
            for candidate in candidate_centers
            if (query, candidate) in utility_by_key
        )
        pairs = tuple(
            (left, right)
            for index, left in enumerate(candidates)
            for right in candidates[index + 1 :]
        )
        if not pairs:
            raise ProtocolError("SCEPTRE fit query has no pairwise contrasts.")
        query_weight = 1.0 / float(len(pairs))
        for left, right in pairs:
            left_row = evidence_by_key[(query, left)]
            right_row = evidence_by_key[(query, right)]
            left_x = _normalize(left_row.values, tuple(means), tuple(scales))
            right_x = _normalize(right_row.values, tuple(means), tuple(scales))
            pair_rows.append(
                _design_vector(left, left_x, candidate_centers)
                - _design_vector(right, right_x, candidate_centers)
            )
            responses.append(
                utility_by_key[(query, left)].mean_bacc
                - utility_by_key[(query, right)].mean_bacc
            )
            weights.append(query_weight)
    design = np.vstack(pair_rows).astype(np.float64, copy=False)
    response = np.asarray(responses, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    root_weight = np.sqrt(weight)
    weighted_design = design * root_weight[:, None]
    weighted_response = response * root_weight
    gram = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_response
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    try:
        coefficients = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram + penalty, hermitian=True) @ rhs
    if not np.all(np.isfinite(coefficients)):
        raise ProtocolError("SCEPTRE pairwise coefficients are non-finite.")
    receipt = canonical_hash(
        {
            "schema_version": "sceptre_pairwise_utility_model_v1",
            "outer_target": outer_target,
            "candidate_centers": list(candidate_centers),
            "feature_names": list(feature_names),
            "feature_means": means.tolist(),
            "feature_scales": scales.tolist(),
            "coefficients": coefficients.tolist(),
            "alpha": alpha,
            "training_query_centers": list(query_centers),
            "training_keys": [list(key) for key in sorted(utility_by_key)],
            "parent_exclusion_receipt_hash": parent_receipt_hash,
            "evidence_transform_receipt_hash": evidence_receipt_hash,
        }
    )
    return PairwiseUtilityModel(
        outer_target=outer_target,
        candidate_centers=candidate_centers,
        feature_names=feature_names,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        alpha=float(alpha),
        training_query_centers=query_centers,
        parent_exclusion_receipt_hash=parent_receipt_hash,
        evidence_receipt_hash=evidence_receipt_hash,
        training_receipt_hash=receipt,
    )


def _normalize(
    values: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    mean = np.asarray(means, dtype=np.float64)
    scale = np.asarray(scales, dtype=np.float64)
    if value.shape != mean.shape or mean.shape != scale.shape or np.any(scale <= 0.0):
        raise ProtocolError("SCEPTRE normalization geometry drifted.")
    return (value - mean) / scale


def _design_vector(
    candidate: str,
    features: np.ndarray,
    candidate_centers: tuple[str, ...],
) -> np.ndarray:
    if candidate not in candidate_centers:
        raise ProtocolError("SCEPTRE design candidate is outside the fit inventory.")
    candidate_index = candidate_centers.index(candidate)
    family = np.zeros(len(candidate_centers), dtype=np.float64)
    family[candidate_index] = 1.0
    interactions = np.zeros(
        (len(candidate_centers), int(features.size)), dtype=np.float64
    )
    interactions[candidate_index] = features
    return np.concatenate((family, features, interactions.ravel()))


__all__ = (
    "AlphaAssessment",
    "EvidenceFeatureRow",
    "NestedLodoFit",
    "NestedLodoFold",
    "PairwiseUtilityModel",
    "fit_nested_lodo_pairwise_ranker",
)
