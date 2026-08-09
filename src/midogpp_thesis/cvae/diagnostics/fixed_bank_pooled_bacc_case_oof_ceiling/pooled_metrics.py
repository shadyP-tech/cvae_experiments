"""Pooled exact BACC and paired whole-case-cluster uncertainty primitives."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

from ...protocol import ProtocolError
from .core_contracts import (
    BinaryLabelRow,
    CaseActionSufficientStatistics,
    SealedProbabilitySurface,
    SufficientStatisticSurface,
    make_statistics_surface,
)
from .core_hashing import canonical_hash, finite, require_sha256
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, UTILITY_ID, action_ids

if TYPE_CHECKING:
    from .case_partitions import CaseFold
    from .decisions import DecisionSeal
    from .permutation_plan import PermutationDecisionPlan
    from .pooled_prior import PooledLocoPrior


@dataclass(frozen=True)
class PooledExactBacc:
    n_positive: int
    true_positive: int
    n_negative: int
    true_negative: int
    sensitivity: float
    specificity: float
    exact_bacc: float
    metric_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.n_positive <= 0 or self.n_negative <= 0:
            raise ProtocolError("Pooled exact BACC requires both classes in the whole scope.")
        if not (0 <= self.true_positive <= self.n_positive):
            raise ProtocolError("Pooled true-positive count is invalid.")
        if not (0 <= self.true_negative <= self.n_negative):
            raise ProtocolError("Pooled true-negative count is invalid.")
        sensitivity = finite(self.sensitivity, "sensitivity")
        specificity = finite(self.specificity, "specificity")
        exact = finite(self.exact_bacc, "exact_bacc")
        expected_sensitivity = self.true_positive / self.n_positive
        expected_specificity = self.true_negative / self.n_negative
        if (
            abs(sensitivity - expected_sensitivity) > 1.0e-12
            or abs(specificity - expected_specificity) > 1.0e-12
            or abs(exact - 0.5 * (expected_sensitivity + expected_specificity)) > 1.0e-12
        ):
            raise ProtocolError("Pooled exact BACC differs from its sufficient statistics.")
        object.__setattr__(self, "sensitivity", sensitivity)
        object.__setattr__(self, "specificity", specificity)
        object.__setattr__(self, "exact_bacc", exact)
        object.__setattr__(self, "metric_hash", canonical_hash(self._unhashed()))

    @property
    def row_count(self) -> int:
        return self.n_positive + self.n_negative

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_exact_bacc_v2",
            "n_positive": self.n_positive,
            "true_positive": self.true_positive,
            "n_negative": self.n_negative,
            "true_negative": self.true_negative,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "exact_bacc": self.exact_bacc,
            "utility_id": UTILITY_ID,
            "per_case_bacc_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "metric_hash": self.metric_hash}


@dataclass(frozen=True)
class PairedClusterContrast:
    challenger_action_id: str
    reference_action_id: str
    case_count: int
    n_positive: int
    n_negative: int
    positive_mean_difference: float
    negative_mean_difference: float
    pooled_bacc_difference: float
    case_influences: tuple[tuple[str, float], ...]
    cluster_variance: float
    variance_floor: float
    contrast_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.challenger_action_id == self.reference_action_id:
            raise ProtocolError("Paired cluster contrast requires distinct actions.")
        if self.case_count < 2 or self.n_positive <= 0 or self.n_negative <= 0:
            raise ProtocolError("Paired cluster contrast needs two cases and both pooled classes.")
        influences = tuple(self.case_influences)
        if len(influences) != self.case_count or len({case for case, _ in influences}) != self.case_count:
            raise ProtocolError("Paired cluster influences must cover each whole case once.")
        for name in (
            "positive_mean_difference",
            "negative_mean_difference",
            "pooled_bacc_difference",
            "cluster_variance",
            "variance_floor",
        ):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.variance_floor <= 0.0 or self.cluster_variance < self.variance_floor:
            raise ProtocolError("Paired cluster variance violates its fixed floor.")
        expected_difference = 0.5 * (
            self.positive_mean_difference + self.negative_mean_difference
        )
        expected_variance = max(
            self.case_count
            / (self.case_count - 1)
            * sum(float(value) ** 2 for _case, value in influences),
            self.variance_floor,
        )
        if (
            abs(self.pooled_bacc_difference - expected_difference) > 1.0e-12
            or abs(self.cluster_variance - expected_variance) > 1.0e-12
            or abs(sum(float(value) for _case, value in influences)) > 1.0e-12
        ):
            raise ProtocolError("Paired cluster contrast mathematical identity drifted.")
        object.__setattr__(self, "case_influences", influences)
        object.__setattr__(self, "contrast_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pooled_bacc_paired_case_cluster_contrast_v2",
            "challenger_action_id": self.challenger_action_id,
            "reference_action_id": self.reference_action_id,
            "case_count": self.case_count,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "positive_mean_difference": self.positive_mean_difference,
            "negative_mean_difference": self.negative_mean_difference,
            "pooled_bacc_difference": self.pooled_bacc_difference,
            "case_influences": [[case, value] for case, value in self.case_influences],
            "cluster_variance": self.cluster_variance,
            "variance_floor": self.variance_floor,
            "uncertainty_unit": "paired_whole_case_cluster",
            "equal_case_utility_weighting": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "contrast_hash": self.contrast_hash}


def pooled_exact_bacc(
    rows: Sequence[CaseActionSufficientStatistics],
) -> PooledExactBacc:
    values = tuple(rows)
    if not values:
        raise ProtocolError("Cannot pool an empty sufficient-statistic scope.")
    n_positive = sum(row.n_positive for row in values)
    n_negative = sum(row.n_negative for row in values)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Pooled exact BACC requires both classes in the whole scope.")
    true_positive = sum(row.true_positive for row in values)
    true_negative = sum(row.true_negative for row in values)
    sensitivity = true_positive / n_positive
    specificity = true_negative / n_negative
    return PooledExactBacc(
        n_positive=n_positive,
        true_positive=true_positive,
        n_negative=n_negative,
        true_negative=true_negative,
        sensitivity=sensitivity,
        specificity=specificity,
        exact_bacc=0.5 * (sensitivity + specificity),
    )


def binary_balanced_accuracy(
    labels: Sequence[int], predictions: Sequence[int]
) -> float:
    """Raw-row equivalence helper; both classes are required in the full scope."""

    if len(labels) != len(predictions) or not labels:
        raise ProtocolError("Balanced-accuracy inputs must be non-empty and aligned.")
    truth = tuple(int(value) for value in labels)
    guessed = tuple(int(value) for value in predictions)
    if set(truth) != {0, 1} or any(value not in (0, 1) for value in guessed):
        raise ProtocolError("Scope-level balanced accuracy requires both binary classes.")
    positive = sum(truth)
    negative = len(truth) - positive
    sensitivity = sum(t == 1 and p == 1 for t, p in zip(truth, guessed)) / positive
    specificity = sum(t == 0 and p == 0 for t, p in zip(truth, guessed)) / negative
    return 0.5 * (float(sensitivity) + float(specificity))


def paired_pooled_difference(
    challenger_rows: Sequence[CaseActionSufficientStatistics],
    reference_rows: Sequence[CaseActionSufficientStatistics],
) -> float:
    challenger, reference = _aligned_pair(challenger_rows, reference_rows)
    return pooled_exact_bacc(challenger).exact_bacc - pooled_exact_bacc(reference).exact_bacc


def paired_whole_case_cluster_contrast(
    challenger_rows: Sequence[CaseActionSufficientStatistics],
    reference_rows: Sequence[CaseActionSufficientStatistics],
    *,
    variance_floor: float,
) -> PairedClusterContrast:
    challenger, reference = _aligned_pair(challenger_rows, reference_rows)
    floor = finite(variance_floor, "variance_floor")
    if floor <= 0.0:
        raise ProtocolError("variance_floor must be positive.")
    if len(challenger) < 2:
        raise ProtocolError("Whole-case cluster variance requires at least two cases.")
    n_positive = sum(row.n_positive for row in challenger)
    n_negative = sum(row.n_negative for row in challenger)
    if n_positive <= 0 or n_negative <= 0:
        raise ProtocolError("Support scope must contain both classes for pooled BACC.")
    positive_differences = tuple(
        left.true_positive - right.true_positive
        for left, right in zip(challenger, reference)
    )
    negative_differences = tuple(
        left.true_negative - right.true_negative
        for left, right in zip(challenger, reference)
    )
    positive_mean = sum(positive_differences) / n_positive
    negative_mean = sum(negative_differences) / n_negative
    influences: list[tuple[str, float]] = []
    for left, positive_difference, negative_difference in zip(
        challenger, positive_differences, negative_differences
    ):
        # Algebraically identical to the predeclared class-specific mean form;
        # an absent case class contributes exactly zero without division by zero.
        positive_term = (
            positive_difference - left.n_positive * positive_mean
        ) / n_positive
        negative_term = (
            negative_difference - left.n_negative * negative_mean
        ) / n_negative
        influences.append((left.case_id, 0.5 * (positive_term + negative_term)))
    m = len(influences)
    sandwich = m / (m - 1) * sum(value * value for _, value in influences)
    return PairedClusterContrast(
        challenger_action_id=challenger[0].action_id,
        reference_action_id=reference[0].action_id,
        case_count=m,
        n_positive=n_positive,
        n_negative=n_negative,
        positive_mean_difference=positive_mean,
        negative_mean_difference=negative_mean,
        pooled_bacc_difference=0.5 * (positive_mean + negative_mean),
        case_influences=tuple(influences),
        cluster_variance=max(sandwich, floor),
        variance_floor=floor,
    )


def score_loco_prior_statistics(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    target_center: str,
) -> SufficientStatisticSurface:
    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ProtocolError("Unknown target center for LOCO prior label access.")
    allowed = tuple(
        sorted(
            {
                identity.case_key
                for identity in probabilities.identities
                if identity.target_center != target
            }
        )
    )
    return _score_allowed_cases(
        probabilities,
        labels,
        allowed_case_keys=allowed,
        label_scope=f"label_derived_LOCO_pooled_prior::heldout_H={target}",
        prerequisite_seal_hash=probabilities.surface_hash,
    )


def score_fold_support_statistics(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    fold: "CaseFold",
    global_prior: "PooledLocoPrior",
) -> SufficientStatisticSurface:
    if (
        getattr(global_prior, "target_center", None) != fold.target_center
        or getattr(global_prior, "sealed_before_h_support_access", None) is not True
        or getattr(global_prior, "probability_surface_hash", None) != probabilities.surface_hash
    ):
        raise ProtocolError("Fold support labels require the matching sealed G_H and pairwise priors.")
    prior_hash = require_sha256(getattr(global_prior, "prior_hash", None), "global_prior.prior_hash")
    return _score_allowed_cases(
        probabilities,
        labels,
        allowed_case_keys=tuple((fold.target_center, case) for case in fold.support_case_ids),
        label_scope=f"fold_local_support::{fold.fold_id}",
        prerequisite_seal_hash=prior_hash,
    )


def score_evaluation_statistics_after_preevaluation_seals(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    decision_seal: "DecisionSeal",
    permutation_plan: "PermutationDecisionPlan",
) -> SufficientStatisticSurface:
    if (
        getattr(decision_seal, "all_fold_decisions_sealed_before_evaluation_labels", None)
        is not True
        or getattr(decision_seal, "probability_surface_hash", None) != probabilities.surface_hash
        or getattr(permutation_plan, "sealed_before_evaluation_labels", None) is not True
        or getattr(permutation_plan, "evaluation_labels_used_to_generate_actions", None) is not False
        or getattr(permutation_plan, "probability_surface_hash", None) != probabilities.surface_hash
        or getattr(permutation_plan, "partition_hash", None)
        != getattr(decision_seal, "partition_hash", None)
    ):
        raise ProtocolError("Evaluation labels require matching observed and null decision seals.")
    decision_hash = require_sha256(
        getattr(decision_seal, "decision_seal_hash", None), "decision_seal_hash"
    )
    plan_hash = require_sha256(getattr(permutation_plan, "plan_hash", None), "plan_hash")
    prerequisite = canonical_hash(
        {
            "schema_version": "fixed_bank_pooled_bacc_preevaluation_seal_pair_v2",
            "decision_seal_hash": decision_hash,
            "permutation_plan_hash": plan_hash,
            "all_observed_and_null_actions_sealed": True,
        }
    )
    allowed = tuple(sorted({identity.case_key for identity in probabilities.identities}))
    return _score_allowed_cases(
        probabilities,
        labels,
        allowed_case_keys=allowed,
        label_scope="terminal_evaluation_after_observed_and_null_decisions_sealed",
        prerequisite_seal_hash=prerequisite,
    )


# Explicit compatibility spelling for orchestration code; the null seal remains mandatory.
score_evaluation_statistics_after_decision_seal = (
    score_evaluation_statistics_after_preevaluation_seals
)


def _score_allowed_cases(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    allowed_case_keys: Sequence[tuple[str, str]],
    label_scope: str,
    prerequisite_seal_hash: str,
) -> SufficientStatisticSurface:
    if probabilities.predictions_globally_sealed_before_labels is not True:
        raise ProtocolError("Labels cannot open before the probability surface seal.")
    require_sha256(prerequisite_seal_hash, "prerequisite_seal_hash")
    allowed = tuple(sorted((str(center), str(case)) for center, case in allowed_case_keys))
    allowed_set = set(allowed)
    if not allowed or len(allowed) != len(allowed_set):
        raise ProtocolError("Label capability must contain unique whole cases.")
    probability_map = probabilities.probabilities()
    identities = {
        (identity.target_center, identity.case_id, identity.sample_id)
        for identity in probabilities.identities
        if identity.case_key in allowed_set
    }
    label_map: dict[tuple[str, str, str], int] = {}
    for row in labels:
        key = (row.target_center, row.case_id, row.sample_id)
        if row.case_key not in allowed_set:
            continue
        if key in label_map:
            raise ProtocolError("Duplicate label row entered a scoped capability.")
        label_map[key] = row.label
    if set(label_map) != identities:
        raise ProtocolError("Scoped labels do not exactly cover the permitted probability rows.")
    samples_by_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    for center, case_id, sample_id in sorted(identities):
        samples_by_case[(center, case_id)].append(sample_id)
    rows: list[CaseActionSufficientStatistics] = []
    for (center, case_id), sample_ids in sorted(samples_by_case.items()):
        truth = tuple(label_map[(center, case_id, sample)] for sample in sample_ids)
        n_positive = sum(truth)
        n_negative = len(truth) - n_positive
        for action in action_ids(center):
            predicted = tuple(
                int(probability_map[(center, case_id, sample, action)] >= 0.5)
                for sample in sample_ids
            )
            rows.append(
                CaseActionSufficientStatistics(
                    target_center=center,
                    case_id=case_id,
                    action_id=action,
                    n_positive=n_positive,
                    true_positive=sum(
                        label == 1 and guess == 1 for label, guess in zip(truth, predicted)
                    ),
                    n_negative=n_negative,
                    true_negative=sum(
                        label == 0 and guess == 0 for label, guess in zip(truth, predicted)
                    ),
                )
            )
    return make_statistics_surface(
        rows,
        allowed_case_keys=allowed,
        label_scope=label_scope,
        prerequisite_seal_hash=prerequisite_seal_hash,
    )


def _aligned_pair(
    challenger_rows: Sequence[CaseActionSufficientStatistics],
    reference_rows: Sequence[CaseActionSufficientStatistics],
) -> tuple[
    tuple[CaseActionSufficientStatistics, ...],
    tuple[CaseActionSufficientStatistics, ...],
]:
    challenger = tuple(sorted(tuple(challenger_rows), key=lambda row: row.case_key))
    reference = tuple(sorted(tuple(reference_rows), key=lambda row: row.case_key))
    if not challenger or len(challenger) != len(reference):
        raise ProtocolError("Paired actions require aligned non-empty whole-case rows.")
    if len({row.action_id for row in challenger}) != 1 or len(
        {row.action_id for row in reference}
    ) != 1:
        raise ProtocolError("Each side of a paired contrast must contain one action.")
    if challenger[0].action_id == reference[0].action_id:
        raise ProtocolError("Paired action contrast requires distinct actions.")
    for left, right in zip(challenger, reference):
        if left.case_key != right.case_key or left.class_counts != right.class_counts:
            raise ProtocolError("Paired action statistics drifted across whole cases.")
    return challenger, reference


def action_rows(
    surface: SufficientStatisticSurface,
    *,
    target_center: str,
    action_id: str,
    case_ids: Sequence[str] | None = None,
) -> tuple[CaseActionSufficientStatistics, ...]:
    allowed_cases = None if case_ids is None else set(str(case) for case in case_ids)
    rows = tuple(
        row
        for row in surface.rows
        if row.target_center == str(target_center)
        and row.action_id == str(action_id)
        and (allowed_cases is None or row.case_id in allowed_cases)
    )
    if allowed_cases is not None and {row.case_id for row in rows} != allowed_cases:
        raise ProtocolError("Requested action statistics do not cover the scoped cases.")
    return rows


__all__ = (
    "PairedClusterContrast",
    "PooledExactBacc",
    "action_rows",
    "binary_balanced_accuracy",
    "paired_pooled_difference",
    "paired_whole_case_cluster_contrast",
    "pooled_exact_bacc",
    "score_evaluation_statistics_after_decision_seal",
    "score_evaluation_statistics_after_preevaluation_seals",
    "score_fold_support_statistics",
    "score_loco_prior_statistics",
)
