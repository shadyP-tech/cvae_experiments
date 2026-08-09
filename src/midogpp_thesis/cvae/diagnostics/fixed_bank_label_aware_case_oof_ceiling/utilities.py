"""Capability-scoped exact and descriptive case utility construction."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Sequence

from ...protocol import ProtocolError
from .core_contracts import (
    BinaryLabelRow,
    CaseActionUtility,
    CaseUtilitySurface,
    SealedProbabilitySurface,
    canonical_utility_rows,
)
from .core_hashing import canonical_hash, require_sha256
from .partitions import CaseFold
from .scientific_constants import BASELINE_ACTION_ID, MIDOGPP_CENTERS, action_ids

if TYPE_CHECKING:
    from .decisions import DecisionSeal
    from .global_prior import LocoGlobalPrior


def binary_balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    if len(labels) != len(predictions) or not labels:
        raise ProtocolError("Balanced accuracy inputs must be non-empty and aligned.")
    truth = tuple(int(value) for value in labels)
    guessed = tuple(int(value) for value in predictions)
    if set(truth) != {0, 1} or any(value not in (0, 1) for value in guessed):
        raise ProtocolError("Case-level balanced accuracy requires both binary classes.")
    sensitivity = sum(t == 1 and p == 1 for t, p in zip(truth, guessed)) / sum(t == 1 for t in truth)
    specificity = sum(t == 0 and p == 0 for t, p in zip(truth, guessed)) / sum(t == 0 for t in truth)
    return 0.5 * (float(sensitivity) + float(specificity))


def soft_binary_balanced_accuracy(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if len(labels) != len(probabilities) or not labels:
        raise ProtocolError("Smooth BACC inputs must be non-empty and aligned.")
    truth = tuple(int(value) for value in labels)
    values = tuple(float(value) for value in probabilities)
    if set(truth) != {0, 1} or any(value < 0.0 or value > 1.0 for value in values):
        raise ProtocolError("Smooth case BACC requires both binary classes and probabilities.")
    positive = [probability for label, probability in zip(truth, values) if label == 1]
    negative = [1.0 - probability for label, probability in zip(truth, values) if label == 0]
    return 0.5 * (sum(positive) / len(positive) + sum(negative) / len(negative))


def score_loco_prior_utilities(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    target_center: str,
) -> CaseUtilitySurface:
    """Open labels from H' != H only for the label-derived LOCO prior G_H."""

    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ProtocolError("Unknown target center for LOCO prior label access.")
    allowed = tuple(
        sorted(
            {
                (identity.target_center, identity.case_id)
                for identity in probabilities.identities
                if identity.target_center != target
            }
        )
    )
    return _score_allowed_cases(
        probabilities,
        labels,
        allowed_case_keys=allowed,
        label_scope=f"label_derived_LOCO_global_prior::heldout_H={target}",
        prerequisite_seal_hash=probabilities.surface_hash,
    )


def score_fold_support_utilities(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    fold: CaseFold,
    global_prior: "LocoGlobalPrior",
) -> CaseUtilitySurface:
    """Open only the four support folds after G_H has been sealed."""

    if (
        getattr(global_prior, "target_center", None) != fold.target_center
        or getattr(global_prior, "sealed_before_h_support_access", None) is not True
        or getattr(global_prior, "probability_surface_hash", None) != probabilities.surface_hash
    ):
        raise ProtocolError("Fold-support label access requires the matching sealed G_H.")
    prior_hash = require_sha256(getattr(global_prior, "prior_hash", None), "global_prior.prior_hash")
    allowed = tuple((fold.target_center, case_id) for case_id in fold.support_case_ids)
    return _score_allowed_cases(
        probabilities,
        labels,
        allowed_case_keys=allowed,
        label_scope=f"fold_local_support::{fold.fold_id}",
        prerequisite_seal_hash=prior_hash,
    )


def score_evaluation_utilities_after_decision_seal(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    decision_seal: "DecisionSeal",
) -> CaseUtilitySurface:
    """Open evaluation labels only after every one of the 45 decisions is sealed."""

    if (
        getattr(decision_seal, "all_fold_decisions_sealed_before_evaluation_labels", None) is not True
        or getattr(decision_seal, "probability_surface_hash", None) != probabilities.surface_hash
    ):
        raise ProtocolError("Evaluation labels require the complete matching decision seal.")
    seal_hash = require_sha256(getattr(decision_seal, "decision_seal_hash", None), "decision_seal_hash")
    allowed = tuple(
        sorted(
            {
                (identity.target_center, identity.case_id)
                for identity in probabilities.identities
            }
        )
    )
    return _score_allowed_cases(
        probabilities,
        labels,
        allowed_case_keys=allowed,
        label_scope="terminal_evaluation_after_all_45_decisions_sealed",
        prerequisite_seal_hash=seal_hash,
    )


def replace_smooth_descriptive(
    surface: CaseUtilitySurface,
    transform: Callable[[CaseActionUtility], float],
) -> CaseUtilitySurface:
    """Create a smooth-only poison control without changing exact identity."""

    rows = canonical_utility_rows(
        CaseActionUtility(
            target_center=row.target_center,
            case_id=row.case_id,
            action_id=row.action_id,
            sample_count=row.sample_count,
            exact_bacc=row.exact_bacc,
            smooth_bacc=float(transform(row)),
            exact_gain_vs_b=row.exact_gain_vs_b,
        )
        for row in surface.rows
    )
    exact_hash = _exact_surface_hash(
        rows,
        label_scope=surface.label_scope,
        prerequisite_seal_hash=surface.prerequisite_seal_hash,
        allowed_case_keys=surface.allowed_case_keys,
    )
    return CaseUtilitySurface(
        rows=rows,
        allowed_case_keys=surface.allowed_case_keys,
        label_scope=surface.label_scope,
        prerequisite_seal_hash=surface.prerequisite_seal_hash,
        exact_surface_hash=exact_hash,
        descriptive_surface_hash=canonical_hash(
            {
                "exact_surface_hash": exact_hash,
                "smooth_bacc": [row.smooth_bacc for row in rows],
            }
        ),
    )


def _score_allowed_cases(
    probabilities: SealedProbabilitySurface,
    labels: Sequence[BinaryLabelRow],
    *,
    allowed_case_keys: Sequence[tuple[str, str]],
    label_scope: str,
    prerequisite_seal_hash: str,
) -> CaseUtilitySurface:
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
    utility_rows: list[CaseActionUtility] = []
    for (center, case_id), sample_ids in sorted(samples_by_case.items()):
        truth = tuple(label_map[(center, case_id, sample)] for sample in sample_ids)
        action_scores: dict[str, tuple[float, float]] = {}
        for action in action_ids(center):
            values = tuple(probability_map[(center, case_id, sample, action)] for sample in sample_ids)
            exact = binary_balanced_accuracy(truth, tuple(int(value >= 0.5) for value in values))
            smooth = soft_binary_balanced_accuracy(truth, values)
            action_scores[action] = (exact, smooth)
        baseline = action_scores[BASELINE_ACTION_ID][0]
        for action in action_ids(center):
            exact, smooth = action_scores[action]
            utility_rows.append(
                CaseActionUtility(
                    target_center=center,
                    case_id=case_id,
                    action_id=action,
                    sample_count=len(sample_ids),
                    exact_bacc=exact,
                    smooth_bacc=smooth,
                    exact_gain_vs_b=exact - baseline,
                )
            )
    canonical = canonical_utility_rows(utility_rows)
    exact_hash = _exact_surface_hash(
        canonical,
        label_scope=label_scope,
        prerequisite_seal_hash=prerequisite_seal_hash,
        allowed_case_keys=allowed,
    )
    return CaseUtilitySurface(
        rows=canonical,
        allowed_case_keys=allowed,
        label_scope=label_scope,
        prerequisite_seal_hash=prerequisite_seal_hash,
        exact_surface_hash=exact_hash,
        descriptive_surface_hash=canonical_hash(
            {
                "exact_surface_hash": exact_hash,
                "smooth_bacc": [row.smooth_bacc for row in canonical],
            }
        ),
    )


def _exact_surface_hash(
    rows: Sequence[CaseActionUtility],
    *,
    label_scope: str,
    prerequisite_seal_hash: str,
    allowed_case_keys: Sequence[tuple[str, str]],
) -> str:
    return canonical_hash(
        {
            "schema_version": "fixed_bank_label_aware_case_utility_exact_v1",
            "label_scope": label_scope,
            "prerequisite_seal_hash": prerequisite_seal_hash,
            "allowed_case_keys": [list(key) for key in sorted(allowed_case_keys)],
            "rows": [row.exact_payload() for row in rows],
        }
    )


__all__ = (
    "binary_balanced_accuracy",
    "replace_smooth_descriptive",
    "score_evaluation_utilities_after_decision_seal",
    "score_fold_support_utilities",
    "score_loco_prior_utilities",
    "soft_binary_balanced_accuracy",
)
