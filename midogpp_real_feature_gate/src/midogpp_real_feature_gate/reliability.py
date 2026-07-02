"""Source-inner pseudo-target reliability scoring for real-feature ensembles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import SOURCE_INNER_RELIABILITY_SCHEMA_VERSION
from .validation import ValidationError


@dataclass(frozen=True)
class UtilityObservation:
    heldout_center: str
    pseudo_target_center: str
    expert_center: str
    utility_family: str
    utility_value: float


@dataclass(frozen=True)
class SourceInnerScores:
    scores: dict[str, float]
    n_eligible_folds: dict[str, int]
    reliability_rows: tuple[dict[str, object], ...]
    fallback_reason_by_expert: dict[str, str]


def source_inner_scores(
    *,
    heldout_center: str,
    candidates: Sequence[str],
    pseudo_targets: Sequence[str],
    observations: Sequence[UtilityObservation],
    utility_family: str,
    epsilon: float = 1e-12,
) -> SourceInnerScores:
    """Compute fold-normalized source-inner scores.

    Each pseudo-target fold is normalized independently over eligible candidate
    experts. The pseudo-target expert itself is always ineligible and is never
    imputed.
    """
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValidationError(f"epsilon must be finite and positive, got {epsilon!r}")
    candidate_ids = tuple(str(candidate) for candidate in candidates)
    pseudo_target_ids = tuple(str(target) for target in pseudo_targets)
    if not candidate_ids:
        raise ValidationError("source-inner reliability requires at least one candidate")
    if not pseudo_target_ids:
        raise ValidationError("source-inner reliability requires at least one pseudo-target fold")

    by_fold_expert: dict[tuple[str, str], UtilityObservation] = {}
    for observation in observations:
        if observation.utility_family != utility_family:
            raise ValidationError(
                f"mixed utility families are not allowed: expected={utility_family!r} got={observation.utility_family!r}"
            )
        key = (str(observation.pseudo_target_center), str(observation.expert_center))
        by_fold_expert[key] = observation

    z_by_expert: dict[str, list[float]] = {candidate: [] for candidate in candidate_ids}
    rows: list[dict[str, object]] = []
    for pseudo_target in pseudo_target_ids:
        eligible_values: dict[str, float] = {}
        for expert in candidate_ids:
            observation = by_fold_expert.get((pseudo_target, expert))
            if expert == pseudo_target or observation is None:
                continue
            utility = float(observation.utility_value)
            if math.isfinite(utility):
                eligible_values[expert] = utility
        mean_value = _mean(tuple(eligible_values.values()))
        std_value = _std(tuple(eligible_values.values()), mean_value)
        zero_variance = bool(eligible_values) and std_value <= epsilon
        for expert in candidate_ids:
            observation = by_fold_expert.get((pseudo_target, expert))
            eligible = expert != pseudo_target and expert in eligible_values
            fallback_reason = ""
            utility_value = float("nan")
            z_value = float("nan")
            if expert == pseudo_target:
                fallback_reason = "pseudo_target_expert_excluded"
            elif observation is None:
                fallback_reason = "missing_utility"
            elif not math.isfinite(float(observation.utility_value)):
                fallback_reason = "invalid_utility"
            elif zero_variance:
                utility_value = float(observation.utility_value)
                z_value = 0.0
                fallback_reason = "zero_fold_std"
            elif eligible:
                utility_value = float(observation.utility_value)
                z_value = float((utility_value - mean_value) / max(std_value, epsilon))

            if eligible:
                z_by_expert[expert].append(z_value)
            rows.append(
                {
                    "schema_version": SOURCE_INNER_RELIABILITY_SCHEMA_VERSION,
                    "heldout_center": heldout_center,
                    "pseudo_target_center": pseudo_target,
                    "expert_center": expert,
                    "eligible": eligible,
                    "utility_family": utility_family,
                    "utility_value": utility_value,
                    "z_iq": z_value,
                    "fold_mean_utility": mean_value,
                    "fold_std_utility": std_value,
                    "fallback_reason": fallback_reason,
                    "fit_used_pseudo_target_center": False,
                    "selection_used_target_labels": False,
                }
            )

    scores: dict[str, float] = {}
    n_eligible: dict[str, int] = {}
    fallback_by_expert: dict[str, str] = {}
    for expert, z_values in z_by_expert.items():
        finite_z = [float(value) for value in z_values if math.isfinite(float(value))]
        n_eligible[expert] = len(finite_z)
        if finite_z:
            scores[expert] = float(sum(finite_z) / len(finite_z))
            fallback_by_expert[expert] = ""
        else:
            scores[expert] = float("nan")
            fallback_by_expert[expert] = "no_eligible_pseudo_target_folds"
    return SourceInnerScores(
        scores=scores,
        n_eligible_folds=n_eligible,
        reliability_rows=tuple(rows),
        fallback_reason_by_expert=fallback_by_expert,
    )


def weight_rows(
    *,
    heldout_center: str,
    method: str,
    row_role: str,
    candidates: Sequence[str],
    utility_family: str,
    scores: SourceInnerScores,
    raw_weights: Mapping[str, float],
    weights: Mapping[str, float],
    fallback_reason: str,
    tau: float,
    cap_min: float | None,
    cap_max: float | None,
    shrinkage: float,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for expert in candidates:
        expert = str(expert)
        expert_fallback = scores.fallback_reason_by_expert.get(expert, "")
        rows.append(
            {
                "schema_version": SOURCE_INNER_RELIABILITY_SCHEMA_VERSION,
                "heldout_center": heldout_center,
                "method": method,
                "row_role": row_role,
                "expert_center": expert,
                "eligible": scores.n_eligible_folds.get(expert, 0) > 0 or bool(fallback_reason),
                "utility_family": utility_family,
                "n_eligible_pseudo_target_folds": scores.n_eligible_folds.get(expert, 0),
                "s_i": scores.scores.get(expert, float("nan")),
                "weight_raw": raw_weights.get(expert, 0.0),
                "w_i_utility": weights.get(expert, 0.0),
                "w_i_preservation": "",
                "fallback_reason": expert_fallback or fallback_reason,
                "tau": tau,
                "cap_min": "" if cap_min is None else cap_min,
                "cap_max": "" if cap_max is None else cap_max,
                "shrinkage": shrinkage,
                "selection_source": "source_inner",
                "target_expert_excluded": expert != heldout_center,
                "selection_used_target_labels": False,
                "fit_used_target_center": False,
            }
        )
    return tuple(rows)


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else math.nan


def _std(values: Sequence[float], mean_value: float) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite or not math.isfinite(mean_value):
        return math.nan
    variance = sum((value - mean_value) ** 2 for value in finite) / len(finite)
    return float(math.sqrt(variance))
