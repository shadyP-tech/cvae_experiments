"""Dense probability aggregation and source-inner weight policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .validation import ValidationError


@dataclass(frozen=True)
class WeightPolicy:
    tau: float = 1.0
    cap_min: float | None = None
    cap_max: float | None = None
    shrinkage: float = 0.0


@dataclass(frozen=True)
class WeightResult:
    weights: dict[str, float]
    raw_weights: dict[str, float]
    fallback_reason: str


def source_inner_softmax_weights(
    candidates: Sequence[str],
    scores: Mapping[str, float],
    *,
    policy: WeightPolicy,
    tie_tolerance: float = 1e-12,
) -> WeightResult:
    """Convert source-inner scores to finite convex weights.

    Missing or non-finite scores are ineligible unless all candidates are missing,
    in which case the predeclared dense uniform fallback is used.
    """
    candidate_ids = tuple(str(candidate) for candidate in candidates)
    if not candidate_ids:
        raise ValidationError("cannot compute ensemble weights without candidates")
    if policy.tau <= 0 or not math.isfinite(policy.tau):
        raise ValidationError(f"tau must be finite and positive, got {policy.tau!r}")
    if not 0.0 <= float(policy.shrinkage) <= 1.0:
        raise ValidationError(f"shrinkage must be in [0, 1], got {policy.shrinkage!r}")

    finite = {
        candidate: float(scores[candidate])
        for candidate in candidate_ids
        if candidate in scores and math.isfinite(float(scores[candidate]))
    }
    if not finite:
        uniform = _uniform(candidate_ids)
        return WeightResult(weights=uniform, raw_weights=uniform, fallback_reason="all_scores_invalid_uniform")

    active = tuple(candidate for candidate in candidate_ids if candidate in finite)
    values = [finite[candidate] for candidate in active]
    if max(values) - min(values) <= tie_tolerance:
        raw = {candidate: 0.0 for candidate in candidate_ids}
        uniform = _uniform(active)
        raw.update(uniform)
        return WeightResult(weights=dict(raw), raw_weights=dict(raw), fallback_reason="all_scores_tied_uniform")

    scaled = [value / policy.tau for value in values]
    max_scaled = max(scaled)
    exp_values = [math.exp(value - max_scaled) for value in scaled]
    denom = sum(exp_values)
    raw_active = {candidate: float(value / denom) for candidate, value in zip(active, exp_values)}
    raw = {candidate: 0.0 for candidate in candidate_ids}
    raw.update(raw_active)
    weights = _apply_regularizers(candidate_ids, active, raw, policy=policy)
    return WeightResult(weights=weights, raw_weights=raw, fallback_reason="")


def uniform_weights(candidates: Sequence[str]) -> WeightResult:
    candidate_ids = tuple(str(candidate) for candidate in candidates)
    weights = _uniform(candidate_ids)
    return WeightResult(weights=weights, raw_weights=dict(weights), fallback_reason="")


def aggregate_positive_probabilities(
    member_probabilities: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
) -> list[float]:
    """Return a convex weighted average of positive-class probabilities."""
    if not member_probabilities:
        raise ValidationError("cannot aggregate an empty member prediction set")
    lengths = {len(values) for values in member_probabilities.values()}
    if len(lengths) != 1:
        raise ValidationError(f"member prediction length mismatch: {sorted(lengths)}")
    missing = sorted(set(member_probabilities).difference(weights))
    if missing:
        raise ValidationError(f"weights missing member predictions: {missing}")
    total_weight = sum(float(weights.get(member, 0.0)) for member in member_probabilities)
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValidationError(f"ensemble weights must sum to 1 over members, got {total_weight}")

    n_items = next(iter(lengths))
    out: list[float] = []
    for idx in range(n_items):
        value = 0.0
        for member, probabilities in member_probabilities.items():
            prob = float(probabilities[idx])
            if not math.isfinite(prob) or prob < 0.0 or prob > 1.0:
                raise ValidationError(f"invalid probability for member={member!r} row={idx}: {prob!r}")
            value += float(weights.get(member, 0.0)) * prob
        out.append(float(value))
    return out


def largest_remainder_allocation(weights: Mapping[str, float], total_budget: int) -> dict[str, int]:
    """Deterministically round convex weights into an integer sample budget."""
    if total_budget < 0:
        raise ValidationError(f"total_budget must be nonnegative, got {total_budget}")
    if not weights:
        raise ValidationError("cannot allocate budget without weights")
    total_weight = sum(float(value) for value in weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValidationError(f"budget weights must sum to 1, got {total_weight}")
    exact = {key: float(value) * int(total_budget) for key, value in weights.items()}
    allocation = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = int(total_budget) - sum(allocation.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - allocation[key]), str(key)))
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def _apply_regularizers(
    candidates: Sequence[str],
    active: Sequence[str],
    raw: Mapping[str, float],
    *,
    policy: WeightPolicy,
) -> dict[str, float]:
    active_set = set(active)
    weights = {candidate: (float(raw[candidate]) if candidate in active_set else 0.0) for candidate in candidates}
    if policy.shrinkage:
        uniform_active = 1.0 / len(active)
        weights = {
            candidate: ((1.0 - policy.shrinkage) * weights[candidate] + policy.shrinkage * uniform_active)
            if candidate in active_set
            else 0.0
            for candidate in candidates
        }
    if policy.cap_min is not None or policy.cap_max is not None:
        cap_min = 0.0 if policy.cap_min is None else float(policy.cap_min)
        cap_max = 1.0 if policy.cap_max is None else float(policy.cap_max)
        if cap_min < 0.0 or cap_max <= 0.0 or cap_min > cap_max:
            raise ValidationError(f"invalid cap range: cap_min={policy.cap_min} cap_max={policy.cap_max}")
        weights = {
            candidate: min(max(weights[candidate], cap_min), cap_max) if candidate in active_set else 0.0
            for candidate in candidates
        }
    total = sum(weights.values())
    if total <= 0.0 or not math.isfinite(total):
        uniform = _uniform(active)
        return {candidate: uniform.get(candidate, 0.0) for candidate in candidates}
    normalized = {candidate: float(value / total) for candidate, value in weights.items()}
    _assert_convex(normalized)
    return normalized


def _uniform(candidates: Sequence[str]) -> dict[str, float]:
    if not candidates:
        raise ValidationError("cannot create uniform weights without candidates")
    weight = 1.0 / len(candidates)
    return {candidate: float(weight) for candidate in candidates}


def _assert_convex(weights: Mapping[str, float]) -> None:
    for key, value in weights.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValidationError(f"non-convex weight for {key!r}: {value!r}")
    total = sum(float(value) for value in weights.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValidationError(f"weights must sum to 1, got {total}")
