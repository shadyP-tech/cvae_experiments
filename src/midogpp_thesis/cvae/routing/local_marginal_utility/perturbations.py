"""Exact uniform-anchored perturbations for local marginal utility estimation.

The perturbation library is deliberately tiny and closed: Stage 90 measures one
one-sided finite difference per legal source around the equal-union control.
The two supported geometries have integer-exact allocations, so prediction
differences cannot be attributed to apportionment noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError


LOCAL_PERTURBATION_EPSILON = Fraction(1, 8)
SUPPORTED_TOTAL_BY_SOURCE_COUNT = {7: 1008, 8: 1024}
CONTROL_ACTION_ID = "control"
BOOST_ACTION_PREFIX = "boost_source_"


@dataclass(frozen=True)
class PerturbationPlan:
    """One exact class-balanced mixture evaluated by the Stage-90 surface."""

    action_id: str
    candidate_sources: tuple[str, ...]
    boosted_source: str | None
    epsilon: float
    weights: Mapping[str, float]
    allocations_per_class: Mapping[str, int]
    total_per_class: int
    maximum_source_weight: float
    effective_source_count: float

    @property
    def is_control(self) -> bool:
        return self.boosted_source is None


def control_action_id() -> str:
    """Return the canonical equal-union action identifier."""

    return CONTROL_ACTION_ID


def boost_action_id(source: str) -> str:
    """Return the canonical identifier for a source-local perturbation."""

    normalized = str(source)
    if not normalized or normalized.strip() != normalized:
        raise ProtocolError("Perturbation source IDs must be nonempty and canonical.")
    return f"{BOOST_ACTION_PREFIX}{normalized}"


def legal_candidate_sources(
    all_sources: Sequence[str],
    *,
    outer_target: str,
    query_cluster: str,
) -> tuple[str, ...]:
    """Return the deterministic legal pool, excluding outer target and query.

    This helper only establishes geometry.  It intentionally accepts no labels,
    utilities, predictions, or scores with which to choose candidates.
    """

    sources = _canonical_sources(all_sources)
    target = str(outer_target)
    query = str(query_cluster)
    if not target or not query or target == query:
        raise ProtocolError("Outer target and query cluster must be distinct.")
    if target not in sources or query not in sources:
        raise ProtocolError("Outer target and query cluster must belong to the source universe.")
    return tuple(source for source in sources if source not in {target, query})


def build_perturbation_library(
    candidate_sources: Sequence[str],
    *,
    total_per_class: int,
    epsilon: float | Fraction = LOCAL_PERTURBATION_EPSILON,
) -> tuple[PerturbationPlan, ...]:
    """Build the control followed by one exact boost per canonical source.

    Only the protocol-locked ``(n=7, total=1008)`` and
    ``(n=8, total=1024)`` geometries are accepted.  All arithmetic is first
    performed as :class:`fractions.Fraction`; a plan is rejected if any desired
    allocation is not an integer.
    """

    sources = _canonical_sources(candidate_sources)
    n_sources = len(sources)
    total = int(total_per_class)
    expected_total = SUPPORTED_TOTAL_BY_SOURCE_COUNT.get(n_sources)
    if expected_total is None or total != expected_total:
        raise ProtocolError(
            "Local utility perturbations require exactly 7/1008 or 8/1024 "
            "candidate/total geometry."
        )
    epsilon_fraction = _epsilon_fraction(epsilon)
    uniform = Fraction(1, n_sources)
    control_weights = {source: uniform for source in sources}
    plans = [
        _plan(
            action_id=control_action_id(),
            sources=sources,
            boosted_source=None,
            epsilon=epsilon_fraction,
            weights=control_weights,
            total=total,
        )
    ]
    for boosted_source in sources:
        weights = {
            source: (Fraction(1, 1) - epsilon_fraction) * uniform
            + (epsilon_fraction if source == boosted_source else Fraction(0, 1))
            for source in sources
        }
        plans.append(
            _plan(
                action_id=boost_action_id(boosted_source),
                sources=sources,
                boosted_source=boosted_source,
                epsilon=epsilon_fraction,
                weights=weights,
                total=total,
            )
        )
    return tuple(plans)


def paired_marginal_utility(
    boosted_utility: float | Sequence[float] | np.ndarray,
    control_utility: float | Sequence[float] | np.ndarray,
    *,
    epsilon: float | Fraction = LOCAL_PERTURBATION_EPSILON,
) -> float | np.ndarray:
    """Return the paired finite-difference target ``(boost-control)/epsilon``."""

    eps = float(_epsilon_fraction(epsilon))
    boosted = np.asarray(boosted_utility, dtype=np.float64)
    control = np.asarray(control_utility, dtype=np.float64)
    if boosted.shape != control.shape or not (
        np.isfinite(boosted).all() and np.isfinite(control).all()
    ):
        raise ProtocolError("Paired marginal utilities require finite aligned inputs.")
    result = (boosted - control) / eps
    if not np.isfinite(result).all():
        raise ProtocolError("Paired marginal utility is non-finite.")
    if result.ndim == 0:
        return float(result)
    result.setflags(write=False)
    return result


def _plan(
    *,
    action_id: str,
    sources: tuple[str, ...],
    boosted_source: str | None,
    epsilon: Fraction,
    weights: Mapping[str, Fraction],
    total: int,
) -> PerturbationPlan:
    if set(weights) != set(sources) or sum(weights.values(), Fraction(0, 1)) != 1:
        raise ProtocolError("Perturbation weights must cover the legal pool and sum to one.")
    exact_counts = {source: weights[source] * total for source in sources}
    if any(value.denominator != 1 for value in exact_counts.values()):
        raise ProtocolError("Perturbation allocation is not integer exact.")
    allocations = {source: int(exact_counts[source]) for source in sources}
    if any(value <= 0 for value in allocations.values()) or sum(allocations.values()) != total:
        raise ProtocolError("Perturbation allocations violate positivity or total count.")
    numeric_weights = {source: float(weights[source]) for source in sources}
    values = np.asarray(tuple(numeric_weights.values()), dtype=np.float64)
    effective = float(1.0 / np.dot(values, values))
    return PerturbationPlan(
        action_id=action_id,
        candidate_sources=sources,
        boosted_source=boosted_source,
        epsilon=float(epsilon),
        weights=numeric_weights,
        allocations_per_class=allocations,
        total_per_class=total,
        maximum_source_weight=float(values.max()),
        effective_source_count=effective,
    )


def _canonical_sources(values: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    if (
        not normalized
        or any(not source or source.strip() != source for source in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ProtocolError("Candidate source IDs must be unique, nonempty, and canonical.")
    return tuple(sorted(normalized))


def _epsilon_fraction(value: float | Fraction) -> Fraction:
    epsilon = value if isinstance(value, Fraction) else Fraction(str(float(value)))
    if epsilon != LOCAL_PERTURBATION_EPSILON:
        raise ProtocolError("The local utility perturbation epsilon is locked to 1/8.")
    return epsilon


__all__ = (
    "BOOST_ACTION_PREFIX",
    "CONTROL_ACTION_ID",
    "LOCAL_PERTURBATION_EPSILON",
    "SUPPORTED_TOTAL_BY_SOURCE_COUNT",
    "PerturbationPlan",
    "boost_action_id",
    "build_perturbation_library",
    "control_action_id",
    "legal_candidate_sources",
    "paired_marginal_utility",
)
