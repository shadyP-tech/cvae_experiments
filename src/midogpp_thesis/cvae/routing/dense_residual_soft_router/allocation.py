"""Deterministic positive Hamilton apportionment for source prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError


ALLOCATION_SEMANTICS = (
    "positive_lower_bound_hamilton_largest_remainder_canonical_source_ties"
)
DEFAULT_TOTAL_PER_CLASS = 1024


@dataclass(frozen=True)
class HamiltonAllocation:
    source_order: tuple[str, ...]
    normalized_weights: Mapping[str, float]
    residual_quotas: Mapping[str, float]
    allocations: Mapping[str, int]
    total: int
    minimum_per_source: int
    remainder_order: tuple[str, ...]
    allocation_semantics: str = ALLOCATION_SEMANTICS


def build_hamilton_allocation(
    weights: Mapping[str, float],
    *,
    total: int = DEFAULT_TOTAL_PER_CLASS,
    minimum_per_source: int = 1,
) -> HamiltonAllocation:
    """Apply Hamilton's largest-remainder method after fixed lower bounds.

    One positive lower bound is reserved for every candidate by default.  The
    remaining seats are apportioned proportionally.  Equal fractional
    remainders are awarded in canonical source-ID order.
    """

    normalized_input: dict[str, float] = {}
    for raw_source, raw_weight in weights.items():
        source = str(raw_source)
        if not source or source in normalized_input:
            raise ProtocolError("Hamilton source keys must be unique and nonempty.")
        normalized_input[source] = float(raw_weight)
    sources = tuple(sorted(normalized_input))
    values = np.asarray([normalized_input[source] for source in sources], dtype=np.float64)
    total_count = int(total)
    lower = int(minimum_per_source)
    if (
        not sources
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
        or lower != 1
        or total_count < lower * len(sources)
    ):
        raise ProtocolError(
            "Hamilton allocation requires the fixed positive lower bound of one."
        )
    normalized = values / float(values.sum())
    residual_total = total_count - lower * len(sources)
    quotas = normalized * float(residual_total)
    floors = np.floor(quotas).astype(np.int64)
    allocations = floors + lower
    remaining = total_count - int(allocations.sum())
    if remaining < 0 or remaining >= len(sources):
        raise ProtocolError("Hamilton floor allocation violated remainder bounds.")
    fractions = quotas - floors
    remainder_indices = tuple(
        sorted(range(len(sources)), key=lambda index: (-fractions[index], sources[index]))
    )
    for index in remainder_indices[:remaining]:
        allocations[index] += 1
    if (
        int(allocations.sum()) != total_count
        or np.any(allocations < lower)
        or (lower > 0 and np.any(allocations <= 0))
    ):
        raise ProtocolError("Hamilton allocation failed its fixed total or lower bound.")
    return HamiltonAllocation(
        source_order=sources,
        normalized_weights={
            source: float(value)
            for source, value in zip(sources, normalized, strict=True)
        },
        residual_quotas={
            source: float(value)
            for source, value in zip(sources, quotas, strict=True)
        },
        allocations={
            source: int(value)
            for source, value in zip(sources, allocations, strict=True)
        },
        total=total_count,
        minimum_per_source=lower,
        remainder_order=tuple(sources[index] for index in remainder_indices),
    )


def hamilton_allocate(
    weights: Mapping[str, float],
    *,
    total: int = DEFAULT_TOTAL_PER_CLASS,
    minimum_per_source: int = 1,
) -> dict[str, int]:
    """Return only the canonical allocation mapping."""

    return dict(
        build_hamilton_allocation(
            weights,
            total=total,
            minimum_per_source=minimum_per_source,
        ).allocations
    )


__all__ = (
    "ALLOCATION_SEMANTICS",
    "DEFAULT_TOTAL_PER_CLASS",
    "HamiltonAllocation",
    "build_hamilton_allocation",
    "hamilton_allocate",
)
