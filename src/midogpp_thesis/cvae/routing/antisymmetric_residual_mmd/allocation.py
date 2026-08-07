"""Deterministic integer allocation for antisymmetric class-paired weights."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError


DEFAULT_TOTAL_PER_CLASS = 1024
ALLOCATION_SEMANTICS = (
    "antisymmetric_balanced_largest_remainder_canonical_source_ties_"
    "strictly_positive_pair"
)


@dataclass(frozen=True)
class AntisymmetricAllocation:
    """Integer realization ``n0=b+k`` and ``n1=b-k`` for every source."""

    source_order: tuple[str, ...]
    delta: Mapping[str, float]
    raw_offsets: Mapping[str, float]
    integer_offsets: Mapping[str, int]
    class_0_allocations: Mapping[str, int]
    class_1_allocations: Mapping[str, int]
    total_per_class: int
    baseline_per_source: int
    adjustment_order: tuple[str, ...]
    allocation_hash: str
    allocation_semantics: str = ALLOCATION_SEMANTICS

    @property
    def k_by_source(self) -> Mapping[str, int]:
        return self.integer_offsets

    @property
    def allocations_by_class(self) -> Mapping[int, Mapping[str, int]]:
        return MappingProxyType(
            {0: self.class_0_allocations, 1: self.class_1_allocations}
        )


def build_antisymmetric_allocation(
    delta: Mapping[object, float],
    *,
    total_per_class: int = DEFAULT_TOTAL_PER_CLASS,
) -> AntisymmetricAllocation:
    """Round continuous residuals into a positive, exactly paired allocation.

    The continuous ideal offset is ``total_per_class * d_e``.  Integer offsets
    are selected by deterministic bounded largest-remainder adjustment, with
    canonical source IDs resolving equal costs.  The returned pair obeys
    ``class_0[e] = baseline + k[e]``,
    ``class_1[e] = baseline - k[e]``, and ``sum(k) = 0`` exactly.
    """

    if not isinstance(delta, Mapping):
        raise ProtocolError("Antisymmetric allocation residuals must be a mapping.")
    normalized: dict[str, float] = {}
    try:
        for raw_source, raw_value in delta.items():
            source = str(raw_source)
            if not source or source.strip() != source or source in normalized:
                raise ProtocolError(
                    "Antisymmetric allocation source keys must be unique and nonempty."
                )
            normalized[source] = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Antisymmetric allocation residuals are invalid.") from exc
    sources = tuple(sorted(normalized))
    values = np.asarray([normalized[source] for source in sources], dtype=np.float64)
    if (
        not sources
        or not np.isfinite(values).all()
        or abs(float(values.sum())) > 1.0e-10
        or isinstance(total_per_class, bool)
        or not isinstance(total_per_class, Integral)
    ):
        raise ProtocolError("Antisymmetric allocation residuals are invalid.")
    total = int(total_per_class)
    if total <= 0 or total % len(sources) != 0:
        raise ProtocolError(
            "Antisymmetric total_per_class must divide the source count exactly."
        )
    baseline = total // len(sources)
    if baseline < 2:
        raise ProtocolError(
            "Antisymmetric allocation cannot preserve strict positivity."
        )
    uniform = 1.0 / float(len(sources))
    class_zero_weights = uniform + values
    class_one_weights = uniform - values
    if (
        float(class_zero_weights.min()) < -1.0e-10
        or float(class_one_weights.min()) < -1.0e-10
        or float(class_zero_weights.max()) > 0.25 + 1.0e-10
        or float(class_one_weights.max()) > 0.25 + 1.0e-10
        or float(np.abs(values).sum()) > 0.25 + 1.0e-10
        or _effective_count(class_zero_weights) < 6.0 - 1.0e-9
        or _effective_count(class_one_weights) < 6.0 - 1.0e-9
    ):
        raise ProtocolError(
            "Antisymmetric allocation residual violates the routing constraints."
        )

    raw_offsets = float(total) * values
    lower = 1 - baseline
    upper = baseline - 1
    clipped = np.clip(raw_offsets, lower, upper)
    integer_offsets = np.floor(clipped).astype(np.int64)
    adjustment_order: list[str] = []
    while int(integer_offsets.sum()) < 0:
        candidates = [
            index
            for index, value in enumerate(integer_offsets)
            if int(value) < upper
        ]
        if not candidates:
            raise ProtocolError("Antisymmetric allocation could not balance offsets.")
        index = min(
            candidates,
            key=lambda candidate: (
                _increment_cost(
                    int(integer_offsets[candidate]), float(raw_offsets[candidate])
                ),
                sources[candidate],
            ),
        )
        integer_offsets[index] += 1
        adjustment_order.append(sources[index])
    while int(integer_offsets.sum()) > 0:
        candidates = [
            index
            for index, value in enumerate(integer_offsets)
            if int(value) > lower
        ]
        if not candidates:
            raise ProtocolError("Antisymmetric allocation could not balance offsets.")
        index = min(
            candidates,
            key=lambda candidate: (
                _decrement_cost(
                    int(integer_offsets[candidate]), float(raw_offsets[candidate])
                ),
                sources[candidate],
            ),
        )
        integer_offsets[index] -= 1
        adjustment_order.append(sources[index])

    class_zero = baseline + integer_offsets
    class_one = baseline - integer_offsets
    if (
        int(integer_offsets.sum()) != 0
        or int(class_zero.sum()) != total
        or int(class_one.sum()) != total
        or np.any(class_zero <= 0)
        or np.any(class_one <= 0)
        or not np.all(class_zero + class_one == 2 * baseline)
    ):
        raise ProtocolError(
            "Antisymmetric integer allocation failed its exact pair contract."
        )
    delta_mapping = _float_mapping(sources, values)
    raw_mapping = _float_mapping(sources, raw_offsets)
    offset_mapping = _integer_mapping(sources, integer_offsets)
    class_zero_mapping = _integer_mapping(sources, class_zero)
    class_one_mapping = _integer_mapping(sources, class_one)
    allocation_hash = _allocation_hash(
        sources=sources,
        offsets=integer_offsets,
        class_zero=class_zero,
        class_one=class_one,
        total=total,
        baseline=baseline,
    )
    return AntisymmetricAllocation(
        source_order=sources,
        delta=MappingProxyType(delta_mapping),
        raw_offsets=MappingProxyType(raw_mapping),
        integer_offsets=MappingProxyType(offset_mapping),
        class_0_allocations=MappingProxyType(class_zero_mapping),
        class_1_allocations=MappingProxyType(class_one_mapping),
        total_per_class=total,
        baseline_per_source=baseline,
        adjustment_order=tuple(adjustment_order),
        allocation_hash=allocation_hash,
    )


def allocate_antisymmetric_counts(
    delta: Mapping[object, float],
    *,
    total_per_class: int = DEFAULT_TOTAL_PER_CLASS,
) -> dict[int, dict[str, int]]:
    """Return only the two allocation mappings."""

    result = build_antisymmetric_allocation(
        delta, total_per_class=total_per_class
    )
    return {
        0: dict(result.class_0_allocations),
        1: dict(result.class_1_allocations),
    }


def antisymmetric_allocate(
    delta: Mapping[object, float],
    *,
    total_per_class: int = DEFAULT_TOTAL_PER_CLASS,
) -> dict[int, dict[str, int]]:
    """Short compatibility alias for :func:`allocate_antisymmetric_counts`."""

    return allocate_antisymmetric_counts(
        delta, total_per_class=total_per_class
    )


def _increment_cost(current: int, target: float) -> float:
    return float((current + 1 - target) ** 2 - (current - target) ** 2)


def _decrement_cost(current: int, target: float) -> float:
    return float((current - 1 - target) ** 2 - (current - target) ** 2)


def _effective_count(weights: np.ndarray) -> float:
    concentration = float(np.dot(weights, weights))
    if not math.isfinite(concentration) or concentration <= 0.0:
        return 0.0
    return float(1.0 / concentration)


def _float_mapping(
    sources: tuple[str, ...], values: np.ndarray
) -> dict[str, float]:
    return {
        source: float(value)
        for source, value in zip(sources, values, strict=True)
    }


def _integer_mapping(
    sources: tuple[str, ...], values: np.ndarray
) -> dict[str, int]:
    return {
        source: int(value)
        for source, value in zip(sources, values, strict=True)
    }


def _allocation_hash(
    *,
    sources: tuple[str, ...],
    offsets: np.ndarray,
    class_zero: np.ndarray,
    class_one: np.ndarray,
    total: int,
    baseline: int,
) -> str:
    payload = {
        "allocation_semantics": ALLOCATION_SEMANTICS,
        "baseline_per_source": baseline,
        "class_0": [int(value) for value in class_zero],
        "class_1": [int(value) for value in class_one],
        "integer_offsets": [int(value) for value in offsets],
        "source_order": list(sources),
        "total_per_class": total,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = (
    "ALLOCATION_SEMANTICS",
    "DEFAULT_TOTAL_PER_CLASS",
    "AntisymmetricAllocation",
    "allocate_antisymmetric_counts",
    "antisymmetric_allocate",
    "build_antisymmetric_allocation",
)
