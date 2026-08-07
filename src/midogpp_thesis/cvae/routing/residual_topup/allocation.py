"""Deterministic Hamilton apportionment of the additive top-up budget."""

from __future__ import annotations

from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ...protocol import ProtocolError
from .contracts import HamiltonTopupAllocation
from .hashing import canonical_sha256


ALLOCATION_SEMANTICS = (
    "nonnegative_hamilton_largest_remainder_canonical_source_ties"
)


def build_hamilton_topup_allocation(
    weights: Mapping[object, float],
    *,
    topup_total_per_class: int,
) -> HamiltonTopupAllocation:
    """Round nonnegative weights to one fixed integer top-up total.

    Equal fractional remainders are resolved by canonical source identifier.
    Unlike full-mixture allocation, zero top-up counts are valid because the
    immutable base already contributes every source.
    """

    if not isinstance(weights, Mapping):
        raise ProtocolError("Residual top-up Hamilton weights must be a mapping.")
    normalized_input: dict[str, float] = {}
    try:
        for raw_source, raw_weight in weights.items():
            source = str(raw_source)
            if (
                not source
                or source.strip() != source
                or source in normalized_input
            ):
                raise ProtocolError(
                    "Residual top-up Hamilton source keys are invalid."
                )
            normalized_input[source] = float(raw_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("Residual top-up Hamilton weights are invalid.") from exc
    sources = tuple(sorted(normalized_input))
    values = np.asarray(
        [normalized_input[source] for source in sources], dtype=np.float64
    )
    if (
        not sources
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
        or isinstance(topup_total_per_class, bool)
        or not isinstance(topup_total_per_class, Integral)
        or int(topup_total_per_class) <= 0
    ):
        raise ProtocolError("Residual top-up Hamilton contract is invalid.")
    total = int(topup_total_per_class)
    normalized = values / float(values.sum())
    quotas = normalized * float(total)
    floors = np.floor(quotas).astype(np.int64)
    remaining = total - int(floors.sum())
    if remaining < 0 or remaining >= len(sources):
        raise ProtocolError("Residual top-up Hamilton remainder bounds drifted.")
    fractions = quotas - floors
    remainder_indices = tuple(
        sorted(
            range(len(sources)),
            key=lambda index: (-fractions[index], sources[index]),
        )
    )
    counts = floors.copy()
    for index in remainder_indices[:remaining]:
        counts[index] += 1
    if int(counts.sum()) != total or np.any(counts < 0):
        raise ProtocolError("Residual top-up Hamilton allocation failed closed.")
    weight_mapping = _float_mapping(sources, normalized)
    quota_mapping = _float_mapping(sources, quotas)
    count_mapping = _integer_mapping(sources, counts)
    allocation_hash = canonical_sha256(
        {
            "allocation_semantics": ALLOCATION_SEMANTICS,
            "source_order": list(sources),
            "counts": count_mapping,
            "topup_total_per_class": total,
        }
    )
    return HamiltonTopupAllocation(
        source_order=sources,
        normalized_weights=MappingProxyType(weight_mapping),
        quotas=MappingProxyType(quota_mapping),
        counts=MappingProxyType(count_mapping),
        topup_total_per_class=total,
        remainder_order=tuple(sources[index] for index in remainder_indices),
        allocation_hash=allocation_hash,
        allocation_semantics=ALLOCATION_SEMANTICS,
    )


def hamilton_topup_allocate(
    weights: Mapping[object, float],
    *,
    topup_total_per_class: int,
) -> dict[str, int]:
    """Return only the canonical integer top-up mapping."""

    return dict(
        build_hamilton_topup_allocation(
            weights, topup_total_per_class=topup_total_per_class
        ).counts
    )


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


__all__ = (
    "ALLOCATION_SEMANTICS",
    "build_hamilton_topup_allocation",
    "hamilton_topup_allocate",
)
