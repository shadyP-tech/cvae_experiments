"""Label-free canonical row-order primitives for CBPUPR.

The physical fixed-bank store may expose rows in storage order.  CBPUPR owns a
single canonical identity order, lexicographic ``(case_id, sample_id)``, and
must apply its permutation to every aligned physical array before any labels
can open.  Contracts use the same primitives to reject non-canonical payloads
at process and persistence boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...protocol import ProtocolError


@dataclass(frozen=True)
class CanonicalCenterRowOrder:
    """One validated canonical permutation and its ordered identities."""

    permutation: tuple[int, ...]
    sample_ids: tuple[str, ...]
    case_ids: tuple[str, ...]


def canonical_center_row_order(
    sample_ids: Sequence[object],
    case_ids: Sequence[object],
    *,
    error_message: str = "Physical center identities cannot be canonicalized.",
) -> CanonicalCenterRowOrder:
    """Validate parallel identities and return their label-free canonical order."""

    samples = tuple(str(value) for value in sample_ids)
    cases = tuple(str(value) for value in case_ids)
    if (
        not samples
        or len(samples) != len(cases)
        or len(samples) != len(set(samples))
        or any(not value for value in (*samples, *cases))
    ):
        raise ProtocolError(error_message)
    permutation = tuple(
        sorted(range(len(samples)), key=lambda index: (cases[index], samples[index]))
    )
    return CanonicalCenterRowOrder(
        permutation=permutation,
        sample_ids=tuple(samples[index] for index in permutation),
        case_ids=tuple(cases[index] for index in permutation),
    )


def require_canonical_center_row_order(
    sample_ids: Sequence[object],
    case_ids: Sequence[object],
    *,
    error_message: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return normalized identities only when already in canonical row order."""

    rows = canonical_center_row_order(
        sample_ids,
        case_ids,
        error_message=error_message,
    )
    if rows.permutation != tuple(range(len(rows.sample_ids))):
        raise ProtocolError(error_message)
    return rows.sample_ids, rows.case_ids


def canonical_sample_ids(
    sample_ids: Sequence[object],
    *,
    error_message: str,
) -> tuple[str, ...]:
    """Return a validated lexicographic sample order for one held case."""

    samples = tuple(str(value) for value in sample_ids)
    if (
        not samples
        or len(samples) != len(set(samples))
        or any(not value for value in samples)
    ):
        raise ProtocolError(error_message)
    return tuple(sorted(samples))


def require_canonical_sample_ids(
    sample_ids: Sequence[object],
    *,
    error_message: str,
) -> tuple[str, ...]:
    """Return normalized per-case sample IDs only when already canonical."""

    samples = tuple(str(value) for value in sample_ids)
    canonical = canonical_sample_ids(samples, error_message=error_message)
    if samples != canonical:
        raise ProtocolError(error_message)
    return canonical


__all__ = (
    "CanonicalCenterRowOrder",
    "canonical_center_row_order",
    "canonical_sample_ids",
    "require_canonical_center_row_order",
    "require_canonical_sample_ids",
)
