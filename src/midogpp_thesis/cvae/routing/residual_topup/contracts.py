"""Immutable contracts for additive residual top-up routing.

This package deliberately contains no experiment-stage or artifact concepts.
It describes only a class-agnostic action around an immutable equal-union
base and the deterministic classwise windows needed to realize that action.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping

from ...protocol import ProtocolError


CLASS_LABELS = (0, 1)
TOPUP_FRACTION_NUMERATOR = 1
TOPUP_FRACTION_DENOMINATOR = 8
MAX_FINAL_SOURCE_WEIGHT = 0.25
MIN_FINAL_EFFECTIVE_SOURCES = 6.0

TARGET_SOURCE_COUNT = 8
TARGET_BASE_PER_SOURCE = 128
TARGET_TOPUP_TOTAL_PER_CLASS = 128
INNER_SOURCE_COUNT = 7
INNER_BASE_PER_SOURCE = 144
INNER_TOPUP_TOTAL_PER_CLASS = 126


@dataclass(frozen=True)
class TopupGeometry:
    """Exact equal-union base and additive per-class budget."""

    source_order: tuple[str, ...]
    base_per_source: int
    topup_total_per_class: int
    class_labels: tuple[int, int] = CLASS_LABELS
    topup_fraction_numerator: int = TOPUP_FRACTION_NUMERATOR
    topup_fraction_denominator: int = TOPUP_FRACTION_DENOMINATOR

    def __post_init__(self) -> None:
        sources = _canonical_sources(self.source_order)
        if sources != self.source_order:
            raise ProtocolError(
                "Residual top-up geometry source_order must be canonical."
            )
        if (
            isinstance(self.base_per_source, bool)
            or not isinstance(self.base_per_source, Integral)
            or isinstance(self.topup_total_per_class, bool)
            or not isinstance(self.topup_total_per_class, Integral)
        ):
            raise ProtocolError("Residual top-up geometry counts must be integers.")
        base = int(self.base_per_source)
        topup = int(self.topup_total_per_class)
        if base <= 0 or topup <= 0 or tuple(self.class_labels) != CLASS_LABELS:
            raise ProtocolError("Residual top-up geometry is invalid.")
        if (
            self.topup_fraction_numerator != TOPUP_FRACTION_NUMERATOR
            or self.topup_fraction_denominator != TOPUP_FRACTION_DENOMINATOR
            or topup * TOPUP_FRACTION_DENOMINATOR
            != self.base_total_per_class * TOPUP_FRACTION_NUMERATOR
        ):
            raise ProtocolError(
                "Residual top-up requires topup_total/base_total == 1/8 exactly."
            )
        object.__setattr__(self, "base_per_source", base)
        object.__setattr__(self, "topup_total_per_class", topup)

    @property
    def source_count(self) -> int:
        return len(self.source_order)

    @property
    def base_total_per_class(self) -> int:
        return self.base_per_source * self.source_count

    @property
    def final_total_per_class(self) -> int:
        return self.base_total_per_class + self.topup_total_per_class

    def to_payload(self) -> dict[str, object]:
        return {
            "source_order": list(self.source_order),
            "source_count": self.source_count,
            "base_per_source": self.base_per_source,
            "base_total_per_class": self.base_total_per_class,
            "topup_total_per_class": self.topup_total_per_class,
            "final_total_per_class": self.final_total_per_class,
            "class_labels": list(self.class_labels),
            "topup_fraction": {
                "numerator": self.topup_fraction_numerator,
                "denominator": self.topup_fraction_denominator,
            },
        }


@dataclass(frozen=True)
class SourceClassWindows:
    """Disjoint base and top-up windows within one source/class stream."""

    base_start: int
    base_stop: int
    topup_start: int
    topup_stop: int

    def __post_init__(self) -> None:
        values = (
            self.base_start,
            self.base_stop,
            self.topup_start,
            self.topup_stop,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in values):
            raise ProtocolError("Residual top-up window bounds must be integers.")
        if not (
            self.base_start == 0
            and self.base_stop > self.base_start
            and self.topup_start == self.base_stop
            and self.topup_stop >= self.topup_start
        ):
            raise ProtocolError(
                "Residual top-up windows must be contiguous and disjoint."
            )

    @property
    def base_count(self) -> int:
        return self.base_stop - self.base_start

    @property
    def topup_count(self) -> int:
        return self.topup_stop - self.topup_start

    @property
    def required_capacity(self) -> int:
        return self.topup_stop

    def to_payload(self) -> dict[str, object]:
        return {
            "base": [self.base_start, self.base_stop],
            "topup": [self.topup_start, self.topup_stop],
            "base_count": self.base_count,
            "topup_count": self.topup_count,
            "required_capacity": self.required_capacity,
        }


@dataclass(frozen=True)
class HamiltonTopupAllocation:
    """Canonical Hamilton realization of one fixed top-up budget."""

    source_order: tuple[str, ...]
    normalized_weights: Mapping[str, float]
    quotas: Mapping[str, float]
    counts: Mapping[str, int]
    topup_total_per_class: int
    remainder_order: tuple[str, ...]
    allocation_hash: str
    allocation_semantics: str

    def to_payload(self) -> dict[str, object]:
        return {
            "source_order": list(self.source_order),
            "normalized_weights": dict(self.normalized_weights),
            "quotas": dict(self.quotas),
            "counts": dict(self.counts),
            "topup_total_per_class": self.topup_total_per_class,
            "remainder_order": list(self.remainder_order),
            "allocation_hash": self.allocation_hash,
            "allocation_semantics": self.allocation_semantics,
        }


@dataclass(frozen=True)
class ResidualTopupAction:
    """Auditable class-agnostic top-up action and its final mixture."""

    geometry: TopupGeometry
    action_kind: str
    direction_semantics: str
    temperature: float | None
    calibrated_energy_by_source: Mapping[str, float]
    direction_weights: Mapping[str, float]
    topup_counts: Mapping[str, int]
    final_counts_by_class: Mapping[int, Mapping[str, int]]
    final_weights_by_class: Mapping[int, Mapping[str, float]]
    windows_by_class: Mapping[int, Mapping[str, SourceClassWindows]]
    effective_source_count_by_class: Mapping[int, float]
    maximum_source_weight: float
    allocation_hash: str
    window_hash: str
    action_hash: str
    density_constraint_semantics: str = "validated_after_addition_without_projection"

    def to_payload(self) -> dict[str, object]:
        return {
            "geometry": self.geometry.to_payload(),
            "action_kind": self.action_kind,
            "direction_semantics": self.direction_semantics,
            "temperature": self.temperature,
            "calibrated_energy_by_source": dict(
                self.calibrated_energy_by_source
            ),
            "direction_weights": dict(self.direction_weights),
            "topup_counts": dict(self.topup_counts),
            "final_counts_by_class": {
                str(label): dict(self.final_counts_by_class[label])
                for label in self.geometry.class_labels
            },
            "final_weights_by_class": {
                str(label): dict(self.final_weights_by_class[label])
                for label in self.geometry.class_labels
            },
            "windows_by_class": {
                str(label): {
                    source: self.windows_by_class[label][source].to_payload()
                    for source in self.geometry.source_order
                }
                for label in self.geometry.class_labels
            },
            "effective_source_count_by_class": {
                str(label): self.effective_source_count_by_class[label]
                for label in self.geometry.class_labels
            },
            "maximum_source_weight": self.maximum_source_weight,
            "allocation_hash": self.allocation_hash,
            "window_hash": self.window_hash,
            "action_hash": self.action_hash,
            "density_constraint_semantics": self.density_constraint_semantics,
        }


def build_topup_geometry(
    candidate_sources: Iterable[object],
    *,
    base_per_source: int,
    topup_total_per_class: int,
) -> TopupGeometry:
    """Build a generic, canonical geometry with the frozen one-eighth ratio."""

    return TopupGeometry(
        source_order=_canonical_sources(candidate_sources),
        base_per_source=base_per_source,
        topup_total_per_class=topup_total_per_class,
    )


def target_topup_geometry(candidate_sources: Iterable[object]) -> TopupGeometry:
    """Build the frozen eight-source, 1024+128 target geometry."""

    sources = _canonical_sources(candidate_sources)
    if len(sources) != TARGET_SOURCE_COUNT:
        raise ProtocolError("Target residual top-up geometry requires eight sources.")
    geometry = build_topup_geometry(
        sources,
        base_per_source=TARGET_BASE_PER_SOURCE,
        topup_total_per_class=TARGET_TOPUP_TOTAL_PER_CLASS,
    )
    return geometry


def inner_topup_geometry(candidate_sources: Iterable[object]) -> TopupGeometry:
    """Build the frozen seven-source, 1008+126 inner geometry."""

    sources = _canonical_sources(candidate_sources)
    if len(sources) != INNER_SOURCE_COUNT:
        raise ProtocolError("Inner residual top-up geometry requires seven sources.")
    geometry = build_topup_geometry(
        sources,
        base_per_source=INNER_BASE_PER_SOURCE,
        topup_total_per_class=INNER_TOPUP_TOTAL_PER_CLASS,
    )
    return geometry


def immutable_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Return a shallow immutable copy for frozen contract construction."""

    return MappingProxyType(dict(values))


def immutable_nested_mapping(
    values: Mapping[int, Mapping[str, object]],
) -> Mapping[int, Mapping[str, object]]:
    """Return an immutable class-to-source mapping."""

    return MappingProxyType(
        {
            int(label): MappingProxyType(dict(source_values))
            for label, source_values in values.items()
        }
    )


def _canonical_sources(values: Iterable[object]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for raw_source in values:
        source = str(raw_source)
        if not source or source.strip() != source or source in normalized:
            raise ProtocolError(
                "Residual top-up source identifiers must be unique and nonempty."
            )
        normalized.add(source)
    if not normalized:
        raise ProtocolError("Residual top-up requires at least one source.")
    return tuple(sorted(normalized))


__all__ = (
    "CLASS_LABELS",
    "INNER_BASE_PER_SOURCE",
    "INNER_SOURCE_COUNT",
    "INNER_TOPUP_TOTAL_PER_CLASS",
    "MAX_FINAL_SOURCE_WEIGHT",
    "MIN_FINAL_EFFECTIVE_SOURCES",
    "TARGET_BASE_PER_SOURCE",
    "TARGET_SOURCE_COUNT",
    "TARGET_TOPUP_TOTAL_PER_CLASS",
    "TOPUP_FRACTION_DENOMINATOR",
    "TOPUP_FRACTION_NUMERATOR",
    "HamiltonTopupAllocation",
    "ResidualTopupAction",
    "SourceClassWindows",
    "TopupGeometry",
    "build_topup_geometry",
    "immutable_mapping",
    "immutable_nested_mapping",
    "inner_topup_geometry",
    "target_topup_geometry",
)
