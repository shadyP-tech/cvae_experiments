"""Canonical effective-sample-size viability for the P-DCAPS v2 bank."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ..contracts import BankViability
from ..identity import ACTION_FAMILIES
from ..physical_actions import (
    B_ACTION_ID,
    PhysicalActionSpec,
    action_library_by_target,
    candidate_sources,
)
from .identity import canonical_hash


MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS = 5.0
EFFECTIVE_SAMPLE_SIZE_FORMULA = "SQUARED_SUM_WEIGHT_OVER_SUM_SQUARED_WEIGHT"


def effective_sample_size_per_class(
    action: PhysicalActionSpec,
) -> tuple[tuple[str, float], ...]:
    """Compute Kish ESS from the frozen per-source row counts and weights."""

    weights = dict(action.sample_weight_by_source)
    rows: list[tuple[str, float]] = []
    for label, counts_by_source in action.counts_by_class:
        weighted_sum = float(
            sum(count * weights[source] for source, count in counts_by_source)
        )
        squared_weight_sum = float(
            sum(
                count * weights[source] * weights[source]
                for source, count in counts_by_source
            )
        )
        if weighted_sum <= 0.0 or squared_weight_sum <= 0.0:
            raise ProtocolError("P-DCAPS v2 physical action ESS is undefined.")
        ess = weighted_sum * weighted_sum / squared_weight_sum
        if not np.isfinite(ess) or ess <= 0.0:
            raise ProtocolError("P-DCAPS v2 physical action ESS is nonfinite.")
        rows.append((str(label), float(ess)))
    if tuple(label for label, _value in rows) != ("0", "1"):
        raise ProtocolError("P-DCAPS v2 physical action class domain drifted.")
    return tuple(rows)


@dataclass(frozen=True)
class CanonicalBankViability:
    """Route-scoped B/I/R bank viability and its exact physical witnesses."""

    target_center: str
    excluded_source_centers: tuple[str, ...]
    rows: tuple[tuple[str, BankViability], ...]
    minimum_effective_sample_size: float
    viability_set_hash: str = field(init=False)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        excluded = tuple(str(value) for value in self.excluded_source_centers)
        rows = tuple((str(family), value) for family, value in self.rows)
        sources = candidate_sources(target)
        if (
            tuple(family for family, _value in rows) != ACTION_FAMILIES
            or len(excluded) != len(set(excluded))
            or any(value not in sources for value in excluded)
            or float(self.minimum_effective_sample_size)
            != MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS
            or any(not isinstance(value, BankViability) for _family, value in rows)
            or any(
                value.minimum_effective_sample_size
                != MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS
                for _family, value in rows
            )
        ):
            raise ProtocolError("P-DCAPS v2 canonical bank viability drifted.")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "excluded_source_centers", excluded)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(
            self,
            "viability_set_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_v2_canonical_bank_viability_v1",
                    "target_center": target,
                    "excluded_source_centers": excluded,
                    "minimum_effective_sample_size_per_class": (
                        MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS
                    ),
                    "formula": EFFECTIVE_SAMPLE_SIZE_FORMULA,
                    "family_viability": tuple(
                        (family, value.to_payload()) for family, value in rows
                    ),
                    "all_families_passed": all(value.passed for _family, value in rows),
                    "labels_used": False,
                }
            ),
        )

    @property
    def passed(self) -> bool:
        return all(value.passed for _family, value in self.rows)

    def as_mapping(self) -> dict[str, BankViability]:
        """Return a fresh ordinary dict; mapping proxies never cross workers."""

        return dict(self.rows)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_v2_canonical_bank_viability_v1",
            "target_center": self.target_center,
            "excluded_source_centers": list(self.excluded_source_centers),
            "minimum_effective_sample_size_per_class": (
                self.minimum_effective_sample_size
            ),
            "formula": EFFECTIVE_SAMPLE_SIZE_FORMULA,
            "family_viability": [
                [family, value.to_payload()] for family, value in self.rows
            ],
            "all_families_passed": self.passed,
            "labels_used": False,
            "viability_set_hash": self.viability_set_hash,
        }


def _family_viability(
    *,
    target_center: str,
    family: str,
    actions: Sequence[PhysicalActionSpec],
    excluded_source_centers: Sequence[str],
) -> BankViability:
    action_rows = tuple(actions)
    if not action_rows:
        raise ProtocolError("P-DCAPS v2 viability family has no physical actions.")
    ess_by_action = tuple(
        (action.action_hash, effective_sample_size_per_class(action))
        for action in action_rows
    )
    minima = tuple(
        (
            label,
            min(dict(values)[label] for _action_hash, values in ess_by_action),
        )
        for label in ("0", "1")
    )
    support_preserved = all(
        all(count > 0 for _source, count in counts)
        for action in action_rows
        for _label, counts in action.counts_by_class
    )
    provenance_hash = canonical_hash(
        {
            "schema_version": "pdcaps_v2_bank_viability_provenance_v1",
            "target_center": target_center,
            "family": family,
            "excluded_source_centers": tuple(excluded_source_centers),
            "physical_action_hashes": tuple(
                action.action_hash for action in action_rows
            ),
            "effective_sample_size_by_action": ess_by_action,
            "per_class_minimum_effective_sample_size": minima,
            "minimum_effective_sample_size_per_class": (
                MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS
            ),
            "formula": EFFECTIVE_SAMPLE_SIZE_FORMULA,
            "row_preserving": True,
            "class_domain_support_preserved": support_preserved,
            "labels_used": False,
        }
    )
    return BankViability(
        True,
        support_preserved,
        minima,
        MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS,
        provenance_hash,
    )


def build_canonical_bank_viability(
    target_center: str,
    *,
    excluded_source_centers: Sequence[str] = (),
) -> CanonicalBankViability:
    """Bind B and the route-eligible A1 library to all three endpoint families."""

    target = str(target_center)
    sources = candidate_sources(target)
    raw_excluded = tuple(str(value) for value in excluded_source_centers)
    excluded = tuple(source for source in sources if source in set(raw_excluded))
    if len(excluded) != len(raw_excluded):
        raise ProtocolError("P-DCAPS v2 viability source exclusion drifted.")
    library = action_library_by_target()[target]
    baseline = tuple(action for action in library if action.action_id == B_ACTION_ID)
    route_a1 = tuple(
        action
        for action in library
        if action.selected_source is not None
        and action.selected_source not in excluded
    )
    rows = (
        (
            ACTION_FAMILIES[0],
            _family_viability(
                target_center=target,
                family=ACTION_FAMILIES[0],
                actions=baseline,
                excluded_source_centers=excluded,
            ),
        ),
        *(
            (
                family,
                _family_viability(
                    target_center=target,
                    family=family,
                    actions=route_a1,
                    excluded_source_centers=excluded,
                ),
            )
            for family in ACTION_FAMILIES[1:]
        ),
    )
    result = CanonicalBankViability(
        target,
        excluded,
        rows,
        MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS,
    )
    if not result.passed:
        raise ProtocolError("P-DCAPS v2 canonical bank viability failed.")
    return result


def viability_mapping(
    target_center: str,
    *,
    excluded_source_centers: Sequence[str] = (),
) -> Mapping[str, BankViability]:
    """Compatibility helper for the package-local action-surface kernel."""

    return build_canonical_bank_viability(
        target_center,
        excluded_source_centers=excluded_source_centers,
    ).as_mapping()


__all__ = (
    "CanonicalBankViability",
    "EFFECTIVE_SAMPLE_SIZE_FORMULA",
    "MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS",
    "build_canonical_bank_viability",
    "effective_sample_size_per_class",
    "viability_mapping",
)
