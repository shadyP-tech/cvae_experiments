"""Strict H/J/d pseudo-route scopes for donor cross-fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import CENTERS


@dataclass(frozen=True, slots=True, order=True)
class PseudoRouteKey:
    outer_center: str
    donor_center: str
    case_id: str

    def __post_init__(self) -> None:
        if (
            self.outer_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_center == self.donor_center
            or not self.case_id
        ):
            raise GovernanceError("SCALE-BP v2 pseudo-route key drifted.")


@dataclass(frozen=True, slots=True)
class PseudoRouteScope:
    key: PseudoRouteKey
    donor_support_case_ids: tuple[str, ...]
    donor_training_case_ids_by_center: Mapping[str, tuple[str, ...]]
    source_excluded_centers: tuple[str, ...]
    scope_hash: str = field(init=False)

    def __post_init__(self) -> None:
        support = tuple(str(case) for case in self.donor_support_case_ids)
        training = {
            str(center): tuple(str(case) for case in cases)
            for center, cases in self.donor_training_case_ids_by_center.items()
        }
        excluded = tuple(str(center) for center in self.source_excluded_centers)
        expected_centers = tuple(
            center
            for center in CENTERS
            if center not in {self.key.outer_center, self.key.donor_center}
        )
        if (
            not support
            or self.key.case_id in support
            or len(support) != len(set(support))
            or tuple(training) != expected_centers
            or any(not cases for cases in training.values())
            or any(len(cases) != len(set(cases)) for cases in training.values())
            or set(excluded) != {self.key.outer_center, self.key.donor_center}
            or len(excluded) != 2
            or excluded
            != tuple(center for center in CENTERS if center in set(excluded))
        ):
            raise GovernanceError("SCALE-BP v2 pseudo H/J/d scope drifted.")
        object.__setattr__(self, "donor_support_case_ids", support)
        object.__setattr__(
            self,
            "donor_training_case_ids_by_center",
            MappingProxyType(training),
        )
        object.__setattr__(self, "source_excluded_centers", excluded)
        object.__setattr__(
            self,
            "scope_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_pseudo_route_scope_v1",
                    "outer_center": self.key.outer_center,
                    "donor_center": self.key.donor_center,
                    "case_id": self.key.case_id,
                    "donor_support_case_ids": support,
                    "donor_training_case_ids_by_center": training,
                    "source_excluded_centers": excluded,
                    "outer_H_excluded": True,
                    "donor_J_excluded_from_prior_fit": True,
                    "held_d_excluded": True,
                }
            ),
        )


def build_pseudo_route_scopes(
    case_ids_by_center: Mapping[str, tuple[str, ...]],
) -> tuple[PseudoRouteScope, ...]:
    cases = {
        str(center): tuple(str(case) for case in center_cases)
        for center, center_cases in case_ids_by_center.items()
    }
    if (
        tuple(cases) != CENTERS
        or any(not center_cases for center_cases in cases.values())
        or any(len(center_cases) != len(set(center_cases)) for center_cases in cases.values())
    ):
        raise GovernanceError("SCALE-BP v2 pseudo case inventory drifted.")
    output: list[PseudoRouteScope] = []
    for outer in CENTERS:
        for donor in CENTERS:
            if donor == outer:
                continue
            for held_case in cases[donor]:
                output.append(
                    PseudoRouteScope(
                        PseudoRouteKey(outer, donor, held_case),
                        tuple(case for case in cases[donor] if case != held_case),
                        {
                            center: tuple(cases[center])
                            for center in CENTERS
                            if center not in {outer, donor}
                        },
                        tuple(
                            center
                            for center in CENTERS
                            if center in {outer, donor}
                        ),
                    )
                )
    return tuple(output)


__all__ = ("PseudoRouteKey", "PseudoRouteScope", "build_pseudo_route_scopes")
