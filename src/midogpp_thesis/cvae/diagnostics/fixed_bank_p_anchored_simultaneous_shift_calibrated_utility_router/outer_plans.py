"""Label-free whole-case exclusion plans for PSSCUR endpoint reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_CASE_COUNTS_BY_CENTER, EXPECTED_OUTER_PLAN_COUNT
from .hashing import canonical_hash, require_sha256


@dataclass(frozen=True, order=True)
class WholeCaseOuterPlan:
    target_center: str
    case_id: str
    group_id: str
    support_case_ids: tuple[str, ...]
    evaluation_sample_ids: tuple[str, ...]
    probability_surface_hash: str
    plan_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        support = tuple(sorted(str(value) for value in self.support_case_ids))
        samples = tuple(sorted(str(value) for value in self.evaluation_sample_ids))
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not self.group_id
            or not support
            or not samples
            or self.case_id in support
            or len(support) != len(set(support))
            or len(samples) != len(set(samples))
        ):
            raise ProtocolError("PSSCUR outer whole-case plan drifted.")
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_sample_ids", samples)
        object.__setattr__(self, "plan_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_outer_plan_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "group_id": self.group_id,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_sample_ids": list(self.evaluation_sample_ids),
            "probability_surface_hash": self.probability_surface_hash,
            "held_case_and_group_excluded": True,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_hash": self.plan_hash}


@dataclass(frozen=True)
class OuterPlanSeal:
    outer_plans: tuple[WholeCaseOuterPlan, ...]
    probability_surface_hash: str
    strict_canonical_topology: bool = True
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        plans = tuple(self.outer_plans)
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        if (
            not plans
            or len({row.key for row in plans}) != len(plans)
            or any(row.probability_surface_hash != self.probability_surface_hash for row in plans)
        ):
            raise ProtocolError("PSSCUR outer plan seal drifted.")
        if self.strict_canonical_topology:
            counts = {
                center: sum(row.target_center == center for row in plans)
                for center in CENTERS
            }
            if len(plans) != EXPECTED_OUTER_PLAN_COUNT or counts != dict(EXPECTED_CASE_COUNTS_BY_CENTER):
                raise ProtocolError("PSSCUR canonical outer plan topology drifted.")
        payload = {
            "schema_version": "fixed_bank_psscur_outer_plan_seal_v1",
            "probability_surface_hash": self.probability_surface_hash,
            "outer_plan_count": len(plans),
            "outer_plan_hashes": [row.plan_hash for row in plans],
            "strict_canonical_topology": self.strict_canonical_topology,
            "double_exclusion_states_used": False,
            "sealed_before_any_label_access": True,
        }
        object.__setattr__(self, "outer_plans", plans)
        object.__setattr__(self, "seal_hash", canonical_hash(payload))

    @property
    def outer_by_key(self) -> Mapping[tuple[str, str], WholeCaseOuterPlan]:
        return MappingProxyType({row.key: row for row in self.outer_plans})

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_psscur_outer_plan_seal_v1",
            "probability_surface_hash": self.probability_surface_hash,
            "outer_plans": [row.to_payload() for row in self.outer_plans],
            "strict_canonical_topology": self.strict_canonical_topology,
            "double_exclusion_states_used": False,
            "seal_hash": self.seal_hash,
        }


def build_outer_plans(
    identities: Sequence[object],
    *,
    probability_surface_hash: str,
    strict_canonical_topology: bool = True,
) -> OuterPlanSeal:
    """Freeze all H-c support/evaluation identities before labels can open."""

    require_sha256(probability_surface_hash, "probability_surface_hash")
    samples: dict[tuple[str, str], set[str]] = {}
    groups: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in identities:
        center = str(getattr(row, "center", getattr(row, "target_center", "")))
        case = str(getattr(row, "case_id", ""))
        sample = str(getattr(row, "sample_id", getattr(row, "evaluation_row_id", "")))
        group = str(getattr(row, "group_id", case))
        key = (center, case, sample)
        if center not in CENTERS or not case or not sample or not group or key in seen:
            raise ProtocolError("PSSCUR outer plan identity row drifted.")
        seen.add(key)
        samples.setdefault((center, case), set()).add(sample)
        if groups.setdefault((center, case), group) != group:
            raise ProtocolError("PSSCUR case maps to multiple group identities.")
    group_cases: dict[tuple[str, str], set[str]] = {}
    for (center, case), group in groups.items():
        group_cases.setdefault((center, group), set()).add(case)
    if any(len(cases) != 1 for cases in group_cases.values()):
        raise ProtocolError("PSSCUR held group spans multiple cases.")
    cases_by_center = {
        center: tuple(sorted(case for observed, case in samples if observed == center))
        for center in CENTERS
    }
    return OuterPlanSeal(
        tuple(
            WholeCaseOuterPlan(
                center,
                case,
                groups[(center, case)],
                tuple(value for value in cases_by_center[center] if value != case),
                tuple(sorted(samples[(center, case)])),
                probability_surface_hash,
            )
            for center in CENTERS
            for case in cases_by_center[center]
        ),
        probability_surface_hash,
        strict_canonical_topology=strict_canonical_topology,
    )


__all__ = ("OuterPlanSeal", "WholeCaseOuterPlan", "build_outer_plans")
