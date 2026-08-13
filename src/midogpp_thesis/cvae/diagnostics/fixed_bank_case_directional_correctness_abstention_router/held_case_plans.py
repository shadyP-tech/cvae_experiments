"""Label-free whole-case leave-one-out route plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_CASE_COUNTS_BY_CENTER, EXPECTED_TOTAL_CASE_COUNT
from .hashing import canonical_hash, require_sha256, require_stable_hash


@dataclass(frozen=True, order=True)
class HeldCasePlan:
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
            raise ProtocolError("Abstention-router held-case plan drifted.")
        require_stable_hash(self.probability_surface_hash, "probability_surface_hash")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_sample_ids", samples)
        object.__setattr__(self, "plan_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cdca_held_case_plan_v1",
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
class HeldCasePlanSeal:
    plans: tuple[HeldCasePlan, ...]
    probability_surface_hash: str
    plan_seal_hash: str = ""

    def __post_init__(self) -> None:
        plans = tuple(self.plans)
        require_stable_hash(self.probability_surface_hash, "probability_surface_hash")
        counts = {center: sum(plan.target_center == center for plan in plans) for center in CENTERS}
        if (
            len(plans) != EXPECTED_TOTAL_CASE_COUNT
            or len({plan.key for plan in plans}) != len(plans)
            or counts != EXPECTED_CASE_COUNTS_BY_CENTER
            or any(plan.probability_surface_hash != self.probability_surface_hash for plan in plans)
        ):
            raise ProtocolError("Abstention-router plan seal requires canonical 218 plans.")
        expected = canonical_hash(self._unhashed(plans))
        if self.plan_seal_hash and require_sha256(self.plan_seal_hash, "plan_seal_hash") != expected:
            raise ProtocolError("Abstention-router plan seal hash drifted.")
        object.__setattr__(self, "plans", plans)
        object.__setattr__(self, "plan_seal_hash", expected)

    def _unhashed(self, plans: tuple[HeldCasePlan, ...] | None = None) -> dict[str, object]:
        rows = self.plans if plans is None else plans
        return {
            "schema_version": "fixed_bank_cdca_all_218_held_case_plan_seal_v1",
            "probability_surface_hash": self.probability_surface_hash,
            "plan_count": len(rows),
            "plans": [plan.to_payload() for plan in rows],
            "sealed_before_route_labels": True,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_seal_hash": self.plan_seal_hash}


def build_held_case_plans(
    identities: Sequence[object],
    *,
    probability_surface_hash: str,
    expected_total_case_count: int | None = EXPECTED_TOTAL_CASE_COUNT,
) -> tuple[HeldCasePlan, ...]:
    require_stable_hash(probability_surface_hash, "probability_surface_hash")
    samples: dict[tuple[str, str], set[str]] = {}
    groups: dict[tuple[str, str], str] = {}
    observed_rows: set[tuple[str, str, str]] = set()
    for row in identities:
        target = str(getattr(row, "target_center", getattr(row, "center", "")))
        case = str(getattr(row, "case_id", ""))
        sample = str(getattr(row, "sample_id", getattr(row, "evaluation_row_id", "")))
        group = str(getattr(row, "group_id", case))
        key = (target, case, sample)
        if target not in CENTERS or not case or not sample or not group or key in observed_rows:
            raise ProtocolError("Abstention-router plan identity drifted.")
        observed_rows.add(key)
        samples.setdefault((target, case), set()).add(sample)
        if groups.setdefault((target, case), group) != group:
            raise ProtocolError("Abstention-router case spans multiple groups.")
    cases_by_center = {
        center: tuple(sorted(case for target, case in samples if target == center))
        for center in CENTERS
    }
    cases_by_group: dict[tuple[str, str], set[str]] = {}
    for (target, case), group in groups.items():
        cases_by_group.setdefault((target, group), set()).add(case)
    if any(len(cases) != 1 for cases in cases_by_group.values()):
        raise ProtocolError(
            "Abstention-router group spans multiple cases; held unit is ambiguous."
        )
    plans = tuple(
        HeldCasePlan(
            center,
            case,
            groups[(center, case)],
            tuple(value for value in cases_by_center[center] if value != case),
            tuple(sorted(samples[(center, case)])),
            probability_surface_hash,
        )
        for center in CENTERS
        for case in cases_by_center[center]
    )
    if expected_total_case_count is not None and len(plans) != int(expected_total_case_count):
        raise ProtocolError("Abstention-router whole-case count drifted.")
    return plans


def seal_held_case_plans(
    plans: Sequence[HeldCasePlan], *, probability_surface_hash: str
) -> HeldCasePlanSeal:
    canonical = tuple(sorted(plans, key=lambda plan: (CENTERS.index(plan.target_center), plan.case_id)))
    return HeldCasePlanSeal(canonical, probability_surface_hash)


__all__ = (
    "HeldCasePlan",
    "HeldCasePlanSeal",
    "build_held_case_plans",
    "seal_held_case_plans",
)
