"""Deterministic whole-case/group leave-one-out plans for all 218 cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import CENTERS, EXPECTED_CASE_COUNTS_BY_CENTER, EXPECTED_TOTAL_CASE_COUNT
from .hashing import canonical_hash, require_sha256, require_stable_hash


@dataclass(frozen=True, order=True)
class WholeCaseLooPlan:
    target_center: str
    case_id: str
    group_id: str
    support_case_ids: tuple[str, ...]
    evaluation_sample_ids: tuple[str, ...]
    probability_surface_hash: str
    plan_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        target = str(self.target_center)
        case = str(self.case_id)
        group = str(self.group_id)
        support = tuple(sorted(str(value) for value in self.support_case_ids))
        samples = tuple(sorted(str(value) for value in self.evaluation_sample_ids))
        if target not in CENTERS or not case or not group:
            raise ProtocolError("DCSE LOO plan identity is malformed.")
        if not support or not samples or len(support) != len(set(support)) or len(samples) != len(set(samples)):
            raise ProtocolError("DCSE LOO support/evaluation identities must be unique and non-empty.")
        if case in support:
            raise ProtocolError("Held case c entered H-minus-c support.")
        require_stable_hash(self.probability_surface_hash, "probability_surface_hash")
        object.__setattr__(self, "target_center", target)
        object.__setattr__(self, "case_id", case)
        object.__setattr__(self, "group_id", group)
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_sample_ids", samples)
        object.__setattr__(self, "plan_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_whole_case_group_loo_plan_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "group_id": self.group_id,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_sample_ids": list(self.evaluation_sample_ids),
            "probability_surface_hash": self.probability_surface_hash,
            "held_case_excluded_from_support": True,
            "held_group_excluded_from_support": True,
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_hash": self.plan_hash}


LooPlan = WholeCaseLooPlan


@dataclass(frozen=True)
class LooPlanSeal:
    plans: tuple[WholeCaseLooPlan, ...]
    probability_surface_hash: str
    plan_seal_hash: str = ""

    def __post_init__(self) -> None:
        plans = tuple(self.plans)
        require_stable_hash(self.probability_surface_hash, "probability_surface_hash")
        expected_keys = tuple(
            (center, plan.case_id)
            for center in CENTERS
            for plan in plans
            if plan.target_center == center
        )
        if (
            len(plans) != EXPECTED_TOTAL_CASE_COUNT
            or len({plan.key for plan in plans}) != EXPECTED_TOTAL_CASE_COUNT
            or tuple(plan.key for plan in plans) != expected_keys
            or any(plan.probability_surface_hash != self.probability_surface_hash for plan in plans)
        ):
            raise ProtocolError("DCSE global LOO seal requires exactly 218 canonical plans.")
        counts = {center: sum(plan.target_center == center for plan in plans) for center in CENTERS}
        if counts != EXPECTED_CASE_COUNTS_BY_CENTER:
            raise ProtocolError("DCSE global LOO plan count by center drifted.")
        expected = canonical_hash(self._unhashed(plans))
        if self.plan_seal_hash:
            if require_sha256(self.plan_seal_hash, "plan_seal_hash") != expected:
                raise ProtocolError("DCSE global LOO plan seal hash drifted.")
        else:
            object.__setattr__(self, "plan_seal_hash", expected)
        object.__setattr__(self, "plans", plans)

    def _unhashed(self, plans: tuple[WholeCaseLooPlan, ...] | None = None) -> dict[str, object]:
        rows = self.plans if plans is None else plans
        return {
            "schema_version": "fixed_bank_dcse_all_218_loo_plan_seal_v1",
            "probability_surface_hash": self.probability_surface_hash,
            "plan_count": len(rows),
            "plans": [plan.to_payload() for plan in rows],
            "all_plans_sealed_before_route_scoped_labels": True,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_seal_hash": self.plan_seal_hash}

    def plan(self, target_center: str, case_id: str) -> WholeCaseLooPlan:
        for value in self.plans:
            if value.key == (str(target_center), str(case_id)):
                return value
        raise KeyError((target_center, case_id))


def build_whole_case_loo_plans(
    identities: Sequence[object],
    *,
    probability_surface_hash: str,
    expected_total_case_count: int | None = EXPECTED_TOTAL_CASE_COUNT,
) -> tuple[WholeCaseLooPlan, ...]:
    """Build one label-blind plan per case, with group identity fail-closed."""

    require_stable_hash(probability_surface_hash, "probability_surface_hash")
    rows = tuple(identities)
    if not rows:
        raise ProtocolError("DCSE cannot build LOO plans from an empty identity surface.")
    samples: dict[tuple[str, str], set[str]] = {}
    group_by_case: dict[tuple[str, str], str] = {}
    cases_by_group: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        target = str(getattr(row, "target_center", getattr(row, "center", "")))
        case = str(getattr(row, "case_id", ""))
        sample = str(getattr(row, "sample_id", getattr(row, "evaluation_row_id", "")))
        group = str(getattr(row, "group_id", case))
        if target not in CENTERS or not case or not sample or not group:
            raise ProtocolError("DCSE LOO identity row is malformed.")
        key = (target, case)
        samples.setdefault(key, set()).add(sample)
        previous = group_by_case.setdefault(key, group)
        if previous != group:
            raise ProtocolError("One DCSE case maps to multiple grouping identities.")
        cases_by_group.setdefault((target, group), set()).add(case)
    if any(len(cases) != 1 for cases in cases_by_group.values()):
        raise ProtocolError("DCSE group spans multiple cases; define the held unit canonically first.")
    if len({(str(getattr(row, "target_center", getattr(row, "center", ""))), str(getattr(row, "case_id", "")), str(getattr(row, "sample_id", getattr(row, "evaluation_row_id", "")))) for row in rows}) != len(rows):
        raise ProtocolError("DCSE LOO identity rows are duplicated.")
    cases_by_center = {
        center: tuple(sorted(case for target, case in samples if target == center))
        for center in CENTERS
    }
    plans = tuple(
        WholeCaseLooPlan(
            target_center=center,
            case_id=case,
            group_id=group_by_case[(center, case)],
            support_case_ids=tuple(value for value in cases_by_center[center] if value != case),
            evaluation_sample_ids=tuple(sorted(samples[(center, case)])),
            probability_surface_hash=probability_surface_hash,
        )
        for center in CENTERS
        for case in cases_by_center[center]
    )
    if expected_total_case_count is not None and len(plans) != int(expected_total_case_count):
        raise ProtocolError(
            f"DCSE expected {expected_total_case_count} whole cases, found {len(plans)}."
        )
    return plans


def seal_loo_plans(
    plans: Sequence[WholeCaseLooPlan], *, probability_surface_hash: str
) -> LooPlanSeal:
    canonical = tuple(
        sorted(
            plans,
            key=lambda plan: (CENTERS.index(plan.target_center), plan.case_id),
        )
    )
    return LooPlanSeal(canonical, probability_surface_hash)


def plan_index(plans: Sequence[WholeCaseLooPlan]) -> Mapping[tuple[str, str], WholeCaseLooPlan]:
    result = {plan.key: plan for plan in plans}
    if len(result) != len(plans):
        raise ProtocolError("DCSE LOO plan index contains duplicates.")
    return MappingProxyType(result)


__all__ = (
    "LooPlan",
    "LooPlanSeal",
    "WholeCaseLooPlan",
    "build_whole_case_loo_plans",
    "plan_index",
    "seal_loo_plans",
)
