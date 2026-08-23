"""Whole-case plans and ephemeral scoped-label rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .identity import canonical_hash


@dataclass(frozen=True)
class BinaryLabel:
    center: str
    case_id: str
    sample_id: str
    value: int
    scope: str

    def __post_init__(self) -> None:
        if (
            str(self.center) not in CENTERS
            or not str(self.case_id)
            or not str(self.sample_id)
            or int(self.value) not in {0, 1}
            or not str(self.scope)
        ):
            raise ProtocolError("P-DCAPS scoped label row drifted.")

    @property
    def key(self) -> tuple[str, str, str]:
        return str(self.center), str(self.case_id), str(self.sample_id)


@dataclass(frozen=True)
class WholeCasePlan:
    target_center: str
    case_id: str
    support_case_ids: tuple[str, ...]
    evaluation_sample_ids: tuple[str, ...]
    physical_surface_hash: str
    center_surface_hash: str
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        support = tuple(sorted(str(value) for value in self.support_case_ids))
        evaluation = tuple(str(value) for value in self.evaluation_sample_ids)
        if (
            str(self.target_center) not in CENTERS
            or str(self.case_id) in support
            or not support
            or not evaluation
            or len(support) != len(set(support))
            or len(evaluation) != len(set(evaluation))
        ):
            raise ProtocolError("P-DCAPS whole-case plan drifted.")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_sample_ids", evaluation)
        object.__setattr__(
            self,
            "plan_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_whole_case_plan_v1",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "support_case_ids": support,
                    "evaluation_sample_ids": evaluation,
                    "physical_surface_hash": self.physical_surface_hash,
                    "center_surface_hash": self.center_surface_hash,
                    "whole_case_excluded": True,
                    "labels_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "target_center": self.target_center,
            "case_id": self.case_id,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_sample_ids": list(self.evaluation_sample_ids),
            "physical_surface_hash": self.physical_surface_hash,
            "center_surface_hash": self.center_surface_hash,
            "whole_case_excluded": True,
            "labels_used": False,
            "plan_hash": self.plan_hash,
        }


@dataclass(frozen=True)
class OrderedPseudoPlan:
    outer_center: str
    scored_center: str
    held_case_id: str
    outer_excluded: str
    scored_excluded: str
    source_plan_hash: str
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            self.outer_center not in CENTERS
            or self.scored_center not in CENTERS
            or self.outer_center == self.scored_center
            or self.outer_excluded != self.outer_center
            or self.scored_excluded != self.scored_center
            or not self.held_case_id
        ):
            raise ProtocolError("P-DCAPS ordered H/J/d plan drifted.")
        object.__setattr__(
            self,
            "plan_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_ordered_pseudo_plan_v1",
                    "outer_center": self.outer_center,
                    "scored_center": self.scored_center,
                    "held_case_id": self.held_case_id,
                    "excluded_outer_center": self.outer_excluded,
                    "excluded_scored_center": self.scored_excluded,
                    "source_plan_hash": self.source_plan_hash,
                    "held_case_role": "SCORED_RESPONSE_ONLY_AFTER_SURFACE_SEAL",
                }
            ),
        )


def build_whole_case_plans(
    rows: Iterable[object],
    *,
    physical_surface_hash: str,
    center_hashes: dict[str, str],
) -> tuple[WholeCasePlan, ...]:
    identities = tuple(
        (
            str(getattr(row, "center")),
            str(getattr(row, "case_id")),
            str(getattr(row, "sample_id")),
        )
        for row in rows
    )
    plans: list[WholeCasePlan] = []
    for center in CENTERS:
        center_rows = tuple(row for row in identities if row[0] == center)
        cases = tuple(sorted({row[1] for row in center_rows}))
        for case in cases:
            plans.append(
                WholeCasePlan(
                    center,
                    case,
                    tuple(value for value in cases if value != case),
                    tuple(row[2] for row in center_rows if row[1] == case),
                    physical_surface_hash,
                    center_hashes[center],
                )
            )
    return tuple(plans)


def require_exact_label_scope(
    labels: Sequence[BinaryLabel],
    *,
    expected_keys: Sequence[tuple[str, str, str]],
    expected_scope: str,
) -> tuple[BinaryLabel, ...]:
    rows = tuple(labels)
    if (
        len(rows) != len(expected_keys)
        or len({row.key for row in rows}) != len(rows)
        or {row.key for row in rows} != set(expected_keys)
        or {row.scope for row in rows} != {str(expected_scope)}
    ):
        raise ProtocolError("P-DCAPS label capability escaped its exact scope.")
    return rows


__all__ = (
    "BinaryLabel",
    "OrderedPseudoPlan",
    "WholeCasePlan",
    "build_whole_case_plans",
    "require_exact_label_scope",
)
