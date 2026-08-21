"""Label-free whole-case and ordered H/J plans for CBPUPR.

The ordinary outer plans own the 218 ``H\\c`` endpoint/posterior routes.  The
role-exclusion plans are a second, label-free topology: for every ordered
``(H, J)`` pair they freeze the seven centers eligible for endpoint-source and
donor-calibration roles when ``J`` is replayed as a pseudo target for ``H``.
The reused ``J-d`` posterior contains frozen, label-free ``A1::source=H``
fingerprint covariates; it does not contain H rows or labels.  Keeping both
topologies in one pre-label seal makes that narrower contract explicit before
any response or evaluation label is opened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT,
    EXPECTED_OUTER_PLAN_COUNT,
)
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
            raise ProtocolError("CBPUPR outer whole-case plan drifted.")
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        object.__setattr__(self, "support_case_ids", support)
        object.__setattr__(self, "evaluation_sample_ids", samples)
        object.__setattr__(self, "plan_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str]:
        return self.target_center, self.case_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_outer_plan_v1",
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


@dataclass(frozen=True, order=True)
class DoubleExclusionPlan:
    """One ordered outer-target/pseudo-target role-exclusion scope."""

    outer_target_center: str
    pseudo_target_center: str
    eligible_source_and_calibration_centers: tuple[str, ...]
    pseudo_case_ids: tuple[str, ...]
    probability_surface_hash: str
    plan_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        eligible = tuple(
            str(value) for value in self.eligible_source_and_calibration_centers
        )
        cases = tuple(sorted(str(value) for value in self.pseudo_case_ids))
        excluded = {self.outer_target_center, self.pseudo_target_center}
        if (
            self.outer_target_center not in CENTERS
            or self.pseudo_target_center not in CENTERS
            or self.outer_target_center == self.pseudo_target_center
            or eligible != tuple(center for center in CENTERS if center not in excluded)
            or not cases
            or len(cases) != len(set(cases))
        ):
            raise ProtocolError("CBPUPR ordered H/J plan drifted.")
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        object.__setattr__(
            self, "eligible_source_and_calibration_centers", eligible
        )
        object.__setattr__(self, "pseudo_case_ids", cases)
        object.__setattr__(self, "plan_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str]:
        return self.outer_target_center, self.pseudo_target_center

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_double_exclusion_plan_v1",
            "outer_target_center": self.outer_target_center,
            "pseudo_target_center": self.pseudo_target_center,
            "eligible_source_and_calibration_centers": list(
                self.eligible_source_and_calibration_centers
            ),
            "pseudo_case_ids": list(self.pseudo_case_ids),
            "probability_surface_hash": self.probability_surface_hash,
            "outer_H_support_rows_or_labels_enter_J_minus_d_posterior_fit_or_normalization": False,
            "outer_H_frozen_label_free_expert_fingerprint_covariates_present": True,
            "posterior_is_outer_H_covariate_invariant": False,
            "outer_H_excluded_from_actionable_endpoint_source_selection": True,
            "outer_H_excluded_from_source_prior_and_donor_calibration_roles": True,
            "pseudo_J_support_rows_enter_J_minus_d_posterior_fit": True,
            "pseudo_case_d_rows_or_labels_enter_own_posterior_fit": False,
            "pseudo_J_excluded_from_actionable_endpoint_source_selection": True,
            "pseudo_J_excluded_from_source_prior_and_donor_calibration_roles": True,
            "pseudo_evaluation_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "plan_hash": self.plan_hash}


@dataclass(frozen=True)
class OuterPlanSeal:
    outer_plans: tuple[WholeCaseOuterPlan, ...]
    probability_surface_hash: str
    strict_canonical_topology: bool = True
    double_exclusion_plans: tuple[DoubleExclusionPlan, ...] = field(init=False)
    seal_hash: str = field(init=False)

    def __post_init__(self) -> None:
        plans = tuple(self.outer_plans)
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        if (
            not plans
            or len({row.key for row in plans}) != len(plans)
            or any(row.probability_surface_hash != self.probability_surface_hash for row in plans)
        ):
            raise ProtocolError("CBPUPR outer plan seal drifted.")
        if self.strict_canonical_topology:
            counts = {
                center: sum(row.target_center == center for row in plans)
                for center in CENTERS
            }
            if len(plans) != EXPECTED_OUTER_PLAN_COUNT or counts != dict(EXPECTED_CASE_COUNTS_BY_CENTER):
                raise ProtocolError("CBPUPR canonical outer plan topology drifted.")
        cases_by_center = {
            center: tuple(
                sorted(row.case_id for row in plans if row.target_center == center)
            )
            for center in CENTERS
        }
        if any(not values for values in cases_by_center.values()):
            raise ProtocolError("CBPUPR H/J plans require cases for every center.")
        double = tuple(
            DoubleExclusionPlan(
                outer,
                pseudo,
                tuple(center for center in CENTERS if center not in {outer, pseudo}),
                cases_by_center[pseudo],
                self.probability_surface_hash,
            )
            for outer in CENTERS
            for pseudo in CENTERS
            if pseudo != outer
        )
        if len(double) != EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT:
            raise ProtocolError("CBPUPR ordered H/J plan count drifted.")
        payload = {
            "schema_version": "fixed_bank_cbpupr_outer_plan_seal_v1",
            "probability_surface_hash": self.probability_surface_hash,
            "outer_plan_count": len(plans),
            "outer_plan_hashes": [row.plan_hash for row in plans],
            "double_exclusion_plan_count": len(double),
            "double_exclusion_plan_hashes": [row.plan_hash for row in double],
            "strict_canonical_topology": self.strict_canonical_topology,
            "double_exclusion_states_used": True,
            "sealed_before_any_label_access": True,
        }
        object.__setattr__(self, "outer_plans", plans)
        object.__setattr__(self, "double_exclusion_plans", double)
        object.__setattr__(self, "seal_hash", canonical_hash(payload))

    @property
    def outer_by_key(self) -> Mapping[tuple[str, str], WholeCaseOuterPlan]:
        return MappingProxyType({row.key: row for row in self.outer_plans})

    @property
    def double_by_key(self) -> Mapping[tuple[str, str], DoubleExclusionPlan]:
        return MappingProxyType({row.key: row for row in self.double_exclusion_plans})

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_cbpupr_outer_plan_seal_v1",
            "probability_surface_hash": self.probability_surface_hash,
            "outer_plans": [row.to_payload() for row in self.outer_plans],
            "double_exclusion_plans": [
                row.to_payload() for row in self.double_exclusion_plans
            ],
            "strict_canonical_topology": self.strict_canonical_topology,
            "double_exclusion_states_used": True,
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
            raise ProtocolError("CBPUPR outer plan identity row drifted.")
        seen.add(key)
        samples.setdefault((center, case), set()).add(sample)
        if groups.setdefault((center, case), group) != group:
            raise ProtocolError("CBPUPR case maps to multiple group identities.")
    group_cases: dict[tuple[str, str], set[str]] = {}
    for (center, case), group in groups.items():
        group_cases.setdefault((center, group), set()).add(case)
    if any(len(cases) != 1 for cases in group_cases.values()):
        raise ProtocolError("CBPUPR held group spans multiple cases.")
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


__all__ = (
    "DoubleExclusionPlan",
    "OuterPlanSeal",
    "WholeCaseOuterPlan",
    "build_outer_plans",
)
