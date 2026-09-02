"""Per-outer admission against an always-B, tie-aware source null."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .contracts import CasePrediction, SourceActionOutcome
from .hashing import canonical_hash
from .effective_menu import EffectiveMenu
from .model import _source_cases


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    min_pooled_top1_excess: float = 0.01
    min_delete_center_top1_excess: float = -0.02
    min_opportunity_top1_accuracy: float = 0.35
    min_opportunity_cases: int = 8

    def __post_init__(self) -> None:
        values = (
            self.min_pooled_top1_excess,
            self.min_delete_center_top1_excess,
            self.min_opportunity_top1_accuracy,
        )
        if any(not math.isfinite(value) for value in values) or int(
            self.min_opportunity_cases
        ) < 1:
            raise ProtocolError("Source-active admission configuration is malformed.")


@dataclass(frozen=True, slots=True)
class OuterAdmission:
    outer_target_id: str
    admitted: bool
    learned_top1_accuracy: float
    always_b_top1_accuracy: float
    pooled_top1_excess: float
    min_delete_center_top1_excess: float
    opportunity_top1_accuracy: float
    opportunity_case_count: int
    case_count: int
    reasons: tuple[str, ...]
    config: AdmissionConfig
    admission_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.admitted != (not self.reasons):
            raise ProtocolError("Outer-admission decision and reasons disagree.")
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(
                {
                    "schema_version": "source_active_outer_admission_v7",
                    "outer_target_id": self.outer_target_id,
                    "admitted": self.admitted,
                    "learned_top1_accuracy": self.learned_top1_accuracy,
                    "always_b_top1_accuracy": self.always_b_top1_accuracy,
                    "pooled_top1_excess": self.pooled_top1_excess,
                    "min_delete_center_top1_excess": self.min_delete_center_top1_excess,
                    "opportunity_top1_accuracy": self.opportunity_top1_accuracy,
                    "opportunity_case_count": self.opportunity_case_count,
                    "case_count": self.case_count,
                    "reasons": self.reasons,
                    "config": self.config,
                    "skill_surface": "POSITIVE_OPPORTUNITY_CASES_ONLY",
                    "whole_policy_safety_delegated_to_nested_calibration": True,
                    "target_evaluation_labels_used": False,
                }
            ),
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_outer_admission(
    predictions: Sequence[CasePrediction],
    observations: Sequence[SourceActionOutcome],
    *,
    config: AdmissionConfig = AdmissionConfig(),
    effective_menus: Sequence[EffectiveMenu] | None = None,
) -> OuterAdmission:
    """Evaluate conditional rank skill on held-source opportunity cases.

    Always-B is the tie-aware null on this surface: once a case has strictly
    positive oracle opportunity, B cannot be a top-1 action.  All-case
    opportunity detection and route-or-B safety are evaluated separately by
    the nested whole-policy risk--coverage calibration.  Keeping those roles
    separate prevents sparse opportunity prevalence from erasing useful
    conditional ordering signal.
    """

    cases = _source_cases(observations, effective_menus)
    by_key = {(row.query_center_id, row.case_id): row for row in predictions}
    expected = {(case.menu.query_center_id, case.menu.case_id) for case in cases}
    if set(by_key) != expected:
        raise ProtocolError("Admission OOF predictions do not cover the source case inventory.")
    learned_by_center: dict[str, list[float]] = defaultdict(list)
    null_by_center: dict[str, list[float]] = defaultdict(list)
    opportunity_by_center: dict[str, list[float]] = defaultdict(list)
    outer = cases[0].menu.outer_target_id
    for case in cases:
        prediction = by_key[(case.menu.query_center_id, case.menu.case_id)]
        if (
            prediction.outer_target_id != outer
            or prediction.query_center_id in prediction.training_center_ids
            or prediction.menu_hash != case.menu.menu_hash
        ):
            raise ProtocolError("Admission received non-OOF or menu-drifted predictions.")
        gains = {row.action.action_id: row.bacc_gain for row in case.outcomes}
        best = max((0.0, *gains.values()))
        selected_id = prediction.top_action_id
        selected_gain = 0.0 if selected_id is None else gains[selected_id]
        learned = float(math.isclose(selected_gain, best, rel_tol=0.0, abs_tol=1e-12))
        null = float(math.isclose(0.0, best, rel_tol=0.0, abs_tol=1e-12))
        center = case.menu.query_center_id
        if best > 0.0:
            learned_by_center[center].append(learned)
            null_by_center[center].append(null)
            opportunity_by_center[center].append(learned)

    centers = tuple(sorted(opportunity_by_center))

    def center_equal(mapping: dict[str, list[float]], subset: Sequence[str]) -> float:
        return _mean([_mean(mapping[center]) for center in subset if mapping.get(center)])

    learned_top1 = center_equal(learned_by_center, centers)
    null_top1 = center_equal(null_by_center, centers)
    pooled_excess = learned_top1 - null_top1
    delete_excesses = tuple(
        center_equal(learned_by_center, tuple(c for c in centers if c != deleted))
        - center_equal(null_by_center, tuple(c for c in centers if c != deleted))
        for deleted in centers
    )
    min_delete = min(delete_excesses) if delete_excesses else 0.0
    opportunity_count = sum(len(values) for values in opportunity_by_center.values())
    opportunity_accuracy = center_equal(opportunity_by_center, centers)
    reasons: list[str] = []
    if pooled_excess < config.min_pooled_top1_excess:
        reasons.append("TOP1_EXCESS_VS_ALWAYS_B_BELOW_FLOOR")
    if min_delete < config.min_delete_center_top1_excess:
        reasons.append("DELETE_CENTER_TOP1_EXCESS_BELOW_FLOOR")
    if opportunity_count < config.min_opportunity_cases:
        reasons.append("INSUFFICIENT_SOURCE_OPPORTUNITY_CASES")
    if opportunity_accuracy < config.min_opportunity_top1_accuracy:
        reasons.append("CONDITIONAL_OPPORTUNITY_TOP1_BELOW_FLOOR")
    return OuterAdmission(
        outer_target_id=outer,
        admitted=not reasons,
        learned_top1_accuracy=learned_top1,
        always_b_top1_accuracy=null_top1,
        pooled_top1_excess=pooled_excess,
        min_delete_center_top1_excess=min_delete,
        opportunity_top1_accuracy=opportunity_accuracy,
        opportunity_case_count=opportunity_count,
        case_count=len(cases),
        reasons=tuple(reasons),
        config=config,
    )


__all__ = ("AdmissionConfig", "OuterAdmission", "evaluate_outer_admission")
