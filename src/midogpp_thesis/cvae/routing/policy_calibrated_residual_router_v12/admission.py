"""Per-outer source-only admission over the HARP v12 case inventory."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash
from .outcome_inventory import CaseOutcomeInventory


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    min_pooled_top1_excess: float = 0.01
    min_delete_center_top1_excess: float = -0.02
    min_opportunity_top1_accuracy: float = 0.35
    min_opportunity_cases: int = 8
    utility_tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if (
            any(
                not math.isfinite(value)
                for value in (
                    self.min_pooled_top1_excess,
                    self.min_delete_center_top1_excess,
                    self.min_opportunity_top1_accuracy,
                    self.utility_tie_tolerance,
                )
            )
            or int(self.min_opportunity_cases) < 1
            or self.utility_tie_tolerance < 0.0
        ):
            raise ProtocolError("HARP v12 admission configuration is malformed.")


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
            raise ProtocolError("HARP v12 admission decision and reasons disagree.")
        object.__setattr__(
            self,
            "admission_hash",
            canonical_hash(
                {
                    "schema_version": "policy_calibrated_outer_admission_v12",
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
                    "ranked_all_actions_before_acceptance": True,
                    "target_evaluation_labels_used": False,
                }
            ),
        )

    def public_payload(self) -> dict[str, object]:
        return {
            "outer_target_id": self.outer_target_id,
            "admitted": self.admitted,
            "learned_top1_accuracy": self.learned_top1_accuracy,
            "always_b_top1_accuracy": self.always_b_top1_accuracy,
            "pooled_top1_excess": self.pooled_top1_excess,
            "min_delete_center_top1_excess": self.min_delete_center_top1_excess,
            "opportunity_top1_accuracy": self.opportunity_top1_accuracy,
            "opportunity_case_count": self.opportunity_case_count,
            "case_count": self.case_count,
            "reasons": list(self.reasons),
            "admission_hash": self.admission_hash,
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_outer_admission(
    inventory: CaseOutcomeInventory,
    *,
    config: AdmissionConfig = AdmissionConfig(),
) -> OuterAdmission:
    if not isinstance(inventory, CaseOutcomeInventory):
        raise ProtocolError("HARP v12 admission requires a case-outcome inventory.")
    outers = {row.key[0] for row in inventory.contexts}
    if len(outers) != 1:
        raise ProtocolError("HARP v12 admission crossed outer targets.")
    learned_by_center: dict[str, list[float]] = defaultdict(list)
    null_by_center: dict[str, list[float]] = defaultdict(list)
    opportunity_count = 0
    for context in inventory.contexts:
        prediction = context.prediction
        best = context.best_bacc_gain
        if best <= config.utility_tie_tolerance:
            continue
        opportunity_count += 1
        selected = prediction.raw_top_action_id
        outcome = context.outcome_for(selected)
        if selected != "B" and outcome is None:
            raise ProtocolError("HARP v12 selected active action lacks an outcome.")
        selected_gain = 0.0 if outcome is None else outcome.bacc_gain
        correct = float(
            math.isclose(
                selected_gain,
                best,
                rel_tol=0.0,
                abs_tol=config.utility_tie_tolerance,
            )
        )
        learned_by_center[prediction.query_center_id].append(correct)
        null_by_center[prediction.query_center_id].append(0.0)
    centers = tuple(sorted(learned_by_center))

    def center_equal(mapping: dict[str, list[float]], selected: Sequence[str]) -> float:
        return _mean([_mean(mapping[center]) for center in selected if mapping.get(center)])

    learned = center_equal(learned_by_center, centers)
    null = center_equal(null_by_center, centers)
    excess = learned - null
    deletes = tuple(
        center_equal(
            learned_by_center, tuple(center for center in centers if center != removed)
        )
        - center_equal(
            null_by_center, tuple(center for center in centers if center != removed)
        )
        for removed in centers
    )
    min_delete = min(deletes) if deletes else 0.0
    reasons: list[str] = []
    if excess < config.min_pooled_top1_excess:
        reasons.append("PAIRWISE_TOP1_EXCESS_VS_ALWAYS_B_BELOW_FLOOR")
    if min_delete < config.min_delete_center_top1_excess:
        reasons.append("PAIRWISE_DELETE_CENTER_TOP1_EXCESS_BELOW_FLOOR")
    if opportunity_count < config.min_opportunity_cases:
        reasons.append("INSUFFICIENT_SOURCE_OPPORTUNITY_CASES")
    if learned < config.min_opportunity_top1_accuracy:
        reasons.append("PAIRWISE_OPPORTUNITY_TOP1_BELOW_FLOOR")
    return OuterAdmission(
        outer_target_id=next(iter(outers)),
        admitted=not reasons,
        learned_top1_accuracy=learned,
        always_b_top1_accuracy=null,
        pooled_top1_excess=excess,
        min_delete_center_top1_excess=min_delete,
        opportunity_top1_accuracy=learned,
        opportunity_case_count=opportunity_count,
        case_count=len(inventory.contexts),
        reasons=tuple(reasons),
        config=config,
    )


__all__ = ("AdmissionConfig", "OuterAdmission", "evaluate_outer_admission")
