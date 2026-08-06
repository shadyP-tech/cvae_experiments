"""Predeclared descriptive summaries and paired policy contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..protocol import ProtocolError
from .contracts import CONTROL_ARM, METADATA_ARM, POLICY_ARMS, UTILITY_ARM
from .scoring import TargetMetricRow


@dataclass(frozen=True)
class ArmSummary:
    policy_id: str
    equal_center_equal_seed_mean_bacc: float
    equal_center_equal_seed_mean_macro_f1: float
    minimum_cell_bacc: float
    maximum_cell_bacc: float
    cell_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage70_arm_summary_v1",
            "policy_id": self.policy_id,
            "equal_center_equal_seed_mean_bacc": self.equal_center_equal_seed_mean_bacc,
            "equal_center_equal_seed_mean_macro_f1": (
                self.equal_center_equal_seed_mean_macro_f1
            ),
            "minimum_cell_bacc": self.minimum_cell_bacc,
            "maximum_cell_bacc": self.maximum_cell_bacc,
            "cell_count": self.cell_count,
            "fresh_confirmatory_evidence": False,
        }


@dataclass(frozen=True)
class PairedDelta:
    comparison_id: str
    policy_id: str
    control_policy_id: str
    target_center: str
    training_seed: int
    generation_seed: int
    bacc_delta: float
    macro_f1_delta: float
    role: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_stage70_paired_delta_v1",
            "comparison_id": self.comparison_id,
            "policy_id": self.policy_id,
            "control_policy_id": self.control_policy_id,
            "target_center": self.target_center,
            "training_seed": self.training_seed,
            "generation_seed": self.generation_seed,
            "bacc_delta": self.bacc_delta,
            "macro_f1_delta": self.macro_f1_delta,
            "role": self.role,
            "paired": True,
            "fresh_confirmatory_evidence": False,
        }


def build_descriptive_contrasts(
    metrics: Sequence[TargetMetricRow],
) -> tuple[tuple[ArmSummary, ...], tuple[PairedDelta, ...]]:
    by_key = {
        (row.policy_id, row.target_center, row.training_seed, row.generation_seed): row
        for row in metrics
    }
    if len(by_key) != 243:
        raise ProtocolError("Stage-70 contrast input must contain 243 unique cells.")
    summaries: list[ArmSummary] = []
    for arm in POLICY_ARMS:
        arm_rows = [row for row in metrics if row.policy_id == arm]
        if len(arm_rows) != 81:
            raise ProtocolError(f"Stage-70 arm {arm} does not contain 81 cells.")
        center_bacc: list[float] = []
        center_f1: list[float] = []
        for center in sorted({row.target_center for row in arm_rows}):
            center_rows = [row for row in arm_rows if row.target_center == center]
            if len(center_rows) != 9:
                raise ProtocolError("Stage-70 center does not retain all nine seed cells.")
            center_bacc.append(sum(row.bacc for row in center_rows) / 9.0)
            center_f1.append(sum(row.macro_f1 for row in center_rows) / 9.0)
        summaries.append(
            ArmSummary(
                policy_id=arm,
                equal_center_equal_seed_mean_bacc=sum(center_bacc) / len(center_bacc),
                equal_center_equal_seed_mean_macro_f1=sum(center_f1) / len(center_f1),
                minimum_cell_bacc=min(row.bacc for row in arm_rows),
                maximum_cell_bacc=max(row.bacc for row in arm_rows),
                cell_count=len(arm_rows),
            )
        )
    deltas: list[PairedDelta] = []
    cell_keys = sorted(
        {
            (row.target_center, row.training_seed, row.generation_seed)
            for row in metrics
        }
    )
    for target, training_seed, generation_seed in cell_keys:
        control = by_key[(CONTROL_ARM, target, training_seed, generation_seed)]
        for arm, comparison_id, role in (
            (
                METADATA_ARM,
                "metadata_max_tie_union_minus_equal_union",
                "sole_predeclared_descriptive_policy_contrast",
            ),
            (
                UTILITY_ARM,
                "utility_regret_minus_equal_union",
                "deterministic_fallback_equivalence_audit",
            ),
        ):
            observed = by_key[(arm, target, training_seed, generation_seed)]
            delta = PairedDelta(
                comparison_id=comparison_id,
                policy_id=arm,
                control_policy_id=CONTROL_ARM,
                target_center=target,
                training_seed=training_seed,
                generation_seed=generation_seed,
                bacc_delta=observed.bacc - control.bacc,
                macro_f1_delta=observed.macro_f1 - control.macro_f1,
                role=role,
            )
            if arm == UTILITY_ARM and (
                delta.bacc_delta != 0.0
                or delta.macro_f1_delta != 0.0
                or observed.prediction_sha256 != control.prediction_sha256
                or observed.probability_sha256 != control.probability_sha256
            ):
                raise ProtocolError("Utility/regret fallback is not exact control equivalence.")
            deltas.append(delta)
    if len(deltas) != 162:
        raise ProtocolError("Stage-70 paired delta count drifted.")
    return tuple(summaries), tuple(deltas)


__all__ = ("ArmSummary", "PairedDelta", "build_descriptive_contrasts")
