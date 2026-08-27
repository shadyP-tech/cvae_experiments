"""Direct additive action utility relative to protected P."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from ..physical.contracts import ACTION_IDS, MetricVector, probability_vector
from .actions import ActionCell, ActionRectangle


@dataclass(frozen=True, slots=True)
class CenterMetricDenominators:
    """Frozen center denominators make case utilities exactly additive."""

    n_positive: int
    n_negative: int
    n_rows: int

    def __post_init__(self) -> None:
        values = (int(self.n_positive), int(self.n_negative), int(self.n_rows))
        if any(value <= 0 for value in values) or values[0] + values[1] != values[2]:
            raise GovernanceError("SCALE-BP v2 center metric denominators drifted.")
        object.__setattr__(self, "n_positive", values[0])
        object.__setattr__(self, "n_negative", values[1])
        object.__setattr__(self, "n_rows", values[2])


@dataclass(frozen=True, slots=True)
class ActionValueRecord:
    target_center: str
    case_id: str
    action_id: str
    value: MetricVector
    structural_noop: bool
    action_hash: str
    label_scope_hash: str
    value_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.action_id not in ACTION_IDS or not self.label_scope_hash:
            raise GovernanceError("SCALE-BP v2 action-value identity drifted.")
        if self.structural_noop and self.value != MetricVector.zeros():
            raise GovernanceError("SCALE-BP v2 structural no-op has nonzero utility.")
        object.__setattr__(
            self,
            "value_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_action_value_record_v1",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "action_id": self.action_id,
                    "value": self.value.to_payload(),
                    "structural_noop": self.structural_noop,
                    "action_hash": self.action_hash,
                    "label_scope_hash": self.label_scope_hash,
                    "raw_labels_persisted": False,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ScoredActionRectangle:
    rectangle: ActionRectangle
    values: tuple[ActionValueRecord, ...]
    score_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(self.values)
        if (
            tuple(value.action_id for value in values) != ACTION_IDS
            or any(
                value.target_center != self.rectangle.target_center
                or value.case_id != self.rectangle.case_id
                for value in values
            )
            or any(
                value.action_hash != self.rectangle.cell(value.action_id).action.action_hash
                for value in values
            )
            or len({value.label_scope_hash for value in values}) != 1
        ):
            raise GovernanceError("SCALE-BP v2 scored action rectangle drifted.")
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "score_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_scored_action_rectangle_v1",
                    "rectangle_hash": self.rectangle.rectangle_hash,
                    "value_hashes": tuple(value.value_hash for value in values),
                    "complete_six_action_surface": True,
                }
            ),
        )


def compute_action_value(
    cell: ActionCell,
    labels: Sequence[int] | np.ndarray,
    *,
    denominators: CenterMetricDenominators,
    label_scope_hash: object,
    epsilon: float = 1.0e-7,
) -> ActionValueRecord:
    truth = np.ascontiguousarray(labels, dtype=np.int8)
    baseline = probability_vector(cell.action.protected_p)
    candidate = probability_vector(
        cell.action.projected, expected_length=len(baseline)
    )
    if (
        truth.shape != baseline.shape
        or not np.isin(truth, (0, 1)).all()
        or not str(label_scope_hash)
        or not 0.0 < float(epsilon) < 0.5
    ):
        raise GovernanceError("SCALE-BP v2 action-value labels or scope drifted.")
    if cell.structural_noop:
        value = MetricVector.zeros()
    else:
        baseline_hard = baseline >= 0.5
        candidate_hard = candidate >= 0.5
        positive = truth == 1
        negative = ~positive
        delta_tp = int(np.sum(candidate_hard & positive)) - int(
            np.sum(baseline_hard & positive)
        )
        delta_tn = int(np.sum((~candidate_hard) & negative)) - int(
            np.sum((~baseline_hard) & negative)
        )
        delta_bacc = 0.5 * (
            delta_tp / denominators.n_positive
            + delta_tn / denominators.n_negative
        )
        delta_brier = float(
            np.sum((candidate - truth) ** 2 - (baseline - truth) ** 2, dtype=np.float64)
            / denominators.n_rows
        )
        clipped_candidate = np.clip(candidate, epsilon, 1.0 - epsilon)
        clipped_baseline = np.clip(baseline, epsilon, 1.0 - epsilon)
        candidate_loss = -(
            truth * np.log(clipped_candidate)
            + (1 - truth) * np.log1p(-clipped_candidate)
        )
        baseline_loss = -(
            truth * np.log(clipped_baseline)
            + (1 - truth) * np.log1p(-clipped_baseline)
        )
        delta_log = float(
            np.sum(candidate_loss - baseline_loss, dtype=np.float64)
            / denominators.n_rows
        )
        value = MetricVector(float(delta_bacc), delta_brier, delta_log)
    return ActionValueRecord(
        cell.target_center,
        cell.case_id,
        cell.action_id,
        value,
        cell.structural_noop,
        cell.action.action_hash,
        str(label_scope_hash),
    )


def score_action_rectangle(
    rectangle: ActionRectangle,
    labels: Sequence[int] | np.ndarray,
    *,
    denominators: CenterMetricDenominators,
    label_scope_hash: object,
) -> ScoredActionRectangle:
    return ScoredActionRectangle(
        rectangle,
        tuple(
            compute_action_value(
                cell,
                labels,
                denominators=denominators,
                label_scope_hash=label_scope_hash,
            )
            for cell in rectangle.cells
        ),
    )


def center_denominators(labels_by_case: Mapping[str, Sequence[int]]) -> CenterMetricDenominators:
    rows = tuple(np.ascontiguousarray(values, dtype=np.int8) for values in labels_by_case.values())
    if not rows or any(not np.isin(row, (0, 1)).all() for row in rows):
        raise GovernanceError("SCALE-BP v2 center labels are empty or invalid.")
    values = np.concatenate(rows)
    return CenterMetricDenominators(
        int(np.sum(values == 1)), int(np.sum(values == 0)), len(values)
    )


__all__ = (
    "ActionValueRecord",
    "CenterMetricDenominators",
    "ScoredActionRectangle",
    "center_denominators",
    "compute_action_value",
    "score_action_rectangle",
)
