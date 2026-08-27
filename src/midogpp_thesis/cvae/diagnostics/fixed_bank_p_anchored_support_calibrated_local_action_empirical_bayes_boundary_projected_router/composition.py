"""Exact direct composition and full-endpoint sensitivity replay."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .action_geometry import HARD_THRESHOLD, canonical_probabilities, probability_hash
from .hashing import canonical_hash, require_sha256
from .identity import ACTION_IDS
from .protocol import ProtocolError
from .selection import ActionCandidate, SelectionDecision


@dataclass(frozen=True, slots=True)
class ComposedAction:
    case_id: str
    mode: str
    baseline_probabilities: tuple[float, ...]
    composed_probabilities: tuple[float, ...]
    selected_action_ids: tuple[str, ...]
    crossing_indices: tuple[int, ...]
    decision_hash: str
    baseline_probability_hash: str = field(init=False)
    composed_probability_hash: str = field(init=False)
    composition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        case_id = str(self.case_id)
        baseline = canonical_probabilities(self.baseline_probabilities)
        composed = canonical_probabilities(
            self.composed_probabilities, expected_length=len(baseline)
        )
        selected = tuple(str(value) for value in self.selected_action_ids)
        crossing_indices = tuple(int(value) for value in self.crossing_indices)
        decision_hash = require_sha256(self.decision_hash, "composition decision hash")
        if (
            not case_id
            or self.mode not in {"boundary", "full_endpoint"}
            or selected != tuple(sorted(set(selected)))
            or any(action_id not in ACTION_IDS for action_id in selected)
            or len(selected) > 2
            or crossing_indices != tuple(sorted(set(crossing_indices)))
            or any(index < 0 or index >= len(baseline) for index in crossing_indices)
        ):
            raise ProtocolError("SCALE-BP composition identity drifted.")
        crossing = np.zeros(len(baseline), dtype=bool)
        crossing[list(crossing_indices)] = True
        actual = (baseline >= HARD_THRESHOLD) != (composed >= HARD_THRESHOLD)
        if (
            not np.array_equal(actual, crossing)
            or not np.array_equal(composed[~crossing], baseline[~crossing])
            or (not selected and (crossing_indices or not np.array_equal(composed, baseline)))
            or (selected and not crossing_indices)
        ):
            raise ProtocolError("SCALE-BP composition changed its exact-P mask contract.")
        baseline_values = tuple(float(value) for value in baseline)
        composed_values = tuple(float(value) for value in composed)
        baseline_hash = probability_hash(baseline)
        composed_hash = probability_hash(composed)
        payload = {
            "schema_version": "scale_bp_composed_action_v1",
            "case_id": case_id,
            "mode": self.mode,
            "row_count": len(baseline),
            "selected_action_ids": selected,
            "crossing_indices": crossing_indices,
            "decision_hash": decision_hash,
            "baseline_probability_hash": baseline_hash,
            "composed_probability_hash": composed_hash,
            "off_mask_exact_p": True,
            "full_endpoint_is_sensitivity_only": self.mode == "full_endpoint",
        }
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "baseline_probabilities", baseline_values)
        object.__setattr__(self, "composed_probabilities", composed_values)
        object.__setattr__(self, "selected_action_ids", selected)
        object.__setattr__(self, "crossing_indices", crossing_indices)
        object.__setattr__(self, "decision_hash", decision_hash)
        object.__setattr__(self, "baseline_probability_hash", baseline_hash)
        object.__setattr__(self, "composed_probability_hash", composed_hash)
        object.__setattr__(self, "composition_hash", canonical_hash(payload))

    @property
    def is_exact_p(self) -> bool:
        return not self.selected_action_ids

    def as_array(self) -> np.ndarray:
        return canonical_probabilities(self.composed_probabilities)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "scale_bp_composed_action_v1",
            "case_id": self.case_id,
            "mode": self.mode,
            "row_count": len(self.baseline_probabilities),
            "selected_action_ids": self.selected_action_ids,
            "crossing_indices": self.crossing_indices,
            "decision_hash": self.decision_hash,
            "baseline_probability_hash": self.baseline_probability_hash,
            "composed_probability_hash": self.composed_probability_hash,
            "off_mask_exact_p": True,
            "full_endpoint_is_sensitivity_only": self.mode == "full_endpoint",
            "composition_hash": self.composition_hash,
        }


def compose_selection(
    portfolio: object,
    candidates: object,
    decision: SelectionDecision,
    *,
    mode: str = "boundary",
) -> ComposedAction:
    """Replay a sealed direct decision; an abstention is exact P."""

    if mode not in {"boundary", "full_endpoint"}:
        raise ProtocolError("SCALE-BP composition mode drifted.")
    baseline = canonical_probabilities(portfolio)
    baseline_hash = probability_hash(baseline)
    rows = tuple(candidates)  # type: ignore[arg-type]
    if any(not isinstance(row, ActionCandidate) for row in rows):
        raise ProtocolError("SCALE-BP composition candidate population drifted.")
    if (
        decision.baseline_probability_hash != baseline_hash
        or tuple(sorted(row.candidate_hash for row in rows)) != decision.candidate_hashes
        or any(
            row.case_id != decision.case_id
            or row.projection.baseline_probability_hash != baseline_hash
            for row in rows
        )
    ):
        raise ProtocolError("SCALE-BP composition lineage drifted.")
    by_id = {row.action_id: row for row in rows}
    if len(by_id) != len(rows) or any(
        action_id not in by_id for action_id in decision.selected_action_ids
    ):
        raise ProtocolError("SCALE-BP selected action does not resolve uniquely.")

    composed = np.array(baseline, dtype=np.float32, copy=True, order="C")
    used_indices: set[int] = set()
    for action_id in decision.selected_action_ids:
        candidate = by_id[action_id]
        crossing = set(candidate.projection.crossing_indices)
        if used_indices.intersection(crossing):
            raise ProtocolError("SCALE-BP selected actions overlap in threshold space.")
        values = (
            candidate.projection.projected_probabilities
            if mode == "boundary"
            else candidate.projection.full_endpoint_probabilities
        )
        value_array = np.asarray(values, dtype=np.float32)
        indices = tuple(sorted(crossing))
        composed[list(indices)] = value_array[list(indices)]
        used_indices.update(crossing)
    return ComposedAction(
        case_id=decision.case_id,
        mode=mode,
        baseline_probabilities=tuple(float(value) for value in baseline),
        composed_probabilities=tuple(float(value) for value in composed),
        selected_action_ids=decision.selected_action_ids,
        crossing_indices=tuple(sorted(used_indices)),
        decision_hash=decision.decision_hash,
    )


__all__ = ("ComposedAction", "compose_selection")
