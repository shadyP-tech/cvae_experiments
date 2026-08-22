"""Deterministic complete-case composition with exact-P fallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .canonical_probabilities import (
    CanonicalProbabilityVector,
    canonical_float32_probabilities,
    canonical_hash,
    exact_p_fallback,
    require_byte_exact_p,
)
from .eligibility import ActionCandidate


@dataclass(frozen=True)
class CompositionResult:
    selected_case_ids: tuple[str, ...]
    selected_candidate_hashes: tuple[str, ...]
    probabilities: CanonicalProbabilityVector
    changed_probability_count: int
    exact_p: bool
    composition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            len(self.selected_case_ids) != len(self.selected_candidate_hashes)
            or len(set(self.selected_case_ids)) != len(self.selected_case_ids)
            or len(set(self.selected_candidate_hashes))
            != len(self.selected_candidate_hashes)
            or self.changed_probability_count < 0
            or self.exact_p != (len(self.selected_case_ids) == 0)
        ):
            raise ProtocolError("CBPUPR composition result drifted.")
        object.__setattr__(
            self,
            "composition_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_composition_v1",
                    "selected_case_ids": list(self.selected_case_ids),
                    "selected_candidate_hashes": list(self.selected_candidate_hashes),
                    "probability_sha256": self.probabilities.sha256,
                    "changed_probability_count": self.changed_probability_count,
                    "exact_p": self.exact_p,
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CompositionResult":
        row = cls(
            tuple(str(value) for value in payload["selected_case_ids"]),  # type: ignore[index]
            tuple(str(value) for value in payload["selected_candidate_hashes"]),  # type: ignore[index]
            CanonicalProbabilityVector.from_payload(payload["probabilities"]),  # type: ignore[arg-type]
            int(payload["changed_probability_count"]),
            bool(payload["exact_p"]),
        )
        if "composition_hash" in payload and str(payload["composition_hash"]) != row.composition_hash:
            raise ProtocolError("CBPUPR composition hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "selected_case_ids": list(self.selected_case_ids),
            "selected_candidate_hashes": list(self.selected_candidate_hashes),
            "probabilities": self.probabilities.to_payload(),
            "changed_probability_count": self.changed_probability_count,
            "exact_p": self.exact_p,
            "composition_hash": self.composition_hash,
        }


def compose_exact_p(portfolio_probabilities: object) -> CompositionResult:
    p = canonical_float32_probabilities(portfolio_probabilities)
    result = exact_p_fallback(p)
    require_byte_exact_p(result, p)
    return CompositionResult(
        (), (), CanonicalProbabilityVector.from_array(result), 0, True
    )


def compose_center_probabilities(
    portfolio_probabilities: object,
    sample_case_ids: Sequence[str],
    selected_candidates: Sequence[ActionCandidate],
) -> CompositionResult:
    p = canonical_float32_probabilities(portfolio_probabilities)
    case_ids = tuple(str(value) for value in sample_case_ids)
    rows = tuple(selected_candidates)
    if len(case_ids) != len(p) or len({row.case_id for row in rows}) != len(rows):
        raise ProtocolError("CBPUPR composition topology drifted.")
    if not rows:
        return compose_exact_p(p)
    composed = p.copy(order="C")
    selected_cases: list[str] = []
    selected_hashes: list[str] = []
    case_array = np.asarray(case_ids, dtype=object)
    for row in rows:
        positions = np.flatnonzero(case_array == row.case_id)
        if not len(positions):
            raise ProtocolError("CBPUPR selected case is absent from center surface.")
        local = row.probabilities.as_array()
        if len(local) != len(positions):
            raise ProtocolError("CBPUPR case candidate probability length drifted.")
        composed[positions] = local
        selected_cases.append(row.case_id)
        selected_hashes.append(row.action_hash)
    changed = int(np.count_nonzero(composed.view(np.uint32) != p.view(np.uint32)))
    composed.setflags(write=False)
    return CompositionResult(
        tuple(selected_cases),
        tuple(selected_hashes),
        CanonicalProbabilityVector.from_array(composed),
        changed,
        False,
    )


__all__ = (
    "CompositionResult",
    "compose_center_probabilities",
    "compose_exact_p",
)
