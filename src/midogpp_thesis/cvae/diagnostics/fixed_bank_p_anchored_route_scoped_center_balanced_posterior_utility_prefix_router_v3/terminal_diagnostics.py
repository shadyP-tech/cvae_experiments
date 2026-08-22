"""Gate-funnel and terminal-only metric diagnostics for CBPUPR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .candidate_runtime import CandidateRuntimeResult
from .canonical_probabilities import canonical_float32_probabilities, canonical_hash
from .decision import RouteDecision
from .eligibility import BRIER_UNSAFE, LOG_UNSAFE, NONPOSITIVE_BACC
from .posterior_expected_utility import FavorableUtility, LOG_CLIP_EPSILON


@dataclass(frozen=True)
class GateFunnel:
    route_count: int
    descriptor_count: int
    crossing_descriptor_count: int
    positive_bacc_descriptor_count: int
    proper_safe_descriptor_count: int
    selected_case_candidate_count: int
    structurally_admissible_center_count: int
    feasible_prefix_center_count: int
    routed_case_count: int
    exact_p_center_count: int
    stage_counts: tuple[tuple[str, int], ...]
    funnel_hash: str = field(init=False)

    def __post_init__(self) -> None:
        values = (
            self.route_count,
            self.descriptor_count,
            self.crossing_descriptor_count,
            self.positive_bacc_descriptor_count,
            self.proper_safe_descriptor_count,
            self.selected_case_candidate_count,
            self.structurally_admissible_center_count,
            self.feasible_prefix_center_count,
            self.routed_case_count,
            self.exact_p_center_count,
        )
        if any(value < 0 for value in values) or len({name for name, _ in self.stage_counts}) != len(self.stage_counts):
            raise ProtocolError("CBPUPR gate funnel drifted.")
        if not (
            self.descriptor_count >= self.crossing_descriptor_count
            >= self.positive_bacc_descriptor_count
            >= self.proper_safe_descriptor_count
            >= self.selected_case_candidate_count
        ):
            raise ProtocolError("CBPUPR gate funnel is not monotone.")
        object.__setattr__(
            self,
            "funnel_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_gate_funnel_v1",
                    "route_count": self.route_count,
                    "descriptor_count": self.descriptor_count,
                    "crossing_descriptor_count": self.crossing_descriptor_count,
                    "positive_bacc_descriptor_count": self.positive_bacc_descriptor_count,
                    "proper_safe_descriptor_count": self.proper_safe_descriptor_count,
                    "selected_case_candidate_count": self.selected_case_candidate_count,
                    "structurally_admissible_center_count": self.structurally_admissible_center_count,
                    "feasible_prefix_center_count": self.feasible_prefix_center_count,
                    "routed_case_count": self.routed_case_count,
                    "exact_p_center_count": self.exact_p_center_count,
                    "stage_counts": [list(row) for row in self.stage_counts],
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "GateFunnel":
        row = cls(
            int(payload["route_count"]),
            int(payload["descriptor_count"]),
            int(payload["crossing_descriptor_count"]),
            int(payload["positive_bacc_descriptor_count"]),
            int(payload["proper_safe_descriptor_count"]),
            int(payload["selected_case_candidate_count"]),
            int(payload["structurally_admissible_center_count"]),
            int(payload["feasible_prefix_center_count"]),
            int(payload["routed_case_count"]),
            int(payload["exact_p_center_count"]),
            tuple(
                (str(value[0]), int(value[1]))
                for value in payload["stage_counts"]  # type: ignore[index]
            ),
        )
        if "funnel_hash" in payload and str(payload["funnel_hash"]) != row.funnel_hash:
            raise ProtocolError("CBPUPR gate funnel hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "route_count": self.route_count,
            "descriptor_count": self.descriptor_count,
            "crossing_descriptor_count": self.crossing_descriptor_count,
            "positive_bacc_descriptor_count": self.positive_bacc_descriptor_count,
            "proper_safe_descriptor_count": self.proper_safe_descriptor_count,
            "selected_case_candidate_count": self.selected_case_candidate_count,
            "structurally_admissible_center_count": self.structurally_admissible_center_count,
            "feasible_prefix_center_count": self.feasible_prefix_center_count,
            "routed_case_count": self.routed_case_count,
            "exact_p_center_count": self.exact_p_center_count,
            "stage_counts": [list(row) for row in self.stage_counts],
            "funnel_hash": self.funnel_hash,
        }


def build_gate_funnel(
    candidate_results: Sequence[CandidateRuntimeResult],
    decisions: Sequence[RouteDecision],
) -> GateFunnel:
    routes = tuple(candidate_results)
    decision_rows = tuple(decisions)
    descriptors = sum(row.descriptor_count for row in routes)
    crossing = sum(len(row.candidates) for row in routes)
    nonpositive = sum(
        NONPOSITIVE_BACC in decision.reason_codes
        for row in routes
        for decision in row.eligibility
    )
    positive = crossing - nonpositive
    proper_unsafe_among_positive = sum(
        NONPOSITIVE_BACC not in decision.reason_codes
        and (
            BRIER_UNSAFE in decision.reason_codes
            or LOG_UNSAFE in decision.reason_codes
        )
        for row in routes
        for decision in row.eligibility
    )
    proper_safe = positive - proper_unsafe_among_positive
    selected = sum(row.selected_candidate is not None for row in routes)
    structural = sum(row.structural_transport.passed for row in decision_rows)
    feasible = sum(row.prefix_selection.authorized for row in decision_rows)
    routed_cases = sum(row.prefix_selection.selected_k for row in decision_rows)
    exact_p = sum(row.composition.exact_p for row in decision_rows)
    stages = (
        ("route_count", len(routes)),
        ("descriptor_count", descriptors),
        ("no_crossing_reject", descriptors - crossing),
        ("crossing_descriptor_count", crossing),
        ("nonpositive_bacc_reject", nonpositive),
        ("positive_bacc_descriptor_count", positive),
        ("proper_unsafe_reject", proper_unsafe_among_positive),
        ("proper_safe_descriptor_count", proper_safe),
        ("selected_case_candidate_count", selected),
        ("structurally_admissible_center_count", structural),
        ("feasible_prefix_center_count", feasible),
        ("routed_case_count", routed_cases),
        ("exact_p_center_count", exact_p),
    )
    return GateFunnel(
        len(routes),
        descriptors,
        crossing,
        positive,
        proper_safe,
        selected,
        structural,
        feasible,
        routed_cases,
        exact_p,
        stages,
    )


@dataclass(frozen=True)
class TerminalMetric:
    center: str
    method_id: str
    row_count: int
    positive_count: int
    negative_count: int
    bacc: float
    brier: float
    log_loss: float

    def __post_init__(self) -> None:
        if (
            not self.center
            or not self.method_id
            or self.row_count != self.positive_count + self.negative_count
            or min(self.positive_count, self.negative_count) <= 0
            or not all(math.isfinite(value) for value in (self.bacc, self.brier, self.log_loss))
        ):
            raise ProtocolError("CBPUPR terminal metric drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "center": self.center,
            "method_id": self.method_id,
            "row_count": self.row_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "bacc": self.bacc,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "terminal_labels_used": True,
        }


def score_terminal_metric(
    *,
    center: str,
    method_id: str,
    labels: Sequence[int],
    probabilities: object,
) -> TerminalMetric:
    p = canonical_float32_probabilities(probabilities)
    y = np.asarray(tuple(labels), dtype=np.int8)
    if y.shape != p.shape or bool(np.any((y != 0) & (y != 1))):
        raise ProtocolError("CBPUPR terminal labels drifted.")
    positive = int(np.sum(y))
    negative = len(y) - positive
    if min(positive, negative) <= 0:
        raise ProtocolError("CBPUPR terminal center lacks both labels.")
    predicted = p >= np.float32(0.5)
    sensitivity = float(np.sum(predicted & (y == 1)) / positive)
    specificity = float(np.sum((~predicted) & (y == 0)) / negative)
    p64 = p.astype(np.float64, copy=False)
    y64 = y.astype(np.float64)
    clipped = np.clip(p64, LOG_CLIP_EPSILON, 1.0 - LOG_CLIP_EPSILON)
    return TerminalMetric(
        str(center),
        str(method_id),
        len(y),
        positive,
        negative,
        0.5 * (sensitivity + specificity),
        float(np.mean((p64 - y64) ** 2, dtype=np.float64)),
        float(
            -np.mean(
                y64 * np.log(clipped) + (1.0 - y64) * np.log(1.0 - clipped),
                dtype=np.float64,
            )
        ),
    )


def favorable_terminal_contrast(
    baseline: TerminalMetric,
    routed: TerminalMetric,
) -> FavorableUtility:
    if (
        baseline.center != routed.center
        or baseline.row_count != routed.row_count
        or baseline.positive_count != routed.positive_count
    ):
        raise ProtocolError("CBPUPR terminal contrast scope drifted.")
    return FavorableUtility(
        routed.bacc - baseline.bacc,
        baseline.brier - routed.brier,
        baseline.log_loss - routed.log_loss,
    )


__all__ = (
    "GateFunnel",
    "TerminalMetric",
    "build_gate_funnel",
    "favorable_terminal_contrast",
    "score_terminal_metric",
)
