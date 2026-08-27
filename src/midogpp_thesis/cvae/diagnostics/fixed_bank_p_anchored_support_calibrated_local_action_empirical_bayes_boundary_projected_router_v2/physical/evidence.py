"""Label-free per-case evidence packets computed before action-value fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..hashing import canonical_hash
from ..protocol import GovernanceError
from .contracts import FeatureVector, HARD_THRESHOLD, binary_entropy
from .endpoints import CaseEndpointSurface
from .geometry import BoundaryAction


EVIDENCE_FEATURE_NAMES = (
    "case_row_count_log1p",
    "direction_branch_rate",
    "crossing_count_log1p",
    "crossing_rate",
    "protected_mean_on_branch",
    "endpoint_mean_on_branch",
    "protected_abs_margin_on_branch",
    "endpoint_abs_margin_on_branch",
    "signed_shift_on_crossings",
    "absolute_shift_on_crossings",
    "protected_entropy_on_crossings",
    "endpoint_entropy_on_crossings",
    "protected_seed_sd_on_crossings",
    "endpoint_seed_sd_on_crossings",
    "protected_vote_disagreement_on_crossings",
    "endpoint_vote_disagreement_on_crossings",
    "crossing_seed_support_fraction",
    "structural_noop",
)
HARMFUL_SWITCH_COUNT_STATUS = "UNAVAILABLE_PRETERMINAL_TARGET_LABELS_CLOSED"


@dataclass(frozen=True, slots=True)
class CaseEvidencePacket:
    target_center: str
    case_id: str
    action_id: str
    descriptor: FeatureVector
    endpoint_surface_hash: str
    action_hash: str
    threshold_switch_count: int
    harmful_switch_count: int | None
    harmful_switch_count_status: str
    packet_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.target_center or not self.case_id or not self.action_id:
            raise GovernanceError("SCALE-BP v2 evidence packet identity drifted.")
        if self.descriptor.names != EVIDENCE_FEATURE_NAMES:
            raise GovernanceError("SCALE-BP v2 evidence feature schema drifted.")
        if (
            isinstance(self.threshold_switch_count, bool)
            or not isinstance(self.threshold_switch_count, int)
            or self.threshold_switch_count < 0
            or self.harmful_switch_count is not None
            or self.harmful_switch_count_status != HARMFUL_SWITCH_COUNT_STATUS
        ):
            raise GovernanceError("SCALE-BP v2 action-switch audit drifted.")
        object.__setattr__(
            self,
            "packet_hash",
            canonical_hash(
                {
                    "schema_version": "scale_bp_v2_case_evidence_packet_v2",
                    "target_center": self.target_center,
                    "case_id": self.case_id,
                    "action_id": self.action_id,
                    "descriptor_hash": self.descriptor.feature_hash,
                    "endpoint_surface_hash": self.endpoint_surface_hash,
                    "action_hash": self.action_hash,
                    "threshold_switch_count": self.threshold_switch_count,
                    "harmful_switch_count": self.harmful_switch_count,
                    "harmful_switch_count_status": self.harmful_switch_count_status,
                    "target_labels_used": False,
                }
            ),
        )


def build_case_evidence_packet(
    surface: CaseEndpointSurface,
    action: BoundaryAction,
) -> CaseEvidencePacket:
    if not np.array_equal(surface.protected_p, action.protected_p):
        raise GovernanceError("SCALE-BP v2 evidence action is not anchored to surface P.")
    baseline = action.protected_p
    endpoint = action.endpoint
    direction_mask = (
        baseline < HARD_THRESHOLD
        if action.direction == "zero_to_one"
        else baseline >= HARD_THRESHOLD
    )
    crossing = np.zeros(len(baseline), dtype=bool)
    crossing[list(action.crossing_indices)] = True
    family_seed = surface.seed_challenger_probabilities[action.action_id]
    p_seed = surface.seed_protected_component_probabilities["P_PROTECTED"]
    p_sd = np.std(p_seed, axis=0, ddof=0, dtype=np.float64)
    endpoint_sd = np.std(family_seed, axis=0, ddof=0, dtype=np.float64)
    p_vote = np.mean(p_seed >= HARD_THRESHOLD, axis=0, dtype=np.float64)
    endpoint_vote = np.mean(family_seed >= HARD_THRESHOLD, axis=0, dtype=np.float64)

    def mean_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask], dtype=np.float64)) if np.any(mask) else 0.0

    vote_disagreement_p = 2.0 * np.minimum(p_vote, 1.0 - p_vote)
    vote_disagreement_endpoint = 2.0 * np.minimum(endpoint_vote, 1.0 - endpoint_vote)
    crossing_per_seed = (p_seed >= HARD_THRESHOLD) != (family_seed >= HARD_THRESHOLD)
    if action.direction == "zero_to_one":
        crossing_per_seed &= p_seed < HARD_THRESHOLD
    else:
        crossing_per_seed &= p_seed >= HARD_THRESHOLD
    values = (
        math.log1p(len(baseline)),
        float(np.mean(direction_mask, dtype=np.float64)),
        math.log1p(len(action.crossing_indices)),
        float(len(action.crossing_indices) / len(baseline)),
        mean_or_zero(baseline, direction_mask),
        mean_or_zero(endpoint, direction_mask),
        mean_or_zero(np.abs(baseline - HARD_THRESHOLD), direction_mask),
        mean_or_zero(np.abs(endpoint - HARD_THRESHOLD), direction_mask),
        mean_or_zero(endpoint - baseline, crossing),
        mean_or_zero(np.abs(endpoint - baseline), crossing),
        mean_or_zero(binary_entropy(baseline), crossing),
        mean_or_zero(binary_entropy(endpoint), crossing),
        mean_or_zero(p_sd, crossing),
        mean_or_zero(endpoint_sd, crossing),
        mean_or_zero(vote_disagreement_p, crossing),
        mean_or_zero(vote_disagreement_endpoint, crossing),
        float(np.mean(crossing_per_seed, dtype=np.float64)),
        float(action.structural_noop),
    )
    return CaseEvidencePacket(
        surface.target_center,
        surface.case_id,
        action.action_id,
        FeatureVector(EVIDENCE_FEATURE_NAMES, values),
        surface.surface_hash,
        action.action_hash,
        len(action.crossing_indices),
        None,
        HARMFUL_SWITCH_COUNT_STATUS,
    )


__all__ = (
    "CaseEvidencePacket",
    "EVIDENCE_FEATURE_NAMES",
    "HARMFUL_SWITCH_COUNT_STATUS",
    "build_case_evidence_packet",
)
