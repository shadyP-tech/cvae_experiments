"""Label-free construction of route-local posterior utility candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .canonical_probabilities import (
    CanonicalProbabilityVector,
    canonical_float32_probabilities,
    canonical_hash,
    require_sha256,
)
from .eligibility import (
    ALTERNATIVE_ORDER,
    DIRECTION_ORDER,
    ActionCandidate,
    EligibilityDecision,
    assess_action,
    select_best_eligible_action,
)
from .posterior_expected_utility import score_posterior_folds


SOURCE_EXCLUSION_ROLE = (
    "actionable_endpoint_source_selection_only_not_posterior_fingerprint_covariates"
)


@dataclass(frozen=True)
class CandidateRuntimeResult:
    outer_center: str
    center: str
    case_id: str
    control_id: str
    descriptor_count: int
    no_crossing_count: int
    candidates: tuple[ActionCandidate, ...]
    eligibility: tuple[EligibilityDecision, ...]
    selected_candidate: ActionCandidate | None
    posterior_model_reference_count: int
    posterior_model_hash: str
    support_capability_hash: str
    source_excluded_centers: tuple[str, ...]
    endpoint_lineage_hash: str
    runtime_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = tuple(sorted(set(str(value) for value in self.source_excluded_centers)))
        expected_selected = select_best_eligible_action(self.candidates)
        if (
            not self.outer_center
            or not self.center
            or not self.case_id
            or not self.control_id
            or self.descriptor_count != len(ALTERNATIVE_ORDER) * len(DIRECTION_ORDER)
            or self.no_crossing_count + len(self.candidates) != self.descriptor_count
            or len(self.candidates) != len(self.eligibility)
            or any(row.candidate_hash != candidate.action_hash for row, candidate in zip(self.eligibility, self.candidates, strict=True))
            or self.posterior_model_reference_count != 1
            or any(
                row.center != self.center
                or row.case_id != self.case_id
                or row.control_id != self.control_id
                for row in self.candidates
            )
            or tuple(assess_action(row) for row in self.candidates) != self.eligibility
            or (
                None if expected_selected is None else expected_selected.action_hash
            )
            != (
                None
                if self.selected_candidate is None
                else self.selected_candidate.action_hash
            )
            or set(excluded)
            != {self.outer_center, self.center}
            or (
                self.selected_candidate is not None
                and self.selected_candidate.action_hash
                not in {row.action_hash for row in self.candidates}
            )
        ):
            raise ProtocolError("CBPUPR candidate runtime result drifted.")
        require_sha256(self.posterior_model_hash, "posterior_model_hash")
        require_sha256(self.support_capability_hash, "support_capability_hash")
        require_sha256(self.endpoint_lineage_hash, "endpoint_lineage_hash")
        object.__setattr__(self, "source_excluded_centers", excluded)
        object.__setattr__(
            self,
            "runtime_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_candidate_runtime_v1",
                    "outer_center": self.outer_center,
                    "center": self.center,
                    "case_id": self.case_id,
                    "control_id": self.control_id,
                    "descriptor_count": self.descriptor_count,
                    "no_crossing_count": self.no_crossing_count,
                    "candidate_hashes": [row.action_hash for row in self.candidates],
                    "eligibility": [row.to_payload() for row in self.eligibility],
                    "selected_candidate_hash": None
                    if self.selected_candidate is None
                    else self.selected_candidate.action_hash,
                    "posterior_model_reference_count": (
                        self.posterior_model_reference_count
                    ),
                    "posterior_fit_increment": 0,
                    "posterior_refit": False,
                    "posterior_model_hash": self.posterior_model_hash,
                    "support_capability_hash": self.support_capability_hash,
                    "source_excluded_centers": list(self.source_excluded_centers),
                    "source_excluded_centers_role": SOURCE_EXCLUSION_ROLE,
                    "endpoint_lineage_hash": self.endpoint_lineage_hash,
                    "support_labels_used_indirectly": True,
                    "held_case_label_used": False,
                    "terminal_evaluation_labels_used": False,
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CandidateRuntimeResult":
        if (
            payload.get("posterior_fit_increment") != 0
            or payload.get("posterior_refit") is not False
            or payload.get("source_excluded_centers_role")
            != SOURCE_EXCLUSION_ROLE
        ):
            raise ProtocolError("CBPUPR candidate runtime attempted a posterior refit.")
        selected_payload = payload.get("selected_candidate")
        row = cls(
            str(payload["outer_center"]),
            str(payload["center"]),
            str(payload["case_id"]),
            str(payload["control_id"]),
            int(payload["descriptor_count"]),
            int(payload["no_crossing_count"]),
            tuple(
                ActionCandidate.from_payload(value)
                for value in payload["candidates"]  # type: ignore[index]
            ),
            tuple(
                EligibilityDecision.from_payload(value)
                for value in payload["eligibility"]  # type: ignore[index]
            ),
            None
            if selected_payload is None
            else ActionCandidate.from_payload(selected_payload),  # type: ignore[arg-type]
            int(payload["posterior_model_reference_count"]),
            str(payload["posterior_model_hash"]),
            str(payload["support_capability_hash"]),
            tuple(str(value) for value in payload["source_excluded_centers"]),  # type: ignore[index]
            str(payload["endpoint_lineage_hash"]),
        )
        if "runtime_hash" in payload and str(payload["runtime_hash"]) != row.runtime_hash:
            raise ProtocolError("CBPUPR candidate runtime hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_center": self.outer_center,
            "center": self.center,
            "case_id": self.case_id,
            "control_id": self.control_id,
            "descriptor_count": self.descriptor_count,
            "no_crossing_count": self.no_crossing_count,
            "candidates": [row.to_payload() for row in self.candidates],
            "eligibility": [row.to_payload() for row in self.eligibility],
            "selected_candidate": None
            if self.selected_candidate is None
            else self.selected_candidate.to_payload(),
            "posterior_model_reference_count": self.posterior_model_reference_count,
            "posterior_fit_increment": 0,
            "posterior_refit": False,
            "posterior_model_hash": self.posterior_model_hash,
            "support_capability_hash": self.support_capability_hash,
            "source_excluded_centers": list(self.source_excluded_centers),
            "source_excluded_centers_role": SOURCE_EXCLUSION_ROLE,
            "endpoint_lineage_hash": self.endpoint_lineage_hash,
            "support_labels_used_indirectly": True,
            "held_case_label_used": False,
            "terminal_evaluation_labels_used": False,
            "runtime_hash": self.runtime_hash,
        }


def directional_candidate_probabilities(
    portfolio_probabilities: object,
    alternative_probabilities: object,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a pure directional action and its threshold-crossing mask."""

    p = canonical_float32_probabilities(portfolio_probabilities)
    alternative = canonical_float32_probabilities(
        alternative_probabilities, expected_length=len(p)
    )
    if direction == "zero_to_one":
        crossing = (p < np.float32(0.5)) & (alternative >= np.float32(0.5))
    elif direction == "one_to_zero":
        crossing = (p >= np.float32(0.5)) & (alternative < np.float32(0.5))
    else:
        raise ProtocolError("CBPUPR directional candidate direction drifted.")
    candidate = p.copy(order="C")
    candidate[crossing] = alternative[crossing]
    candidate.setflags(write=False)
    return candidate, crossing


def build_case_candidates(
    *,
    center: str,
    case_id: str,
    portfolio_probabilities: object,
    alternative_probabilities: Mapping[str, object],
    posterior_eta: object,
    control_id: str = "IDENTITY",
    support_n_positive: float = 0.0,
    support_n_negative: float = 0.0,
    support_row_count: int = 0,
    posterior_model_hash: str | None = None,
    support_capability_hash: str | None = None,
    outer_center: str | None = None,
    source_excluded_centers: Sequence[str] | None = None,
    endpoint_lineage_hash: str | None = None,
) -> CandidateRuntimeResult:
    """Build and select the six fixed B/I/R-by-direction candidates.

    The v1 workload fits exactly one H-minus-c posterior per fingerprint
    control.  The singleton sequence passed to ``score_posterior_folds`` is a
    sealed posterior realization, not an inner cross-validation ensemble.
    """

    if tuple(alternative_probabilities) != ALTERNATIVE_ORDER:
        raise ProtocolError("CBPUPR alternative endpoint order drifted.")
    outer = str(center) if outer_center is None else str(outer_center)
    excluded = tuple(
        sorted(
            set(
                str(value)
                for value in (
                    (outer, center)
                    if source_excluded_centers is None
                    else source_excluded_centers
                )
            )
        )
    )
    if set(excluded) != {outer, str(center)}:
        raise ProtocolError(
            "CBPUPR endpoint candidates require exact outer-H/target-J source exclusion."
        )
    p = canonical_float32_probabilities(portfolio_probabilities)
    eta = np.asarray(posterior_eta, dtype=np.float64)
    if (
        eta.shape != p.shape
        or not np.isfinite(eta).all()
        or bool(np.any((eta < 0.0) | (eta > 1.0)))
    ):
        raise ProtocolError("CBPUPR held-case posterior prediction drifted.")
    posterior_hash = canonical_hash(
        {
            "schema_version": "cbpupr_singleton_route_posterior_v1",
            "center": str(center),
            "case_id": str(case_id),
            "control_id": str(control_id),
            "eta": eta.tolist(),
            "support_n_positive": float(support_n_positive),
            "support_n_negative": float(support_n_negative),
            "support_row_count": int(support_row_count),
            "whole_case_excluded": True,
            "inner_crossfit_used": False,
        }
    )
    bound_model_hash = (
        posterior_hash if posterior_model_hash is None else str(posterior_model_hash)
    )
    bound_capability_hash = (
        canonical_hash(
            {
                "schema_version": "cbpupr_route_support_capability_reference_v1",
                "center": str(center),
                "held_case_id": str(case_id),
                "support_n_positive": float(support_n_positive),
                "support_n_negative": float(support_n_negative),
                "support_row_count": int(support_row_count),
                "held_case_excluded": True,
            }
        )
        if support_capability_hash is None
        else str(support_capability_hash)
    )
    require_sha256(bound_model_hash, "posterior_model_hash")
    require_sha256(bound_capability_hash, "support_capability_hash")
    bound_endpoint_hash = (
        canonical_hash(
            {
                "schema_version": "cbpupr_endpoint_source_exclusion_lineage_v1",
                "outer_center": outer,
                "target_center": str(center),
                "source_excluded_centers": list(excluded),
                "portfolio": CanonicalProbabilityVector.from_array(p).sha256,
                "alternatives": {
                    key: CanonicalProbabilityVector.from_array(value).sha256
                    for key, value in alternative_probabilities.items()
                },
            }
        )
        if endpoint_lineage_hash is None
        else str(endpoint_lineage_hash)
    )
    require_sha256(bound_endpoint_hash, "endpoint_lineage_hash")

    candidates: list[ActionCandidate] = []
    no_crossing = 0
    for alternative_id in ALTERNATIVE_ORDER:
        alternative = alternative_probabilities[alternative_id]
        for direction in DIRECTION_ORDER:
            probabilities, crossing = directional_candidate_probabilities(
                p, alternative, direction
            )
            if not bool(np.any(crossing)):
                no_crossing += 1
                continue
            action_id = f"{alternative_id}::{direction}"
            estimate = score_posterior_folds(
                center=str(center),
                case_id=str(case_id),
                action_id=action_id,
                direction=direction,
                control_id=str(control_id),
                portfolio_probabilities=p,
                candidate_probabilities=probabilities,
                posterior_folds=(eta,),
                posterior_hash=posterior_hash,
                support_n_positive=support_n_positive,
                support_n_negative=support_n_negative,
                support_row_count=support_row_count,
            )
            candidates.append(
                ActionCandidate(
                    str(center),
                    str(case_id),
                    alternative_id,
                    direction,
                    str(control_id),
                    CanonicalProbabilityVector.from_array(probabilities),
                    estimate,
                )
            )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda row: (
                ALTERNATIVE_ORDER.index(row.alternative_id),
                DIRECTION_ORDER.index(row.direction),
                row.action_hash,
            ),
        )
    )
    eligibility = tuple(assess_action(row) for row in ordered)
    return CandidateRuntimeResult(
        outer,
        str(center),
        str(case_id),
        str(control_id),
        len(ALTERNATIVE_ORDER) * len(DIRECTION_ORDER),
        no_crossing,
        ordered,
        eligibility,
        select_best_eligible_action(ordered),
        1,
        bound_model_hash,
        bound_capability_hash,
        excluded,
        bound_endpoint_hash,
    )


__all__ = (
    "CandidateRuntimeResult",
    "SOURCE_EXCLUSION_ROLE",
    "build_case_candidates",
    "directional_candidate_probabilities",
)
