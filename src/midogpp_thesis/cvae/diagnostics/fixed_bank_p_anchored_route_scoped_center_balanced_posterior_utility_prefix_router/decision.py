"""Primary aggregate prefix decision with structural-only hard transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .candidate_runtime import CandidateRuntimeResult
from .canonical_probabilities import canonical_hash
from .composition import CompositionResult, compose_center_probabilities, compose_exact_p
from .policy_prefixes import PrefixCandidate, PrefixSelection, select_prefix
from .transport_geometry import StructuralTransportGate
from .utility_calibration import CenterBalancedUtilityCalibration


PRIMARY_METHOD_ID = "CBPUPR_UNIFIED_PREFIX"
ABSTAIN_TO_P = "ABSTAIN_TO_EXACT_P"
ROUTE_PREFIX = "ROUTE_SELECTED_PREFIX"


@dataclass(frozen=True)
class RouteDecision:
    center: str
    method_id: str
    action: str
    reason_codes: tuple[str, ...]
    prefix_selection: PrefixSelection
    composition: CompositionResult
    structural_transport: StructuralTransportGate
    utility_calibration_hash: str
    candidate_runtime_hashes: tuple[str, ...]
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        routed = self.action == ROUTE_PREFIX
        if (
            not self.center
            or not self.method_id
            or self.action not in (ABSTAIN_TO_P, ROUTE_PREFIX)
            or not self.reason_codes
            or routed != self.prefix_selection.authorized
            or (routed and not self.structural_transport.passed)
            or routed == self.composition.exact_p
            or not self.candidate_runtime_hashes
            or len(set(self.candidate_runtime_hashes))
            != len(self.candidate_runtime_hashes)
        ):
            raise ProtocolError("CBPUPR route decision contract drifted.")
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_route_decision_v1",
                    "center": self.center,
                    "method_id": self.method_id,
                    "action": self.action,
                    "reason_codes": list(self.reason_codes),
                    "prefix_selection_hash": self.prefix_selection.selection_hash,
                    "composition_hash": self.composition.composition_hash,
                    "structural_transport_hash": self.structural_transport.gate_hash,
                    "utility_calibration_hash": self.utility_calibration_hash,
                    "candidate_runtime_hashes": list(self.candidate_runtime_hashes),
                    "policy_replay_bias_used": False,
                }
            ),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RouteDecision":
        row = cls(
            str(payload["center"]),
            str(payload["method_id"]),
            str(payload["action"]),
            tuple(str(value) for value in payload["reason_codes"]),  # type: ignore[index]
            PrefixSelection.from_payload(payload["prefix_selection"]),  # type: ignore[arg-type]
            CompositionResult.from_payload(payload["composition"]),  # type: ignore[arg-type]
            StructuralTransportGate.from_payload(payload["structural_transport"]),  # type: ignore[arg-type]
            str(payload["utility_calibration_hash"]),
            tuple(str(value) for value in payload["candidate_runtime_hashes"]),  # type: ignore[index]
        )
        if "decision_hash" in payload and str(payload["decision_hash"]) != row.decision_hash:
            raise ProtocolError("CBPUPR route decision hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "center": self.center,
            "method_id": self.method_id,
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "prefix_selection": self.prefix_selection.to_payload(),
            "composition": self.composition.to_payload(),
            "structural_transport": self.structural_transport.to_payload(),
            "utility_calibration_hash": self.utility_calibration_hash,
            "candidate_runtime_hashes": list(self.candidate_runtime_hashes),
            "policy_replay_bias_used": False,
            "decision_hash": self.decision_hash,
        }


def make_route_decision(
    *,
    center: str,
    portfolio_probabilities: object,
    sample_case_ids: Sequence[str],
    candidate_results: Sequence[CandidateRuntimeResult],
    utility_calibration: CenterBalancedUtilityCalibration,
    structural_transport: StructuralTransportGate,
    method_id: str = PRIMARY_METHOD_ID,
) -> RouteDecision:
    runtime_rows = tuple(candidate_results)
    rows = tuple(
        result.selected_candidate
        for result in runtime_rows
        if result.selected_candidate is not None
    )
    if (
        utility_calibration.outer_center != str(center)
        or structural_transport.target_center != str(center)
        or any(row.center != str(center) for row in rows)
        or not runtime_rows
        or len({result.case_id for result in runtime_rows}) != len(runtime_rows)
        or any(
            result.outer_center != str(center)
            or result.center != str(center)
            or set(result.source_excluded_centers) != {str(center)}
            for result in runtime_rows
        )
    ):
        raise ProtocolError("CBPUPR decision route scope drifted.")
    prefix_candidates = tuple(
        PrefixCandidate(
            row,
            utility_calibration.correct(row.estimate.utility),
            utility_calibration.calibration_hash,
        )
        for row in rows
    )
    # The single frozen calibration is the donor-center utility bias above.
    # Pseudo PolicyReplay residuals are diagnostic holdouts and never introduce
    # a second policy-level correction into primary authorization.
    selection = select_prefix(prefix_candidates)
    if not structural_transport.passed:
        composition = compose_exact_p(portfolio_probabilities)
        return RouteDecision(
            str(center),
            str(method_id),
            ABSTAIN_TO_P,
            structural_transport.reason_codes,
            # Force the selection to baseline so decision and composition agree.
            select_prefix(()),
            composition,
            structural_transport,
            utility_calibration.calibration_hash,
            tuple(sorted(result.runtime_hash for result in runtime_rows)),
        )
    if not selection.authorized:
        return RouteDecision(
            str(center),
            str(method_id),
            ABSTAIN_TO_P,
            ("NO_FEASIBLE_AGGREGATE_PREFIX",),
            selection,
            compose_exact_p(portfolio_probabilities),
            structural_transport,
            utility_calibration.calibration_hash,
            tuple(sorted(result.runtime_hash for result in runtime_rows)),
        )
    selected = tuple(row.candidate for row in selection.selected_candidates)
    return RouteDecision(
        str(center),
        str(method_id),
        ROUTE_PREFIX,
        ("STRUCTURAL_AND_AGGREGATE_UTILITY_PASS",),
        selection,
        compose_center_probabilities(
            portfolio_probabilities, sample_case_ids, selected
        ),
        structural_transport,
        utility_calibration.calibration_hash,
        tuple(sorted(result.runtime_hash for result in runtime_rows)),
    )


__all__ = (
    "ABSTAIN_TO_P",
    "PRIMARY_METHOD_ID",
    "ROUTE_PREFIX",
    "RouteDecision",
    "make_route_decision",
)
