"""Total, reason-coded route diagnostic decisions for PCSI-RACR."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .calibration import RouteCalibration
from .constants import (
    PRIMARY_METHOD_ID,
    PROJECTED_NO_ENVELOPE_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    RAW_OBSERVED_MAX_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID,
)
from .contracts import TargetRouteKey
from .hashing import canonical_hash, require_sha256
from .policy_selection import CaseCandidatePolicy
from .transport import RouteTransportScreen


CHANGE = "CHANGE"
ABSTAIN_TO_P = "ABSTAIN_TO_P"
REASON_CHANGE = "STRICT_ALL_COORDINATE_MARGIN_POSITIVE"
REASON_NO_ACTION = "NO_ELIGIBLE_ACTION"
REASON_TRANSPORT = "TARGET_TRANSPORT_FAILED"
REASON_CALIBRATION = "INCOMPLETE_OR_INVALID_OUTER_CALIBRATION"
REASON_MARGIN = "EQUALITY_OR_NONPOSITIVE_CORRECTED_COORDINATE"
REASON_NONFINITE = "NONFINITE_INPUT"


@dataclass(frozen=True, order=True)
class RouteDiagnosticDecision:
    policy_id: str
    route: TargetRouteKey
    geometry_id: str
    outcome: str
    reason_code: str
    candidate_policy_hash: str
    transport_screen_hash: str
    calibration_hash: str
    predicted_favorable_vector: tuple[float, float, float]
    diagnostic_margin: tuple[float, float, float]
    corrected_vector: tuple[float, float, float]
    decision_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        if (
            self.policy_id
            not in {
                PRIMARY_METHOD_ID,
                RAW_OBSERVED_MAX_METHOD_ID,
                PROJECTED_NO_ENVELOPE_METHOD_ID,
            }
            or self.geometry_id not in {PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID}
            or self.outcome not in {CHANGE, ABSTAIN_TO_P}
            or not self.reason_code
        ):
            raise ProtocolError("PCSI-RACR decision identity drifted.")
        for digest in (
            self.candidate_policy_hash,
            self.transport_screen_hash,
            self.calibration_hash,
        ):
            require_sha256(digest, "route_decision_binding_hash")
        predicted = tuple(float(value) for value in self.predicted_favorable_vector)
        margin = tuple(float(value) for value in self.diagnostic_margin)
        corrected = tuple(float(value) for value in self.corrected_vector)
        if (
            len(predicted) != 3
            or len(margin) != 3
            or len(corrected) != 3
            or any(not math.isfinite(value) for value in (*predicted, *margin, *corrected))
            or (self.outcome == CHANGE and not all(value > 0.0 for value in corrected))
        ):
            raise ProtocolError("PCSI-RACR decision vector drifted.")
        object.__setattr__(self, "predicted_favorable_vector", predicted)
        object.__setattr__(self, "diagnostic_margin", margin)
        object.__setattr__(self, "corrected_vector", corrected)
        object.__setattr__(self, "decision_hash", canonical_hash(self._unhashed()))

    @property
    def changed(self) -> bool:
        return self.outcome == CHANGE

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_route_diagnostic_decision_v1",
            "policy_id": self.policy_id,
            "outer_center": self.route.outer_center,
            "case_id": self.route.case_id,
            "geometry_id": self.geometry_id,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "candidate_policy_hash": self.candidate_policy_hash,
            "transport_screen_hash": self.transport_screen_hash,
            "calibration_hash": self.calibration_hash,
            "predicted_favorable_vector": list(self.predicted_favorable_vector),
            "diagnostic_margin": list(self.diagnostic_margin),
            "corrected_vector": list(self.corrected_vector),
            "consumed_test": True,
            "eligible_for_promotion": False,
            "may_feed": False,
            "claim_boundary": "NON_GUARANTEE_CONSUMED_TEST_ONLY",
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "decision_hash": self.decision_hash}


def make_route_decision(
    candidate: CaseCandidatePolicy,
    screen: RouteTransportScreen,
    calibration: RouteCalibration,
    *,
    policy_id: str,
    no_envelope: bool = False,
) -> RouteDiagnosticDecision:
    route = TargetRouteKey(candidate.target_center, candidate.case_id)
    if (
        screen.outer_center != route.outer_center
        or screen.candidate_center != route.outer_center
        or screen.candidate_case_id != route.case_id
        or calibration.outer_center != route.outer_center
        or calibration.geometry_id != candidate.geometry_id
    ):
        raise ProtocolError("PCSI-RACR decision crossed a route scope.")
    predicted = tuple(float(value) for value in candidate.predicted_favorable_endpoint_vector)
    margin = (0.0, 0.0, 0.0) if no_envelope else calibration.margin
    corrected = tuple(predicted[index] - margin[index] for index in range(3))
    if any(not math.isfinite(value) for value in (*predicted, *margin, *corrected)):
        reason = REASON_NONFINITE
        outcome = ABSTAIN_TO_P
        corrected = tuple(value if math.isfinite(value) else 0.0 for value in corrected)
    elif not candidate.changed:
        reason = REASON_NO_ACTION
        outcome = ABSTAIN_TO_P
    elif not screen.passed:
        reason = REASON_TRANSPORT
        outcome = ABSTAIN_TO_P
    elif not no_envelope and not calibration.valid:
        reason = REASON_CALIBRATION
        outcome = ABSTAIN_TO_P
    elif all(value > 0.0 for value in corrected):
        reason = REASON_CHANGE
        outcome = CHANGE
    else:
        reason = REASON_MARGIN
        outcome = ABSTAIN_TO_P
    return RouteDiagnosticDecision(
        policy_id,
        route,
        candidate.geometry_id,
        outcome,
        reason,
        candidate.policy_hash,
        screen.screen_hash,
        calibration.calibration_hash,
        predicted,
        margin,
        corrected,
    )


__all__ = (
    "ABSTAIN_TO_P",
    "CHANGE",
    "RouteDiagnosticDecision",
    "make_route_decision",
)
