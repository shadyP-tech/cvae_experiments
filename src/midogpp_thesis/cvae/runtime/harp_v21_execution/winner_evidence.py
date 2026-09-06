"""Label-free verification of the signed-score / harm-gate decision boundary."""
from collections.abc import Mapping
import math

from ...protocol import ProtocolError
from ...routing.harp_protocol import require_sha256
from .gate_replay import replay_gate_prediction


def validate_winner_evidence(payload: Mapping, *, gate_model: Mapping,
                             admitted: bool, threshold: float, routed: bool) -> None:
    """Check declared winner evidence against the authenticated complete policy.

    The gate is replayed on its sealed features. Physical recipe replay is
    separate; the ranker and candidate model are not fitted by this validator.
    """
    if payload.get("route_threshold") != threshold:
        raise ProtocolError("HARP v21 route threshold escaped its frozen policy.")
    if payload.get("admitted") is not admitted:
        raise ProtocolError("HARP v21 decision admission escaped its frozen policy.")
    present = payload.get("winner_composite_hash") is not None
    if not present:
        if routed:
            raise ProtocolError("HARP v21 routed action lacks complete winner evidence.")
        if admitted and payload.get("fallback_reason") != "NO_FEASIBLE_POSITIVE_GAIN_CANDIDATE":
            raise ProtocolError("HARP v21 empty winner fallback reason drifted.")
        if payload.get("winner_gate_prediction_payload") is not None:
            raise ProtocolError("HARP v21 gate prediction has no winner evidence.")
        return
    for name in ("winner_composite_hash", "winner_gate_model_hash", "winner_gate_prediction_hash"):
        require_sha256(payload.get(name), name=name)
    try:
        signed = float(payload["winner_risk_adjusted_score"])
        harm = float(payload["winner_gate_harm_probability"])
        score = float(payload["winner_gate_score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError("HARP v21 winner scores are absent or malformed.") from exc
    if (not all(math.isfinite(v) for v in (signed, harm, score))
        or not 0 <= harm <= 1 or abs(score - (1-harm)) > 1e-10
        or payload["winner_gate_model_hash"] != gate_model["model_hash"]):
        raise ProtocolError("HARP v21 winner gate scores or model binding drifted.")
    replay_gate_prediction(payload, gate_model)
    available = payload["winner_gate_prediction_payload"]["calibration_available"]
    should_route = admitted and signed > 0 and available and score >= threshold
    if routed != should_route:
        raise ProtocolError("HARP v21 decision violates the admitted winner rule in either direction.")
    if admitted:
        expected_reason = (None if should_route else "NONPOSITIVE_PREDICTED_GAIN"
                           if signed <= 0 else "WINNER_GATE_UNAVAILABLE" if not available
                           else "WINNER_GATE_BELOW_THRESHOLD")
        if payload.get("fallback_reason") != expected_reason:
            raise ProtocolError("HARP v21 winner fallback reason drifted.")
    if routed and (
        signed <= 0 or score < threshold or payload.get("route_score") != score
        or payload.get("winner_arm_id") != payload.get("selected_arm_id")
        or payload.get("winner_composite_hash") != payload.get("composite_hash")
        or payload.get("admitted") is not True
        or payload.get("fallback_reason") is not None
        or payload.get("prediction_changed") is not True
    ):
        raise ProtocolError("HARP v21 routed action violates its frozen winner/gate rule.")
