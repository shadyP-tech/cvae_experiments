"""Shared validation of the signed winner and nonnegative gate decision seam."""
from ...protocol import ProtocolError
from .hashing import require_sha256, canonical_hash
from .contracts import finite


def winner_evidence_payload(record):
    names = ("winner_arm_id", "winner_composite_hash", "winner_safe_benefit_score",
             "winner_gate_harm_probability", "winner_gate_model_hash", "winner_gate_prediction_hash")
    values = {name: getattr(record, name) for name in names}
    if any(value is not None for value in values.values()):
        if any(value is None for value in values.values()):
            raise ProtocolError("HARP v19 winner decision evidence is incomplete.")
        for name in ("winner_composite_hash", "winner_gate_model_hash", "winner_gate_prediction_hash"):
            require_sha256(values[name], name=name)
        s = finite(values["winner_safe_benefit_score"], name="signed winner safe benefit")
        q = finite(values["winner_gate_harm_probability"], name="winner gate harm")
        if not 0 <= q <= 1:
            raise ProtocolError("HARP v19 winner gate harm is outside [0,1].")
        if record.composite.route_selected and (s <= 0 or 1-q < record.route_threshold
            or abs(record.route_score-(1-q)) > 1e-10
            or values["winner_composite_hash"] != record.composite.composite_hash
            or values["winner_arm_id"] != record.composite.arm_id):
            raise ProtocolError("HARP v19 route violates its sealed winner/gate rule.")
        values["winner_gate_score"] = 1-q
        transcript = getattr(record, "winner_gate_prediction_payload", None)
        if transcript is None:
            raise ProtocolError("HARP v19 winner decision lacks its complete gate transcript.")
        transcript = dict(transcript)
        digest = transcript.pop("prediction_hash", None)
        if (digest != canonical_hash(transcript) or digest != values["winner_gate_prediction_hash"]
            or transcript.get("model_hash") != values["winner_gate_model_hash"]
            or transcript.get("composite_hash") != values["winner_composite_hash"]
            or transcript.get("harm_probability") != q
            or abs(transcript.get("route_score", -1)-(1-q)) > 1e-10):
            raise ProtocolError("HARP v19 complete gate transcript drifted from its decision.")
        transcript["prediction_hash"] = digest
        values["winner_gate_prediction_payload"] = transcript
        is_source_selection = hasattr(record, "policy_enabled")
        enabled = record.policy_enabled if is_source_selection else getattr(record, "admitted", False)
        if is_source_selection or enabled:
            should_route = enabled and s > 0 and 1-q >= record.route_threshold
            if bool(record.composite.route_selected) != should_route:
                raise ProtocolError("HARP v19 enabled/admitted route does not implement the complete winner rule.")
            expected = (None if should_route else
                        "NO_SAFE_INNER_OOF_POLICY" if not enabled else
                        "NONPOSITIVE_SAFE_BENEFIT" if s <= 0 else "WINNER_GATE_BELOW_THRESHOLD")
            if record.fallback_reason != expected:
                raise ProtocolError("HARP v19 enabled/admitted winner fallback reason drifted.")
    else:
        if getattr(record, "winner_gate_prediction_payload", None) is not None:
            raise ProtocolError("HARP v19 gate transcript has no bound winner evidence.")
        from .candidate_prediction import POLICY_ARM_ID
        if getattr(record, "requested_arm_id", None) == POLICY_ARM_ID:
            expected = ("NO_PREDICTION_CHANGING_CANDIDATE" if record.policy_enabled
                        else "NO_SAFE_INNER_OOF_POLICY")
            if record.composite.route_selected or record.fallback_reason != expected:
                raise ProtocolError("HARP v19 complete policy lacks winner evidence or an explicit empty-winner fallback.")
        values["winner_gate_score"] = None
        values["winner_gate_prediction_payload"] = None
    return values


def decision_evidence(winner, gate):
    if winner is None or gate is None:
        return {}
    return {"winner_arm_id": winner.arm_id,
        "winner_composite_hash": winner.candidate.composite.composite_hash,
        "winner_safe_benefit_score": winner.safe_benefit_score,
        "winner_gate_harm_probability": gate.harm_probability,
        "winner_gate_model_hash": gate.model_hash,
        "winner_gate_prediction_hash": gate.prediction_hash,
        "winner_gate_prediction_payload": tuple(gate.public_payload().items())}
