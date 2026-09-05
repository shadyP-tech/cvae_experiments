"""Recompute complete model identities from fresh manifests without unpickling."""
from collections.abc import Mapping
from ...protocol import ProtocolError
from .hashing import canonical_hash


def _model_hash(payload, name):
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"HARP v19 {name} payload is absent.")
    body = dict(payload)
    advertised = body.pop("model_hash", None)
    if advertised != canonical_hash(body):
        raise ProtocolError(f"HARP v19 {name} model hash drifted.")
    return advertised


def verify_complete_model_payload(payload: Mapping) -> str:
    """Bind the gate, both proposer copies, action model, and complete rule."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "harp_v19_complete_nested_winner_learner":
        raise ProtocolError("HARP v19 requires the complete nested winner model schema.")
    gate = payload.get("winner_gate")
    _model_hash(gate, "winner gate")
    proposer = payload.get("proposer")
    _model_hash(proposer, "ranker-stacked proposer")
    _model_hash(payload.get("action_model"), "actual action")
    if (canonical_hash(payload.get("action_model")) != canonical_hash(proposer.get("action_model"))
        or canonical_hash(payload.get("proposal_model")) != canonical_hash(proposer.get("proposal_model"))):
        raise ProtocolError("HARP v19 duplicate proposer components drifted.")
    keys = tuple(tuple(key) for key in payload.get("training_case_keys", ()))
    if not keys or tuple(tuple(key) for key in gate.get("training_case_keys", ())) != keys:
        raise ProtocolError("HARP v19 complete model/gate training scopes drifted.")
    if payload.get("policy_rule") != "MAX_NONBASELINE_SAFE_BENEFIT_THEN_POSITIVE_SCORE_AND_ONE_MINUS_WINNER_HARM_GE_TAU_ELSE_EXACT_B" or payload.get("runnerup_after_gate_veto") is not False:
        raise ProtocolError("HARP v19 complete winner decision rule drifted.")
    return _model_hash(payload, "complete learner")


__all__ = ("verify_complete_model_payload",)
