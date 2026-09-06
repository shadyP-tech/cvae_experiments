"""Reconstruct scientific components and verify the frozen fit/calibration split."""
from collections.abc import Mapping
from ...protocol import ProtocolError
from .hashing import canonical_hash
from .calibration_split import proposer_calibration_partition


def _model_hash(payload, name):
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"HARP v21 {name} payload is absent.")
    body = dict(payload)
    advertised = body.pop("model_hash", None)
    if advertised != canonical_hash(body):
        raise ProtocolError(f"HARP v21 {name} model hash drifted.")
    return advertised


def _keys(payload, name):
    return tuple(tuple(key) for key in payload.get(name, ()))


def verify_complete_model_payload(payload: Mapping) -> str:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "harp_v21_complete_nested_winner_learner":
        raise ProtocolError("HARP v21 requires the complete frozen winner model schema.")
    from .patch_evidence import PatchEvidenceModel
    from .outcome_model import ActionOutcomeModel
    from .winner_gate import WinnerGateModel
    from .risk_selection import POLICY_RULE, selection_contract
    gate, proposer = payload.get("winner_gate"), payload.get("proposer")
    _model_hash(gate, "winner gate")
    _model_hash(proposer, "frozen proposer")
    _model_hash(payload.get("action_model"), "actual action")
    if (canonical_hash(payload.get("action_model")) != canonical_hash(proposer.get("action_model"))
        or canonical_hash(payload.get("proposal_model")) != canonical_hash(proposer.get("proposal_model"))):
        raise ProtocolError("HARP v21 duplicate proposer components drifted.")
    patch = payload["action_model"].get("patch_evidence_model")
    _model_hash(patch, "patch evidence")
    restored = PatchEvidenceModel.from_payload(patch)
    ActionOutcomeModel.from_payload(payload["action_model"])
    gate_model = WinnerGateModel.from_payload(gate)
    keys = _keys(payload, "training_case_keys")
    if len(keys) != len(set(keys)) or tuple(sorted(keys)) != keys:
        raise ProtocolError("HARP v21 complete model fitting scope drifted.")
    fitting, calibration = proposer_calibration_partition(keys)
    if (restored.training_case_keys != fitting or gate_model.training_case_keys != calibration
        or _keys(proposer, "training_case_keys") != fitting
        or _keys(payload["proposal_model"], "training_case_keys") != fitting
        or _keys(payload, "proposer_fit_case_keys") != fitting
        or _keys(payload, "calibration_case_keys") != calibration
        or _keys(gate_model.fit_audit, "normalization_scope_case_keys") != keys
        or restored.variant != payload.get("evidence_variant")):
        raise ProtocolError("HARP v21 complete model fit/calibration partitions drifted.")
    receipts = payload.get("winner_fit_receipts", ())
    if len(receipts) != 1:
        raise ProtocolError("HARP v21 requires one frozen proposer calibration receipt.")
    receipt = receipts[0]
    if (_keys(receipt, "training_case_keys") != fitting or _keys(receipt, "held_case_keys") != calibration
        or receipt.get("proposer_hash") != proposer.get("model_hash")
        or receipt.get("proposer_refitted_after_calibration") is not False
        or payload.get("proposer_refitted_after_calibration") is not False):
        raise ProtocolError("HARP v21 calibrated proposer identity drifted.")
    seals = gate_model.fit_audit.get("all_winner_seals", ())
    if (tuple(sorted(tuple(row["case_key"]) for row in seals)) != calibration
        or tuple(row["winner_seal_hash"] for row in seals) != tuple(receipt.get("pretruth_winner_seal_hashes", ()))
        or any(row["proposer_hash"] != proposer["model_hash"] or _keys(row, "training_case_keys") != fitting for row in seals)):
        raise ProtocolError("HARP v21 calibration seals do not bind the deployed proposer.")
    for seal in seals:
        body = dict(seal)
        if body.pop("winner_seal_hash", None) != canonical_hash(body):
            raise ProtocolError("HARP v21 pretruth calibration winner seal drifted.")
    if (canonical_hash(payload.get("risk_selection")) != canonical_hash(selection_contract())
        or payload.get("policy_rule") != POLICY_RULE or payload.get("runnerup_after_gate_veto") is not False):
        raise ProtocolError("HARP v21 complete winner decision rule drifted.")
    return _model_hash(payload, "complete learner")


__all__ = ("verify_complete_model_payload",)
