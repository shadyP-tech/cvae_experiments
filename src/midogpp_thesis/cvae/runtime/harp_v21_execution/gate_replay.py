"""Independent binary-gate replay without fitting, unpickling, or labels."""
from collections.abc import Mapping

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash


def replay_gate_prediction(payload: Mapping, gate_model: Mapping) -> None:
    """Authenticate the exact input and independently evaluate frozen coefficients."""
    transcript = payload.get("winner_gate_prediction_payload")
    if not isinstance(transcript, Mapping):
        raise ProtocolError("HARP v21 winner evidence lacks its gate prediction transcript.")
    body = dict(transcript)
    digest = body.pop("prediction_hash", None)
    if (digest != canonical_hash(body) or digest != payload["winner_gate_prediction_hash"]
        or body.get("model_hash") != gate_model["model_hash"]
        or body.get("composite_hash") != payload["winner_composite_hash"]
        or body.get("score_definition") != "ONE_MINUS_WINNER_HARM_PROBABILITY"
        or body.get("feature_values_are_raw_before_standardization") is not True):
        raise ProtocolError("HARP v21 gate prediction transcript binding drifted.")
    try:
        names = tuple(body["feature_names"])
        raw = np.asarray(body["feature_values"], dtype=float)
        observed = np.asarray([body[name] for name in
            ("safe_probability", "harm_probability", "remaining_probability")], dtype=float)
        available = bool(gate_model["participating_case_keys"])
        if (names != tuple(gate_model["feature_names"]) or raw.shape != (len(names),)
            or len(set(names)) != len(names) or not np.isfinite(raw).all()
            or not np.isfinite(observed).all() or np.any(observed < 0)
            or np.any(observed > 1) or abs(float(observed.sum())-1) > 1e-10
            or body.get("calibration_available") is not available):
            raise ValueError("feature, probability, or calibration schema")
        if available:
            means = np.asarray(gate_model["means"], dtype=float)
            scales = np.asarray(gate_model["scales"], dtype=float)
            coefficients = np.asarray(gate_model["coefficients"], dtype=float)
            if (means.shape != raw.shape or scales.shape != raw.shape
                or coefficients.shape != (len(names)+1,)
                or not all(np.isfinite(v).all() for v in (means, scales, coefficients))
                or np.any(scales <= 0)):
                raise ValueError("binary gate model dimensions")
            logit = float(coefficients @ np.concatenate(([1.0], (raw-means)/scales)))
            if not np.isfinite(logit):
                raise ValueError("nonfinite gate logit")
            # Stable scalar sigmoid; avoid depending on the scientific predictor.
            q = 1.0/(1.0+np.exp(-logit)) if logit >= 0 else np.exp(logit)/(1.0+np.exp(logit))
            expected = np.asarray((0., q, 1.-q))
            if ("predicted_gain" not in names
                or raw[names.index("predicted_gain")] != payload["winner_risk_adjusted_score"]):
                raise ValueError("winner gain feature")
        else:
            expected = np.asarray((0.0, 1.0, 0.0))
        if (not np.allclose(observed, expected, rtol=1e-10, atol=1e-12)
            or body["harm_probability"] != payload["winner_gate_harm_probability"]
            or abs(float(body["route_score"])-(1-observed[1])) > 1e-10):
            raise ValueError("gate prediction differs from coefficient replay")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ProtocolError("HARP v21 sealed winner gate prediction failed independent replay.") from exc
