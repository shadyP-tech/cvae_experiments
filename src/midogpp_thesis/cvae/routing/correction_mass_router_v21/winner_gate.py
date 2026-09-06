"""Six-feature harm calibration of one unchanged, independently fitted selector."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import numpy as np
from scipy.special import expit
from ...protocol import ProtocolError
from .candidate_prediction import unthresholded_winner
from .calibration_estimator import fit_harm_logistic
from .hashing import canonical_hash, require_sha256
from .winner_records import winner_features


@dataclass(frozen=True, slots=True)
class WinnerGatePrediction:
    composite_hash: str
    safe_probability: float
    harm_probability: float
    remaining_probability: float
    model_hash: str
    feature_names: tuple[str, ...] = ()
    feature_values: tuple[float, ...] = ()
    calibration_available: bool = True
    prediction_hash: str = field(init=False)

    def __post_init__(self):
        require_sha256(self.composite_hash, name="winner gate composite hash")
        require_sha256(self.model_hash, name="winner gate model hash")
        if (len(self.feature_names) != len(self.feature_values)
            or len(self.feature_names) != len(set(self.feature_names))
            or any(not isinstance(name, str) or not name for name in self.feature_names)
            or any(not np.isfinite(v) for v in self.feature_values)
            or type(self.calibration_available) is not bool):
            raise ProtocolError("HARP v21 winner gate feature transcript is malformed.")
        values = self.safe_probability, self.harm_probability, self.remaining_probability
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in values) or abs(sum(values)-1) > 1e-8:
            raise ProtocolError("HARP v21 winner gate probabilities are incoherent.")
        object.__setattr__(self, "prediction_hash", canonical_hash(self._payload()))

    @property
    def route_score(self):
        return 1.0-self.harm_probability

    def _payload(self):
        return {"composite_hash": self.composite_hash,
            "safe_probability": self.safe_probability, "harm_probability": self.harm_probability,
            "remaining_probability": self.remaining_probability, "route_score": self.route_score,
            "safe_positive_probability_estimated": False,
            "model_hash": self.model_hash, "score_definition": "ONE_MINUS_WINNER_HARM_PROBABILITY",
            "feature_names": self.feature_names, "feature_values": self.feature_values,
            "calibration_available": self.calibration_available,
            "feature_values_are_raw_before_standardization": True}

    def public_payload(self):
        return {**self._payload(), "prediction_hash": self.prediction_hash}


@dataclass(frozen=True, slots=True)
class WinnerGateModel:
    feature_names: tuple
    means: tuple
    scales: tuple
    coefficients: tuple
    training_case_keys: tuple
    participating_case_keys: tuple
    fit_audit: dict
    model_hash: str = field(init=False)

    def __post_init__(self):
        d = len(self.feature_names)
        if (not self.training_case_keys or len(self.training_case_keys) != len(set(self.training_case_keys))
            or len(self.participating_case_keys) != len(set(self.participating_case_keys))
            or not set(self.participating_case_keys).issubset(self.training_case_keys)
            or len(self.feature_names) != len(set(self.feature_names)) or d not in (0, 6)
            or len(self.means) != d or len(self.scales) != d or len(self.coefficients) != d+1
            or any(not np.isfinite(v) for v in (*self.means, *self.coefficients))
            or any(not np.isfinite(v) or v <= 0 for v in self.scales)):
            raise ProtocolError("HARP v21 winner gate parameters or fitting scope are malformed.")
        object.__setattr__(self, "model_hash", canonical_hash(self._payload()))

    def predict(self, menu, candidates):
        if (menu.center_id, menu.case_id) in self.training_case_keys:
            raise ProtocolError("HARP v21 winner gate prediction includes a calibration fitting case.")
        winner = unthresholded_winner(candidates)
        if winner is None:
            return None
        features = dict(winner_features(menu, candidates))
        raw = np.asarray([features[name] for name in self.feature_names])
        available = bool(self.participating_case_keys)
        matrix = np.concatenate(([1.0], (raw-self.means)/self.scales))
        harm = float(expit(matrix @ self.coefficients)) if available else 1.0
        return WinnerGatePrediction(winner.candidate.composite.composite_hash,
            0.0, harm, 1.0-harm, self.model_hash, self.feature_names, tuple(map(float, raw)), available)

    def _payload(self):
        return {"schema_version": "harp_v21_frozen_selector_binary_harm_gate",
            "feature_names": self.feature_names, "means": self.means, "scales": self.scales,
            "coefficients": self.coefficients, "training_case_keys": self.training_case_keys,
            "participating_case_keys": self.participating_case_keys, "fit_audit": self.fit_audit,
            "response": "SELECTED_ACTION_NEGATIVE_BACC_GAIN", "safe_positive_probability_estimated": False,
            "full_population_weights_conditioned_on_participation": True,
            "in_sample_score_is_not_confidence_bound": True}

    def public_payload(self):
        return {**self._payload(), "model_hash": self.model_hash}

    @classmethod
    def from_payload(cls, payload):
        model = cls(tuple(payload["feature_names"]), tuple(payload["means"]), tuple(payload["scales"]),
            tuple(payload["coefficients"]), tuple(tuple(k) for k in payload["training_case_keys"]),
            tuple(tuple(k) for k in payload["participating_case_keys"]), payload["fit_audit"])
        if model.model_hash != payload.get("model_hash"):
            raise ProtocolError("HARP v21 winner gate reconstruction drifted.")
        return model


def fit_winner_gate(seals, outcomes, *, training_case_keys, population_case_keys=None, ridge_alpha=.01):
    """Fit on calibration cases; retain full fitting-population center weights."""
    training = tuple(training_case_keys)
    population = training if population_case_keys is None else tuple(population_case_keys)
    if (len(population) != len(set(population)) or not set(training).issubset(population)
        or tuple(sorted(row.case_key for row in seals)) != tuple(sorted(training))):
        raise ProtocolError("HARP v21 winner training must contain one sealed record per calibration case.")
    participants = tuple(row for row in seals if row.winner is not None)
    outcomes_by_hash = {row.composite.composite_hash: row for row in outcomes}
    if (len(outcomes_by_hash) != len(outcomes)
        or set(outcomes_by_hash) != {row.winner.candidate.composite.composite_hash for row in participants}):
        raise ProtocolError("HARP v21 winner outcome join is incomplete.")
    keys = tuple(row.case_key for row in participants)
    names = tuple(sorted({name for row in participants for name, _ in row.features}))
    centers = Counter(center for center, _ in population)
    records = []
    retained_mass = 0.0
    if participants:
        weights = np.asarray([1.0/len(centers)/centers[center] for center, _ in keys])
        retained_mass = float(weights.sum())
        weights /= retained_mass
        raw = np.asarray([[dict(row.features)[name] for name in names] for row in participants])
        means = weights @ raw
        scales = np.sqrt(weights @ ((raw-means)**2))
        scales[scales < 1e-12] = 1.0
        matrix = np.column_stack((np.ones(len(raw)), (raw-means)/scales))
        targets = []
        for row, weight in zip(participants, weights, strict=True):
            outcome = outcomes_by_hash[row.winner.candidate.composite.composite_hash]
            target = int(outcome.harmed)
            targets.append(target)
            records.append({**row.public_payload(), "harm_target": target,
                "outcome": outcome.public_payload(), "fit_weight": float(weight)})
        coefficients = fit_harm_logistic(matrix, targets, weights, alpha=ridge_alpha)
    else:
        means, scales, coefficients = np.zeros(len(names)), np.ones(len(names)), np.zeros(len(names)+1)
    audit = {"records": tuple(records), "scope_case_count": len(seals),
        "participating_case_count": len(participants), "empty_menu_case_count": len(seals)-len(participants),
        "all_winner_seals": tuple(row.public_payload() for row in seals),
        "all_winners_sealed_before_any_winner_truth": True,
        "normalization_scope_case_keys": population,
        "full_population_case_weights_before_participation_filter": True,
        "retained_population_weight_mass": retained_mass,
        "ridge_alpha_mean_loss": ridge_alpha, "sample_weight_sum": 1.0 if participants else 0.0,
        "raw_labels_persisted": False}
    return WinnerGateModel(names, tuple(map(float, means)), tuple(map(float, scales)), tuple(map(float, coefficients)),
                           training, keys, audit)
