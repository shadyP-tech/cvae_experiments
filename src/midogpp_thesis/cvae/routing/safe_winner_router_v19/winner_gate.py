"""One case-weighted coherent gate fitted on held-out unthresholded winners."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import numpy as np

from ...protocol import ProtocolError
from .candidate_prediction import unthresholded_winner
from .estimators import fit_softmax_ridge, predict_softmax
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
    prediction_hash: str = field(init=False)

    def __post_init__(self):
        require_sha256(self.composite_hash, name="winner gate composite hash")
        require_sha256(self.model_hash, name="winner gate model hash")
        if (len(self.feature_names) != len(self.feature_values)
            or len(self.feature_names) != len(set(self.feature_names))
            or any(not isinstance(name, str) or not name for name in self.feature_names)
            or any(not np.isfinite(value) for value in self.feature_values)):
            raise ProtocolError("HARP v19 winner gate raw feature transcript is malformed.")
        values = (self.safe_probability, self.harm_probability, self.remaining_probability)
        if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values) or abs(sum(values) - 1) > 1e-8:
            raise ProtocolError("HARP v19 winner gate probabilities are incoherent.")
        object.__setattr__(self, "prediction_hash", canonical_hash(self._payload()))

    @property
    def route_score(self):
        return 1.0 - self.harm_probability

    def _payload(self):
        return {"composite_hash": self.composite_hash,
            "safe_probability": self.safe_probability, "harm_probability": self.harm_probability,
            "remaining_probability": self.remaining_probability, "route_score": self.route_score,
            "model_hash": self.model_hash, "score_definition": "ONE_MINUS_WINNER_HARM_PROBABILITY",
            "feature_names": self.feature_names, "feature_values": self.feature_values,
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
        dimension = len(self.feature_names)
        if (not self.training_case_keys
            or len(self.training_case_keys) != len(set(self.training_case_keys))
            or len(self.participating_case_keys) != len(set(self.participating_case_keys))
            or not set(self.participating_case_keys).issubset(self.training_case_keys)
            or len(self.feature_names) != len(set(self.feature_names))
            or len(self.means) != dimension or len(self.scales) != dimension
            or len(self.coefficients) != 3
            or any(len(row) != dimension+1 for row in self.coefficients)
            or any(not np.isfinite(value) for row in self.coefficients for value in row)
            or any(not np.isfinite(value) for value in self.means)
            or any(not np.isfinite(value) or value <= 0 for value in self.scales)):
            raise ProtocolError("HARP v19 winner gate parameters or fitting scope are malformed.")
        object.__setattr__(self, "model_hash", canonical_hash(self._payload()))

    def predict(self, menu, candidates):
        winner = unthresholded_winner(candidates)
        if winner is None:
            return None
        if not self.participating_case_keys:
            raw = np.asarray([], dtype=float)
            values = (0.0, 1.0, 0.0)
        else:
            features = dict(winner_features(menu, candidates))
            raw = np.asarray([features.get(name, 0.0) for name in self.feature_names])
            matrix = np.concatenate(([1.0], (raw - self.means) / self.scales))[None, :]
            values = tuple(float(value) for value in predict_softmax(matrix, np.asarray(self.coefficients))[0])
        return WinnerGatePrediction(winner.candidate.composite.composite_hash, *values, self.model_hash,
                                    self.feature_names, tuple(map(float, raw)))

    def _payload(self):
        return {"schema_version": "harp_v19_complete_winner_gate",
            "feature_names": self.feature_names, "means": self.means, "scales": self.scales,
            "coefficients": self.coefficients, "training_case_keys": self.training_case_keys,
            "participating_case_keys": self.participating_case_keys, "fit_audit": self.fit_audit,
            "categories": ("SAFE_POSITIVE", "HARMFUL", "REMAINING"),
            "one_center_balanced_weight_per_participating_case": True,
            "negative_score_and_harmful_winners_retained": True,
            "in_sample_score_is_not_confidence_bound": True}

    def public_payload(self):
        return {**self._payload(), "model_hash": self.model_hash}


def fit_winner_gate(seals, outcomes, *, training_case_keys, ridge_alpha=.01):
    outcomes_by_hash = {row.composite.composite_hash: row for row in outcomes}
    participants = tuple(row for row in seals if row.winner is not None)
    names = tuple(sorted({name for row in participants for name, _ in row.features}))
    keys = tuple(row.case_key for row in participants)
    if len(keys) != len(set(keys)) or tuple(sorted(row.case_key for row in seals)) != tuple(sorted(training_case_keys)):
        raise ProtocolError("HARP v19 winner training must contain one sealed record per scope case.")
    if set(outcomes_by_hash) != {row.winner.candidate.composite.composite_hash for row in participants}:
        raise ProtocolError("HARP v19 winner outcome join is incomplete.")
    records = []
    if participants:
        centers = Counter(center for center, _ in keys)
        weights = np.asarray([1.0 / len(centers) / centers[center] for center, _ in keys])
        raw = np.asarray([[dict(row.features).get(name, 0.0) for name in names] for row in participants])
        means = np.sum(raw * weights[:, None], axis=0)
        scales = np.sqrt(np.sum((raw - means) ** 2 * weights[:, None], axis=0))
        scales[scales < 1e-12] = 1.0
        matrix = np.column_stack((np.ones(len(raw)), (raw - means) / scales))
        targets = []
        for row, weight in zip(participants, weights, strict=True):
            outcome = outcomes_by_hash[row.winner.candidate.composite.composite_hash]
            target = 0 if outcome.safe_positive else 1 if outcome.harmed else 2
            targets.append(target)
            records.append({**row.public_payload(), "category": target,
                            "outcome": outcome.public_payload(), "fit_weight": float(weight)})
        coefficients = fit_softmax_ridge(matrix, np.asarray(targets), weights, alpha=ridge_alpha)
    else:
        means, scales, coefficients = np.zeros(len(names)), np.ones(len(names)), np.zeros((3, len(names) + 1))
    audit = {"records": tuple(records), "scope_case_count": len(seals),
        "participating_case_count": len(participants), "empty_menu_case_count": len(seals) - len(participants),
        "all_winner_seals": tuple(row.public_payload() for row in seals),
        "all_winners_sealed_before_any_winner_truth": True,
        "normalization_scope_case_keys": tuple(training_case_keys),
        "empty_menu_cases_retained_in_outcome_normalizer": True,
        "ridge_alpha_mean_loss": ridge_alpha, "sample_weight_sum": 1.0 if participants else 0.0,
        "raw_labels_persisted": False}
    return WinnerGateModel(names, tuple(map(float, means)), tuple(map(float, scales)), tuple(tuple(map(float, row)) for row in coefficients),
                           tuple(training_case_keys), keys, audit)
