"""Immutable package products for orchestration and persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from .constants import CENTERS, FEATURE_NAMES, METHOD_IDS, SEED_PAIR_COUNT, a1_action_id, candidate_sources
from .hashing import canonical_hash, finite, nonempty_text, require_sha256, require_stable_hash


@dataclass(frozen=True, order=True)
class SeedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    seed_pair_ordinal: int
    probability: float
    probability_store_hash: str

    def __post_init__(self) -> None:
        if self.target_center not in CENTERS or self.seed_pair_ordinal not in range(SEED_PAIR_COUNT):
            raise ProtocolError("Flip-router seed probability identity drifted.")
        nonempty_text(self.case_id, "case_id"); nonempty_text(self.sample_id, "sample_id")
        value = finite(self.probability, "probability")
        if not 0.0 <= value <= 1.0:
            raise ProtocolError("Flip-router probability lies outside [0,1].")
        require_stable_hash(self.probability_store_hash, "probability_store_hash")
        object.__setattr__(self, "probability", value)

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class AggregatedProbabilityRow:
    target_center: str
    case_id: str
    sample_id: str
    action_id: str
    probability_mean: float
    probability_sd: float
    seed_pair_count: int
    seed_probabilities: tuple[float, ...]
    row_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        probabilities = tuple(finite(value, "seed_probability") for value in self.seed_probabilities)
        if (
            self.target_center not in CENTERS
            or self.seed_pair_count != SEED_PAIR_COUNT
            or len(probabilities) != SEED_PAIR_COUNT
            or not 0.0 <= self.probability_mean <= 1.0
            or self.probability_sd < 0.0
        ):
            raise ProtocolError("Flip-router exact-nine aggregate drifted.")
        object.__setattr__(self, "seed_probabilities", probabilities)
        object.__setattr__(self, "row_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.target_center, self.case_id, self.sample_id, self.action_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "target_center": self.target_center, "case_id": self.case_id,
            "sample_id": self.sample_id, "action_id": self.action_id,
            "probability_mean": self.probability_mean, "probability_sd": self.probability_sd,
            "seed_pair_count": self.seed_pair_count,
            "seed_probabilities": list(self.seed_probabilities),
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "row_hash": self.row_hash}


@dataclass(frozen=True)
class ExactNineProbabilitySurface:
    rows: tuple[AggregatedProbabilityRow, ...]
    probability_store_hash: str
    surface_hash: str

    def __post_init__(self) -> None:
        require_stable_hash(self.probability_store_hash, "probability_store_hash")
        require_sha256(self.surface_hash, "surface_hash")
        if len({row.key for row in self.rows}) != len(self.rows):
            raise ProtocolError("Flip-router probability surface contains duplicates.")
        expected = canonical_hash({
            "schema_version": "fixed_bank_flip_router_exact_nine_surface_v1",
            "probability_store_hash": self.probability_store_hash,
            "rows": [row.to_payload() for row in self.rows],
            "predictions_sealed_before_labels": True,
        })
        if self.surface_hash != expected:
            raise ProtocolError("Flip-router probability surface hash drifted.")


@dataclass(frozen=True, order=True)
class CaseActionFeature:
    target_center: str
    case_id: str
    action_id: str
    selected_source: str
    values: tuple[float, ...]
    feature_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        values = tuple(finite(value, "flip feature") for value in self.values)
        if self.target_center not in CENTERS or len(values) != len(FEATURE_NAMES):
            raise ProtocolError("Flip-router case-action feature drifted.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.action_id

    def _unhashed(self) -> dict[str, object]:
        return {
            "target_center": self.target_center, "case_id": self.case_id,
            "action_id": self.action_id, "selected_source": self.selected_source,
            "feature_names": list(FEATURE_NAMES), "values": list(self.values),
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "feature_hash": self.feature_hash}


@dataclass(frozen=True)
class PrelabelSurface:
    features: tuple[CaseActionFeature, ...]
    probability_surface_hash: str
    prediction_seal_hash: str
    feature_surface_hash: str

    def __post_init__(self) -> None:
        require_sha256(self.probability_surface_hash, "probability_surface_hash")
        require_stable_hash(self.prediction_seal_hash, "prediction_seal_hash")
        require_sha256(self.feature_surface_hash, "feature_surface_hash")
        if len({row.key for row in self.features}) != len(self.features):
            raise ProtocolError("Flip-router prelabel feature keys are duplicated.")
        expected = canonical_hash({
            "schema_version": "fixed_bank_flip_router_prelabel_surface_v1",
            "probability_surface_hash": self.probability_surface_hash,
            "prediction_seal_hash": self.prediction_seal_hash,
            "features": [row.to_payload() for row in self.features],
            "labels_used": False,
        })
        if self.feature_surface_hash != expected:
            raise ProtocolError("Flip-router feature surface hash drifted.")


@dataclass(frozen=True, order=True)
class ContributionTarget:
    target_center: str
    case_id: str
    action_id: str
    selected_source: str
    tp_delta: int
    tn_delta: int
    n_positive: int
    n_negative: int

    def __post_init__(self) -> None:
        if (
            self.target_center not in CENTERS
            or self.n_positive < 0 or self.n_negative < 0
            or abs(self.tp_delta) > self.n_positive
            or abs(self.tn_delta) > self.n_negative
        ):
            raise ProtocolError("Flip-router contribution target drifted.")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.action_id

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True, order=True)
class MethodDecision:
    target_center: str
    fold_ordinal: int
    case_id: str
    method_id: str
    action_id: str
    challenger_action_id: str
    predicted_gain: float
    gain_standard_error: float
    lower_confidence_bound: float
    decision_source: str
    evaluation_labels_used: bool = False

    def __post_init__(self) -> None:
        legal_actions = {"B", "U", *(a1_action_id(source) for source in candidate_sources(self.target_center))}
        if (
            self.target_center not in CENTERS
            or self.method_id not in METHOD_IDS[:-2]
            or self.evaluation_labels_used is not False
            or not self.case_id
            or self.action_id not in legal_actions
            or self.challenger_action_id not in legal_actions
            or (
                self.method_id in {"F_G", "F_S", "F_P"}
                and self.action_id not in {"B", self.challenger_action_id}
            )
            or (
                self.method_id in {"G_static", "S_static"}
                and self.action_id != self.challenger_action_id
            )
        ):
            raise ProtocolError("Flip-router method decision drifted.")
        for role in ("predicted_gain", "gain_standard_error", "lower_confidence_bound"):
            finite(getattr(self, role), role)

    def to_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class DecisionBundle:
    decisions: tuple[MethodDecision, ...]
    fold_seal_hashes: Mapping[tuple[str, int], str]
    decision_bundle_hash: str

    def __post_init__(self) -> None:
        hashes = dict(self.fold_seal_hashes)
        expected_keys = {(center, fold) for center in CENTERS for fold in range(5)}
        if set(hashes) != expected_keys or any(not require_sha256(value, "fold seal") for value in hashes.values()):
            raise ProtocolError("Flip-router fold decision seal coverage drifted.")
        expected = canonical_hash({
            "schema_version": "fixed_bank_flip_router_decision_bundle_v1",
            "decisions": [row.to_payload() for row in self.decisions],
            "fold_seals": {f"{key[0]}::{key[1]}": value for key, value in sorted(hashes.items())},
            "evaluation_labels_used": False,
        })
        if self.decision_bundle_hash != expected:
            raise ProtocolError("Flip-router decision bundle hash drifted.")
        object.__setattr__(self, "fold_seal_hashes", MappingProxyType(hashes))


__all__ = (
    "AggregatedProbabilityRow", "CaseActionFeature", "ContributionTarget", "DecisionBundle",
    "ExactNineProbabilitySurface", "MethodDecision", "PrelabelSurface", "SeedProbabilityRow",
)
