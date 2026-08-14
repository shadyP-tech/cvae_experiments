"""Persistable label-free method probabilities sealed before terminal labels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

from ...protocol import ProtocolError
from .constants import CENTERS, HARD_THRESHOLD, PRE_TERMINAL_METHOD_IDS, candidate_sources
from .hashing import canonical_hash, require_sha256


@dataclass(frozen=True, order=True)
class MethodPrediction:
    target_center: str
    case_id: str
    sample_id: str
    method_id: str
    probability: float
    hard_prediction: int
    baseline_hard_prediction: int
    endpoint_identity: str
    selected_source: str | None
    selected_sources_by_arm: tuple[str | None, ...]
    reason: str
    prediction_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        probability = float(self.probability)
        hard = int(self.hard_prediction)
        baseline = int(self.baseline_hard_prediction)
        selected = None if self.selected_source is None else str(self.selected_source)
        arm_sources = tuple(None if value is None else str(value) for value in self.selected_sources_by_arm)
        legal = candidate_sources(self.target_center) if self.target_center in CENTERS else ()
        if (
            self.target_center not in CENTERS
            or not self.case_id
            or not self.sample_id
            or self.method_id not in PRE_TERMINAL_METHOD_IDS
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or hard not in (0, 1)
            or baseline not in (0, 1)
            or hard != int(probability >= HARD_THRESHOLD)
            or not self.endpoint_identity
            or selected not in (None, *legal)
            or any(source not in (None, *legal) for source in arm_sources)
            or not self.reason
        ):
            raise ProtocolError("OGDE method prediction drifted.")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "hard_prediction", hard)
        object.__setattr__(self, "baseline_hard_prediction", baseline)
        object.__setattr__(self, "selected_source", selected)
        object.__setattr__(self, "selected_sources_by_arm", arm_sources)
        object.__setattr__(self, "prediction_hash", canonical_hash(self._unhashed()))

    @property
    def key(self) -> tuple[str, str, str]:
        return self.target_center, self.case_id, self.sample_id

    @property
    def method_key(self) -> tuple[str, str, str, str]:
        return self.method_id, *self.key

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_ogde_method_prediction_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "sample_id": self.sample_id,
            "method_id": self.method_id,
            "probability": self.probability,
            "hard_prediction": self.hard_prediction,
            "baseline_hard_prediction": self.baseline_hard_prediction,
            "endpoint_identity": self.endpoint_identity,
            "selected_source": self.selected_source,
            "selected_sources_by_arm": list(self.selected_sources_by_arm),
            "reason": self.reason,
            "terminal_labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "prediction_hash": self.prediction_hash}

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "MethodPrediction":
        row = cls(
            str(payload["target_center"]), str(payload["case_id"]), str(payload["sample_id"]),
            str(payload["method_id"]), float(payload["probability"]), int(payload["hard_prediction"]),
            int(payload["baseline_hard_prediction"]), str(payload["endpoint_identity"]),
            None if payload["selected_source"] is None else str(payload["selected_source"]),
            tuple(None if value is None else str(value) for value in payload["selected_sources_by_arm"]),  # type: ignore[union-attr]
            str(payload["reason"]),
        )
        if require_sha256(payload["prediction_hash"], "prediction_hash") != row.prediction_hash:
            raise ProtocolError("OGDE method prediction hash drifted after reload.")
        return row


__all__ = ("MethodPrediction",)
