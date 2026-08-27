"""Neutral label-free evidence contracts shared by SCEPTRE development modules."""

from __future__ import annotations

from dataclasses import dataclass
import math

from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.protocol import ProtocolError


@dataclass(frozen=True, slots=True)
class EvidenceFeatureRow:
    """Label-free evidence for one ordered query/candidate source family."""

    query_center: str
    candidate_center: str
    feature_names: tuple[str, ...]
    values: tuple[float, ...]
    labels_consumed: bool = False
    feature_scope: str = "LABEL_FREE_PREDECISION_EVIDENCE"

    def __post_init__(self) -> None:
        if self.query_center not in CENTERS or self.candidate_center not in CENTERS:
            raise ProtocolError("SCEPTRE evidence row has an unknown center.")
        if self.query_center == self.candidate_center:
            raise ProtocolError("SCEPTRE evidence row contains forbidden q == e.")
        if not self.feature_names or len(self.feature_names) != len(self.values):
            raise ProtocolError("SCEPTRE evidence feature geometry is invalid.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ProtocolError("SCEPTRE evidence feature names are not unique.")
        if any(not name for name in self.feature_names):
            raise ProtocolError("SCEPTRE evidence feature name is empty.")
        forbidden_tokens = (
            "label",
            "truth",
            "target_y",
            "bacc",
            "f1",
            "utility",
            "outcome",
            "realized",
            "confusion",
            "delta_tp",
            "delta_tn",
        )
        normalized_names = tuple(name.lower() for name in self.feature_names)
        if any(
            token in name for name in normalized_names for token in forbidden_tokens
        ):
            raise ProtocolError("SCEPTRE evidence feature schema exposes label utility.")
        if self.labels_consumed is not False:
            raise ProtocolError("SCEPTRE evidence features must remain label-free.")
        if self.feature_scope != "LABEL_FREE_PREDECISION_EVIDENCE":
            raise ProtocolError("SCEPTRE evidence feature scope drifted.")
        if any(not math.isfinite(float(value)) for value in self.values):
            raise ProtocolError("SCEPTRE evidence feature is non-finite.")

    @property
    def key(self) -> tuple[str, str]:
        return self.query_center, self.candidate_center


__all__ = ("EvidenceFeatureRow",)
