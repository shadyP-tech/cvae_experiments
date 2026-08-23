"""Canonical pseudo-response evidence DTO for P-DCAPS v3 admission."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ...protocol import ProtocolError
from ..fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    FavorableUtility,
)
from .identity import canonical_hash, require_sha256
from .nullable_statistics import finite_float, strict_bool


PSEUDO_EVIDENCE_SCHEMA = "pdcaps_v3_pseudo_policy_evidence_v1"


@dataclass(frozen=True)
class PseudoPolicyEvidence:
    """Label-free, pseudo-response-only evidence for one donor center."""

    outer_center: str
    donor_center: str
    predicted: FavorableUtility
    realized: FavorableUtility
    routed: bool
    jointly_safe: bool
    endpoint_oracle_bacc_gain: float
    absolute_oracle_regret: float
    legacy_realized: FavorableUtility
    legacy_routed: bool
    legacy_jointly_safe: bool
    legacy_absolute_oracle_regret: float
    evidence_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        donor = str(self.donor_center)
        if (
            not outer
            or not donor
            or outer == donor
            or not isinstance(self.predicted, FavorableUtility)
            or not isinstance(self.realized, FavorableUtility)
            or not isinstance(self.legacy_realized, FavorableUtility)
        ):
            raise ProtocolError("P-DCAPS v3 pseudo admission evidence drifted.")
        routed = strict_bool(self.routed, "pseudo routed flag")
        safe = strict_bool(self.jointly_safe, "pseudo joint-safety flag")
        legacy_routed = strict_bool(
            self.legacy_routed, "legacy pseudo routed flag"
        )
        legacy_safe = strict_bool(
            self.legacy_jointly_safe, "legacy pseudo joint-safety flag"
        )
        endpoint = finite_float(
            self.endpoint_oracle_bacc_gain, "endpoint oracle BACC gain"
        )
        regret = finite_float(self.absolute_oracle_regret, "absolute regret")
        legacy_regret = finite_float(
            self.legacy_absolute_oracle_regret, "legacy absolute regret"
        )
        if endpoint < 0.0 or regret < 0.0 or legacy_regret < 0.0:
            raise ProtocolError("P-DCAPS v3 pseudo admission evidence drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(self, "donor_center", donor)
        object.__setattr__(self, "routed", routed)
        object.__setattr__(self, "jointly_safe", safe)
        object.__setattr__(self, "legacy_routed", legacy_routed)
        object.__setattr__(self, "legacy_jointly_safe", legacy_safe)
        object.__setattr__(self, "endpoint_oracle_bacc_gain", endpoint)
        object.__setattr__(self, "absolute_oracle_regret", regret)
        object.__setattr__(
            self, "legacy_absolute_oracle_regret", legacy_regret
        )
        object.__setattr__(
            self,
            "evidence_hash",
            canonical_hash(
                {
                    "schema_version": PSEUDO_EVIDENCE_SCHEMA,
                    **self._science_payload(),
                    "target_labels_used": False,
                }
            ),
        )

    def _science_payload(self) -> dict[str, object]:
        return {
            "outer_center": self.outer_center,
            "donor_center": self.donor_center,
            "predicted": self.predicted.to_payload(),
            "realized": self.realized.to_payload(),
            "routed": self.routed,
            "jointly_safe": self.jointly_safe,
            "endpoint_oracle_bacc_gain": self.endpoint_oracle_bacc_gain,
            "absolute_oracle_regret": self.absolute_oracle_regret,
            "legacy_realized": self.legacy_realized.to_payload(),
            "legacy_routed": self.legacy_routed,
            "legacy_jointly_safe": self.legacy_jointly_safe,
            "legacy_absolute_oracle_regret": self.legacy_absolute_oracle_regret,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": PSEUDO_EVIDENCE_SCHEMA,
            **self._science_payload(),
            "target_labels_used": False,
            "evidence_hash": self.evidence_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PseudoPolicyEvidence":
        expected = {
            "schema_version",
            "outer_center",
            "donor_center",
            "predicted",
            "realized",
            "routed",
            "jointly_safe",
            "endpoint_oracle_bacc_gain",
            "absolute_oracle_regret",
            "legacy_realized",
            "legacy_routed",
            "legacy_jointly_safe",
            "legacy_absolute_oracle_regret",
            "target_labels_used",
            "evidence_hash",
        }
        if (
            not isinstance(payload, Mapping)
            or set(payload) != expected
            or payload.get("schema_version") != PSEUDO_EVIDENCE_SCHEMA
            or payload.get("target_labels_used") is not False
        ):
            raise ProtocolError("P-DCAPS v3 pseudo evidence schema drifted.")

        def utility(role: str) -> FavorableUtility:
            row = payload[role]
            if not isinstance(row, Mapping) or set(row) != {
                "bacc_gain",
                "brier_gain",
                "log_gain",
            }:
                raise ProtocolError(
                    "P-DCAPS v3 pseudo evidence utility schema drifted."
                )
            return FavorableUtility(
                finite_float(row["bacc_gain"], f"{role} BACC gain"),
                finite_float(row["brier_gain"], f"{role} Brier gain"),
                finite_float(row["log_gain"], f"{role} log gain"),
            )

        result = cls(
            str(payload["outer_center"]),
            str(payload["donor_center"]),
            utility("predicted"),
            utility("realized"),
            strict_bool(payload["routed"], "pseudo routed flag"),
            strict_bool(payload["jointly_safe"], "pseudo joint-safety flag"),
            finite_float(
                payload["endpoint_oracle_bacc_gain"],
                "endpoint oracle BACC gain",
            ),
            finite_float(payload["absolute_oracle_regret"], "absolute regret"),
            utility("legacy_realized"),
            strict_bool(payload["legacy_routed"], "legacy pseudo routed flag"),
            strict_bool(
                payload["legacy_jointly_safe"],
                "legacy pseudo joint-safety flag",
            ),
            finite_float(
                payload["legacy_absolute_oracle_regret"],
                "legacy absolute regret",
            ),
        )
        if result.evidence_hash != require_sha256(
            payload["evidence_hash"], "persisted pseudo evidence hash"
        ):
            raise ProtocolError("P-DCAPS v3 pseudo evidence hash drifted.")
        return result


__all__ = ("PSEUDO_EVIDENCE_SCHEMA", "PseudoPolicyEvidence")
