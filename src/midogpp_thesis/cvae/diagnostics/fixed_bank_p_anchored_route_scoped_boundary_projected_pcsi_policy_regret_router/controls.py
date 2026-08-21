"""Frozen, sibling-owned PCSI-RACR control specifications."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...protocol import ProtocolError
from .constants import (
    PRIMARY_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    PROJECTED_NO_ENVELOPE_METHOD_ID,
    RAW_OBSERVED_MAX_METHOD_ID,
    UNPROJECTED_GEOMETRY_ID,
)
from .hashing import canonical_hash


@dataclass(frozen=True, order=True)
class ControlSpec:
    policy_id: str
    geometry_id: str
    uses_blocked_fingerprint: bool
    requires_positive_bacc_prediction: bool
    uses_policy_regret: bool
    control_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "control_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_control_spec_v1",
            "policy_id": self.policy_id,
            "geometry_id": self.geometry_id,
            "uses_blocked_fingerprint": self.uses_blocked_fingerprint,
            "requires_positive_bacc_prediction": self.requires_positive_bacc_prediction,
            "uses_policy_regret": self.uses_policy_regret,
            "predecessor_artifact_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "control_hash": self.control_hash}


CONTROL_SPECS = (
    ControlSpec(PRIMARY_METHOD_ID, PROJECTION_GEOMETRY_ID, False, False, True),
    ControlSpec(
        PROJECTED_NO_ENVELOPE_METHOD_ID,
        PROJECTION_GEOMETRY_ID,
        False,
        False,
        False,
    ),
    ControlSpec(
        RAW_OBSERVED_MAX_METHOD_ID,
        UNPROJECTED_GEOMETRY_ID,
        False,
        False,
        True,
    ),
)


def control_spec(policy_id: str) -> ControlSpec:
    try:
        return next(row for row in CONTROL_SPECS if row.policy_id == policy_id)
    except StopIteration as exc:
        raise ProtocolError("PCSI-RACR requested an unknown control.") from exc


__all__ = ("CONTROL_SPECS", "ControlSpec", "control_spec")
