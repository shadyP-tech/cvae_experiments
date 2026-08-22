"""Thin public facade and explicit fail-closed calibration outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .hashing import canonical_hash

from .policy_calibration import PolicyReplay
from .utility_calibration import (
    CenterBalancedUtilityCalibration,
    UtilityReplay,
    build_center_balanced_utility_calibration,
)


@dataclass(frozen=True)
class UnsupportedCalibration:
    """Auditable non-estimate used when fewer than six donor centers support it."""

    calibration_kind: str
    outer_center: str
    excluded_centers: tuple[str, ...]
    supported_donor_centers: tuple[str, ...]
    reason_code: str = "FEWER_THAN_SIX_SUPPORTED_DONOR_CENTERS"
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = tuple(sorted(set(self.excluded_centers)))
        supported = tuple(sorted(set(self.supported_donor_centers)))
        if (
            self.calibration_kind not in {"candidate_utility", "policy_replay"}
            or self.outer_center not in excluded
            or any(center in excluded for center in supported)
            or len(supported) >= 6
        ):
            raise ProtocolError("CBPUPR unsupported calibration contract drifted.")
        object.__setattr__(self, "excluded_centers", excluded)
        object.__setattr__(self, "supported_donor_centers", supported)
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "fixed_bank_cbpupr_unsupported_calibration_v1",
                    "calibration_kind": self.calibration_kind,
                    "outer_center": self.outer_center,
                    "excluded_centers": list(excluded),
                    "supported_donor_centers": list(supported),
                    "reason_code": self.reason_code,
                    "bias_estimated": False,
                    "forces_exact_P": True,
                    "finite_sample_coverage_claimed": False,
                }
            ),
        )

    @property
    def supported(self) -> bool:
        return False

    def to_payload(self) -> dict[str, object]:
        return {
            "calibration_kind": self.calibration_kind,
            "outer_center": self.outer_center,
            "excluded_centers": list(self.excluded_centers),
            "supported_donor_centers": list(self.supported_donor_centers),
            "reason_code": self.reason_code,
            "bias_estimated": False,
            "forces_exact_P": True,
            "calibration_hash": self.calibration_hash,
        }


def supported_donor_centers(
    replays: Sequence[UtilityReplay], *, excluded_centers: Sequence[str]
) -> tuple[str, ...]:
    excluded = set(str(value) for value in excluded_centers)
    return tuple(
        sorted(
            {
                row.donor_center
                for row in replays
                if row.donor_center not in excluded
            }
        )
    )

__all__ = (
    "CenterBalancedUtilityCalibration",
    "PolicyReplay",
    "UnsupportedCalibration",
    "UtilityReplay",
    "build_center_balanced_utility_calibration",
    "supported_donor_centers",
)
