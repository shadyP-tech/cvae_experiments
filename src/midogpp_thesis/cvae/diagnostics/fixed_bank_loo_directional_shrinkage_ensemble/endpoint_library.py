"""Frozen nine-arm DCSE and matched-G endpoint libraries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...protocol import ProtocolError
from .constants import ARM_IDS, K_GRID, W_RATIONAL_GRID
from .hashing import canonical_hash
from .products import EndpointArm


@dataclass(frozen=True)
class EndpointLibrary:
    method_id: str
    arms: tuple[EndpointArm, ...]
    support_score_source: str
    library_hash: str = field(init=False)

    def __post_init__(self) -> None:
        arms = tuple(self.arms)
        if self.method_id not in {"DCSE_LOO", "G_directional_matched"}:
            raise ProtocolError("DCSE endpoint library method identity drifted.")
        expected_source = "H_minus_c_directional_gain" if self.method_id == "DCSE_LOO" else "G_directional_prior"
        if self.support_score_source != expected_source:
            raise ProtocolError("DCSE endpoint library support-score source drifted.")
        if tuple(arm.arm_id for arm in arms) != ARM_IDS or len({arm.arm_hash for arm in arms}) != 9:
            raise ProtocolError("DCSE endpoint library must preserve the exact nine distinct arms.")
        object.__setattr__(self, "arms", arms)
        object.__setattr__(self, "library_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_dcse_endpoint_library_v1",
            "method_id": self.method_id,
            "arms": [arm.to_payload() for arm in self.arms],
            "support_score_source": self.support_score_source,
            "candidate_prescreen": "rank_G_then_retain_top_K",
            "OFF_score": [0, 1],
            "hidden_arm_selection": False,
            "duplicates_preserved_across_arms": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "library_hash": self.library_hash}


def build_endpoint_arms() -> tuple[EndpointArm, ...]:
    return tuple(
        EndpointArm(k, numerator, denominator)
        for k in K_GRID
        for numerator, denominator in W_RATIONAL_GRID
    )


def build_endpoint_library(method_id: str = "DCSE_LOO") -> EndpointLibrary:
    method = str(method_id)
    return EndpointLibrary(
        method_id=method,
        arms=build_endpoint_arms(),
        support_score_source=(
            "H_minus_c_directional_gain"
            if method == "DCSE_LOO"
            else "G_directional_prior"
        ),
    )


def build_matched_endpoint_libraries() -> tuple[EndpointLibrary, EndpointLibrary]:
    return build_endpoint_library("DCSE_LOO"), build_endpoint_library("G_directional_matched")


def validate_arm_subset(arms: Sequence[EndpointArm]) -> tuple[EndpointArm, ...]:
    values = tuple(arms)
    if not values or any(arm.arm_id not in ARM_IDS for arm in values) or len({arm.arm_id for arm in values}) != len(values):
        raise ProtocolError("DCSE arm subset is empty, duplicated, or outside the frozen library.")
    return values


__all__ = (
    "EndpointLibrary",
    "build_endpoint_arms",
    "build_endpoint_library",
    "build_matched_endpoint_libraries",
    "validate_arm_subset",
)
