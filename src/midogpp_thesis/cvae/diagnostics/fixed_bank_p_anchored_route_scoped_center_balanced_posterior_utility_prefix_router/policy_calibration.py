"""Diagnostic holdout summaries for complete-case pseudo policies.

The primary router has exactly one calibration estimand, implemented in
``utility_calibration``.  PolicyReplay rows assess that calibration after a
leave-J pseudo policy has been selected; their residual summary is diagnostic
only and must never be applied as a second authorization bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ...protocol import ProtocolError
from .canonical_probabilities import canonical_hash
from .posterior_expected_utility import FavorableUtility


@dataclass(frozen=True)
class PolicyReplay:
    outer_center: str
    donor_center: str
    selected_k: int
    selected_candidate_hashes: tuple[str, ...]
    predicted_utility: FavorableUtility
    realized_utility: FavorableUtility
    calibration_excluded_centers: tuple[str, ...]
    policy_hash: str
    replay_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = tuple(sorted(set(self.calibration_excluded_centers)))
        if (
            not self.outer_center
            or not self.donor_center
            or self.outer_center == self.donor_center
            or self.selected_k < 0
            or self.selected_k != len(self.selected_candidate_hashes)
            or len(set(self.selected_candidate_hashes))
            != len(self.selected_candidate_hashes)
            or self.outer_center not in excluded
            or self.donor_center not in excluded
            or not self.policy_hash
        ):
            raise ProtocolError("CBPUPR policy replay violates H/J reconstruction.")
        object.__setattr__(self, "calibration_excluded_centers", excluded)
        object.__setattr__(
            self,
            "replay_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_policy_replay_v1",
                    "outer_center": self.outer_center,
                    "donor_center": self.donor_center,
                    "selected_k": self.selected_k,
                    "selected_candidate_hashes": list(self.selected_candidate_hashes),
                    "predicted_utility": self.predicted_utility.to_payload(),
                    "realized_utility": self.realized_utility.to_payload(),
                    "calibration_excluded_centers": list(excluded),
                    "policy_hash": self.policy_hash,
                }
            ),
        )

    @property
    def overprediction(self) -> FavorableUtility:
        residual = self.predicted_utility - self.realized_utility
        return FavorableUtility(*(max(value, 0.0) for value in residual.as_tuple()))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PolicyReplay":
        row = cls(
            str(payload["outer_center"]),
            str(payload["donor_center"]),
            int(payload["selected_k"]),
            tuple(str(value) for value in payload["selected_candidate_hashes"]),  # type: ignore[index]
            FavorableUtility.from_payload(payload["predicted_utility"]),  # type: ignore[arg-type]
            FavorableUtility.from_payload(payload["realized_utility"]),  # type: ignore[arg-type]
            tuple(str(value) for value in payload["calibration_excluded_centers"]),  # type: ignore[index]
            str(payload["policy_hash"]),
        )
        if "replay_hash" in payload and str(payload["replay_hash"]) != row.replay_hash:
            raise ProtocolError("CBPUPR policy replay hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_center": self.outer_center,
            "donor_center": self.donor_center,
            "selected_k": self.selected_k,
            "selected_candidate_hashes": list(self.selected_candidate_hashes),
            "predicted_utility": self.predicted_utility.to_payload(),
            "realized_utility": self.realized_utility.to_payload(),
            "overprediction": self.overprediction.to_payload(),
            "calibration_excluded_centers": list(self.calibration_excluded_centers),
            "policy_hash": self.policy_hash,
            "replay_hash": self.replay_hash,
        }

__all__ = ("PolicyReplay",)
