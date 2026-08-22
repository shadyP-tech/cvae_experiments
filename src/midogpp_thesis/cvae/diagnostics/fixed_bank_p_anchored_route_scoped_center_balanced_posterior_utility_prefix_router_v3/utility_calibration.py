"""Center-balanced conditional-overprediction calibration for case utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .canonical_probabilities import canonical_hash
from .posterior_expected_utility import FavorableUtility


MIN_SUPPORTED_DONOR_CENTER_COUNT = 6


@dataclass(frozen=True)
class UtilityReplay:
    """Sealed pseudo-case prediction and later-opened realised utility.

    ``outer_center`` is H and ``donor_center`` is the replayed J.  The sealed
    target-local posterior for case d is trained only on J-minus-d, so no H
    rows or labels enter its fit and d is excluded as a whole case.  Frozen,
    label-free A1::source=H fingerprint covariates remain present.  The H/J
    lineage wrapper records the narrower guarantee that neither H nor J can
    contribute endpoint-source, source-prior, or donor-calibration roles.  J is
    intentionally present in its own J-minus-d support posterior.
    """

    outer_center: str
    donor_center: str
    case_id: str
    candidate_hash: str
    predicted_utility: FavorableUtility
    realized_utility: FavorableUtility
    lineage_excluded_centers: tuple[str, ...]
    control_id: str
    replay_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = tuple(
            sorted(set(str(value) for value in self.lineage_excluded_centers))
        )
        if (
            not self.outer_center
            or not self.donor_center
            or self.outer_center == self.donor_center
            or not self.case_id
            or not self.candidate_hash
            or not self.control_id
            or self.outer_center not in excluded
            or self.donor_center not in excluded
        ):
            raise ProtocolError("CBPUPR utility replay violates H/J/d exclusion.")
        object.__setattr__(self, "lineage_excluded_centers", excluded)
        object.__setattr__(
            self,
            "replay_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_utility_replay_v1",
                    "outer_center": self.outer_center,
                    "donor_center": self.donor_center,
                    "case_id": self.case_id,
                    "candidate_hash": self.candidate_hash,
                    "predicted_utility": self.predicted_utility.to_payload(),
                    "realized_utility": self.realized_utility.to_payload(),
                    "lineage_excluded_centers": list(excluded),
                    "posterior_support_scope": "donor_center_minus_whole_case",
                    "control_id": self.control_id,
                }
            ),
        )

    @property
    def overprediction(self) -> FavorableUtility:
        residual = self.predicted_utility - self.realized_utility
        return FavorableUtility(*(max(value, 0.0) for value in residual.as_tuple()))

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "UtilityReplay":
        row = cls(
            str(payload["outer_center"]),
            str(payload["donor_center"]),
            str(payload["case_id"]),
            str(payload["candidate_hash"]),
            FavorableUtility.from_payload(payload["predicted_utility"]),  # type: ignore[arg-type]
            FavorableUtility.from_payload(payload["realized_utility"]),  # type: ignore[arg-type]
            tuple(
                str(value)
                for value in payload.get(
                    "lineage_excluded_centers",
                    payload.get("fit_excluded_centers", ()),
                )
            ),
            str(payload["control_id"]),
        )
        if "replay_hash" in payload and str(payload["replay_hash"]) != row.replay_hash:
            raise ProtocolError("CBPUPR utility replay hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_center": self.outer_center,
            "donor_center": self.donor_center,
            "case_id": self.case_id,
            "candidate_hash": self.candidate_hash,
            "predicted_utility": self.predicted_utility.to_payload(),
            "realized_utility": self.realized_utility.to_payload(),
            "overprediction": self.overprediction.to_payload(),
            "lineage_excluded_centers": list(self.lineage_excluded_centers),
            "posterior_support_scope": "donor_center_minus_whole_case",
            "control_id": self.control_id,
            "replay_hash": self.replay_hash,
        }


@dataclass(frozen=True)
class CenterBalancedUtilityCalibration:
    outer_center: str
    calibration_excluded_centers: tuple[str, ...]
    supported_donor_centers: tuple[str, ...]
    case_count_by_center: tuple[tuple[str, int], ...]
    mean_overprediction_by_center: tuple[tuple[str, FavorableUtility], ...]
    bias: FavorableUtility
    replay_hashes: tuple[str, ...]
    calibration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        excluded = tuple(sorted(set(self.calibration_excluded_centers)))
        donors = tuple(sorted(set(self.supported_donor_centers)))
        counts = tuple(sorted(self.case_count_by_center))
        center_rows = tuple(sorted(self.mean_overprediction_by_center, key=lambda row: row[0]))
        if (
            not self.outer_center
            or self.outer_center not in excluded
            or len(donors) < MIN_SUPPORTED_DONOR_CENTER_COUNT
            or any(center in excluded for center in donors)
            or tuple(center for center, _ in counts) != donors
            or tuple(center for center, _ in center_rows) != donors
            or any(count <= 0 for _, count in counts)
            or not self.replay_hashes
            or len(set(self.replay_hashes)) != len(self.replay_hashes)
        ):
            raise ProtocolError("CBPUPR center-balanced calibration support drifted.")
        object.__setattr__(self, "calibration_excluded_centers", excluded)
        object.__setattr__(self, "supported_donor_centers", donors)
        object.__setattr__(self, "case_count_by_center", counts)
        object.__setattr__(self, "mean_overprediction_by_center", center_rows)
        object.__setattr__(
            self,
            "calibration_hash",
            canonical_hash(
                {
                    "schema_version": "cbpupr_center_balanced_utility_calibration_v1",
                    "outer_center": self.outer_center,
                    "calibration_excluded_centers": list(excluded),
                    "supported_donor_centers": list(donors),
                    "case_count_by_center": [list(row) for row in counts],
                    "mean_overprediction_by_center": [
                        [center, utility.to_payload()] for center, utility in center_rows
                    ],
                    "bias": self.bias.to_payload(),
                    "replay_hashes": list(self.replay_hashes),
                    "finite_sample_coverage_claimed": False,
                }
            ),
        )

    def correct(self, utility: FavorableUtility) -> FavorableUtility:
        return utility - self.bias

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "CenterBalancedUtilityCalibration":
        row = cls(
            str(payload["outer_center"]),
            tuple(str(value) for value in payload["calibration_excluded_centers"]),  # type: ignore[index]
            tuple(str(value) for value in payload["supported_donor_centers"]),  # type: ignore[index]
            tuple(
                (str(value[0]), int(value[1]))
                for value in payload["case_count_by_center"]  # type: ignore[index]
            ),
            tuple(
                (str(value[0]), FavorableUtility.from_payload(value[1]))
                for value in payload["mean_overprediction_by_center"]  # type: ignore[index]
            ),
            FavorableUtility.from_payload(payload["bias"]),  # type: ignore[arg-type]
            tuple(str(value) for value in payload["replay_hashes"]),  # type: ignore[index]
        )
        if "calibration_hash" in payload and str(payload["calibration_hash"]) != row.calibration_hash:
            raise ProtocolError("CBPUPR utility calibration hash drifted.")
        return row

    def to_payload(self) -> dict[str, object]:
        return {
            "outer_center": self.outer_center,
            "calibration_excluded_centers": list(self.calibration_excluded_centers),
            "supported_donor_centers": list(self.supported_donor_centers),
            "case_count_by_center": [list(row) for row in self.case_count_by_center],
            "mean_overprediction_by_center": [
                [center, utility.to_payload()]
                for center, utility in self.mean_overprediction_by_center
            ],
            "bias": self.bias.to_payload(),
            "replay_hashes": list(self.replay_hashes),
            "calibration_hash": self.calibration_hash,
            "finite_sample_coverage_claimed": False,
        }


def build_center_balanced_utility_calibration(
    replays: Sequence[UtilityReplay],
    *,
    outer_center: str,
    calibration_excluded_centers: Sequence[str] | None = None,
    minimum_supported_centers: int = MIN_SUPPORTED_DONOR_CENTER_COUNT,
) -> CenterBalancedUtilityCalibration:
    """Median of donor-center mean positive overprediction biases.

    For a target route the excluded set is ``{H}``.  For a pseudo-J policy it
    is ``{H, J}``, implementing the predeclared leave-J reconstruction.
    """

    excluded = tuple(
        sorted(
            set(
                str(value)
                for value in (
                    (outer_center,)
                    if calibration_excluded_centers is None
                    else calibration_excluded_centers
                )
            )
        )
    )
    if str(outer_center) not in excluded:
        raise ProtocolError("CBPUPR utility calibration must exclude outer H.")
    rows = tuple(replays)
    if not rows:
        raise ProtocolError("CBPUPR utility calibration replay set is empty.")
    if any(row.outer_center != str(outer_center) for row in rows):
        raise ProtocolError("CBPUPR utility calibration mixed outer centers.")
    if any(row.donor_center in excluded for row in rows):
        raise ProtocolError("CBPUPR leave-J calibration consumed an excluded donor.")
    if (
        len({(row.donor_center, row.case_id) for row in rows}) != len(rows)
        or len({row.control_id for row in rows}) != 1
    ):
        raise ProtocolError("CBPUPR utility calibration mixed controls or cases.")

    grouped: dict[str, list[UtilityReplay]] = {}
    for row in rows:
        grouped.setdefault(row.donor_center, []).append(row)
    donors = tuple(sorted(grouped))
    if len(donors) < max(int(minimum_supported_centers), MIN_SUPPORTED_DONOR_CENTER_COUNT):
        raise ProtocolError("CBPUPR calibration has fewer than six donor centers.")

    center_means: list[tuple[str, FavorableUtility]] = []
    counts: list[tuple[str, int]] = []
    for center in donors:
        matrix = np.asarray(
            [row.overprediction.as_tuple() for row in grouped[center]], dtype=np.float64
        )
        values = np.mean(matrix, axis=0, dtype=np.float64)
        center_means.append(
            (center, FavorableUtility(*(float(value) for value in values)))
        )
        counts.append((center, len(grouped[center])))
    bias_values = np.median(
        np.asarray([value.as_tuple() for _, value in center_means], dtype=np.float64),
        axis=0,
    )
    return CenterBalancedUtilityCalibration(
        str(outer_center),
        excluded,
        donors,
        tuple(counts),
        tuple(center_means),
        FavorableUtility(*(float(value) for value in bias_values)),
        tuple(sorted(row.replay_hash for row in rows)),
    )


__all__ = (
    "CenterBalancedUtilityCalibration",
    "MIN_SUPPORTED_DONOR_CENTER_COUNT",
    "UtilityReplay",
    "build_center_balanced_utility_calibration",
)
