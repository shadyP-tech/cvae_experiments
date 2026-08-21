"""Observed donor-case regret envelopes for route-scoped decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    CENTERS,
    PROJECTION_GEOMETRY_ID,
    UNPROJECTED_GEOMETRY_ID,
)
from .case_regret import PseudoCaseReplay
from .hashing import canonical_hash, require_sha256


@dataclass(frozen=True, order=True)
class DonorCaseEnvelope:
    geometry_id: str
    outer_center: str
    donor_center: str
    case_ids: tuple[str, ...]
    coordinate_maxima: tuple[float, float, float]
    argmax_case_ids: tuple[str, str, str]
    upper_median: tuple[float, float, float]
    replay_hashes: tuple[str, ...]
    envelope_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        cases = tuple(self.case_ids)
        maxima = tuple(float(value) for value in self.coordinate_maxima)
        medians = tuple(float(value) for value in self.upper_median)
        if (
            self.geometry_id not in {PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID}
            or self.outer_center not in CENTERS
            or self.donor_center not in CENTERS
            or self.outer_center == self.donor_center
            or not cases
            or len(cases) != len(set(cases))
            or len(maxima) != 3
            or len(medians) != 3
            or any(not math.isfinite(value) or value < 0.0 for value in (*maxima, *medians))
            or any(case not in cases for case in self.argmax_case_ids)
            or len(self.replay_hashes) != len(cases)
        ):
            raise ProtocolError("PCSI-RACR donor envelope drifted.")
        for digest in self.replay_hashes:
            require_sha256(digest, "envelope_replay_hash")
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "coordinate_maxima", maxima)
        object.__setattr__(self, "upper_median", medians)
        object.__setattr__(self, "envelope_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_observed_donor_case_envelope_v1",
            "geometry_id": self.geometry_id,
            "outer_center": self.outer_center,
            "donor_center": self.donor_center,
            "case_ids": list(self.case_ids),
            "coordinate_maxima": list(self.coordinate_maxima),
            "argmax_case_ids": list(self.argmax_case_ids),
            "upper_median_descriptive": list(self.upper_median),
            "replay_hashes": list(self.replay_hashes),
            "method": "OBSERVED_DONOR_CASE_ENVELOPE",
            "conformal": False,
            "finite_sample_coverage": False,
            "tail_probability_claimed": False,
            "dependent_leave_one_case_replays": True,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "envelope_hash": self.envelope_hash}


@dataclass(frozen=True, order=True)
class RouteCalibration:
    geometry_id: str
    outer_center: str
    donor_envelopes: tuple[DonorCaseEnvelope, ...]
    margin: tuple[float, float, float]
    valid: bool
    invalid_reasons: tuple[str, ...]
    calibration_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        envelopes = tuple(self.donor_envelopes)
        margin = tuple(float(value) for value in self.margin)
        expected = tuple(center for center in CENTERS if center != self.outer_center)
        observed_donors = tuple(row.donor_center for row in envelopes)
        if (
            self.geometry_id not in {PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID}
            or self.outer_center not in CENTERS
            or (
                self.valid
                and observed_donors != expected
            )
            or (
                not self.valid
                and any(donor not in expected for donor in observed_donors)
            )
            or any(
                row.outer_center != self.outer_center
                or row.geometry_id != self.geometry_id
                for row in envelopes
            )
            or len(margin) != 3
            or any(not math.isfinite(value) or value < 0.0 for value in margin)
            or bool(self.valid) == bool(self.invalid_reasons)
        ):
            raise ProtocolError("PCSI-RACR route calibration drifted.")
        object.__setattr__(self, "donor_envelopes", envelopes)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(self, "calibration_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_route_calibration_v1",
            "geometry_id": self.geometry_id,
            "outer_center": self.outer_center,
            "envelope_hashes": [row.envelope_hash for row in self.donor_envelopes],
            "margin": list(self.margin),
            "valid": self.valid,
            "invalid_reasons": list(self.invalid_reasons),
            "interpretation": (
                "coordinatewise_maximum_positive_overprediction_observed_in_"
                "finite_dependent_consumed_test_pseudo_replays"
            ),
            "NON_GUARANTEE_CONSUMED_TEST_ONLY": True,
            "confidence_bound": False,
            "calibrated_uncertainty": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "calibration_hash": self.calibration_hash}


@dataclass(frozen=True, order=True)
class DescriptorMatchedAnnotation:
    geometry_id: str
    outer_center: str
    target_case_id: str
    matched_case_ids: tuple[tuple[str, str], ...]
    margin: tuple[float, float, float]
    match_table_hash: str
    annotation_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        expected = tuple(center for center in CENTERS if center != self.outer_center)
        if (
            self.geometry_id not in {PROJECTION_GEOMETRY_ID, UNPROJECTED_GEOMETRY_ID}
            or self.outer_center not in CENTERS
            or not self.target_case_id
            or tuple(center for center, _case in self.matched_case_ids) != expected
            or len(self.margin) != 3
            or any(not math.isfinite(value) or value < 0.0 for value in self.margin)
        ):
            raise ProtocolError("PCSI-RACR descriptor-match annotation drifted.")
        require_sha256(self.match_table_hash, "descriptor_match_table_hash")
        object.__setattr__(self, "annotation_hash", canonical_hash(self._unhashed()))

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_racr_descriptor_matched_annotation_v1",
            "geometry_id": self.geometry_id,
            "outer_center": self.outer_center,
            "target_case_id": self.target_case_id,
            "matched_case_ids": [list(row) for row in self.matched_case_ids],
            "margin": list(self.margin),
            "match_table_hash": self.match_table_hash,
            "unscored_annotation_only": True,
            "similarity_implies_utility": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._unhashed(), "annotation_hash": self.annotation_hash}


def build_route_calibration(
    replays: Sequence[PseudoCaseReplay],
    *,
    geometry_id: str,
    outer_center: str,
    expected_case_ids_by_center: Mapping[str, Sequence[str]],
) -> RouteCalibration:
    rows = tuple(replays)
    invalid: list[str] = []
    envelopes: list[DonorCaseEnvelope] = []
    for donor in CENTERS:
        if donor == outer_center:
            continue
        expected_cases = tuple(sorted(str(value) for value in expected_case_ids_by_center[donor]))
        donor_rows = tuple(
            sorted(
                (
                    row
                    for row in rows
                    if row.geometry_id == geometry_id
                    and row.route.outer_center == outer_center
                    and row.route.donor_center == donor
                ),
                key=lambda row: (row.route.case_id, row.replay_hash),
            )
        )
        if tuple(row.route.case_id for row in donor_rows) != expected_cases:
            invalid.append(f"INCOMPLETE_DONOR_CASES::{donor}")
            continue
        residual = np.asarray(
            [row.overprediction_residual for row in donor_rows], dtype=np.float64
        )
        maxima = []
        argmax = []
        upper = []
        for coordinate in range(3):
            values = residual[:, coordinate]
            index = min(
                range(len(values)),
                key=lambda row_index: (-float(values[row_index]), donor_rows[row_index].route.case_id),
            )
            maxima.append(max(0.0, float(values[index])))
            argmax.append(donor_rows[index].route.case_id)
            ordered = np.sort(values, kind="mergesort")
            upper_index = len(ordered) // 2
            upper.append(max(0.0, float(ordered[upper_index])))
        envelopes.append(
            DonorCaseEnvelope(
                geometry_id,
                outer_center,
                donor,
                expected_cases,
                tuple(maxima),
                tuple(argmax),
                tuple(upper),
                tuple(row.replay_hash for row in donor_rows),
            )
        )
    if invalid:
        # Keep the topology explicit while preventing any incomplete observed
        # set from changing a target route.
        return RouteCalibration(
            geometry_id,
            outer_center,
            tuple(envelopes),
            (0.0, 0.0, 0.0),
            False,
            tuple(invalid),
        )
    margin = tuple(
        max(row.coordinate_maxima[coordinate] for row in envelopes)
        for coordinate in range(3)
    )
    return RouteCalibration(
        geometry_id,
        outer_center,
        tuple(envelopes),
        margin,
        True,
        (),
    )


__all__ = (
    "DescriptorMatchedAnnotation",
    "DonorCaseEnvelope",
    "RouteCalibration",
    "build_route_calibration",
)
