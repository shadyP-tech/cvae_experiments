"""Nested pseudo-safe policy calibration with symmetric fit reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ....expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ....protocol import ProtocolError
from ..identity import METRICS, canonical_hash
from .contracts import PrefixSurface
from .envelope import PolicyEnvelope, PolicyOOFResidual, build_policy_envelope
from .ridge import PolicyCalibration, fit_policy_calibration
from .runtime import (
    NestedPolicyCalibration,
    fit_nested_policy_calibrations,
    observations_from_surfaces,
)


@dataclass(frozen=True)
class PolicyCalibrationFamilies:
    outer_center: str
    target: NestedPolicyCalibration
    pseudo_calibrations_by_center: tuple[tuple[str, PolicyCalibration], ...]
    pseudo_envelopes_by_center: tuple[tuple[str, PolicyEnvelope], ...]
    numerical_metric_fit_count: int
    serialized_model_count: int
    plan_hash: str = field(init=False)

    def __post_init__(self) -> None:
        donors = tuple(center for center in CENTERS if center != self.outer_center)
        calibration_centers = tuple(
            center for center, _row in self.pseudo_calibrations_by_center
        )
        envelope_centers = tuple(
            center for center, _row in self.pseudo_envelopes_by_center
        )
        if (
            self.target.outer_center != self.outer_center
            or calibration_centers != donors
            or envelope_centers != donors
            or any(
                calibration.scored_center != center
                for center, calibration in self.pseudo_calibrations_by_center
            )
            or any(
                envelope.excluded_scored_center != center
                for center, envelope in self.pseudo_envelopes_by_center
            )
            or int(self.numerical_metric_fit_count)
            != len(METRICS)
            * (
                1
                + len(donors)
                + len(donors) * (len(donors) - 1) // 2
            )
            or int(self.serialized_model_count)
            != len(METRICS)
            * (1 + len(donors) + len(donors) * (len(donors) - 1))
        ):
            raise ProtocolError("P-DCAPS optimized policy calibration plan drifted.")
        object.__setattr__(
            self,
            "plan_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_policy_calibration_families_v1",
                    "outer_center": self.outer_center,
                    "target_nested_hash": self.target.nested_hash,
                    "pseudo_calibration_hashes": tuple(
                        (center, row.calibration_hash)
                        for center, row in self.pseudo_calibrations_by_center
                    ),
                    "pseudo_envelope_hashes": tuple(
                        (center, row.envelope_hash)
                        for center, row in self.pseudo_envelopes_by_center
                    ),
                    "numerical_metric_fit_count": self.numerical_metric_fit_count,
                    "serialized_model_count": self.serialized_model_count,
                    "unordered_exclusion_pair_fit_reuse": True,
                }
            ),
        )

    @property
    def pseudo_calibrations(self) -> Mapping[str, PolicyCalibration]:
        return dict(self.pseudo_calibrations_by_center)

    @property
    def pseudo_envelopes(self) -> Mapping[str, PolicyEnvelope]:
        return dict(self.pseudo_envelopes_by_center)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_policy_calibration_families_v1",
            "outer_center": self.outer_center,
            "target": self.target.to_payload(),
            "pseudo_calibrations_by_center": [
                [center, row.to_payload()]
                for center, row in self.pseudo_calibrations_by_center
            ],
            "pseudo_envelopes_by_center": [
                [center, row.to_payload()]
                for center, row in self.pseudo_envelopes_by_center
            ],
            "numerical_metric_fit_count": self.numerical_metric_fit_count,
            "serialized_model_count": self.serialized_model_count,
            "unordered_exclusion_pair_fit_reuse": True,
            "plan_hash": self.plan_hash,
        }


def _rebind_policy_scored_center(
    calibration: PolicyCalibration,
    *,
    scored_center: str,
) -> PolicyCalibration:
    scored = str(scored_center)
    if scored not in calibration.excluded_centers or scored == calibration.outer_center:
        raise ProtocolError("P-DCAPS policy model rebind exclusion drifted.")
    additional = tuple(
        center
        for center in calibration.excluded_centers
        if center not in {calibration.outer_center, scored}
    )
    return PolicyCalibration(
        calibration.outer_center,
        scored,
        calibration.excluded_centers,
        calibration.supported_centers,
        calibration.models,
        calibration.observation_hashes,
        calibration.observation_weights,
        additional,
    )


def build_optimized_policy_calibration_families(
    response_surfaces: Sequence[PrefixSurface],
    *,
    outer_center: str,
) -> PolicyCalibrationFamilies:
    """Fit target and pseudo-safe H/J/K policy calibrations and envelopes."""

    surfaces = tuple(response_surfaces)
    outer = str(outer_center)
    if outer not in CENTERS:
        raise ProtocolError("P-DCAPS optimized policy outer center drifted.")
    donors = tuple(center for center in CENTERS if center != outer)
    if tuple(surface.provenance.route_center for surface in surfaces) != donors:
        raise ProtocolError("P-DCAPS optimized policy donor inventory drifted.")
    observations = observations_from_surfaces(surfaces)
    target = fit_nested_policy_calibrations(surfaces, outer_center=outer)
    single = {str(row.scored_center): row for row in target.oof_calibrations}

    oriented: dict[tuple[str, str], PolicyCalibration] = {}
    for left_index, left in enumerate(donors):
        for right in donors[left_index + 1 :]:
            right_scored = fit_policy_calibration(
                observations,
                outer_center=outer,
                scored_center=right,
                additional_excluded_centers=(left,),
            )
            oriented[(left, right)] = right_scored
            oriented[(right, left)] = _rebind_policy_scored_center(
                right_scored, scored_center=left
            )

    envelopes: list[tuple[str, PolicyEnvelope]] = []
    for context in donors:
        residuals: list[PolicyOOFResidual] = []
        for scored in donors:
            if scored == context:
                continue
            model = oriented[(context, scored)]
            for row in observations:
                if row.center != scored:
                    continue
                residuals.append(
                    PolicyOOFResidual(
                        outer,
                        scored,
                        row.route_hash,
                        row.cell.cell_hash,
                        model.predict(row.cell),
                        row.cell.realized_utility,  # type: ignore[arg-type]
                        model.calibration_hash,
                        model.excluded_centers,
                    )
                )
        envelopes.append(
            (
                context,
                build_policy_envelope(
                    residuals,
                    outer_center=outer,
                    excluded_scored_center=context,
                ),
            )
        )
    numerical = len(METRICS) * (
        1 + len(donors) + len(donors) * (len(donors) - 1) // 2
    )
    serialized = len(METRICS) * (
        1 + len(donors) + len(donors) * (len(donors) - 1)
    )
    return PolicyCalibrationFamilies(
        outer,
        target,
        tuple((center, single[center]) for center in donors),
        tuple(envelopes),
        numerical,
        serialized,
    )


__all__ = (
    "PolicyCalibrationFamilies",
    "build_optimized_policy_calibration_families",
)
