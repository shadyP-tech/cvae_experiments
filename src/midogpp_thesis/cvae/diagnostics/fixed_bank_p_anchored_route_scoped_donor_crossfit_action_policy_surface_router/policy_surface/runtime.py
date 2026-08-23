"""Complete prefix construction and nested policy-surface calibration runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from ....protocol import ProtocolError
from ..contracts import FavorableUtility
from ..identity import ACTION_STRATA, canonical_hash
from .contracts import (
    PolicyAction,
    PolicyObservation,
    PolicySurfaceProvenance,
    PrefixCell,
    PrefixSurface,
    require_complete_responses,
)
from .envelope import (
    PolicyEnvelope,
    PolicyOOFResidual,
    apply_policy_envelope,
    build_policy_envelope,
)
from .ridge import PolicyCalibration, fit_policy_calibration
from .selection import CalibratedPrefixCell, PolicySelection, select_policy_prefix


@dataclass(frozen=True)
class NestedPolicyCalibration:
    outer_center: str
    final_calibration: PolicyCalibration
    oof_calibrations: tuple[PolicyCalibration, ...]
    oof_residuals: tuple[PolicyOOFResidual, ...]
    envelope: PolicyEnvelope
    nested_hash: str = field(init=False)

    def __post_init__(self) -> None:
        outer = str(self.outer_center)
        scored = tuple(row.scored_center for row in self.oof_calibrations)
        residual_centers = tuple(sorted({row.scored_center for row in self.oof_residuals}))
        if (
            self.final_calibration.outer_center != outer
            or self.final_calibration.scored_center is not None
            or self.final_calibration.excluded_centers != (outer,)
            or not scored
            or any(value is None for value in scored)
            or tuple(sorted(str(value) for value in scored)) != residual_centers
            or any(row.outer_center != outer for row in self.oof_calibrations)
            or any(row.outer_center != outer for row in self.oof_residuals)
            or self.envelope.outer_center != outer
        ):
            raise ProtocolError("P-DCAPS nested policy calibration drifted.")
        object.__setattr__(self, "outer_center", outer)
        object.__setattr__(
            self,
            "nested_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_nested_policy_calibration_v1",
                    "outer_center": outer,
                    "final_calibration_hash": self.final_calibration.calibration_hash,
                    "oof_calibration_hashes": tuple(
                        row.calibration_hash for row in self.oof_calibrations
                    ),
                    "oof_residual_hashes": tuple(
                        row.residual_hash for row in self.oof_residuals
                    ),
                    "envelope_hash": self.envelope.envelope_hash,
                    "nested_leave_scored_center": True,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_nested_policy_calibration_v1",
            "outer_center": self.outer_center,
            "final_calibration": self.final_calibration.to_payload(),
            "oof_calibrations": [row.to_payload() for row in self.oof_calibrations],
            "oof_residuals": [row.to_payload() for row in self.oof_residuals],
            "envelope": self.envelope.to_payload(),
            "nested_leave_scored_center": True,
            "nested_hash": self.nested_hash,
        }


def policy_action_from_selection(selection: object) -> PolicyAction | None:
    """Adapt the action layer's immutable selection without importing it.

    Exact-P case decisions intentionally disappear from the nonzero prefix
    candidate list.  The adapter is duck-typed so the policy layer owns no
    action-layer DTO or persistence dependency.
    """

    exact_p = getattr(selection, "exact_p_fallback", None)
    selected_key = getattr(selection, "selected_action_key", None)
    utility = getattr(selection, "selected_utility", None)
    selection_hash = getattr(selection, "selection_hash", None)
    if exact_p is True:
        if selected_key is not None or utility != FavorableUtility.zeros():
            raise ProtocolError("P-DCAPS exact-P action selection drifted.")
        return None
    if (
        exact_p is not False
        or selected_key is None
        or not isinstance(utility, FavorableUtility)
        or not isinstance(selection_hash, str)
    ):
        raise ProtocolError("P-DCAPS policy action-selection adapter drifted.")
    route_key = getattr(selected_key, "route_key", None)
    if not hasattr(route_key, "held_case_id"):
        raise ProtocolError("P-DCAPS selected action lacks route lineage.")
    return PolicyAction(
        route_key,
        str(route_key.held_case_id),
        str(getattr(selected_key, "action_key_hash", "")),
        str(getattr(selected_key, "family", "")),
        str(getattr(selected_key, "direction", "")),
        utility,
        selection_hash,
    )


def build_prefix_surface(
    actions: Sequence[PolicyAction],
    *,
    surface_role: str,
    outer_center: str,
    route_center: str,
    action_surface_seal_hash: str,
) -> PrefixSurface:
    """Seal all k=0..n prefix descriptors without reading responses."""

    rows = tuple(actions)
    role = str(surface_role)
    outer = str(outer_center)
    route = str(route_center)
    if len({row.case_id for row in rows}) != len(rows) or len(
        {row.action_hash for row in rows}
    ) != len(rows):
        raise ProtocolError("P-DCAPS policy surface repeats a case or action.")
    expected_scored = None if role == "target" else route
    if any(
        row.route_key.surface_role != role
        or row.route_key.outer_center != outer
        or row.route_key.route_center != route
        or row.route_key.excluded_outer_center != outer
        or row.route_key.excluded_scored_center != expected_scored
        for row in rows
    ):
        raise ProtocolError("P-DCAPS policy action H/J provenance drifted.")
    ranked = tuple(
        sorted(
            rows,
            key=lambda row: (
                -row.predicted_utility.bacc_gain,
                row.case_id,
                row.action_hash,
            ),
        )
    )
    provenance = PolicySurfaceProvenance(
        role,
        outer,
        route,
        outer,
        expected_scored,
        action_surface_seal_hash,
        tuple(row.route_key.exclusion_hash for row in ranked),
        tuple(row.route_key.fit_scope_hash for row in ranked),
    )
    cells: list[PrefixCell] = []
    for k in range(len(ranked) + 1):
        selected = ranked[:k]
        predicted = FavorableUtility.zeros()
        for row in selected:
            predicted = predicted + row.predicted_utility
        positive = np.asarray(
            [max(row.predicted_utility.bacc_gain, 0.0) for row in selected],
            dtype=np.float64,
        )
        max_share = (
            0.0
            if positive.size == 0 or float(positive.sum()) <= 0.0
            else float(positive.max() / positive.sum())
        )
        proportions = tuple(
            0.0
            if not selected
            else sum(row.stratum == stratum for row in selected) / len(selected)
            for stratum in ACTION_STRATA
        )
        cells.append(
            PrefixCell(
                provenance,
                k,
                len(ranked),
                tuple(row.action_hash for row in selected),
                predicted,
                0.0 if not ranked else k / len(ranked),
                max_share,
                proportions,
                None,
            )
        )
    return PrefixSurface(provenance, ranked, tuple(cells))


def attach_prefix_responses(
    surface: PrefixSurface,
    realized_utility_by_action_hash: Mapping[str, FavorableUtility],
) -> PrefixSurface:
    """Attach a complete pseudo response only after the descriptor seal exists."""

    if surface.responses_available:
        raise ProtocolError("P-DCAPS policy response surface was opened twice.")
    responses = {
        str(key): value for key, value in realized_utility_by_action_hash.items()
    }
    expected = {row.action_hash for row in surface.ranked_actions}
    if set(responses) != expected or any(
        not isinstance(value, FavorableUtility) for value in responses.values()
    ):
        raise ProtocolError("P-DCAPS policy response rectangle is incomplete or extra.")
    cells = []
    for cell in surface.cells:
        realized = FavorableUtility.zeros()
        for action_hash in cell.ordered_action_hashes:
            realized = realized + responses[action_hash]
        cells.append(cell.with_realized_utility(realized))
    response_surface = PrefixSurface(
        surface.provenance, surface.ranked_actions, tuple(cells)
    )
    if response_surface.surface_hash != surface.surface_hash:
        raise ProtocolError("P-DCAPS response attachment changed the descriptor seal.")
    return response_surface


def strip_prefix_responses(surface: PrefixSurface) -> PrefixSurface:
    """Return the byte-identical descriptor surface without response values."""

    if not surface.responses_available:
        return surface
    descriptor = PrefixSurface(
        surface.provenance,
        surface.ranked_actions,
        tuple(replace(cell, realized_utility=None) for cell in surface.cells),
    )
    if descriptor.surface_hash != surface.surface_hash:
        raise ProtocolError("P-DCAPS response stripping changed a descriptor seal.")
    return descriptor


def observations_from_surfaces(
    surfaces: Sequence[PrefixSurface],
) -> tuple[PolicyObservation, ...]:
    rows = tuple(surfaces)
    require_complete_responses(rows)
    observations = tuple(
        PolicyObservation(cell, surface.surface_hash)
        for surface in rows
        for cell in surface.cells
    )
    if len({row.observation_hash for row in observations}) != len(observations):
        raise ProtocolError("P-DCAPS policy response observations repeat.")
    return observations


def fit_nested_policy_calibrations(
    response_surfaces: Sequence[PrefixSurface],
    *,
    outer_center: str,
) -> NestedPolicyCalibration:
    """Fit every leave-J model, its OOF envelope, then the final H model."""

    surfaces = tuple(response_surfaces)
    outer = str(outer_center)
    require_complete_responses(surfaces)
    if any(
        surface.provenance.surface_role != "pseudo"
        or surface.provenance.outer_center != outer
        for surface in surfaces
    ):
        raise ProtocolError("P-DCAPS nested policy surfaces drifted from outer H.")
    observations = observations_from_surfaces(surfaces)
    centers = tuple(sorted({row.center for row in observations}))
    if len(centers) < 3:
        # At least two fitting centers must remain after holding J out.
        raise ProtocolError("P-DCAPS nested policy calibration lacks donor centers.")
    oof_models = tuple(
        fit_policy_calibration(
            observations, outer_center=outer, scored_center=scored_center
        )
        for scored_center in centers
    )
    residuals: list[PolicyOOFResidual] = []
    for model in oof_models:
        assert model.scored_center is not None
        for row in observations:
            if row.center != model.scored_center:
                continue
            residuals.append(
                PolicyOOFResidual(
                    outer,
                    row.center,
                    row.route_hash,
                    row.cell.cell_hash,
                    model.predict(row.cell),
                    row.cell.realized_utility,  # type: ignore[arg-type]
                    model.calibration_hash,
                    model.excluded_centers,
                )
            )
    envelope = build_policy_envelope(residuals, outer_center=outer)
    final_model = fit_policy_calibration(observations, outer_center=outer)
    return NestedPolicyCalibration(
        outer,
        final_model,
        oof_models,
        tuple(residuals),
        envelope,
    )


def calibrate_prefix_surface(
    surface: PrefixSurface,
    nested: NestedPolicyCalibration,
) -> tuple[CalibratedPrefixCell, ...]:
    """Predict and correct one target surface, preserving exact P exactly."""

    if (
        surface.provenance.surface_role != "target"
        or surface.provenance.outer_center != nested.outer_center
        or surface.provenance.route_center != nested.outer_center
        or surface.responses_available
    ):
        raise ProtocolError("P-DCAPS target policy surface role drifted.")
    result = []
    for cell in surface.cells:
        predicted = nested.final_calibration.predict(cell)
        if cell.k == 0:
            corrected = FavorableUtility.zeros()
            correction = FavorableUtility.zeros()
            count = 0
        else:
            corrected, count = apply_policy_envelope(predicted, nested.envelope)
            correction = nested.envelope.correction
        result.append(
            CalibratedPrefixCell(
                cell,
                predicted,
                correction,
                corrected,
                nested.final_calibration.calibration_hash,
                nested.envelope.envelope_hash,
                count,
            )
        )
    return tuple(result)


def calibrate_prefix_surface_with(
    surface: PrefixSurface,
    calibration: PolicyCalibration,
    envelope: PolicyEnvelope,
) -> tuple[CalibratedPrefixCell, ...]:
    """Calibrate a target-H or pseudo-J surface with explicit exclusions."""

    provenance = surface.provenance
    expected_scored = (
        None if provenance.surface_role == "target" else provenance.route_center
    )
    if (
        surface.responses_available
        or calibration.outer_center != provenance.outer_center
        or calibration.scored_center != expected_scored
        or envelope.outer_center != provenance.outer_center
        or envelope.excluded_scored_center != expected_scored
    ):
        raise ProtocolError("P-DCAPS explicit policy calibration lineage drifted.")
    result = []
    for cell in surface.cells:
        predicted = calibration.predict(cell)
        if cell.k == 0:
            corrected = FavorableUtility.zeros()
            correction = FavorableUtility.zeros()
            count = 0
        else:
            corrected, count = apply_policy_envelope(predicted, envelope)
            correction = envelope.correction
        result.append(
            CalibratedPrefixCell(
                cell,
                predicted,
                correction,
                corrected,
                calibration.calibration_hash,
                envelope.envelope_hash,
                count,
            )
        )
    return tuple(result)


def calibrate_and_select_prefix_with(
    surface: PrefixSurface,
    calibration: PolicyCalibration,
    envelope: PolicyEnvelope,
) -> PolicySelection:
    return select_policy_prefix(
        surface,
        calibrate_prefix_surface_with(surface, calibration, envelope),
    )


def calibrate_and_select_prefix(
    surface: PrefixSurface,
    nested: NestedPolicyCalibration,
) -> PolicySelection:
    calibrated = calibrate_prefix_surface(surface, nested)
    return select_policy_prefix(surface, calibrated)


__all__ = (
    "NestedPolicyCalibration",
    "attach_prefix_responses",
    "build_prefix_surface",
    "calibrate_and_select_prefix",
    "calibrate_and_select_prefix_with",
    "calibrate_prefix_surface",
    "calibrate_prefix_surface_with",
    "fit_nested_policy_calibrations",
    "observations_from_surfaces",
    "policy_action_from_selection",
    "strip_prefix_responses",
)
