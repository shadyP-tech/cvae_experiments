"""Pure nested H/J action-and-policy surface engine for P-DCAPS.

The runner owns artifact and label capabilities; this engine owns only sealed
DTOs.  Consequently it is testable without paths, CUDA, worker handles, raw
labels, or predecessor-router state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...protocol import ProtocolError
from .action_surface import (
    ActionCalibrationFamilies,
    ActionResponse,
    ActionStratumReliability,
    CalibratedActionSelection,
    SealedActionSurface,
    build_action_reliability_by_stratum,
    build_optimized_action_calibration_families,
    calibrate_and_select_actions,
)
from .contracts import RouteKey
from .identity import canonical_hash, require_sha256
from .policy_surface import (
    NestedPolicyCalibration,
    PolicyCalibrationFamilies,
    PolicyAction,
    PolicySelection,
    PrefixSurface,
    attach_prefix_responses,
    build_prefix_surface,
    calibrate_and_select_prefix,
    calibrate_and_select_prefix_with,
    build_optimized_policy_calibration_families,
    policy_action_from_selection,
    strip_prefix_responses,
)
from .target_local_runtime import POSTERIOR_CONTROL_IDS


@dataclass(frozen=True)
class RouteActionDecision:
    route_key: RouteKey
    reliability_hashes: tuple[str, ...]
    selection: CalibratedActionSelection
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        hashes = tuple(str(value) for value in self.reliability_hashes)
        if (
            self.selection.route_key != self.route_key
            or len(hashes) != 6
            or len(set(hashes)) != len(hashes)
        ):
            raise ProtocolError("P-DCAPS route action decision drifted.")
        object.__setattr__(self, "reliability_hashes", hashes)
        object.__setattr__(
            self,
            "decision_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_route_action_decision_v1",
                    "route_key": self.route_key.to_payload(),
                    "reliability_hashes": hashes,
                    "selection_hash": self.selection.selection_hash,
                    "held_case_response_used": False,
                }
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_route_action_decision_v1",
            "route_key": self.route_key.to_payload(),
            "reliability_hashes": list(self.reliability_hashes),
            "selection": self.selection.to_payload(),
            "held_case_response_used": False,
            "decision_hash": self.decision_hash,
        }


@dataclass(frozen=True)
class OuterActionPolicyResult:
    outer_center: str
    action_surface_seal_hash: str
    physical_surface_hash: str
    posterior_control_id: str
    calibration_families: ActionCalibrationFamilies
    target_reliabilities: tuple[ActionStratumReliability, ...]
    pseudo_reliabilities_by_center: tuple[
        tuple[str, tuple[ActionStratumReliability, ...]], ...
    ]
    target_action_decisions: tuple[RouteActionDecision, ...]
    pseudo_action_decisions: tuple[RouteActionDecision, ...]
    pseudo_policy_response_surfaces: tuple[PrefixSurface, ...]
    policy_calibration_families: PolicyCalibrationFamilies
    pseudo_policy_selections_by_center: tuple[tuple[str, PolicySelection], ...]
    nested_policy_calibration: NestedPolicyCalibration
    target_policy_surface: PrefixSurface
    target_policy_selection: PolicySelection
    result_hash: str = field(init=False)

    def __post_init__(self) -> None:
        action_seal_hash = require_sha256(
            self.action_surface_seal_hash, "action-surface seal hash"
        )
        physical_hash = require_sha256(
            self.physical_surface_hash, "physical surface hash"
        )
        control_id = str(self.posterior_control_id)
        donors = tuple(center for center in CENTERS if center != self.outer_center)
        pseudo_reliability_centers = tuple(
            center for center, _rows in self.pseudo_reliabilities_by_center
        )
        pseudo_surface_centers = tuple(
            row.provenance.route_center
            for row in self.pseudo_policy_response_surfaces
        )
        pseudo_selection_centers = tuple(
            center for center, _row in self.pseudo_policy_selections_by_center
        )
        if (
            self.outer_center not in CENTERS
            or control_id not in POSTERIOR_CONTROL_IDS
            or self.calibration_families.outer_center != self.outer_center
            or self.calibration_families.plan_hash == ""
            or len(self.target_reliabilities) != 6
            or pseudo_reliability_centers != donors
            or pseudo_surface_centers != donors
            or pseudo_selection_centers != donors
            or any(
                not row.responses_available
                for row in self.pseudo_policy_response_surfaces
            )
            or self.target_policy_surface.responses_available
            or self.target_policy_surface.provenance.route_center
            != self.outer_center
            or self.target_policy_selection.surface_hash
            != self.target_policy_surface.surface_hash
            or self.nested_policy_calibration.outer_center != self.outer_center
            or self.policy_calibration_families.outer_center != self.outer_center
            or self.policy_calibration_families.target.nested_hash
            != self.nested_policy_calibration.nested_hash
            or any(
                selection.surface_hash != surface.surface_hash
                or center != surface.provenance.route_center
                for (center, selection), surface in zip(
                    self.pseudo_policy_selections_by_center,
                    self.pseudo_policy_response_surfaces,
                    strict=True,
                )
            )
            or any(
                row.provenance.action_surface_seal_hash != action_seal_hash
                for row in self.pseudo_policy_response_surfaces
            )
            or self.target_policy_surface.provenance.action_surface_seal_hash
            != action_seal_hash
        ):
            raise ProtocolError("P-DCAPS outer action-policy result drifted.")
        object.__setattr__(self, "action_surface_seal_hash", action_seal_hash)
        object.__setattr__(self, "physical_surface_hash", physical_hash)
        object.__setattr__(self, "posterior_control_id", control_id)
        object.__setattr__(
            self,
            "result_hash",
            canonical_hash(
                {
                    "schema_version": "pdcaps_outer_action_policy_result_v3",
                    "outer_center": self.outer_center,
                    "action_surface_seal_hash": action_seal_hash,
                    "physical_surface_hash": physical_hash,
                    "posterior_control_id": control_id,
                    "action_calibration_plan_hash": self.calibration_families.plan_hash,
                    "target_reliability_hashes": tuple(
                        row.reliability_hash for row in self.target_reliabilities
                    ),
                    "pseudo_reliability_hashes": tuple(
                        (
                            center,
                            tuple(row.reliability_hash for row in values),
                        )
                        for center, values in self.pseudo_reliabilities_by_center
                    ),
                    "target_action_decision_hashes": tuple(
                        row.decision_hash for row in self.target_action_decisions
                    ),
                    "pseudo_action_decision_hashes": tuple(
                        row.decision_hash for row in self.pseudo_action_decisions
                    ),
                    "pseudo_policy_response_surface_hashes": tuple(
                        row.response_surface_hash
                        for row in self.pseudo_policy_response_surfaces
                    ),
                    "policy_calibration_plan_hash": self.policy_calibration_families.plan_hash,
                    "pseudo_policy_selection_hashes": tuple(
                        (center, row.selection_hash)
                        for center, row in self.pseudo_policy_selections_by_center
                    ),
                    "nested_policy_calibration_hash": self.nested_policy_calibration.nested_hash,
                    "target_policy_surface_hash": self.target_policy_surface.surface_hash,
                    "target_policy_selection_hash": self.target_policy_selection.selection_hash,
                    "target_labels_used": False,
                }
            ),
        )

    @property
    def target_selected_policy_actions(self) -> tuple[str, ...]:
        return self.target_policy_selection.selected_cell.cell.ordered_action_hashes

    @property
    def target_action_only_actions(self) -> tuple[str, ...]:
        return tuple(
            action.action_hash for action in self.target_policy_surface.ranked_actions
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "pdcaps_outer_action_policy_result_v3",
            "outer_center": self.outer_center,
            "action_surface_seal_hash": self.action_surface_seal_hash,
            "physical_surface_hash": self.physical_surface_hash,
            "posterior_control_id": self.posterior_control_id,
            "calibration_families": self.calibration_families.to_payload(),
            "target_reliabilities": [
                row.to_payload() for row in self.target_reliabilities
            ],
            "pseudo_reliabilities_by_center": [
                [center, [row.to_payload() for row in rows]]
                for center, rows in self.pseudo_reliabilities_by_center
            ],
            "target_action_decisions": [
                row.to_payload() for row in self.target_action_decisions
            ],
            "pseudo_action_decisions": [
                row.to_payload() for row in self.pseudo_action_decisions
            ],
            "pseudo_policy_response_surfaces": [
                row.to_payload() for row in self.pseudo_policy_response_surfaces
            ],
            "policy_calibration_families": self.policy_calibration_families.to_payload(),
            "pseudo_policy_selections_by_center": [
                [center, row.to_payload()]
                for center, row in self.pseudo_policy_selections_by_center
            ],
            "nested_policy_calibration": self.nested_policy_calibration.to_payload(),
            "target_policy_surface": self.target_policy_surface.to_payload(),
            "target_policy_selection": self.target_policy_selection.to_payload(),
            "target_labels_used": False,
            "result_hash": self.result_hash,
        }


def _select_routes(
    routes: Sequence[object],
    *,
    models: Sequence[object],
    reliabilities: tuple[ActionStratumReliability, ...],
) -> tuple[RouteActionDecision, ...]:
    decisions: list[RouteActionDecision] = []
    for route in sorted(tuple(routes), key=lambda row: row.route_key):
        _calibrated, selection = calibrate_and_select_actions(
            route.predictions,
            models,
            reliabilities,
            empty_route_key=route.route_key,
        )
        decisions.append(
            RouteActionDecision(
                route.route_key,
                tuple(row.reliability_hash for row in reliabilities),
                selection,
            )
        )
    return tuple(decisions)


def _policy_actions(
    decisions: Sequence[RouteActionDecision],
) -> tuple[PolicyAction, ...]:
    rows = tuple(
        row
        for row in (
            policy_action_from_selection(decision.selection)
            for decision in decisions
        )
        if row is not None
    )
    return rows


def fit_outer_action_policy_surface(
    action_surface: SealedActionSurface,
    action_responses: Sequence[ActionResponse],
    *,
    outer_center: str,
    minimum_reliability_center_count: int = 6,
    require_complete_center_inventory: bool = True,
) -> OuterActionPolicyResult:
    """Fit and apply the complete nested action/policy surface for one H.

    ``action_responses`` must be the complete pseudo-only rectangle for H.
    The target routes remain response-free throughout this function.
    """

    outer = str(outer_center)
    if outer not in CENTERS:
        raise ProtocolError("P-DCAPS outer engine center drifted.")
    routes = tuple(
        row for row in action_surface.routes if row.route_key.outer_center == outer
    )
    target_routes = tuple(
        row for row in routes if row.route_key.surface_role == "target"
    )
    pseudo_routes = tuple(
        row for row in routes if row.route_key.surface_role == "pseudo"
    )
    donors = tuple(center for center in CENTERS if center != outer)
    observed_donors = tuple(
        center
        for center in CENTERS
        if center in {row.route_key.route_center for row in pseudo_routes}
    )
    if (
        not target_routes
        or not pseudo_routes
        or any(row.route_key.route_center != outer for row in target_routes)
        or (
            require_complete_center_inventory
            and observed_donors != donors
        )
    ):
        raise ProtocolError("P-DCAPS H/J route inventory is incomplete.")

    predictions = tuple(
        prediction for route in routes for prediction in route.predictions
    )
    pseudo_predictions = tuple(
        prediction for route in pseudo_routes for prediction in route.predictions
    )
    responses = tuple(action_responses)
    if (
        any(row.key.route_key.outer_center != outer for row in responses)
        or any(row.key.route_key.surface_role != "pseudo" for row in responses)
        or {row.prediction_hash for row in responses}
        != {row.prediction_hash for row in pseudo_predictions}
        or len({row.response_hash for row in responses}) != len(responses)
    ):
        raise ProtocolError("P-DCAPS pseudo action-response rectangle drifted.")

    calibration = build_optimized_action_calibration_families(
        predictions,
        responses,
        outer_center=outer,
    )
    target_reliability = build_action_reliability_by_stratum(
        predictions,
        responses,
        calibration.target_reliability_oof,
        outer_center=outer,
        minimum_center_count=int(minimum_reliability_center_count),
    )
    target_decisions = _select_routes(
        target_routes,
        models=calibration.target_models,
        reliabilities=target_reliability,
    )

    pseudo_reliability_rows: list[
        tuple[str, tuple[ActionStratumReliability, ...]]
    ] = []
    pseudo_decisions: list[RouteActionDecision] = []
    pseudo_surfaces: list[PrefixSurface] = []
    response_by_action_key = {
        row.key.action_key_hash: row for row in responses
    }
    for donor in donors:
        donor_routes = tuple(
            row for row in pseudo_routes if row.route_key.route_center == donor
        )
        reliability = build_action_reliability_by_stratum(
            predictions,
            responses,
            calibration.pseudo_reliability_oof(donor),
            outer_center=outer,
            scored_center=donor,
            minimum_center_count=int(minimum_reliability_center_count),
        )
        pseudo_reliability_rows.append((donor, reliability))
        decisions = _select_routes(
            donor_routes,
            models=calibration.pseudo_models[donor],
            reliabilities=reliability,
        )
        pseudo_decisions.extend(decisions)
        actions = _policy_actions(decisions)
        descriptor_surface = build_prefix_surface(
            actions,
            surface_role="pseudo",
            outer_center=outer,
            route_center=donor,
            action_surface_seal_hash=action_surface.action_surface_seal_hash,
        )
        realized_by_action = {}
        for action in actions:
            try:
                response = response_by_action_key[action.action_hash]
            except KeyError as exc:
                raise ProtocolError(
                    "P-DCAPS selected pseudo action lacks its post-seal response."
                ) from exc
            if response.key.route_key.route_center != donor:
                raise ProtocolError("P-DCAPS pseudo policy response crossed J.")
            realized_by_action[action.action_hash] = response.realized_utility
        pseudo_surfaces.append(
            attach_prefix_responses(descriptor_surface, realized_by_action)
        )

    policy_calibration = build_optimized_policy_calibration_families(
        pseudo_surfaces, outer_center=outer
    )
    pseudo_policy_selections = tuple(
        (
            donor,
            calibrate_and_select_prefix_with(
                strip_prefix_responses(surface),
                policy_calibration.pseudo_calibrations[donor],
                policy_calibration.pseudo_envelopes[donor],
            ),
        )
        for donor, surface in zip(donors, pseudo_surfaces, strict=True)
    )
    nested_policy = policy_calibration.target
    target_surface = build_prefix_surface(
        _policy_actions(target_decisions),
        surface_role="target",
        outer_center=outer,
        route_center=outer,
        action_surface_seal_hash=action_surface.action_surface_seal_hash,
    )
    target_selection = calibrate_and_select_prefix(target_surface, nested_policy)
    return OuterActionPolicyResult(
        outer,
        action_surface.action_surface_seal_hash,
        action_surface.physical_surface_hash,
        action_surface.posterior_control_id,
        calibration,
        target_reliability,
        tuple(pseudo_reliability_rows),
        target_decisions,
        tuple(pseudo_decisions),
        tuple(pseudo_surfaces),
        policy_calibration,
        pseudo_policy_selections,
        nested_policy,
        target_surface,
        target_selection,
    )


__all__ = (
    "OuterActionPolicyResult",
    "RouteActionDecision",
    "fit_outer_action_policy_surface",
)
