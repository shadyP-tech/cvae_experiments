from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    ActionDraft,
    ActionResponse,
    RouteActionDraftSurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    BankViability,
    FavorableUtility,
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.engine import (
    OuterActionPolicyResult,
    fit_outer_action_policy_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.identity import (
    ACTION_STRATA,
    METHOD_MENU,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.legacy_control import (
    seal_legacy_control,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.method_controls import (
    MethodControlDecision,
    build_action_only_method_decision,
    build_cyclic_poison_method_decision,
    build_legacy_method_decision,
    build_policy_only_method_decision,
    build_primary_method_decision,
    build_protected_method_decision,
    compose_method_prediction,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.preterminal import (
    PreterminalOutputHashes,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.surface_set import (
    seal_action_surface_set,
)
from midogpp_thesis.cvae.protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _viability(value: object) -> BankViability:
    return BankViability(
        True,
        True,
        (("0", 30.0), ("1", 30.0)),
        5.0,
        _hash((value, "viability")),
    )


def _route(*, outer: str, center: str, target: bool) -> RouteKey:
    return RouteKey(
        "target" if target else "pseudo",
        outer,
        outer if target else center,
        f"case-{center}",
        outer,
        None if target else center,
        _hash((outer, center, target, "fit")),
    )


def _draft_route(
    *, outer: str, center: str, target: bool, control_id: str
) -> RouteActionDraftSurface:
    route = _route(outer=outer, center=center, target=target)
    baseline = np.asarray([0.2, 0.8], dtype=np.float32)
    drafts = []
    center_index = CENTERS.index(center)
    for stratum_index, (family, direction) in enumerate(ACTION_STRATA):
        action = (
            np.asarray([0.7, 0.8], dtype=np.float32)
            if direction == "zero_to_one"
            else np.asarray([0.2, 0.3], dtype=np.float32)
        )
        level = 0.01 * (center_index + 1) + 0.001 * (stratum_index + 1)
        drafts.append(
            ActionDraft(
                route,
                family,
                direction,
                f"{family}::{direction}",
                action,
                FavorableUtility(level, level / 2.0, level / 3.0),
                0.5,
                _viability((center, family, direction)),
                _hash((center, "endpoint")),
                _hash((center, control_id, "posterior")),
            )
        )
    return RouteActionDraftSurface(
        route,
        (f"{center}-a", f"{center}-b"),
        baseline,
        tuple(drafts),
        _hash((center, "endpoint")),
        _hash((center, control_id, "posterior")),
        _hash("physical"),
        control_id,
    )


def _outer_result(sealed, *, outer: str = "0"):
    responses = []
    for prediction in sealed.predictions:
        route = prediction.key.route_key
        if route.outer_center != outer or route.surface_role != "pseudo":
            continue
        center_index = CENTERS.index(route.route_center)
        stratum_index = ACTION_STRATA.index(prediction.key.stratum)
        level = 0.012 * (center_index + 1) + 0.001 * (stratum_index + 1)
        responses.append(
            ActionResponse(
                prediction.key,
                prediction.prediction_hash,
                FavorableUtility(level, level / 2.0, level / 3.0),
                2,
                10,
                10,
                20,
                _hash("P"),
                _hash((route.route_center, "rows")),
            )
        )
    return fit_outer_action_policy_surface(
        sealed,
        responses,
        outer_center=outer,
    )


@pytest.fixture(scope="module")
def method_sources():
    inventory = ExpectedRouteInventory.focused_fixture(
        tuple(
            (center, f"case-{center}", sample_id)
            for center in CENTERS
            for sample_id in (f"{center}-a", f"{center}-b")
        )
    )
    identity_routes = tuple(
        _draft_route(
            outer=outer,
            center=center,
            target=center == outer,
            control_id="IDENTITY",
        )
        for outer in CENTERS
        for center in CENTERS
    )
    cyclic_routes = tuple(
        _draft_route(
            outer=outer,
            center=center,
            target=center == outer,
            control_id="WITHIN_CASE_CYCLIC_SHIFT",
        )
        for outer in CENTERS
        for center in CENTERS
    )
    surface_set = seal_action_surface_set(
        identity_routes,
        expected_inventory=inventory,
        cyclic_routes=cyclic_routes,
    )
    identity_result = _outer_result(surface_set.identity)
    cyclic_result = _outer_result(surface_set.cyclic)
    return surface_set, identity_result, cyclic_result


def test_fixed_menu_derives_each_method_from_its_declared_source(
    method_sources,
) -> None:
    surface_set, identity, cyclic = method_sources
    identity_surface = surface_set.identity
    cyclic_surface = surface_set.cyclic
    assert cyclic_surface is not None
    identity_legacy = seal_legacy_control(identity)
    cyclic_legacy = seal_legacy_control(cyclic)
    decisions = (
        build_protected_method_decision(identity),
        build_primary_method_decision(identity, identity_legacy),
        build_action_only_method_decision(identity),
        build_policy_only_method_decision(identity),
        build_legacy_method_decision(identity, identity_legacy),
        build_cyclic_poison_method_decision(
            identity, cyclic, surface_set, cyclic_legacy
        ),
    )

    assert tuple(row.method_id for row in decisions) == METHOD_MENU
    assert decisions[0].selected_action_hashes == ()
    assert decisions[2].selected_action_hashes == identity.target_action_only_actions
    assert decisions[3].selected_action_hashes == identity.target_selected_policy_actions
    assert decisions[4].selected_action_hashes == (
        identity_legacy.surface.target_decision.selected_action_hashes
    )
    assert decisions[5].posterior_control_id == "WITHIN_CASE_CYCLIC_SHIFT"
    assert decisions[5].source_result_hash == cyclic.result_hash
    assert decisions[5].joint_surface_set_seal_hash == (
        surface_set.surface_set_seal_hash
    )
    assert decisions[5].source_action_surface_seal_hash != (
        decisions[5].identity_action_surface_seal_hash
    )
    assert all(row.to_payload()["routing_authorized"] is False for row in decisions)
    assert all(row.to_payload()["promotion_allowed"] is False for row in decisions)
    assert decisions[1].outer_admission_applied is True
    assert decisions[5].outer_admission_applied is True
    assert decisions[2].outer_admission_applied is False
    assert decisions[3].outer_admission_applied is False

    identity_routes = tuple(
        row
        for row in identity_surface.routes
        if row.route_key.surface_role == "target"
        and row.route_key.outer_center == identity.outer_center
    )
    cyclic_routes = tuple(
        row
        for row in cyclic_surface.routes
        if row.route_key.surface_role == "target"
        and row.route_key.outer_center == cyclic.outer_center
    )
    order = tuple(sample for row in identity_routes for sample in row.sample_ids)
    compositions = []
    for decision in decisions:
        routes = cyclic_routes if decision is decisions[-1] else identity_routes
        composed = compose_method_prediction(
            routes,
            center_sample_order=order,
            decision=decision,
        )
        assert composed.prediction.method_id == decision.method_id
        assert composed.prediction.selected_action_hashes == (
            decision.selected_action_hashes
        )
        assert composed.to_payload()["terminal_diagnostic_only"] is True
        compositions.append(composed)
    assert np.array_equal(
        compose_method_prediction(
            identity_routes,
            center_sample_order=order,
            decision=decisions[0],
        ).prediction.probabilities,
        identity_routes[0].baseline_probabilities,
    )
    preterminal = PreterminalOutputHashes.from_runtime(
        surface_set=surface_set,
        identity_results=(identity,),
        cyclic_results=(cyclic,),
        identity_legacy_controls=(identity_legacy,),
        cyclic_legacy_controls=(cyclic_legacy,),
        method_decisions=decisions,
        method_compositions=tuple(compositions),
    )
    assert preterminal.centers == (identity.outer_center,)
    assert len(preterminal.method_decision_hashes) == len(METHOD_MENU)
    assert len(preterminal.output_bundle_hash) == 64
    with pytest.raises(ProtocolError, match="method inventory"):
        PreterminalOutputHashes.from_runtime(
            surface_set=surface_set,
            identity_results=(identity,),
            cyclic_results=(cyclic,),
            identity_legacy_controls=(identity_legacy,),
            cyclic_legacy_controls=(cyclic_legacy,),
            method_decisions=decisions[:-1],
            method_compositions=tuple(compositions),
        )


def test_method_controls_fail_closed_on_wrong_source_or_seal(method_sources) -> None:
    surface_set, identity, cyclic = method_sources
    identity_surface = surface_set.identity
    identity_legacy = seal_legacy_control(identity)
    cyclic_legacy = seal_legacy_control(cyclic)

    with pytest.raises(ProtocolError, match="fixed menu"):
        MethodControlDecision("CALLER_METHOD", identity, identity)
    with pytest.raises(ProtocolError, match="identity posterior"):
        build_action_only_method_decision(cyclic)
    with pytest.raises(ProtocolError, match="distinct same-topology cyclic"):
        build_cyclic_poison_method_decision(
            identity,
            identity,
            surface_set,
            identity_legacy,
        )
    with pytest.raises(ProtocolError, match="same-run|lineage|decision"):
        build_cyclic_poison_method_decision(
            identity,
            cyclic,
            surface_set,
            identity_legacy,
        )
    with pytest.raises(ProtocolError, match="same-run|lineage|decision"):
        build_legacy_method_decision(identity, cyclic_legacy)

    action_only = build_action_only_method_decision(identity)
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(action_only, selected_action_hashes=(_hash("caller"),))

    identity_routes = tuple(
        row
        for row in identity_surface.routes
        if row.route_key.surface_role == "target"
        and row.route_key.outer_center == identity.outer_center
    )
    cyclic_decision = build_cyclic_poison_method_decision(
        identity,
        cyclic,
        surface_set,
        cyclic_legacy,
    )
    order = tuple(sample for row in identity_routes for sample in row.sample_ids)
    with pytest.raises(ProtocolError, match="source surface"):
        compose_method_prediction(
            identity_routes,
            center_sample_order=order,
            decision=cyclic_decision,
        )


def test_policy_contract_freezes_all_six_method_semantics() -> None:
    from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.config_payloads import (
        canonical_policy_menu_payload,
    )

    controls = canonical_policy_menu_payload()["method_controls"]
    fixed_menu = controls["fixed_menu"]
    assert tuple(fixed_menu) == METHOD_MENU
    assert controls["caller_selected_action_hashes_permitted"] is False
    assert controls["typed_source_and_seal_binding_required"] is True
    assert controls["terminal_diagnostic_only"] is True
    assert controls["routing_authorized"] is False
    assert controls["promotion_allowed"] is False
    assert fixed_menu["P_DCAPS_ACTION_ONLY"]["admission_H"] == (
        "bypassed_terminal_ablation"
    )
    assert fixed_menu["P_DCAPS_POLICY_ONLY"]["admission_H"] == (
        "bypassed_terminal_ablation"
    )
    cyclic = fixed_menu["P_DCAPS_CYCLIC_POISON"]
    assert cyclic["expected_posterior_control_id"] == (
        "WITHIN_CASE_CYCLIC_SHIFT"
    )
    assert cyclic["same_physical_surface_as_identity_required"] is True
    assert cyclic["distinct_action_surface_seal_from_identity_required"] is True
