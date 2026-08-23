from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.action_surface import (
    ActionDraft,
    ResponseDenominators,
    RouteActionDraftSurface,
    seal_action_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.route_support import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.contracts import (
    BankViability,
    FavorableUtility,
    RouteKey,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    CANONICAL_CASE_COUNT,
    CANONICAL_ROW_COUNT,
    ExpectedRouteInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.lifecycle import (
    PDCAPSLabelLifecycle,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _canonical_keys() -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for case_ordinal in range(CANONICAL_CASE_COUNT):
        center = CENTERS[case_ordinal % len(CENTERS)]
        case_id = f"case-{case_ordinal:03d}"
        row_count = 46 if case_ordinal < 118 else 45
        rows.extend(
            (center, case_id, f"sample-{case_ordinal:03d}-{row:02d}")
            for row in range(row_count)
        )
    assert len(rows) == CANONICAL_ROW_COUNT
    return tuple(rows)


def _draft_routes(
    inventory: ExpectedRouteInventory,
) -> tuple[RouteActionDraftSurface, ...]:
    physical_hash = _hash("physical")
    rows: list[RouteActionDraftSurface] = []
    for outer in inventory.centers:
        for case in inventory.cases:
            target = case.center == outer
            route = RouteKey(
                "target" if target else "pseudo",
                outer,
                outer if target else case.center,
                case.case_id,
                outer,
                None if target else case.center,
                _hash((outer, case.center, case.case_id, "fit")),
            )
            rows.append(
                RouteActionDraftSurface(
                    route,
                    case.sample_ids,
                    np.full(len(case.sample_ids), 0.25, dtype=np.float32),
                    (),
                    _hash((outer, case.case_id, "endpoint")),
                    _hash((outer, case.case_id, "posterior")),
                    physical_hash,
                    "IDENTITY",
                )
            )
    return tuple(rows)


def test_canonical_inventory_binds_all_cases_rows_and_outer_routes() -> None:
    keys = _canonical_keys()
    inventory = ExpectedRouteInventory.from_label_free_keys(
        keys,
        manifest_sha256=_hash("manifest"),
        row_order_hash=_hash("row-order"),
    )
    routes = _draft_routes(inventory)

    assert inventory.centers == CENTERS
    assert inventory.case_count == 218
    assert inventory.row_count == 9_928
    assert inventory.target_route_count == 218
    assert inventory.pseudo_route_count == 1_744
    assert inventory.total_route_count == 1_962
    assert len(routes) == 1_962
    assert len(inventory.validate_draft_routes(routes)) == 64

    surface = seal_action_surface(
        routes,
        expected_outer_centers=CENTERS,
        expected_inventory_hash=inventory.inventory_hash,
    )
    assert surface.expected_inventory_hash == inventory.inventory_hash
    with pytest.raises(ProtocolError, match="global action surface"):
        replace(surface, expected_inventory_hash=_hash("foreign-inventory"))


def test_inventory_and_lifecycle_reject_missing_route_or_row() -> None:
    keys = _canonical_keys()
    inventory = ExpectedRouteInventory.from_label_free_keys(
        keys,
        manifest_sha256=_hash("manifest"),
        row_order_hash=_hash("row-order"),
    )
    routes = _draft_routes(inventory)

    with pytest.raises(ProtocolError, match="route/case inventory"):
        inventory.validate_draft_routes(routes[:-1])
    lifecycle = PDCAPSLabelLifecycle(
        lambda _keys, _scope: (),
        protocol_hash=_hash("protocol"),
        expected_inventory=inventory,
    )
    lifecycle.begin_support()
    with pytest.raises(ProtocolError, match="route/case inventory"):
        lifecycle.seal_actions(routes[:-1])

    first = routes[0]
    shortened = RouteActionDraftSurface(
        first.route_key,
        first.sample_ids[:-1],
        first.baseline_probabilities[:-1],
        (),
        first.endpoint_hash,
        first.posterior_prediction_hash,
        first.physical_surface_hash,
        first.posterior_control_id,
    )
    with pytest.raises(ProtocolError, match="route row inventory"):
        inventory.validate_draft_routes((shortened, *routes[1:]))

    with pytest.raises(ProtocolError, match="canonical test inventory"):
        ExpectedRouteInventory.from_label_free_keys(
            keys[:-1],
            manifest_sha256=_hash("manifest"),
            row_order_hash=_hash("row-order"),
        )


def test_identity_and_cyclic_surfaces_share_one_label_grant_and_joint_seal() -> None:
    inventory = ExpectedRouteInventory.focused_fixture(
        (
            ("0", "case-0", "0-a"),
            ("0", "case-0", "0-b"),
            ("1", "case-1", "1-a"),
            ("1", "case-1", "1-b"),
        )
    )
    identity = _draft_routes(inventory)
    cyclic = tuple(
        replace(
            row,
            posterior_prediction_hash=_hash(
                (row.route_key.outer_center, row.route_key.held_case_id, "cyclic")
            ),
            posterior_control_id="WITHIN_CASE_CYCLIC_SHIFT",
        )
        for row in identity
    )
    loader_calls: list[str] = []

    def loader(keys: object, scope: str) -> tuple[BinaryLabel, ...]:
        loader_calls.append(scope)
        return tuple(
            BinaryLabel(*key, index % 2, scope)
            for index, key in enumerate(tuple(keys))  # type: ignore[arg-type]
        )

    lifecycle = PDCAPSLabelLifecycle(
        loader,
        protocol_hash=_hash("protocol"),
        expected_inventory=inventory,
    )
    lifecycle.begin_support()
    with pytest.raises(ProtocolError, match="identity/cyclic surface set"):
        lifecycle.seal_actions(identity)

    lifecycle = PDCAPSLabelLifecycle(
        loader,
        protocol_hash=_hash("protocol"),
        expected_inventory=inventory,
    )
    lifecycle.begin_support()
    lifecycle.seal_actions(identity, cyclic_control_routes=cyclic)
    assert lifecycle.action_surface_set.control_ids == (
        "IDENTITY",
        "WITHIN_CASE_CYCLIC_SHIFT",
    )
    assert (
        lifecycle.action_surface_set.identity.action_surface_seal_hash
        != lifecycle.action_surface_set.cyclic.action_surface_seal_hash  # type: ignore[union-attr]
    )
    lifecycle.begin_pseudo_responses()
    pseudo_route = next(
        row.route_key
        for row in identity
        if row.route_key.surface_role == "pseudo"
    )
    responses = lifecycle.open_pseudo_control_action_responses(
        pseudo_route,
        denominators=ResponseDenominators(1, 1),
    )
    assert tuple(control_id for control_id, _rows in responses) == (
        "IDENTITY",
        "WITHIN_CASE_CYCLIC_SHIFT",
    )
    assert len(loader_calls) == 1

    mislabeled = tuple(
        replace(row, posterior_control_id="WITHIN_CASE_CYCLIC_SHIFT")
        for row in identity
    )
    second = PDCAPSLabelLifecycle(
        loader,
        protocol_hash=_hash("protocol"),
        expected_inventory=inventory,
    )
    second.begin_support()
    with pytest.raises(ProtocolError, match="identity/cyclic surface set"):
        second.seal_actions(identity, cyclic_control_routes=mislabeled)


def test_v2_derives_pseudo_denominators_inside_lifecycle_and_rejects_forgery() -> None:
    inventory = ExpectedRouteInventory.focused_fixture(
        (
            ("0", "case-0", "0-a"),
            ("0", "case-0", "0-b"),
            ("1", "case-1", "1-a"),
            ("1", "case-1", "1-b"),
        )
    )
    original = _draft_routes(inventory)
    identity = tuple(
        replace(
            row,
            drafts=(
                ActionDraft(
                    row.route_key,
                    "B",
                    "zero_to_one",
                    "B::zero_to_one",
                    np.asarray([0.75, 0.25], dtype=np.float32),
                    FavorableUtility(0.1, 0.1, 0.1),
                    0.5,
                    BankViability(
                        True,
                        True,
                        (("0", 8.0), ("1", 8.0)),
                        5.0,
                        _hash("viability"),
                    ),
                    row.endpoint_hash,
                    row.posterior_prediction_hash,
                ),
            ),
        )
        if row.route_key.outer_center == "0"
        and row.route_key.route_center == "1"
        else row
        for row in original
    )
    cyclic = tuple(
        replace(
            row,
            drafts=tuple(
                replace(
                    draft,
                    posterior_prediction_hash=_hash(
                        (
                            row.route_key.outer_center,
                            row.route_key.held_case_id,
                            "cyclic",
                        )
                    ),
                )
                for draft in row.drafts
            ),
            posterior_prediction_hash=_hash(
                (row.route_key.outer_center, row.route_key.held_case_id, "cyclic")
            ),
            posterior_control_id="WITHIN_CASE_CYCLIC_SHIFT",
        )
        for row in identity
    )
    grants: list[str] = []

    def loader(keys: object, scope: str) -> tuple[BinaryLabel, ...]:
        grants.append(scope)
        return tuple(
            BinaryLabel(*key, index % 2, scope)
            for index, key in enumerate(tuple(keys))  # type: ignore[arg-type]
        )

    lifecycle = PDCAPSLabelLifecycle(
        loader,
        protocol_hash=_hash("protocol"),
        expected_inventory=inventory,
        require_derived_response_denominators=True,
    )
    lifecycle.begin_support()
    lifecycle.open_support_labels(
        center="1",
        held_case_id="case-1",
        keys=(("1", "support-case", "s-a"), ("1", "support-case", "s-b")),
    )
    lifecycle.seal_actions(identity, cyclic_control_routes=cyclic)
    lifecycle.begin_pseudo_responses()
    pseudo_route = next(
        row.route_key
        for row in identity
        if row.route_key.outer_center == "0" and row.route_key.route_center == "1"
    )
    with pytest.raises(ProtocolError, match="derives response denominators"):
        lifecycle.open_pseudo_control_action_responses(
            pseudo_route, denominators=ResponseDenominators(999, 1)
        )
    responses = lifecycle.open_pseudo_control_action_responses_derived(pseudo_route)
    assert tuple(control for control, _rows in responses) == (
        "IDENTITY",
        "WITHIN_CASE_CYCLIC_SHIFT",
    )
    assert {
        (row.positive_denominator, row.negative_denominator)
        for _control, rows in responses
        for row in rows
    } == {(2, 2)}
    assert grants.count("PSEUDO::<0,1,case-1>") == 1
