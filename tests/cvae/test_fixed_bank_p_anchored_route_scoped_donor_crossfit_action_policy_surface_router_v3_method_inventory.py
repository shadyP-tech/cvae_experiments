from __future__ import annotations

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.legacy_control import (
    seal_legacy_control,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3 import (
    method_controls as v3_method_controls,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.admission import (
    build_outer_admission,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3.identity import (
    CYCLIC_METHOD_ID,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    P_METHOD_ID,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from tests.cvae.test_fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_admission import (
    _build_method_sources,
    _evidence,
)


@pytest.fixture(scope="module")
def six_method_sources():
    surface_set, identity, cyclic = _build_method_sources()
    return (
        surface_set,
        identity,
        cyclic,
        seal_legacy_control(identity),
        seal_legacy_control(cyclic),
    )


def _menu(sources):
    surface_set, identity, cyclic, identity_legacy, cyclic_legacy = sources
    return v3_method_controls.build_fixed_method_menu(
        identity_result=identity,
        cyclic_result=cyclic,
        surface_set=surface_set,
        identity_legacy_control=identity_legacy,
        cyclic_legacy_control=cyclic_legacy,
    )


def _target_routes(surface, outer_center: str):
    return tuple(
        row
        for row in surface.routes
        if row.route_key.surface_role == "target"
        and row.route_key.outer_center == outer_center
    )


def _protected_bytes(routes, order: tuple[str, ...]) -> bytes:
    by_sample = {
        sample_id: np.float32(value)
        for row in routes
        for sample_id, value in zip(
            row.sample_ids, row.baseline_probabilities, strict=True
        )
    }
    return np.ascontiguousarray(
        [by_sample[sample_id] for sample_id in order], dtype=np.float32
    ).tobytes(order="C")


def test_v3_constructs_complete_fixed_six_method_inventory(
    six_method_sources,
) -> None:
    decisions = _menu(six_method_sources)
    assert tuple(row.method_id for row in decisions) == METHOD_MENU
    assert len({row.decision_hash for row in decisions}) == len(METHOD_MENU)
    assert all(
        isinstance(row, v3_method_controls.AdmissionControlledMethodDecision)
        for row in decisions
    )
    assert decisions[0].method_id == P_METHOD_ID
    assert decisions[0].selected_action_hashes == ()
    assert decisions[0].exact_p_fallback is True
    assert decisions[0].reason == "EXACT_P_PROTECTED"
    assert decisions[1].method_id == PRIMARY_METHOD_ID
    assert decisions[-1].method_id == CYCLIC_METHOD_ID
    assert tuple(row.outer_admission_applied for row in decisions) == (
        False,
        True,
        False,
        False,
        False,
        True,
    )
    assert tuple(row.outer_admission_passed for row in decisions) == (
        None,
        True,
        None,
        None,
        None,
        True,
    )
    assert decisions[0].legacy_control_seal_hash is None
    assert decisions[2].legacy_control_seal_hash is None
    assert decisions[3].legacy_control_seal_hash is None
    assert decisions[1].legacy_control_seal_hash is not None
    assert decisions[4].legacy_control_seal_hash is not None
    assert decisions[5].legacy_control_seal_hash is not None
    assert decisions[-1].posterior_control_id == "WITHIN_CASE_CYCLIC_SHIFT"
    assert decisions[-1].joint_surface_set_seal_hash is not None
    assert all(row.to_payload()["routing_authorized"] is False for row in decisions)
    assert all(row.to_payload()["promotion_allowed"] is False for row in decisions)
    assert all(row.to_payload()["target_labels_used"] is False for row in decisions)
    assert tuple(
        row.to_payload()["nullable_admission_statistics"] for row in decisions
    ) == (False, True, False, False, False, True)


def test_all_six_methods_compose_from_their_typed_source(
    six_method_sources,
) -> None:
    surface_set, identity, cyclic, _identity_legacy, _cyclic_legacy = (
        six_method_sources
    )
    decisions = _menu(six_method_sources)
    identity_routes = _target_routes(surface_set.identity, identity.outer_center)
    assert surface_set.cyclic is not None
    cyclic_routes = _target_routes(surface_set.cyclic, cyclic.outer_center)
    order = tuple(
        reversed(
            tuple(sample for row in identity_routes for sample in row.sample_ids)
        )
    )
    compositions = tuple(
        v3_method_controls.compose_method_prediction(
            cyclic_routes if row.method_id == CYCLIC_METHOD_ID else identity_routes,
            center_sample_order=order,
            decision=row,
        )
        for row in decisions
    )
    assert tuple(row.prediction.method_id for row in compositions) == METHOD_MENU
    assert len(
        {row.method_composition_hash for row in compositions}
    ) == len(METHOD_MENU)
    for decision, composition in zip(decisions, compositions, strict=True):
        assert composition.prediction.selected_action_hashes == (
            decision.selected_action_hashes
        )
        assert composition.prediction.selection_enabled == (
            decision.composition_selection_enabled
        )
        assert composition.to_payload()["terminal_diagnostic_only"] is True
        if decision.exact_p_fallback:
            routes = (
                cyclic_routes
                if decision.method_id == CYCLIC_METHOD_ID
                else identity_routes
            )
            assert composition.prediction.probabilities.tobytes(order="C") == (
                _protected_bytes(routes, order)
            )


def test_protected_primary_and_cyclic_are_exact_p_when_admission_fails(
    six_method_sources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface_set, identity, cyclic, _identity_legacy, _cyclic_legacy = (
        six_method_sources
    )
    failed = build_outer_admission(
        identity.outer_center,
        _evidence(constant_metric="bacc", constant_side="realized"),
    )
    assert failed.passed is False
    monkeypatch.setattr(
        v3_method_controls,
        "build_admission_from_pseudo_policies",
        lambda result, control: failed,
    )
    decisions = _menu(six_method_sources)
    exact_p_methods = {
        row.method_id: row
        for row in decisions
        if row.method_id in {P_METHOD_ID, PRIMARY_METHOD_ID, CYCLIC_METHOD_ID}
    }
    assert set(exact_p_methods) == {
        P_METHOD_ID,
        PRIMARY_METHOD_ID,
        CYCLIC_METHOD_ID,
    }
    assert all(row.exact_p_fallback for row in exact_p_methods.values())
    assert all(not row.composition_selection_enabled for row in exact_p_methods.values())

    assert surface_set.cyclic is not None
    for method_id, decision in exact_p_methods.items():
        routes = _target_routes(
            surface_set.cyclic
            if method_id == CYCLIC_METHOD_ID
            else surface_set.identity,
            identity.outer_center,
        )
        order = tuple(
            reversed(tuple(sample for row in routes for sample in row.sample_ids))
        )
        composed = v3_method_controls.compose_method_prediction(
            routes,
            center_sample_order=order,
            decision=decision,
        )
        assert composed.prediction.selected_action_hashes == ()
        assert composed.prediction.selection_enabled is False
        assert composed.prediction.probabilities.tobytes(order="C") == (
            _protected_bytes(routes, order)
        )


def test_fixed_menu_rejects_caller_method_and_wrong_control_source(
    six_method_sources,
) -> None:
    _surface_set, identity, _cyclic, _identity_legacy, cyclic_legacy = (
        six_method_sources
    )
    with pytest.raises(ProtocolError, match="fixed menu"):
        v3_method_controls.AdmissionControlledMethodDecision(
            "CALLER_METHOD", identity, identity
        )
    with pytest.raises(ProtocolError, match="same-run|lineage|decision"):
        v3_method_controls.build_legacy_method_decision(
            identity, cyclic_legacy
        )

