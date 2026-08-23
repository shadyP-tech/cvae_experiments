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
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.identity import (
    CYCLIC_METHOD_ID,
    METHOD_MENU,
    PRIMARY_METHOD_ID,
    P_METHOD_ID,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v4.terminal import (
    exact_shared_center_max_sign_flip,
    midrank_spearman,
)
from tests.cvae.test_fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router_v3_admission import (
    _build_method_sources,
    _evidence,
)


def _target_routes(surface, outer_center: str):
    return tuple(
        row
        for row in surface.routes
        if row.route_key.surface_role == "target"
        and row.route_key.outer_center == outer_center
    )


def _protected_bytes(routes, order: tuple[str, ...]) -> bytes:
    protected = {
        sample: np.float32(value)
        for route in routes
        for sample, value in zip(
            route.sample_ids, route.baseline_probabilities, strict=True
        )
    }
    return np.ascontiguousarray(
        [protected[sample] for sample in order], dtype=np.float32
    ).tobytes(order="C")


def test_v4_failed_nullable_admission_composes_primary_and_cyclic_as_exact_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surface_set, identity, cyclic = _build_method_sources()
    failed = build_outer_admission(
        identity.outer_center,
        _evidence(constant_metric="bacc", constant_side="realized"),
    )
    monkeypatch.setattr(
        v3_method_controls,
        "build_admission_from_pseudo_policies",
        lambda result, control: failed,
    )
    decisions = v3_method_controls.build_fixed_method_menu(
        identity_result=identity,
        cyclic_result=cyclic,
        surface_set=surface_set,
        identity_legacy_control=seal_legacy_control(identity),
        cyclic_legacy_control=seal_legacy_control(cyclic),
    )
    assert surface_set.cyclic is not None
    for decision in decisions:
        if decision.method_id not in {
            P_METHOD_ID,
            PRIMARY_METHOD_ID,
            CYCLIC_METHOD_ID,
        }:
            continue
        routes = _target_routes(
            surface_set.cyclic
            if decision.method_id == CYCLIC_METHOD_ID
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
        assert decision.exact_p_fallback is True
        assert composed.prediction.selection_enabled is False
        assert composed.prediction.probabilities.tobytes(order="C") == (
            _protected_bytes(routes, order)
        )


def test_v4_terminal_inference_remains_descriptive_and_tie_aware() -> None:
    assert midrank_spearman((1.0, 1.0, 2.0), (3.0, 3.0, 1.0)) == pytest.approx(
        -1.0
    )
    metrics = {
        method: {
            center: {
                "center_bacc": 0.5
                if method == P_METHOD_ID
                else 0.5 + 0.01 * METHOD_MENU.index(method)
            }
            for center in ("0", "1", "2", "3", "5", "6", "7", "8", "9")
        }
        for method in METHOD_MENU
    }
    result = exact_shared_center_max_sign_flip(metrics)
    assert result["null_replicate_count"] == 512
    assert result["descriptive_only"] is True
    assert result["formal_claim_authorized"] is False
    assert result["nominal_significance_claimed"] is False

