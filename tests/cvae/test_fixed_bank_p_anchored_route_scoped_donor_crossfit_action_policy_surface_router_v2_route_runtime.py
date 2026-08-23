from __future__ import annotations

import hashlib
import pickle

import numpy as np

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.inventory import (
    ExpectedRouteInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.label_firewall import (
    LabelPhase,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.lifecycle import (
    PDCAPSLabelLifecycle,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.physical_actions import (
    B_ACTION_ID,
    U_ACTION_ID,
    action_library_by_target,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.physical_contracts import (
    CenterPhysicalSurface,
    PhysicalSurface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.route_support import (
    BinaryLabel,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.route_planning import (
    build_route_plan_inventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.route_runtime import (
    ZERO_DONOR_PRIOR_POLICY_ID,
    build_route_runtime,
    open_all_pseudo_responses,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_route_scoped_donor_crossfit_action_policy_surface_router.v2.viability import (
    MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS,
    build_canonical_bank_viability,
)
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import (
    CENTERS,
)


def _hash(value: object) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _center_surface(center: str) -> CenterPhysicalSurface:
    samples = tuple(
        f"{center}-{case}-{label}"
        for case in ("case-a", "case-b")
        for label in ("n", "p")
    )
    cases = tuple(
        f"case-{case}"
        for case in ("a", "a", "b", "b")
    )
    inverse = np.asarray([0.8, 0.2, 0.8, 0.2], dtype=np.float32)
    correct = np.asarray([0.2, 0.8, 0.2, 0.8], dtype=np.float32)
    arrays = []
    for action in action_library_by_target()[center]:
        base = (
            inverse
            if action.action_id in {B_ACTION_ID, U_ACTION_ID}
            else correct
        )
        seeds = np.stack(
            [
                np.clip(base + (seed - 4) * 0.001, 0.01, 0.99)
                for seed in range(9)
            ]
        ).astype(np.float32)
        arrays.append((action.action_id, seeds))
    return CenterPhysicalSurface(
        center,
        samples,
        cases,
        tuple(arrays),
        _hash("prediction-store"),
    )


def _physical_surface() -> PhysicalSurface:
    return PhysicalSurface(
        tuple(_center_surface(center) for center in CENTERS),
        _hash("prediction-store"),
    )


def _inventory(surface: PhysicalSurface) -> ExpectedRouteInventory:
    keys = tuple(
        (center, case_id, sample_id)
        for center in ("0", "1")
        for case_id, sample_id in zip(
            surface.center(center).case_ids,
            surface.center(center).sample_ids,
            strict=True,
        )
    )
    return ExpectedRouteInventory.focused_fixture(keys)


def _loader(keys, scope: str):
    return tuple(
        BinaryLabel(
            center,
            case_id,
            sample_id,
            0 if sample_id.endswith("-n") else 1,
            scope,
        )
        for center, case_id, sample_id in keys
    )


def test_route_runtime_builds_exact_h_j_d_inventory_and_binds_zero_prior() -> None:
    physical = _physical_surface()
    inventory = _inventory(physical)
    plans = build_route_plan_inventory(inventory, physical)
    assert len(plans.plans) == inventory.total_route_count == 8
    assert sum(row.route_key.surface_role == "target" for row in plans.plans) == 4
    assert sum(row.route_key.surface_role == "pseudo" for row in plans.plans) == 4
    assert all(
        row.endpoint_excluded_source_centers
        == (() if row.route_key.surface_role == "target" else (row.route_key.outer_center,))
        for row in plans.plans
    )

    target_viability = build_canonical_bank_viability("0")
    pseudo_viability = build_canonical_bank_viability(
        "0", excluded_source_centers=("1",)
    )
    assert target_viability.passed and pseudo_viability.passed
    assert target_viability.minimum_effective_sample_size == (
        MINIMUM_EFFECTIVE_SAMPLE_SIZE_PER_CLASS
    )
    assert dict(target_viability.rows)["B"].per_class_effective_sample_size == (
        ("0", 1024.0),
        ("1", 1024.0),
    )

    lifecycle = PDCAPSLabelLifecycle(
        _loader,
        protocol_hash=_hash("v2-protocol"),
        expected_inventory=inventory,
        require_derived_response_denominators=True,
    )
    runtime = build_route_runtime(
        physical_surface=physical,
        lifecycle=lifecycle,
        route_plans=plans,
    )
    assert lifecycle.phase == LabelPhase.ACTION_SURFACE_SEALED
    assert len(runtime.posterior_fits) == 2 * inventory.case_count == 8
    assert len(runtime.route_bindings) == inventory.total_route_count == 8
    assert runtime.surface_set.control_ids == (
        "IDENTITY",
        "WITHIN_CASE_CYCLIC_SHIFT",
    )
    assert all(
        row.donor_prior_policy.policy_id == ZERO_DONOR_PRIOR_POLICY_ID
        and all(value == 0.0 for _source, _direction, value in row.donor_prior_policy.values)
        for row in runtime.route_bindings
    )
    assert all(
        len(row.donor_prior_policy.values)
        == (16 if row.route_key.surface_role == "target" else 14)
        for row in runtime.route_bindings
    )
    restored = pickle.loads(pickle.dumps(runtime))
    assert restored.runtime_hash == runtime.runtime_hash

    responses = open_all_pseudo_responses(lifecycle, runtime)
    assert lifecycle.phase == LabelPhase.PSEUDO_RESPONSE
    assert len(responses.opened_route_keys) == inventory.pseudo_route_count
    assert tuple(control for control, _rows in responses.responses_by_control) == (
        "IDENTITY",
        "WITHIN_CASE_CYCLIC_SHIFT",
    )
    assert all(
        row.key.route_key.surface_role == "pseudo"
        for _control, rows in responses.responses_by_control
        for row in rows
    )
    assert len(responses.runtime_hash) == 64
