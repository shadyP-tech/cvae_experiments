"""Real pooled scientific fitting through two label-free process reconstructions."""
from __future__ import annotations

import numpy as np
from dataclasses import replace
from types import SimpleNamespace
import pytest

from test_harp_v18_support_runtime import _fit_config, _physical_menu, _source_labels
from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from midogpp_thesis.cvae.routing.case_conditional_composite_router_v18 import (
    CompositeKind, Direction, RouteDecision, build_soft_topk_composite, build_baseline_composite,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v18_execution.branch_recipe import validate_branch_recipe
from midogpp_thesis.cvae.runtime.harp_v18_execution.contracts import PrelabelRouteSet
from midogpp_thesis.cvae.runtime.harp_v18_execution.production import _bind_model_artifact, _target_action_artifact
from midogpp_thesis.cvae.runtime.harp_v18_execution.menu_root_binding import CenterMenuRootBinding
from midogpp_thesis.cvae.runtime.harp_v18_execution.support_model_artifacts import (
    build_support_outcome_artifact, build_support_router_artifact, build_support_target_routes,
)
from midogpp_thesis.cvae.runtime.harp_v18_execution.support_target_adapter import compile_support_target_menus, route_target_bundle
from midogpp_thesis.cvae.runtime.harp_v18_execution.stores import (
    write_artifact_value, write_label_free_outer_menu, write_prelabel_routes,
)
from midogpp_thesis.cvae.runtime.harp_v18_execution.support_validation import (
    POOLED_POLICY_ARTIFACT_ROLE, TARGET_EVALUATION_ACTION_ARTIFACT_ROLE,
    run_two_fresh_pooled_policy_validations,
)


@pytest.fixture(scope="module")
def real_fitted_surface():
    physical = tuple(_physical_menu(center) for center in CENTERS)
    bundles = tuple(compile_support_target_menus(menu) for menu in physical)
    source = build_support_outcome_artifact(bundles, {bundle.center_id: _source_labels(bundle) for bundle in bundles})
    fitted = build_support_router_artifact(source, config=_fit_config())
    assert fitted.state.policy.admitted
    config_hash = "9" * 64
    fitted = _bind_model_artifact(fitted, config_hash=config_hash, centers=CENTERS)
    target = _target_action_artifact(bundles, fitted, config_hash=config_hash, centers=CENTERS)
    routes = build_support_target_routes(bundles, fitted, target_action_hash=target.manifest["target_action_hash"])
    return physical, bundles, fitted, target, routes, config_hash


def _validate_durable(tmp_path, fixture, routes):
    physical, bundles, fitted, target, _, config_hash = fixture
    common = tmp_path / "physical"
    roots = {center: common / f"center_{center}" for center in CENTERS}
    receipts = tuple(write_label_free_outer_menu(roots[menu.outer_target_id], menu) for menu in physical)
    binding = CenterMenuRootBinding.create(common_parent=common, centers=CENTERS, menu_roots=roots, menus=physical, receipts=receipts)
    model_root, target_root, route_root = tmp_path / "model", tmp_path / "target", tmp_path / "routes"
    write_artifact_value(model_root, fitted, role=POOLED_POLICY_ARTIFACT_ROLE)
    write_artifact_value(target_root, target, role=TARGET_EVALUATION_ACTION_ARTIFACT_ROLE)
    write_prelabel_routes(route_root, routes)
    # Validators reconstruct public recipes and never deserialize the memory-only
    # source capability or open any target-label endpoint.
    first, second = run_two_fresh_pooled_policy_validations(
        route_root=route_root, menu_binding=binding, model_root=model_root,
        target_action_root=target_root, expected_center_ids=CENTERS,
        expected_config_hash=config_hash,
    )
    return first, second


def test_real_pooled_policy_survives_two_independent_label_free_reconstructions(tmp_path, real_fitted_surface):
    routes = real_fitted_surface[4]
    assert any(case.decision_payload["route_selected"] for case in routes.cases)
    first, second = _validate_durable(tmp_path, real_fitted_surface, routes)
    assert first["process_id"] != second["process_id"]
    assert first["reconstruction_hash"] == second["reconstruction_hash"]
    assert first["evaluation_labels_opened"] is second["evaluation_labels_opened"] is False


@pytest.mark.parametrize("kind", (CompositeKind.D01_ONLY, CompositeKind.D10_ONLY))
def test_directional_only_runtime_recipe_roundtrip(kind):
    bundle = compile_support_target_menus(_physical_menu(CENTERS[0]))
    menu = bundle.target_menus[0]
    composite = build_soft_topk_composite(
        menu, d01_ranked_actions=menu.actions_for(Direction.D01)[:2],
        d10_ranked_actions=menu.actions_for(Direction.D10)[:2],
        k=2, mixing_lambda=.5, kind=kind,
    )
    decision = RouteDecision(composite, composite.arm_id, .1, 0., "a" * 64, True)
    class FixedPolicy:
        policy_hash = "a" * 64
        model_hash = "b" * 64
        def route_menu(self, _menu):
            return decision
    case = route_target_bundle(bundle, FixedPolicy(), decisions=(decision,))[0]
    assert case.routed_probabilities.tobytes() == b"".join(bytes.fromhex(value) for value in composite.probability_hex)


def test_both_directional_only_recipes_survive_fresh_process_validation(tmp_path, real_fitted_surface):
    # A reconstruction fixture deliberately controls the recipe. This does not
    # claim that the learned model selected these branches; that separate
    # model-to-route path is exercised in the preceding integration test.
    _, bundles, fitted, target, original, _ = real_fitted_surface
    cases = []
    for index, bundle in enumerate(bundles):
        menu = bundle.target_menus[0]
        if index < 2:
            composite = build_soft_topk_composite(menu,
                d01_ranked_actions=menu.actions_for(Direction.D01)[:2],
                d10_ranked_actions=menu.actions_for(Direction.D10)[:2], k=2, mixing_lambda=.5,
                kind=CompositeKind.D01_ONLY if index==0 else CompositeKind.D10_ONLY)
            decision = RouteDecision(composite, composite.arm_id,.1,0.,fitted.state.policy.policy_hash,True)
        else:
            composite = build_baseline_composite(menu)
            decision = RouteDecision(composite,"B",0.,0.,fitted.state.policy.policy_hash,True,"FIXTURE_B")
        cases.extend(route_target_bundle(bundle,fitted.state.policy,decisions=(decision,)))
    routes = replace(original,cases=tuple(cases))
    first,second = _validate_durable(tmp_path,real_fitted_surface,routes)
    assert first["reconstruction_hash"]==second["reconstruction_hash"]


@pytest.mark.parametrize("tamper", ("direction", "component_id", "family", "unused_branch"))
def test_directional_recipe_tampering_is_rejected(tamper):
    baseline=np.asarray((.2,.3,.7,.8),dtype=np.float32)
    component=np.asarray((.2,.8,.7,.8),dtype=np.float32)
    kwargs=dict(direction="D01",component_ids=("HXE:1:D01",),components=(component,),
                baseline=baseline,routed=component.copy(),
                payload={"composite_kind":"D01_ONLY","composite_k":1},require_family=True)
    if tamper=="direction": kwargs["direction"]="D10"
    if tamper=="component_id": kwargs["component_ids"]=("HXE:1:D10",)
    if tamper=="family": kwargs["payload"]={"composite_kind":"BOTH","composite_k":1}
    if tamper=="unused_branch": kwargs["routed"][2]=.1
    with pytest.raises(ProtocolError): validate_branch_recipe(**kwargs)
