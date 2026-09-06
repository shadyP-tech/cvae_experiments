"""Regressions for the v21 scientific population and executed-action features."""
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.correction_mass_router_v21 import (
    CompositeKind, Direction, LabelFreeAction, SurfaceRole, build_exact_u_composite,
    build_soft_topk_composite, float32_probability_hex, fit_action_outcome_model,
)
from midogpp_thesis.cvae.routing.correction_mass_router_v21.estimators import (
    fit_mean_ridge, fit_softmax_ridge, predict_softmax,
)
from midogpp_thesis.cvae.routing.correction_mass_router_v21.features import (
    RawFeatureCache, current_raw_feature_cache, exact_composite_features,
    fit_composite_feature_scope, use_raw_feature_cache,
)
from midogpp_thesis.cvae.routing.correction_mass_router_v21.outcome_targets import category_target
from test_harp_v21_science_foundation import candidates, make_menu, truth_for


def test_features_measure_executed_direction_and_ignore_unexecuted_primitive_statistics():
    menu = make_menu(baseline=(.4, .6), d01=(.7, .6), d10=(.4, .3), uniform=(.7, .3))
    menu = replace(menu, actions=tuple(replace(action,
        feature_names=("threshold_flip_fraction", "surface_boundary_distance_min", "compatibility"),
        feature_values=(1., 999., 1.)) for action in menu.actions))
    composite = build_soft_topk_composite(menu, d01_ranked_actions=("D01::E",),
        d10_ranked_actions=("D10::E",), k=1, mixing_lambda=1., kind=CompositeKind.D01_ONLY)
    features = exact_composite_features(menu, composite)
    assert features["hard_change_fraction"] == .5
    assert features["d01_flip_count"] == 1
    assert features["d10_flip_count"] == 0
    assert features["d10_flip_delta_mean"] == 0
    assert features["action_margin_min"] == pytest.approx(.1, abs=1e-7)
    assert fit_composite_feature_scope((menu,)).feature_names == ("compatibility",)


def test_two_donor_mixture_statistics_are_not_averaged_primitive_statistics():
    menu = make_menu(baseline=(.2, .2), d01=(.9, .2), d10=None, uniform=(.8, .8))
    first = menu.actions_for(Direction.D01)[0]
    second = replace(first, arm_id="D01::F", donor_id="F", action_probability_hex=float32_probability_hex((.2, .9)))
    menu = replace(menu, actions=(*menu.actions, second))
    composite = build_soft_topk_composite(menu, d01_ranked_actions=("D01::E", "D01::F"),
        d10_ranked_actions=(), k=2, mixing_lambda=1., kind=CompositeKind.D01_ONLY)
    features = exact_composite_features(menu, composite)
    assert features["d01_flip_count"] == 2
    assert features["delta_std"] == 0
    assert features["d01_flip_delta_std"] == 0
    assert features["d01_action_margin_median"] == pytest.approx(.05, abs=1e-7)
    assert features["d01_selected_donor_std_on_flips"] == pytest.approx(.35, abs=1e-7)
    assert features["d01_selected_donor_disagreement_on_flips"] == .5


def test_candidate_population_retains_full_scope_weights_after_filtering():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.outcome_targets import prepare_candidate_population
    menus = (make_menu('a',center='A'),make_menu('b',center='A'),make_menu('c',center='B'),
             make_menu('noop',center='A',d01=(.3,.8),d10=(.2,.7),uniform=(.3,.7)))
    capability=truth_for(menus,((0,1),)*4)
    profiles,_=capability.derive_training_surface(menus)
    composites=tuple(r.composite for m in menus for r in candidates(m) if r.eligible)
    population=prepare_candidate_population(menus,composites,capability.score_composites(composites),
                                             normalization_profiles=profiles)
    totals=defaultdict(float)
    for c,w in zip(population.composites,population.row_weights,strict=True):
        totals[(c.center_id,c.case_id)]+=w
    assert totals == pytest.approx({('A','a'):.2,('A','b'):.2,('B','c'):.6})
    assert population.excluded_baseline_count == 4
    assert population.excluded_no_hard_change_count > 0


def test_outcome_prediction_obeys_gain_identity_and_cannot_claim_harm_probability():
    menus=tuple(make_menu(f'fit{i}') for i in range(4))
    model=fit_action_outcome_model(menus,**evidence_fit_arguments(menus))
    held=make_menu('held',role=SurfaceRole.TARGET_EVALUATION)
    predicted=model.predict_composites(held,tuple(r.composite for r in candidates(held) if r.eligible))
    for p in predicted:
        assert p.predicted_gain == pytest.approx(.5*(p.predicted_class_0_gain+p.predicted_class_1_gain))
        assert not p.public_payload()['candidate_harm_probability_estimated']
        assert not p.public_payload()['risk_adjusted_score_is_confidence_bound']
    with pytest.raises(ProtocolError,match='fitted case'):
        model.predict_composites(menus[0],(build_exact_u_composite(menus[0]),))


def test_action_and_evidence_payload_reconstruct_and_reject_tampering():
    import copy,json
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.outcome_model import ActionOutcomeModel
    menus=(make_menu('a'),make_menu('b'))
    model=fit_action_outcome_model(menus,**evidence_fit_arguments(menus))
    payload=json.loads(json.dumps(model.public_payload()))
    assert ActionOutcomeModel.from_payload(payload).model_hash == model.model_hash
    poison=copy.deepcopy(payload);poison['patch_evidence_model']['mass_coefficients'][0][0]+=1.
    with pytest.raises(ProtocolError,match='seal differs'):
        ActionOutcomeModel.from_payload(poison)


def test_expected_effect_algebra_matches_dependent_joint_labels_including_missing_classes():
    from itertools import product
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.outcome_model import effect_arrays
    labels=np.asarray(tuple(product((0,1),repeat=4)))
    # Deliberately dependent labels, with all-zero/all-one configurations retained.
    weights=np.arange(1,17,dtype=float)**2;weights/=weights.sum()
    mass=[]
    for y in labels:
        row=np.zeros((4,2))
        for k,v in ((0,1.25),(1,2.5)):
            count=np.sum(y==k)
            if count:
                row[y==k,k]=v/count
        mass.append(row)
    baseline=np.asarray([.1,.6,.4,.9])
    actions=np.asarray(tuple(product((.2,.8),repeat=4)))
    estimated=effect_arrays(baseline,actions,weights@labels,np.einsum('i,ijk->jk',weights,mass))
    expected=[np.zeros(len(actions)) for _ in range(5)]
    for y,w,m in zip(labels,weights,mass,strict=True):
        # Evaluate realized recalls and losses directly, with arbitrary dependence.
        hard=actions>=.5;basehard=baseline>=.5
        g=[]
        for k,v in ((0,1.25),(1,2.5)):
            mask=y==k
            g.append(v*np.mean((hard[:,mask]==k).astype(float)-(basehard[mask]==k).astype(float),axis=1)
                     if mask.any() else np.zeros(len(actions)))
        observed=[.5*(g[0]+g[1]),*g,
                  np.mean((actions-y)**2-(baseline-y)**2,axis=1),
                  np.mean(-y*np.log(actions)-(1-y)*np.log1p(-actions)
                          +y*np.log(baseline)+(1-y)*np.log1p(-baseline),axis=1)]
        for j,values in enumerate(observed):
            expected[j]+=w*values
    assert np.asarray(estimated) == pytest.approx(np.asarray(expected),abs=1e-14)
    noop=effect_arrays(baseline,np.array([[.2,.7,.3,.8]]),weights@labels,np.einsum('i,ijk->jk',weights,mass))
    assert noop[0][0] == noop[1][0] == noop[2][0] == 0.
    assert noop[3][0] != 0 and noop[4][0] != 0


def test_mean_loss_regularization_is_invariant_to_weight_rescaling():
    matrix = np.column_stack((np.ones(6), np.arange(6)-2.5))
    response = np.asarray([0., 0., .2, .6, 1., 1.])
    weights = np.asarray([1., 2., 1., 3., 1., 2.])
    assert fit_mean_ridge(matrix, response, weights) == pytest.approx(fit_mean_ridge(matrix, response, weights*100))
    categories = np.asarray([1,1,2,2,0,0])
    first = fit_softmax_ridge(matrix, categories, weights)
    second = fit_softmax_ridge(matrix, categories, weights*100)
    assert predict_softmax(matrix, first) == pytest.approx(predict_softmax(matrix, second))


def test_raw_feature_cache_is_bounded_pure_and_scoped_to_one_execution():
    menu = make_menu("cache")
    composite = build_exact_u_composite(menu)
    expected = exact_composite_features(menu, composite)
    first = RawFeatureCache(max_entries=1)
    with use_raw_feature_cache(first):
        actual = exact_composite_features(menu, composite)
        actual["hard_change_fraction"] = -999  # Caller cannot mutate cached data.
        assert exact_composite_features(menu, composite) == expected
        other = make_menu("another")
        exact_composite_features(other, build_exact_u_composite(other))
        assert first.public_payload()["entry_count"] == 1
        assert first.public_payload()["hits"] == 1
    assert current_raw_feature_cache() is None
    second = RawFeatureCache()
    with use_raw_feature_cache(second):
        assert exact_composite_features(menu, composite) == expected
        assert second.public_payload()["hits"] == 0
        assert second.public_payload()["misses"] == 1

from test_harp_v21_science_foundation import evidence_fit_arguments
