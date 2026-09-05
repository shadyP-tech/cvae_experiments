"""Regressions for the v20 scientific population and executed-action features."""
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.risk_aligned_router_v20 import (
    CompositeKind, Direction, LabelFreeAction, SurfaceRole, build_exact_u_composite,
    build_soft_topk_composite, float32_probability_hex, fit_action_outcome_model,
)
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.estimators import (
    fit_mean_ridge, fit_softmax_ridge, predict_softmax,
)
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.features import (
    RawFeatureCache, current_raw_feature_cache, exact_composite_features,
    fit_composite_feature_scope, use_raw_feature_cache,
)
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.outcome_targets import category_target
from test_harp_v20_science_foundation import candidates, make_menu, truth_for


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


def test_candidate_population_excludes_baseline_noops_and_rebalances_participating_cases():
    menus = (make_menu("a", center="A"), make_menu("b", center="A"), make_menu("c", center="B"),
             make_menu("noop", center="A", d01=(.3,.8), d10=(.2,.7), uniform=(.3,.7)))
    capability = truth_for(menus, ((0, 1),)*4)
    profiles, _ = capability.derive_training_surface(menus)
    composites = tuple(row.composite for menu in menus for row in candidates(menu) if row.eligible)
    truth = capability.score_composites(composites)
    model = fit_action_outcome_model(menus, composites, truth, normalization_profiles=profiles, **evidence_fit_arguments(menus))
    by_hash = {c.composite_hash: c for c in composites}
    totals = defaultdict(float)
    for hashed, weight in zip(model.training_composite_hashes, model.row_weights, strict=True):
        composite = by_hash[hashed]
        assert composite.kind is not CompositeKind.B and composite.prediction_changed
        totals[(composite.center_id, composite.case_id)] += weight
    assert totals == pytest.approx({("A", "a"): .25, ("A", "b"): .25, ("B", "c"): .5})
    assert model.training_case_keys == tuple(sorted((m.center_id, m.case_id) for m in menus))
    assert model.population_exclusions[0] == 4
    assert model.population_exclusions[1] > 0


def test_normalizer_and_magnitude_bound_include_cases_excluded_from_candidate_fit():
    menus = tuple(make_menu(f"c{i}", uniform=(.8,.2) if i < 3 else (.3,.7)) for i in range(4))
    capability = truth_for(menus, ((0,0),(0,0),(0,0),(0,1)))
    profiles, _ = capability.derive_training_surface(menus)
    composites = tuple(build_exact_u_composite(m) for m in menus)
    outcomes = capability.score_composites(composites)
    fitted = fit_action_outcome_model(menus, composites[:3], outcomes[:3], normalization_profiles=profiles, **evidence_fit_arguments(menus))
    assert fitted.gain_magnitude_bound == 2.5
    assert len(fitted.normalization_payload["case_keys"]) == 4
    with pytest.raises(ProtocolError, match="full-scope"):
        fit_action_outcome_model(menus, composites[:3], outcomes[:3])


def test_category_distribution_and_signed_gain_head_are_coherent():
    menus = tuple(make_menu(f"fit{i}") for i in range(6))
    capability = truth_for(menus, ((0,1),(1,0),(0,0),(1,1),(0,1),(1,0)))
    composites = tuple(row.composite for m in menus for row in candidates(m) if row.eligible)
    outcomes = capability.score_composites(composites)
    assert set(category_target(o) for o in outcomes) == {0, 1, 2}
    model = fit_action_outcome_model(menus, composites, outcomes, **evidence_fit_arguments(menus))
    target = make_menu("held", role=SurfaceRole.TARGET_EVALUATION)
    predicted = model.predict_composites(target, tuple(row.composite for row in candidates(target) if row.eligible))
    for p in predicted:
        assert p.safe_positive_probability + p.predicted_harm + p.remaining_probability == pytest.approx(1.)
        assert p.safe_gain_magnitude >= 0 and p.harm_gain_magnitude >= 0
        assert p.risk_adjusted_score == pytest.approx(p.predicted_gain)
        assert not p.public_payload()["risk_adjusted_score_is_confidence_bound"]
    with pytest.raises(ProtocolError, match="fitted case"):
        model.predict_composites(menus[0], (build_exact_u_composite(menus[0]),))


def test_empty_fit_has_no_fabricated_rows_and_cannot_propose_safe_benefit():
    menus = (make_menu("train0"), make_menu("train1"))
    capability = truth_for(menus, ((0,1),(0,1)))
    profiles, _ = capability.derive_training_surface(menus)
    model = fit_action_outcome_model(menus, (), (), normalization_profiles=profiles, **evidence_fit_arguments(menus))
    assert model.empty_population and model.row_weights == model.training_outcome_hashes == ()
    target = make_menu("held", role=SurfaceRole.TARGET_EVALUATION)
    predicted = model.predict_composites(target, (build_exact_u_composite(target),))[0]
    assert predicted.risk_adjusted_score == predicted.safe_positive_probability == 0
    assert predicted.predicted_harm == 1


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

from test_harp_v20_science_foundation import evidence_fit_arguments
