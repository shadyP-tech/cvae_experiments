"""Regressions for evidence isolation, risk-aware winners, and sparse coverage."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.risk_aligned_router_v20 import RouterFitConfig, SurfaceRole
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.candidate_prediction import CandidatePrediction, unthresholded_winner
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.fit_cache import ScopedFitCache
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.frontier import failed_constraints
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.patch_evidence import (
    evidence_descriptor, seal_patch_evidence, sketch_virchow2,
)
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.proposer import fit_proposer
from midogpp_thesis.cvae.routing.risk_aligned_router_v20.risk_selection import apply_risk_penalty
from test_harp_v20_science_foundation import make_menu, truth_for, build_exact_u_composite
from test_harp_v20_nested_policy import _source_surface, _config, _prediction


def test_fixed_sketch_is_row_local_and_does_not_fit_on_target_features():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(5,3840)).astype(np.float32)
    target = rng.normal(size=(3,3840)).astype(np.float32)
    saved = source.copy()
    expected = sketch_virchow2(source)
    assert np.array_equal(expected, sketch_virchow2(np.concatenate((source,target)))[:5])
    assert np.array_equal(expected, sketch_virchow2(np.concatenate((source,target*100)))[:5])
    assert np.array_equal(source,saved)
    assert expected.shape == (5,64) and expected.dtype == np.float32
    with pytest.raises(ProtocolError,match='Virchow2_3840'):
        sketch_virchow2(source[:,:64])


def test_held_labels_cannot_shape_patch_evidence_and_fitted_case_is_rejected():
    menus = tuple(make_menu(c) for c in ('a','b','held'))
    first = truth_for(menus,((0,1),(0,1),(0,1)))
    poison = truth_for(menus,((0,1),(0,1),(1,0)))
    a = first.scoped(menus[:2]).fit_patch_evidence(menus[:2])
    b = poison.scoped(menus[:2]).fit_patch_evidence(menus[:2])
    assert a.model_hash == b.model_hash
    held = seal_patch_evidence(a,menus[2])
    assert (menus[2].center_id,menus[2].case_id) not in held.training_case_keys
    with pytest.raises(ProtocolError,match='fitted case'):
        a.predict(menus[0])
    with pytest.raises(ProtocolError,match='leaked its case'):
        replace(held,training_case_keys=(*a.training_case_keys,held.case_key))
    wrong = replace(menus[0],surface_role=SurfaceRole.TARGET_EVALUATION,
                    actions=tuple(replace(v,surface_role=SurfaceRole.TARGET_EVALUATION) for v in menus[0].actions))
    with pytest.raises(ProtocolError):
        first.scoped((wrong,))
    scores = first.score_patch_controls((held,))
    assert scores[0]['used_for_policy_selection'] is False
    assert scores[0]['control_prediction_hash'] == held.prediction_hash
    altered = replace(menus[2],patch_features=tuple(tuple(9. for _ in range(64)) for _ in menus[2].sample_ids))
    with pytest.raises(ProtocolError,match='exact scoped capability'):
        first.fit_patch_evidence((*menus[:2],altered))
    with pytest.raises(ProtocolError,match='menu changed'):
        first.scoped((*menus[:2],altered)).score_patch_controls((held,))


def test_patch_expected_proper_losses_include_nonflipping_probability_changes():
    menu = make_menu(uniform=(.3,.7))
    composite = build_exact_u_composite(menu)
    p = np.asarray([.7,.3])
    values = evidence_descriptor(menu,composite,p)
    assert values[:4] == (0.,0.,0.,0.)
    # Enumerate the conditional Bernoulli loss rather than repeat the formula.
    from midogpp_thesis.cvae.routing.risk_aligned_router_v20.contracts import decode_probability_hex
    base=np.asarray(decode_probability_hex(menu.baseline_probability_hex))
    action=np.asarray(decode_probability_hex(composite.probability_hex))
    expected_brier=sum(np.mean(weight*((action-y)**2-(base-y)**2)) for y,weight in ((1,p),(0,1-p)))
    expected_log=sum(np.mean(weight*(-np.log(action if y else 1-action)+np.log(base if y else 1-base)))
                     for y,weight in ((1,p),(0,1-p)))
    assert values[4] == pytest.approx(expected_brier)
    assert values[5] == pytest.approx(expected_log)
    assert values[4] != 0 and values[5] != 0


def test_risk_penalty_changes_winner_before_gate_instead_of_only_vetoing():
    menus,_ = _source_surface()
    raw = [r for r in _prediction(menus[0]) if r.eligible_for_winner][:2]
    estimates = [replace(raw[0].prediction,predicted_gain=.03,predicted_harm=.8,
                         predicted_brier_delta=.02,predicted_logloss_delta=.04),
                 replace(raw[1].prediction,predicted_gain=.02,predicted_harm=.1,
                         predicted_brier_delta=-.01,predicted_logloss_delta=-.02)]
    candidates=tuple(CandidatePrediction(row.candidate,replace(p,risk_adjusted_score=p.predicted_gain))
                     for row,p in zip(raw,estimates,strict=True))
    assert unthresholded_winner(candidates).arm_id == raw[0].arm_id
    adjusted=tuple(CandidatePrediction(row.candidate,apply_risk_penalty(p,1.))
                   for row,p in zip(raw,estimates,strict=True))
    assert unthresholded_winner(adjusted).arm_id == raw[1].arm_id


def test_proposer_cache_reuses_only_scope_invariant_fits_and_keeps_baseline_zero():
    menus,cap = _source_surface(centers=2,cases_per_center=3)
    config=_config();cache=ScopedFitCache()
    stronger=replace(config,risk_penalty_scale=2.)
    a=fit_proposer(menus,cap,config=config,cache=cache)
    assert fit_proposer(menus,cap,config=stronger,cache=cache) is a
    assert cache.key('complete_learner',menus,cap,config) != cache.key('complete_learner',menus,cap,stronger)
    held = replace(menus[0],case_id='external',actions=tuple(replace(v,case_id='external') for v in menus[0].actions))
    baseline=next(r for r in a.candidate_predictions(held,stronger) if r.arm_id=='B')
    assert baseline.risk_adjusted_score == 0 and not baseline.eligible_for_winner


def test_two_safe_cases_cannot_satisfy_declared_inner_selection_coverage():
    records=tuple(SimpleNamespace(center_id='C0',route_selected=True,bacc_gain=.1,harm=False,
                      brier_delta=-.01,log_loss_delta=-.01) for _ in range(2))
    assert not failed_constraints(records)
    failed=failed_constraints(records,minimum_cases=18,minimum_centers=6,minimum_cases_per_center=2)
    assert set(failed)=={'INSUFFICIENT_ROUTED_CASES','INSUFFICIENT_CENTER_COVERAGE'}


def test_risk_scale_outside_predeclared_grid_is_rejected():
    with pytest.raises(ProtocolError):
        replace(RouterFitConfig(),risk_penalty_scale=100.)


def test_physical_patch_features_roundtrip_and_poisoned_bytes_fail_closed(tmp_path):
    from test_harp_v20_support_runtime import _physical_menu
    from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
    from midogpp_thesis.cvae.runtime.harp_v20_execution.stores import write_label_free_outer_menu, read_label_free_outer_menu
    menu=_physical_menu(CENTERS[0],source_cases=2)
    receipt=write_label_free_outer_menu(tmp_path/'menu',menu)
    restored=read_label_free_outer_menu(tmp_path/'menu')
    assert restored.menu_hash==menu.menu_hash
    for role in ('source_train','target'):
        assert np.array_equal(restored.patch_features[role],menu.patch_features[role])
        assert not restored.patch_features[role].flags.writeable
    with np.load(receipt.npz_path,allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    arrays['patch_source_train'][0,0]=123.
    np.savez_compressed(receipt.npz_path,**arrays)
    with pytest.raises(ProtocolError):
        read_label_free_outer_menu(tmp_path/'menu')


def test_adapter_rejects_missing_patch_surface_before_any_label_capability():
    from test_harp_v20_support_runtime import _physical_menu
    from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
    from midogpp_thesis.cvae.runtime.harp_v20_execution.support_target_adapter import compile_support_target_menus
    menu=replace(_physical_menu(CENTERS[0]),patch_features={})
    with pytest.raises(ProtocolError,match='seal before source labels'):
        compile_support_target_menus(menu)
