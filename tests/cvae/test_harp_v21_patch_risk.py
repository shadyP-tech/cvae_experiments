"""Regressions for evidence isolation, risk-aware winners, and sparse coverage."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.correction_mass_router_v21 import RouterFitConfig, SurfaceRole
from midogpp_thesis.cvae.routing.correction_mass_router_v21.candidate_prediction import CandidatePrediction, unthresholded_winner
from midogpp_thesis.cvae.routing.correction_mass_router_v21.fit_cache import ScopedFitCache
from midogpp_thesis.cvae.routing.correction_mass_router_v21.frontier import failed_constraints
from midogpp_thesis.cvae.routing.correction_mass_router_v21.patch_evidence import (
    evidence_descriptor, seal_patch_evidence, sketch_virchow2,
)
from midogpp_thesis.cvae.routing.correction_mass_router_v21.proposer import fit_proposer
from test_harp_v21_science_foundation import make_menu, truth_for, build_exact_u_composite


def test_canonical_full_features_are_row_local_and_do_not_fit_on_target_features():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(5,3840)).astype(np.float32)
    target = rng.normal(size=(3,3840)).astype(np.float32)
    saved = source.copy()
    expected = sketch_virchow2(source)
    assert np.array_equal(expected, sketch_virchow2(np.concatenate((source,target)))[:5])
    assert np.array_equal(expected, sketch_virchow2(np.concatenate((source,target*100)))[:5])
    assert np.array_equal(source,saved)
    assert np.array_equal(source,expected) and not expected.flags.writeable
    assert expected.shape == (5,3840) and expected.dtype == np.float32
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
    altered = replace(menus[2],patch_features=tuple(tuple(9. for _ in range(3840)) for _ in menus[2].sample_ids))
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
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.contracts import decode_probability_hex
    base=np.asarray(decode_probability_hex(menu.baseline_probability_hex))
    action=np.asarray(decode_probability_hex(composite.probability_hex))
    expected_brier=sum(np.mean(weight*((action-y)**2-(base-y)**2)) for y,weight in ((1,p),(0,1-p)))
    expected_log=sum(np.mean(weight*(-np.log(action if y else 1-action)+np.log(base if y else 1-base)))
                     for y,weight in ((1,p),(0,1-p)))
    assert values[4] == pytest.approx(expected_brier)
    assert values[5] == pytest.approx(expected_log)
    assert values[4] != 0 and values[5] != 0


def test_sparse_full_embedding_signal_improves_held_correction_identification():
    """Held cases have new identities; signal is one retained coordinate only."""
    def informative(case, labels, role=SurfaceRole.SOURCE_TRAIN_DEVELOPMENT):
        menu=make_menu(case,role=role)
        features=np.zeros((2,3840),dtype=np.float32)
        features[:,3179]=2*np.asarray(labels)-1
        return replace(menu,patch_features=features)
    labels=tuple((0,1) if i%2 else (1,0) for i in range(16))
    menus=tuple(informative(f'train{i}',y) for i,y in enumerate(labels))
    capability=truth_for(menus,labels)
    held=informative('external',(1,0),SurfaceRole.TARGET_EVALUATION)
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.outcome_model import ActionOutcomeModel
    predicted={}
    for variant in ('baseline','calibrated_baseline','embedding_residual'):
        evidence=capability.fit_correction_evidence(menus,variant=variant)
        predicted[variant]=ActionOutcomeModel(evidence).predict_composites(held,(build_exact_u_composite(held),))[0]
        if variant=='baseline':
            from midogpp_thesis.cvae.routing.correction_mass_router_v21.contracts import decode_probability_hex
            assert np.array_equal(evidence.predict(held),decode_probability_hex(held.baseline_probability_hex))
    assert predicted['embedding_residual'].predicted_gain > .2
    assert predicted['baseline'].predicted_gain < 0
    assert predicted['calibrated_baseline'].predicted_gain < 0
    assert abs(predicted['embedding_residual'].predicted_gain-1) < abs(predicted['baseline'].predicted_gain-1)
    assert predicted['embedding_residual'].predicted_brier_delta < 0


def test_correction_targets_include_source_support_factor_and_singleclass_zero():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.evidence.targets import correction_targets
    menus=tuple(make_menu(f'c{i}') for i in range(3))
    y=tuple(np.asarray(row) for row in ((0,0),(0,0),(0,1)))
    total,q,v=correction_targets(menus,y)
    assert total == pytest.approx(np.array([[1,0],[1,0],[1,3]]))
    assert v == pytest.approx(np.array([[1,3],[1,3],[1,3]]))
    assert q[0] == pytest.approx(np.array([[.5,0],[.5,0]]))
    assert np.sum(q[2]*total[2],axis=0) == pytest.approx([1,3])


def test_missing_features_fail_before_label_callback():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.patch_evidence import fit_correction_evidence
    menus=(replace(make_menu('a'),patch_features=()),)
    reads=[]
    def labels(menu):
        reads.append(menu.case_id)
        return np.asarray([0,1])
    with pytest.raises(ProtocolError,match='sealed patch features'):
        fit_correction_evidence(menus,labels,variant='baseline')
    assert reads == []


def test_inference_uses_neither_center_identity_nor_case_identity_as_features():
    menus=tuple(make_menu(case) for case in ('a','b'))
    capability=truth_for(menus,((0,1),(1,0)))
    for variant in ('baseline','calibrated_baseline','embedding_residual'):
        model=capability.fit_correction_evidence(menus,variant=variant)
        one=make_menu('external',center='unseen_center',role=SurfaceRole.TARGET_EVALUATION)
        two=make_menu('another',center='different_center',role=SurfaceRole.TARGET_EVALUATION)
        assert np.array_equal(model.predict(one),model.predict(two))
        assert np.array_equal(model.predict_masses(one),model.predict_masses(two))
        assert model.public_payload()['center_id_is_model_feature'] is False


def test_mass_allocation_objective_additionally_weights_source_support(monkeypatch):
    """Rare-support cases must change the allocation fit's population mixture."""
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.evidence import fitting
    menus=(make_menu('a',baseline=(.1,.8),d10=(.1,.2)),make_menu('b',baseline=(.1,.8),d10=(.1,.2)),
           make_menu('c',center='C1',baseline=(.1,.8),d10=(.1,.2)))
    labels=((0,0),(0,1),(1,0))
    capability=truth_for(menus,labels)
    values=[]
    def intercept_objective(objective,width):
        values.append(objective(np.zeros(width))[0])
        return np.zeros(width)
    monkeypatch.setattr(fitting,'_minimize',intercept_objective)
    capability.fit_correction_evidence(menus,variant='baseline')
    # Class-one fitting cases: C0:b has baseweight1/4 ×v2 =1/2;
    # C1:c has baseweight1/2 ×v1 =1/2. Their normalized targets
    # select opposite patches. Missing-class C0:a supplies no u target.
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.contracts import decode_probability_hex
    b=np.asarray(decode_probability_hex(menus[0].baseline_probability_hex));u=b/b.sum()
    expected=-.5*np.log(u[1])-.5*np.log(u[0])
    naive=-(1/3)*np.log(u[1])-(2/3)*np.log(u[0])
    assert values[1] == pytest.approx(expected)
    assert abs(values[1]-naive) > .1


def test_held_correction_diagnostics_seal_masses_and_flip_conditioned_evidence():
    menus=tuple(make_menu(case) for case in ('a','b','held'))
    capability=truth_for(menus,((0,1),(0,1),(1,0)))
    model=capability.scoped(menus[:2]).fit_correction_evidence(menus[:2],variant='baseline')
    held=seal_patch_evidence(model,menus[2])
    scored=capability.scoped((menus[2],)).score_patch_controls((held,),composites=(build_exact_u_composite(menus[2]),))[0]
    evidence=scored['correction_evidence'];diagnostics=scored['correction_diagnostics']
    assert evidence['normalized_masses'] == held.normalized_masses
    assert diagnostics['union_d01_flip_calibration']['observed_correctness'] == 1.
    assert diagnostics['union_d10_flip_calibration']['observed_correctness'] == 1.
    assert diagnostics['sealed_action_effect_diagnostics'][0]['observed_gain'] == 1.
    assert diagnostics['used_for_policy_selection'] is False
    with pytest.raises(ProtocolError,match='malformed'):
        replace(held,normalized_masses=((-1.,0.),(1.,1.)))


def test_two_safe_cases_cannot_satisfy_declared_inner_selection_coverage():
    records=tuple(SimpleNamespace(center_id='C0',route_selected=True,bacc_gain=.1,harm=False,
                      brier_delta=-.01,log_loss_delta=-.01) for _ in range(2))
    assert not failed_constraints(records)
    failed=failed_constraints(records,minimum_cases=18,minimum_centers=6,minimum_cases_per_center=2)
    assert set(failed)=={'INSUFFICIENT_ROUTED_CASES','INSUFFICIENT_CENTER_COVERAGE'}


def test_physical_patch_features_roundtrip_and_poisoned_bytes_fail_closed(tmp_path):
    from test_harp_v21_support_runtime import _physical_menu
    from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
    from midogpp_thesis.cvae.runtime.harp_v21_execution.stores import write_label_free_outer_menu, read_label_free_outer_menu
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
    from test_harp_v21_support_runtime import _physical_menu
    from midogpp_thesis.cvae.expert_bank.uniform_b_v2_promotion.contracts import CENTERS
    from midogpp_thesis.cvae.runtime.harp_v21_execution.support_target_adapter import compile_support_target_menus
    menu=replace(_physical_menu(CENTERS[0]),patch_features={})
    with pytest.raises(ProtocolError,match='seal before source labels'):
        compile_support_target_menus(menu)
