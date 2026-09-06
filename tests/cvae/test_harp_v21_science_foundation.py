from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.correction_mass_router_v21 import (
    CompositeKind, Direction, LabelFreeAction, LabelFreeCaseMenu, RouterFitConfig,
    SealedOOFSelection, SupportTruthCapability, SurfaceRole, build_baseline_composite,
    build_candidate_composites, build_exact_u_composite, build_soft_topk_composite,
    float32_probability_hex, fit_action_outcome_model, fit_proposal_model,
)
from midogpp_thesis.cvae.routing.case_equal_metrics import (
    aggregate_case_equal_metrics, case_class_support_counts, case_metrics,
)


def make_menu(case='case', *, center='C0', baseline=(.2,.8), d01=(.8,.8),
              d10=(.2,.2), uniform=(.8,.2), role=SurfaceRole.SOURCE_TRAIN_DEVELOPMENT):
    sample_ids=tuple(f'{center}:{case}:s{i}' for i in range(len(baseline)))
    base=float32_probability_hex(baseline)
    actions=[]
    for arm,direction,donor,proba in [('D01::E',Direction.D01,'E',d01),
                                     ('D10::E',Direction.D10,'E',d10),
                                     ('U_FULL',Direction.FULL,None,uniform)]:
        if proba is not None:
            actions.append(LabelFreeAction(role,center,case,arm,direction,donor,
                ('margin_shift','compatibility'),(float(sum(proba)-sum(baseline)),1.),
                sample_ids,base,float32_probability_hex(proba)))
    return LabelFreeCaseMenu(role,center,case,sample_ids,base,tuple(actions),tuple(tuple(float(i%2) for _ in range(3840)) for i in range(len(sample_ids))))


def truth_for(menus, labels):
    return SupportTruthCapability({(m.center_id,m.case_id):dict(zip(m.sample_ids,y,strict=True))
                                   for m,y in zip(menus,labels,strict=True)}).scoped(menus)


def ranked(menu):
    return SimpleNamespace(d01_ranked_action_ids=tuple(a.arm_id for a in menu.actions_for(Direction.D01)),
                           d10_ranked_action_ids=tuple(a.arm_id for a in menu.actions_for(Direction.D10)))


def candidates(menu,config=None):
    cfg=config or RouterFitConfig(k_values=(1,),lambda_values=(1.,),required_source_case_count=None,
                                  required_source_center_count=None)
    return build_candidate_composites(menu,ranked(menu),cfg)


def seal(composite,index=0):
    return SealedOOFSelection(index,composite,composite.arm_id,.1,0.,(('TRAIN','other'),),'1'*64)


@pytest.mark.parametrize('kind,expected',[(CompositeKind.D01_ONLY,(.8,.8)),
                                         (CompositeKind.D10_ONLY,(.2,.2)),
                                         (CompositeKind.BOTH,(.8,.2))])
def test_branchwise_composites_preserve_unselected_baseline_bytes(kind,expected):
    menu=make_menu()
    result=build_soft_topk_composite(menu,d01_ranked_actions=('D01::E',),d10_ranked_actions=('D10::E',),
                                    k=1,mixing_lambda=1.,kind=kind)
    assert result.probability_hex == float32_probability_hex(expected)
    if kind is CompositeKind.D01_ONLY:
        assert result.probability_hex[1] == menu.baseline_probability_hex[1]
        assert not result.d10_action_ids
    if kind is CompositeKind.D10_ONLY:
        assert result.probability_hex[0] == menu.baseline_probability_hex[0]
        assert not result.d01_action_ids


def test_unavailable_d10_does_not_eliminate_d01_or_another_case():
    limited=make_menu('limited',d10=None)
    rows={r.arm_id:r for r in candidates(limited)}
    assert rows['D01_ONLY_K1_L1.00'].eligible
    assert rows['D10_ONLY_K1_L1.00'].ineligible_reason == 'INSUFFICIENT_D10_ACTIVE_DONORS'
    assert rows['BOTH_K1_L1.00'].ineligible_reason == 'INSUFFICIENT_D10_ACTIVE_DONORS'
    assert any(r.eligible and r.kind is CompositeKind.D10_ONLY for r in candidates(make_menu('other')))


def test_candidate_deduplication_is_exact_and_label_free():
    rows=candidates(make_menu())
    both=next(r for r in rows if r.kind is CompositeKind.BOTH)
    assert both.duplicate_of == 'U_FULL'  # exact uniform precedes identical BOTH
    assert not both.eligible
    assert len(rows)==5
    assert len(candidates(make_menu(),RouterFitConfig()))==38


def test_malformed_ranking_identity_is_not_silently_called_infeasible():
    menu=make_menu()
    with pytest.raises(ProtocolError,match='lacks the requested arm'):
        build_candidate_composites(menu,SimpleNamespace(d01_ranked_action_ids=('foreign',),
            d10_ranked_action_ids=()),RouterFitConfig())


def test_signed_outcomes_retain_harm_and_proper_losses():
    menu=make_menu()
    cap=truth_for((menu,),((0,1),))
    profiles,primitives=cap.derive_training_surface((menu,))
    assert all(o.bacc_gain<0 and o.brier_delta>0 and o.log_loss_delta>0 for o in primitives)
    comp=build_exact_u_composite(menu)
    outcome=cap.score_composites((comp,))[0]
    assert outcome.bacc_gain == -1
    assert outcome.class_0_gain == outcome.class_1_gain == -1
    assert outcome.harmed and not outcome.safe_positive


def test_singleclass_source_gain_matches_terminal_not_mean_of_case_bacc():
    # A single-class case loses recall while a mixed case gains another class.
    # Legacy mean-case BACC gives +.25; aligned supporting-case BACC gives -.25.
    single=make_menu('single',baseline=(.8,),d01=None,d10=(.2,),uniform=(.2,))
    mixed=make_menu('mixed',baseline=(.2,.8),d01=None,d10=(.2,.2),uniform=(.2,.2))
    menus=(single,mixed)
    labels=((0,),(0,1))
    cap=truth_for(menus,labels)
    selections=tuple(seal(build_exact_u_composite(m),i) for i,m in enumerate(menus))
    records=cap.score_selections(selections)
    gain=np.mean([r.bacc_gain for r in records])
    support=case_class_support_counts(labels)
    metrics={}
    for name in ('baseline_probability_hex','probability_hex'):
        probs=[np.asarray([np.frombuffer(bytes.fromhex(x),dtype='<f4')[0] for x in getattr(s.composite,name)])
               for s in selections]
        metrics[name]=aggregate_case_equal_metrics(tuple(case_metrics(p,y,total_case_count=2,
            class_support_case_counts=support) for p,y in zip(probs,labels,strict=True)))
    metric_name=next(k for k in metrics['probability_hex'] if 'bacc' in k)
    assert gain == pytest.approx(metrics['probability_hex'][metric_name]-metrics['baseline_probability_hex'][metric_name])
    assert gain == -.25
    legacy_mean_case_gain = (1.0 + (0.0-1.0)/2)/2
    assert legacy_mean_case_gain == .25


def test_case_contributions_above_one_are_not_clipped():
    # One class-one case supports all its center's class-one recall weight.
    menus=tuple(make_menu(f'c{i}',baseline=(.2,.2),d01=(.8,.2),d10=None,uniform=(.8,.2)) for i in range(4))
    cap=truth_for(menus,((1,0),(0,0),(0,0),(0,0)))
    result=cap.score_selections(tuple(seal(build_exact_u_composite(m),i) for i,m in enumerate(menus)))
    assert result[0].bacc_gain == 2.0
    assert result[0].class_1_gain == 1.0


def test_all_selections_must_be_sealed_before_joint_truth_normalization():
    menus=(make_menu('a'),make_menu('b'))
    cap=truth_for(menus,((0,1),(0,1)))
    with pytest.raises(ProtocolError,match='all scoring-scope selections'):
        cap.score_selected(seal(build_baseline_composite(menus[0])))
    assert cap.selected_score_count == 0
    rows=cap.score_selections(tuple(seal(build_baseline_composite(m),i) for i,m in enumerate(menus)))
    assert cap.selected_score_count == 2
    assert all(row.bacc_gain==0 for row in rows)


def test_truth_rejects_wrong_menu_identity_and_target_role():
    menu=make_menu()
    cap=truth_for((menu,),((0,1),))
    with pytest.raises(ProtocolError,match='authenticated source menu'):
        cap.score_composites((replace(build_baseline_composite(menu),menu_hash='2'*64),))
    target=make_menu(role=SurfaceRole.TARGET_EVALUATION)
    with pytest.raises(ProtocolError,match='target menu'):
        cap.scoped((target,))
    with pytest.raises(ProtocolError,match='cannot be serialized'):
        pickle.dumps(cap)
    assert 'raw_labels' not in json.dumps(cap.public_payload()).replace('raw_labels_persisted','').replace('raw_labels_public','')


def test_scoped_truth_cannot_read_an_outside_case():
    menus=(make_menu('a'),make_menu('b'))
    cap=truth_for(menus,((0,1),(0,1))).scoped((menus[0],))
    with pytest.raises(ProtocolError,match='outside its capability'):
        cap.scoped((menus[1],))


def test_action_model_is_exact_frozen_evidence_wrapper_and_has_no_candidate_heads():
    menus=tuple(make_menu(f'c{i}') for i in range(4))
    arguments=evidence_fit_arguments(menus)
    fitted=fit_action_outcome_model(menus,**arguments)
    assert not fitted.public_payload()['candidate_outcome_heads_fitted']
    assert fitted.training_case_keys == tuple((m.center_id,m.case_id) for m in menus)
    new=make_menu('new',role=SurfaceRole.TARGET_EVALUATION)
    available=tuple(r.composite for r in candidates(new) if r.eligible)
    pred=fitted.predict_composites(new,available)
    assert pred[0].predicted_gain == 0
    assert all(p.predicted_gain<0 for p in pred[1:])
    assert all(p.predicted_brier_delta>0 for p in pred[1:])
    assert all(not p.public_payload()['per_action_safety_guarantee'] for p in pred)
    with pytest.raises(ProtocolError,match='exact scoped correction evidence'):
        fit_action_outcome_model(menus[:2],**arguments)


def test_proposal_predictions_are_honest_and_ignore_poison_outside_scope():
    menus=tuple(make_menu(f'c{i}') for i in range(3))
    cap=truth_for(menus,((0,1),(0,1),(1,0)))
    training=menus[:2]
    profiles,outcomes=cap.scoped(training).derive_training_surface(training)
    model=fit_proposal_model(training,profiles,outcomes)
    assert model.predict_menu(menus[2]).d01_ranked_action_ids == ('D01::E',)
    with pytest.raises(ProtocolError,match='fitted case'):
        model.predict_menu(menus[0])
    poisoned=truth_for(menus,((0,1),(0,1),(0,1)))
    p2,o2=poisoned.scoped(training).derive_training_surface(training)
    assert fit_proposal_model(training,p2,o2).model_hash == model.model_hash


def test_exact_flip_aggregation_can_select_opposite_directions_without_free_heads():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.outcome_model import effect_arrays
    baseline=np.array([.2,.8])
    actions=np.array([[.8,.8],[.2,.2]])
    # Same action menu, opposite shared correction evidence: directional preference reverses.
    first=effect_arrays(baseline,actions,np.array([.9,.9]),np.array([[.05,.9],[.1,.9]]))[0]
    second=effect_arrays(baseline,actions,np.array([.1,.1]),np.array([[.9,.05],[.9,.1]]))[0]
    assert first[0]>first[1] and second[1]>second[0]


def evidence_fit_arguments(menus):
    # Explicit fabricated source labels; no action-outcome fitting or patch stack.
    menus=tuple(menus)
    capability=truth_for(menus,[tuple(i%2 for i in range(len(m.sample_ids))) for m in menus])
    return dict(evidence_model=capability.fit_correction_evidence(menus,variant='baseline'))
