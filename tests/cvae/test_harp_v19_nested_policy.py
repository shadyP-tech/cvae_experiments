from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import pytest

from midogpp_thesis.cvae.routing.safe_winner_router_v19.contracts import (
    AdmissionStatus, CompositeKind, Direction, LabelFreeAction, LabelFreeCaseMenu,
    RouterFitConfig, SurfaceRole, float32_probability_hex,
)
from midogpp_thesis.cvae.routing.safe_winner_router_v19.composition import (
    build_baseline_composite, build_candidate_composites, build_soft_topk_composite,
)
from midogpp_thesis.cvae.routing.safe_winner_router_v19.truth import SupportTruthCapability
from midogpp_thesis.cvae.routing.safe_winner_router_v19.stacked_fitting import (
    CandidatePrediction, HeldCandidatePrediction as _HeldCandidatePrediction, choose_candidate as _choose_candidate, fit_stacked_science_model,
)
from midogpp_thesis.cvae.routing.safe_winner_router_v19.outcome_model import ActionOutcomePrediction
from midogpp_thesis.cvae.routing.safe_winner_router_v19.candidate_prediction import unthresholded_winner
from midogpp_thesis.cvae.routing.safe_winner_router_v19.winner_gate import WinnerGatePrediction


def _gate(candidates):
    winner = unthresholded_winner(candidates)
    if winner is None:
        return None
    harm = max(.1, float(winner.prediction.predicted_harm))
    return WinnerGatePrediction(winner.candidate.composite.composite_hash,
        1-harm, harm, 0., "f"*64)


def choose_candidate(menu, candidates, threshold, **kwargs):
    kwargs.setdefault("winner_prediction", _gate(candidates))
    return _choose_candidate(menu, candidates, threshold, **kwargs)


def HeldCandidatePrediction(fold, menu, candidates, keys, model_hash):
    return _HeldCandidatePrediction(fold, menu, candidates, keys, model_hash, _gate(candidates))

from midogpp_thesis.cvae.routing.safe_winner_router_v19.frontier import build_candidate_frontier, seal_selections
from midogpp_thesis.cvae.routing.safe_winner_router_v19.crossfit import nested_source_crossfit
from midogpp_thesis.cvae.routing.safe_winner_router_v19.admission import build_source_only_admission

def _action(
    role: SurfaceRole,
    center: str,
    case: str,
    arm: str,
    direction: Direction,
    donor: str | None,
    baseline: tuple[float, ...],
    probability: tuple[float, ...],
    feature: float,
) -> LabelFreeAction:
    samples = tuple(f"{case}:s{index}" for index in range(len(baseline)))
    return LabelFreeAction(
        surface_role=role,
        center_id=center,
        case_id=case,
        arm_id=arm,
        direction=direction,
        donor_id=donor,
        feature_names=("compatibility", "margin_shift"),
        feature_values=(feature, feature * 0.5),
        sample_ids=samples,
        baseline_probability_hex=float32_probability_hex(baseline),
        action_probability_hex=float32_probability_hex(probability),
    )


def _menu(
    role: SurfaceRole,
    center: str,
    case: str,
    *,
    donors: tuple[str, ...],
    baseline: tuple[float, ...] = (0.2, 0.3, 0.7, 0.8),
) -> LabelFreeCaseMenu:
    actions: list[LabelFreeAction] = []
    for ordinal, donor in enumerate(donors):
        actions.append(
            _action(
                role,
                center,
                case,
                f"D01::{donor}",
                Direction.D01,
                donor,
                baseline,
                (0.2, 0.8 - 0.05 * ordinal, 0.7, 0.8),
                2.0 - ordinal,
            )
        )
        actions.append(
            _action(
                role,
                center,
                case,
                f"D10::{donor}",
                Direction.D10,
                donor,
                baseline,
                (0.2, 0.3, 0.7, 0.2 + 0.05 * ordinal),
                2.0 - ordinal,
            )
        )
    actions.append(
        _action(
            role,
            center,
            case,
            "U_FULL",
            Direction.FULL,
            None,
            baseline,
            (0.1, 0.6, 0.6, 0.4),
            0.0,
        )
    )
    samples = tuple(f"{case}:s{index}" for index in range(len(baseline)))
    return LabelFreeCaseMenu(
        surface_role=role,
        center_id=center,
        case_id=case,
        sample_ids=samples,
        baseline_probability_hex=float32_probability_hex(baseline),
        actions=tuple(actions),
    )


def _source_surface(
    *, centers: int = 3, cases_per_center: int = 4
) -> tuple[tuple[LabelFreeCaseMenu, ...], SupportTruthCapability]:
    center_ids = tuple(f"C{index}" for index in range(centers))
    menus: list[LabelFreeCaseMenu] = []
    truth: dict[tuple[str, str], dict[str, int]] = {}
    for center in center_ids:
        donors = tuple(value for value in center_ids if value != center)
        for ordinal in range(cases_per_center):
            case = f"{center}:case{ordinal}"
            menu = _menu(
                SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,
                center,
                case,
                donors=donors,
            )
            menus.append(menu)
            truth[(center, case)] = dict(zip(menu.sample_ids, (0, 1, 1, 0), strict=True))
    return tuple(menus), SupportTruthCapability(truth)



def _config() -> RouterFitConfig:
    return RouterFitConfig(outer_folds=2, inner_folds=2, stack_folds=2, winner_folds=2,
        opportunity_ridge_alphas=(1.0,), ranker_ridge_alphas=(1.0,),
        k_values=(1,2), lambda_values=(1.0,), route_thresholds=(0.0,.005),
        required_source_case_count=None,required_source_center_count=None,
        minimum_routed_oof_cases=2,minimum_routed_oof_centers=1,
        minimum_routed_oof_cases_per_center=1,bootstrap_replicates=32)

def _prediction(menu, *, direction="D01_ONLY"):
    proposal=SimpleNamespace(d01_ranked_action_ids=tuple(a.arm_id for a in menu.actions_for(Direction.D01)),
        d10_ranked_action_ids=tuple(a.arm_id for a in menu.actions_for(Direction.D10)))
    values=[]
    for candidate in build_candidate_composites(menu,proposal,_config()):
        gain=.2 if candidate.kind.value==direction else -.1
        values.append(CandidatePrediction(candidate,None if candidate.composite is None else ActionOutcomePrediction(candidate.composite.composite_hash,gain,0.,-.1,-.1,.9,gain,gain,gain-.1,safe_benefit_score=gain)))
    return tuple(values)

def test_one_ineligible_case_does_not_prune_other_cases_or_frontier():
    good=_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0","good",donors=("C1",))
    missing=_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0","missing",donors=())
    held=tuple(HeldCandidatePrediction(0,m,_prediction(m),(("C1","training"),),"a"*64) for m in (good,missing))
    selections=seal_selections(held,0.)
    assert selections[0].composite.kind is CompositeKind.D01_ONLY
    assert selections[1].composite.kind is CompositeKind.B
    cap=SupportTruthCapability({(m.center_id,m.case_id):dict(zip(m.sample_ids,(0,1,1,0),strict=True)) for m in (good,missing)})
    frontier,oracle=build_candidate_frontier(held,cap,thresholds=(0.,.5),stage="fixture")
    arm=selections[0].composite.arm_id
    atzero=next(row for row in frontier if row["arm_id"]==arm and row["threshold"]==0.)
    assert atzero["ineligible_count"]==1 and atzero["eligible_count"]==1
    assert atzero["route_count"]==1 and atzero["baseline_fallback_count"]==1
    assert len(frontier)==(len(held[0].candidates)+1)*2
    assert all("failed_constraints" in row and "utility_risk_moments" in row for row in frontier)
    assert all("pretruth_candidate_prediction_seal_hash" in row for row in frontier)
    assert oracle["oracle_used_for_selection"] is False

def test_opposite_cases_select_opposite_branches_without_individual_risk_veto():
    menus=tuple(_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0",case,donors=("C1",)) for case in ("a","b"))
    a,_,_=choose_candidate(menus[0],_prediction(menus[0],direction="D01_ONLY"),0.)
    b,_,_=choose_candidate(menus[1],_prediction(menus[1],direction="D10_ONLY"),0.)
    assert a.kind is CompositeKind.D01_ONLY and b.kind is CompositeKind.D10_ONLY
    assert a.probability_hex[2:]==a.baseline_probability_hex[2:]
    assert b.probability_hex[:2]==b.baseline_probability_hex[:2]
    bad=tuple(CandidatePrediction(row.candidate,None if row.candidate.composite is None else ActionOutcomePrediction(row.candidate.composite.composite_hash,.9,.8,-.1,-.1,.9,.9,.9,.8,safe_benefit_score=.9)) for row in _prediction(menus[0]))
    assert choose_candidate(menus[0],bad,0.)[0].kind is not CompositeKind.B
    assert choose_candidate(menus[0],bad,.3)[0].kind is CompositeKind.B

def test_nonadmission_happens_before_bootstrap(monkeypatch):
    from midogpp_thesis.cvae.routing.safe_winner_router_v19 import admission
    menus,cap=_source_surface(centers=2,cases_per_center=4)
    held=tuple(HeldCandidatePrediction(0,m,_prediction(m),(("other","training"),),"b"*64) for m in menus)
    selections=seal_selections(held,1.)
    records=cap.scoped(menus).score_selections(selections)
    monkeypatch.setattr(admission,"approximate_source_oof_bounds",lambda *args,**kwargs:pytest.fail("bootstrap reached"))
    result=build_source_only_admission(records,config=_config())
    assert result.status is AdmissionStatus.NO_NONZERO_SAFE_OOF_COVERAGE
    assert result.bounds is None and result.bootstrap_performed is False

def test_full_nested_held_label_poisoning_preserves_model_threshold_and_composite():
    menus,cap=_source_surface(centers=2,cases_per_center=8)
    first=nested_source_crossfit(menus,(),(),cap,config=_config())
    poisoned_key=first.outer_fold_case_keys[0][0]
    labels={(m.center_id,m.case_id):dict(zip(m.sample_ids,(1,0,1,0) if (m.center_id,m.case_id)==poisoned_key else (0,1,1,0),strict=True)) for m in menus}
    second=nested_source_crossfit(menus,(),(),SupportTruthCapability(labels),config=_config())
    assert first.fold_choices[0].choice_hash==second.fold_choices[0].choice_hash
    for center,case in first.outer_fold_case_keys[0]:
        a,b=first.selection_for(center,case),second.selection_for(center,case)
        assert a.selection_hash==b.selection_hash
        assert a.model_hash==b.model_hash and a.composite.composite_hash==b.composite.composite_hash
    assert first.final_inner_fold_hash and first.all_outer_prediction_seal_hash
    assert any(row.route_selected for row in first.records)
    assert all("failed_constraints" in row for row in first.frontier_rows)

def test_stack_proposal_and_calibrator_fit_boundaries_are_disjoint():
    menus,cap=_source_surface(centers=2,cases_per_center=6)
    model=fit_stacked_science_model(menus,cap,config=_config())
    for receipt in model.stacking_receipts:
        assert not set(receipt["training_case_keys"]) & set(receipt["held_case_keys"])
        assert receipt["held_candidates_sealed_before_truth"]
    assert set(model.action_model.training_case_keys)==set(model.training_case_keys)
    assert set(model.proposal_model.training_case_keys)==set(model.training_case_keys)


def test_fold_frontier_can_report_single_class_using_complete_oof_normalizer():
    a=_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0","single",donors=("C1",))
    b=_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0","both",donors=("C1",))
    cap=SupportTruthCapability({("C0","single"):dict(zip(a.sample_ids,(0,0,0,0),strict=True)),
        ("C0","both"):dict(zip(b.sample_ids,(0,1,1,0),strict=True))})
    held=(HeldCandidatePrediction(0,a,_prediction(a),(("C1","training"),),"c"*64),)
    frontier,_=build_candidate_frontier(held,cap,thresholds=(0.,),stage="single-fold",normalization_menus=(a,b))
    assert frontier and frontier[0]["normalization_case_keys"]==(("C0","single"),("C0","both"))
    assert frontier[0]["summary_case_keys"]==(("C0","single"),)


def test_routed_risk_reporting_is_not_diluted_by_baseline_fallbacks():
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.frontier import policy_moments
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.records import SelectedOOFRecord
    menus,cap=_source_surface(centers=2,cases_per_center=4)
    held=tuple(HeldCandidatePrediction(0,m,_prediction(m),(("other","training"),),"d"*64) for m in menus)
    selections=list(seal_selections(held,1.))
    selections[0]=seal_selections(held[:1],0.)[0]
    records=list(cap.scoped(menus).score_selections(tuple(selections)))
    records[0]=SelectedOOFRecord(records[0].selection,-.1,.02,.1)
    moments=policy_moments(records)
    assert moments["routed_harm"]==1.
    assert moments["routed_brier_delta"]==pytest.approx(.02)
    assert moments["routed_logloss_delta"]==pytest.approx(.1)
    assert moments["equal_center_route_coverage"]==pytest.approx(1/8)


def test_incidental_single_route_center_does_not_veto_qualifying_coverage():
    menus,cap=_source_surface(centers=3,cases_per_center=4)
    held=tuple(HeldCandidatePrediction(0,m,_prediction(m),(("other","training"),),"e"*64) for m in menus)
    selections=tuple(seal_selections((row,),1. if i>=9 else 0.)[0] for i,row in enumerate(held))
    records=cap.scoped(menus).score_selections(selections)
    config=replace(_config(),minimum_routed_oof_cases=9,minimum_routed_oof_centers=2,minimum_routed_oof_cases_per_center=2)
    result=build_source_only_admission(records,config=config)
    assert result.routed_case_count==9 and result.routed_center_count==3
    assert result.qualifying_routed_center_count==2
    assert result.bootstrap_performed
    assert result.public_payload()["qualifying_routed_center_count"]==2


def test_audit_only_primitive_aggregates_cannot_change_fitted_policy():
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.policy import fit_source_router
    menus,cap=_source_surface(centers=2,cases_per_center=8)
    profiles,outcomes=cap.scoped(menus).derive_training_surface(menus)
    first=fit_source_router(menus,cap,config=_config(),case_profiles=profiles,action_outcomes=outcomes)
    poisoned_profiles=tuple(replace(row,d01_opportunity_count=0,d10_opportunity_count=0) for row in profiles)
    poisoned_outcomes=tuple(replace(row,bacc_gain=-.5,brier_delta=.5,log_loss_delta=1.,class_0_gain=-1.,class_1_gain=0.) for row in outcomes)
    second=fit_source_router(menus,cap,config=_config(),case_profiles=poisoned_profiles,action_outcomes=poisoned_outcomes)
    assert first.model_hash==second.model_hash
    assert first.policy_hash==second.policy_hash
    assert first.crossfit.result_hash==second.crossfit.result_hash
    assert first.route_threshold==second.route_threshold


def test_candidate_estimates_cannot_bind_a_different_composite():
    from midogpp_thesis.cvae.protocol import ProtocolError
    menu=_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0","binding",donors=("C1",))
    value=next(row for row in _prediction(menu) if row.candidate.eligible)
    with pytest.raises(ProtocolError,match="actual composite"):
        CandidatePrediction(value.candidate,replace(value.prediction,composite_hash="0"*64))


def test_bootstrap_records_missing_class_support_with_conservative_gain():
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.admission import approximate_source_oof_bounds
    menus=tuple(_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0",case,donors=("C1",)) for case in ("only-zero","only-one"))
    cap=SupportTruthCapability({("C0",menu.case_id):dict(zip(menu.sample_ids,(label,)*4,strict=True)) for label,menu in enumerate(menus)})
    held=tuple(HeldCandidatePrediction(0,m,_prediction(m),(("C1","train"),),"a"*64) for m in menus)
    records=cap.scoped(menus).score_selections(seal_selections(held,1.))
    bounds=approximate_source_oof_bounds(records,config=_config())
    assert bounds.missing_class_support_replicates>0
    assert bounds.gain_lower<0.
    assert bounds.public_payload()["class_support_denominators_recomputed_per_replicate"] is True


def test_optimized_frontier_matches_authenticated_selected_record_replay():
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.frontier import _summary
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.hashing import canonical_hash
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.records import SealedOOFSelection
    from midogpp_thesis.cvae.routing.safe_winner_router_v19.stacked_fitting import POLICY_ARM_ID
    menus,cap=_source_surface(centers=2,cases_per_center=3)
    held=tuple(HeldCandidatePrediction(0,m,_prediction(m),(("other","training"),),"b"*64) for m in menus)
    thresholds=(0.,.005,.5)
    frontier,_=build_candidate_frontier(held,cap,thresholds=thresholds,stage="numeric-reference")
    for row in frontier:
        arm_id,threshold=row["arm_id"],row["threshold"]
        if arm_id==POLICY_ARM_ID:
            selections=seal_selections(held,threshold)
        else:
            selections=[]
            for case in held:
                value=next(value for value in case.candidates if value.arm_id==arm_id)
                composite=value.candidate.composite if value.screened and value.route_score>=threshold else build_baseline_composite(case.menu)
                selections.append(SealedOOFSelection(0,composite,arm_id,value.route_score,threshold,case.training_case_keys,case.model_hash))
        expected=_summary(cap.scoped(menus).score_selections(tuple(selections)))
        assert all(row[key]==value for key,value in expected.items())
        body=dict(row);digest=body.pop("frontier_row_hash")
        assert canonical_hash(body)==digest
        assert "PREDICTED_NONPOSITIVE_SAFE_BENEFIT" in row["eligibility_and_screen_counts"] or arm_id==POLICY_ARM_ID or row["eligible_count"]==0
        assert row["prediction_means_are_equal_center_case_means"]


def test_promising_margin_only_composite_cannot_inflate_routing_coverage():
    menu=_menu(SurfaceRole.SOURCE_TRAIN_DEVELOPMENT,"C0","margin-only",donors=("C1",))
    config=replace(_config(),lambda_values=(.25,))
    proposal=SimpleNamespace(d01_ranked_action_ids=tuple(a.arm_id for a in menu.actions_for(Direction.D01)),
        d10_ranked_action_ids=tuple(a.arm_id for a in menu.actions_for(Direction.D10)))
    values=[]
    for candidate in build_candidate_composites(menu,proposal,config):
        gain=.5 if candidate.kind is CompositeKind.D01_ONLY else -.1
        prediction=None if candidate.composite is None else ActionOutcomePrediction(candidate.composite.composite_hash,gain,0.,-.1,-.1,.9,gain,gain,gain-.1,safe_benefit_score=gain)
        values.append(CandidatePrediction(candidate,prediction))
    margin=next(row for row in values if row.candidate.kind is CompositeKind.D01_ONLY and row.candidate.eligible)
    assert margin.candidate.composite.probability_changed
    assert not margin.hard_prediction_changed and not margin.screened
    assert choose_candidate(menu,values,0.)[0].kind is CompositeKind.B
    held=(HeldCandidatePrediction(0,menu,tuple(values),(("C1","training"),),"c"*64),)
    cap=SupportTruthCapability({("C0",menu.case_id):dict(zip(menu.sample_ids,(0,1,1,0),strict=True))})
    frontier,_=build_candidate_frontier(held,cap,thresholds=(0.,),stage="structural-noop")
    report=next(row for row in frontier if row["arm_id"]==margin.arm_id)
    assert report["eligibility_and_screen_counts"]["NO_HARD_PREDICTION_CHANGE"]==1
    assert report["route_count"]==0
