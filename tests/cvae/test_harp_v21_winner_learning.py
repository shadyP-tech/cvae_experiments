"""Behavioral checks for complete winner nesting and abstention evidence."""
from dataclasses import replace
import copy

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.correction_mass_router_v21.candidate_prediction import (
    CandidatePrediction, HeldCandidatePrediction, choose_candidate, unthresholded_winner,
)
from midogpp_thesis.cvae.routing.correction_mass_router_v21.contracts import CompositeKind
from midogpp_thesis.cvae.routing.correction_mass_router_v21.fit_cache import ScopedFitCache
from midogpp_thesis.cvae.routing.correction_mass_router_v21.frontier import build_candidate_frontier, seal_selections
from midogpp_thesis.cvae.routing.correction_mass_router_v21.learning import fit_stacked_science_model
from midogpp_thesis.cvae.routing.correction_mass_router_v21.model_integrity import verify_complete_model_payload
from midogpp_thesis.cvae.routing.correction_mass_router_v21.truth import SupportTruthCapability
from midogpp_thesis.cvae.routing.correction_mass_router_v21.winner_gate import WinnerGatePrediction, fit_winner_gate
from midogpp_thesis.cvae.routing.correction_mass_router_v21.winner_records import SealedWinner
from test_harp_v21_nested_policy import _source_surface, _config, _prediction


def _gate(winner, harm=.2):
    return WinnerGatePrediction(winner.candidate.composite.composite_hash, 1-harm, harm, 0., "e"*64)


def test_gate_veto_never_promotes_safer_runnerup_and_threshold_is_inclusive():
    menus, _ = _source_surface()
    candidates = _prediction(menus[0])
    winner = unthresholded_winner(candidates)
    gate = _gate(winner, harm=.7)
    composite, score, reason = choose_candidate(menus[0], candidates, .4, winner_prediction=gate)
    assert composite.kind is CompositeKind.B and reason == "WINNER_GATE_BELOW_THRESHOLD"
    composite, score, reason = choose_candidate(menus[0], candidates, gate.route_score, winner_prediction=gate)
    assert composite.arm_id == winner.arm_id and reason is None
    assert choose_candidate(menus[0], candidates, 0.)[2] == "MISSING_COMPLETE_WINNER_GATE"


def test_full_population_case_weights_are_retained_after_participation_filter():
    menus, _ = _source_surface(centers=2, cases_per_center=3)
    population = (*menus[:3], menus[-1])  # 3:1 full-population center sizes.
    calibration = (population[0], population[-1])
    labels = {(m.center_id,m.case_id):dict(zip(m.sample_ids,(0,0,1,1),strict=True)) for m in population}
    cap = SupportTruthCapability(labels).scoped(population)
    seals = tuple(SealedWinner(menu, _prediction(menu), (("fit","excluded"),), "f"*64, 0)
                  for menu in calibration)
    outcomes = cap.score_composites(tuple(row.winner.candidate.composite for row in seals))
    model = fit_winner_gate(seals, outcomes,
        training_case_keys=tuple((m.center_id,m.case_id) for m in calibration),
        population_case_keys=cap.case_keys)
    records = model.fit_audit["records"]
    assert all(row["harm_target"] == 1 for row in records)
    assert [row["fit_weight"] for row in records] == pytest.approx([.25,.75])
    assert model.fit_audit["all_winners_sealed_before_any_winner_truth"]
    held = population[1]
    assert model.predict(held, _prediction(held)).harm_probability > .99


def test_complete_gate_receipts_remove_validation_from_all_upstream_fits_and_cache():
    menus, cap = _source_surface(centers=2,cases_per_center=6)
    cache = ScopedFitCache(maximum_entries=4)
    model = fit_stacked_science_model(menus,cap,config=_config(),cache=cache)
    assert fit_stacked_science_model(menus,cap,config=_config(),cache=cache) is model
    assert cache.hits > 0 and len(cache._models) <= 4
    for receipt in model.winner_fit_receipts:
        held = set(receipt["held_case_keys"])
        assert not held.intersection(receipt["training_case_keys"])
        assert receipt["proposer_hash"] == model.proposer.model_hash
        assert receipt["proposer_refitted_after_calibration"] is False
    payload = model.public_payload()
    from midogpp_thesis.cvae.routing.harp_protocol import canonical_hash as strict_runtime_hash
    assert len(strict_runtime_hash(payload)) == 64  # No numpy scalars in public fits.
    assert verify_complete_model_payload(payload) == model.model_hash
    held_menu = next(m for m in menus if (m.center_id,m.case_id) not in model.winner_gate.training_case_keys)
    gate_prediction = model.winner_gate.predict(held_menu, _prediction(held_menu))
    assert gate_prediction.feature_names == model.winner_gate.feature_names
    from scipy.special import expit
    import numpy as np
    raw = np.asarray(gate_prediction.feature_values)
    matrix = np.concatenate(([1.], (raw-model.winner_gate.means)/model.winner_gate.scales))[None,:]
    replay = float(expit(matrix @ model.winner_gate.coefficients)[0])
    assert replay == pytest.approx(gate_prediction.harm_probability)
    broken = copy.deepcopy(payload)
    broken["winner_gate"]["coefficients"] = tuple(value+.1 for value in broken["winner_gate"]["coefficients"])
    with pytest.raises(ProtocolError,match="winner gate model hash"):
        verify_complete_model_payload(broken)
    # Changing source truth must not retrieve the old fit from a matching menu.
    changed = SupportTruthCapability({(m.center_id,m.case_id):dict(zip(m.sample_ids,(0,0,1,1),strict=True)) for m in menus})
    assert fit_stacked_science_model(menus,changed,config=_config(),cache=cache).model_hash != model.model_hash


def test_empty_action_population_fits_conservative_gate_and_exact_baseline():
    menus, cap = _source_surface(centers=2,cases_per_center=4)
    noops = tuple(replace(menu, actions=tuple(replace(action,
        action_probability_hex=action.baseline_probability_hex) for action in menu.actions)) for menu in menus)
    model = fit_stacked_science_model(noops,cap,config=_config())
    assert not model.winner_gate.participating_case_keys
    assert model.winner_gate.fit_audit["empty_menu_case_count"] == len(model.winner_gate.training_case_keys)
    assert not model.action_model.empty_population  # This architecture fits evidence rather than candidate outcomes.


def test_frontier_predicted_harm_is_matched_to_exact_routed_subset():
    menus, cap = _source_surface(centers=2,cases_per_center=3)
    held = tuple(HeldCandidatePrediction(0,m,_prediction(m),(("other","train"),),"a"*64,
        _gate(unthresholded_winner(_prediction(m)), harm=.1 if i==0 else .8)) for i,m in enumerate(menus))
    frontier, diagnostics = build_candidate_frontier(held,cap,thresholds=(0.,.5),stage="fixture")
    policy = next(row for row in frontier if row["arm_id"] == "SAFE_WINNER_ACTION_POLICY" and row["threshold"]==.5)
    assert policy["route_count"]==1
    assert policy["prediction_means_among_routed_candidates"]["predicted_harm"]==pytest.approx(.1)
    assert policy["prediction_means_among_available_candidates"]["predicted_harm"]==pytest.approx((.1+5*.8)/6)
    assert len(diagnostics["candidate_prediction_outcome_joins"])==sum(len(row.candidates) for row in held)
    assert len(diagnostics["winner_gate_diagnostics"])==len(held)
    selection = seal_selections(held,.5)[0]
    assert selection.winner_risk_adjusted_score == pytest.approx(.2)
    assert selection.winner_gate_harm_probability == pytest.approx(.1)
    assert selection.public_payload()["winner_gate_score"] == pytest.approx(.9)
    with pytest.raises(ProtocolError,match="sealed winner/gate rule"):
        replace(selection,winner_risk_adjusted_score=-.1)


def test_admitted_decision_requires_routing_iff_complete_winner_rule_passes():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.records import RouteDecision
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.decision_evidence import decision_evidence
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.composition import build_baseline_composite
    menus, _ = _source_surface()
    candidates = _prediction(menus[0])
    winner = unthresholded_winner(candidates)
    gate = _gate(winner)
    with pytest.raises(ProtocolError,match="does not implement"):
        RouteDecision(build_baseline_composite(menus[0]), "SAFE_WINNER_ACTION_POLICY", gate.route_score,
            .5, "a"*64, True, "WINNER_GATE_BELOW_THRESHOLD", **decision_evidence(winner, gate))
    result = RouteDecision(winner.candidate.composite,"SAFE_WINNER_ACTION_POLICY",gate.route_score,
        .5,"a"*64,True,**decision_evidence(winner,gate))
    transcript = result.public_payload()["winner_gate_prediction_payload"]
    assert transcript["prediction_hash"] == gate.prediction_hash
    assert transcript["feature_names"] == gate.feature_names


def test_source_selection_binds_enabled_state_and_complete_rule():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.composition import build_baseline_composite
    menus, _ = _source_surface()
    candidates = _prediction(menus[0])
    winner = unthresholded_winner(candidates)
    held = (HeldCandidatePrediction(0, menus[0], candidates, (("other","train"),), "a"*64, _gate(winner)),)
    selected = seal_selections(held, .5, enabled=True)[0]
    assert selected.policy_enabled and selected.composite.route_selected
    with pytest.raises(ProtocolError, match="does not implement"):
        replace(selected, composite=build_baseline_composite(menus[0]), fallback_reason="WINNER_GATE_BELOW_THRESHOLD")
    disabled = seal_selections(held, .5, enabled=False)[0]
    assert disabled.policy_enabled is False
    assert disabled.fallback_reason == "NO_SAFE_INNER_OOF_POLICY"
    assert disabled.selection_hash != selected.selection_hash
    assert disabled.public_payload()["policy_enabled"] is False
    with pytest.raises(ProtocolError, match="does not implement"):
        replace(disabled, policy_enabled=True)
    with pytest.raises(ProtocolError, match="fallback reason drifted"):
        replace(disabled, fallback_reason="NONPOSITIVE_PREDICTED_GAIN")


def test_calibration_label_poison_changes_gate_but_never_frozen_proposer():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.calibration_split import proposer_calibration_partition
    menus, cap = _source_surface(centers=2, cases_per_center=6)
    model = fit_stacked_science_model(menus, cap, config=_config())
    _, calibration = proposer_calibration_partition(cap.case_keys)
    labels = {(m.center_id,m.case_id):dict(zip(m.sample_ids,
        (0,0,1,1) if (m.center_id,m.case_id) in calibration else (0,1,1,0), strict=True)) for m in menus}
    poisoned = fit_stacked_science_model(menus, SupportTruthCapability(labels), config=_config())
    assert model.proposer.model_hash == poisoned.proposer.model_hash
    assert model.action_model.model_hash == poisoned.action_model.model_hash
    assert model.winner_gate.model_hash != poisoned.winner_gate.model_hash
    assert model.model_hash != poisoned.model_hash
    assert set(model.proposer.training_case_keys).isdisjoint(calibration)


def test_unavailable_calibration_cannot_route_even_at_threshold_zero():
    menus, _ = _source_surface()
    candidates = _prediction(menus[0])
    winner = unthresholded_winner(candidates)
    gate = WinnerGatePrediction(winner.candidate.composite.composite_hash,
        0., 1., 0., "a"*64, calibration_available=False)
    composite, score, reason = choose_candidate(menus[0], candidates, 0., winner_prediction=gate)
    assert composite.kind is CompositeKind.B
    assert score == 0. and reason == "WINNER_GATE_UNAVAILABLE"
    held = (HeldCandidatePrediction(0, menus[0], candidates, (("other","fit"),), "b"*64, gate),)
    selection = seal_selections(held, 0.)[0]
    assert not selection.composite.route_selected
    assert selection.public_payload()["winner_gate_prediction_payload"]["calibration_available"] is False


def test_candidate_feasibility_uses_gain_and_each_proper_loss_before_ranking():
    menus, _ = _source_surface()
    candidates = _prediction(menus[0])
    winner = unthresholded_winner(candidates)
    for mutation in ({"predicted_gain": -.1}, {"predicted_brier_delta": .002001},
                     {"predicted_logloss_delta": .005001}):
        modified = tuple(CandidatePrediction(row.candidate,
            replace(row.prediction, **mutation) if row is winner else row.prediction) for row in candidates)
        assert all(row is not winner for row in modified)
        rejected = next(row for row in modified if row.arm_id == winner.arm_id)
        assert not rejected.eligible_for_winner
    assert len(model_feature_names()) == 6


def model_feature_names():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.winner_records import winner_features
    menus, _ = _source_surface()
    return dict(winner_features(menus[0], _prediction(menus[0])))


def test_complete_learner_rejects_fit_and_calibration_identities_even_retagged_target():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.contracts import SurfaceRole
    menus, cap = _source_surface(centers=2, cases_per_center=6)
    model = fit_stacked_science_model(menus, cap, config=_config())
    for scope in (model.proposer.training_case_keys, model.winner_gate.training_case_keys):
        source = next(m for m in menus if (m.center_id,m.case_id) == scope[0])
        target = replace(source, surface_role=SurfaceRole.TARGET_EVALUATION,
            actions=tuple(replace(a, surface_role=SurfaceRole.TARGET_EVALUATION) for a in source.actions))
        for menu in (source, target):
            for predict in (lambda: model.predict_menu(menu),
                            lambda: model.candidate_predictions(menu, _config()),
                            lambda: model.winner_prediction(menu, _prediction(menu))):
                with pytest.raises(ProtocolError, match="fitting or calibration case"):
                    predict()
    # Calibration construction remains valid through the unchanged proposer.
    calibration = next(m for m in menus if (m.center_id,m.case_id) == model.winner_gate.training_case_keys[0])
    assert model.proposer.candidate_predictions(calibration, _config())


def test_gate_rejects_own_calibration_case_even_without_a_winner_or_after_retagging():
    from midogpp_thesis.cvae.routing.correction_mass_router_v21.contracts import SurfaceRole
    from test_harp_v21_nested_policy import _menu
    menus, cap = _source_surface(centers=2, cases_per_center=6)
    model = fit_stacked_science_model(menus, cap, config=_config())
    source = next(m for m in menus if (m.center_id,m.case_id) == model.winner_gate.training_case_keys[0])
    target = replace(source, surface_role=SurfaceRole.TARGET_EVALUATION,
        actions=tuple(replace(a, surface_role=SurfaceRole.TARGET_EVALUATION) for a in source.actions))
    for menu in (source, target):
        for candidates in ((), _prediction(menu)):
            with pytest.raises(ProtocolError, match="calibration fitting case"):
                model.winner_gate.predict(menu, candidates)
    held = _menu(SurfaceRole.TARGET_EVALUATION, "C0", "new-target-case", donors=("C1",))
    predictions = model.candidate_predictions(held, _config())
    assert model.predict_menu(held)
    assert model.winner_prediction(held, predictions) is not None
