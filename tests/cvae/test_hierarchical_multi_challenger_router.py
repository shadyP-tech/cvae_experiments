from __future__ import annotations

import math

import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.hierarchical_multi_challenger import (
    ActionScore,
    CandidateMenu,
    DirectionalCalibration,
    DirectionalDonorRow,
    DirectionalLogitModel,
    SOURCE_ALPHA,
    SupportActionScore,
    baseline_action_score,
    build_calibration_observation,
    build_candidate_menu,
    fit_direction_calibration,
    fit_directional_logit,
    permute_complete_case_feature_blocks,
    predict_direction,
    score_action_against_baseline,
    select_action_with_margin,
)
from midogpp_thesis.cvae.routing.hierarchical_multi_challenger.hashing import (
    canonical_hash,
    fitted_numeric_fingerprint,
)
from midogpp_thesis.cvae.routing.threshold_flip_case_router import ContributionTarget


FEATURES = ("count", "margin")


def _donor_rows(direction: str = "0to1") -> tuple[DirectionalDonorRow, ...]:
    result = []
    sources = {"1": "2", "2": "3", "3": "1"}
    for query in ("1", "2", "3"):
        for case_ordinal in range(4):
            for action_ordinal, source in enumerate((sources[query], "4")):
                trials = 2 + case_ordinal + action_ordinal
                successes = min(
                    trials,
                    1 + case_ordinal + (1 if direction == "0to1" else 0),
                )
                result.append(
                    DirectionalDonorRow(
                        model_target="0",
                        query_center=query,
                        candidate_source=source,
                        case_id=f"{query}-case-{case_ordinal}",
                        action_id=f"A1::source={source}",
                        feature_case_id=f"{query}-case-{case_ordinal}",
                        direction=direction,
                        success_count=successes,
                        trial_count=trials,
                        feature_names=FEATURES,
                        values=(float(trials), float(case_ordinal - 1)),
                    )
                )
    return tuple(result)


def _model(direction: str, *, coefficient: float = 2.0) -> DirectionalLogitModel:
    payload = {
        "target": "0",
        "direction": direction,
        "purpose": "unit-test",
    }
    covariance = tuple(
        tuple(1.0e-8 if row == column else 0.0 for column in range(7))
        for row in range(7)
    )
    return DirectionalLogitModel(
        model_target="0",
        family="R",
        direction=direction,
        feature_names=("signal",),
        feature_mean=(0.0,),
        feature_scale=(1.0,),
        candidate_sources=("1", "2", "3"),
        query_centers=("1", "2"),
        coefficients=(0.0, coefficient, 0.1, -0.1, 0.0, 0.0, 0.0),
        covariance=covariance,
        feature_alpha=1.0,
        source_alpha=SOURCE_ALPHA,
        query_alpha=4.0,
        intercept_alpha=0.25,
        training_row_count=20,
        training_trial_count=80,
        training_case_clusters=("1::a", "2::b"),
        provenance_hash=canonical_hash(payload),
    )


def _menu(anchor: str = "A1::source=1") -> CandidateMenu:
    ranked = tuple(
        SupportActionScore(action, gain, gain * 0.8, 10)
        for action, gain in (
            ("A1::source=1", 0.03),
            ("A1::source=2", 0.02),
            ("A1::source=3", 0.01),
            ("A1::source=4", -0.01),
        )
    )
    return CandidateMenu(
        ("B", "A1::source=1", "A1::source=2", "A1::source=3"),
        anchor,
        ranked,
        3,
    )


def _calibrations(menu: CandidateMenu, *, valid: bool = True):
    return {
        direction: DirectionalCalibration(
            direction=direction,
            offset=0.0,
            offset_variance=1.0e-8 if valid else 0.0,
            success_count=20 if valid else 0,
            trial_count=40 if valid else 0,
            row_count=10 if valid else 0,
            case_count=4 if valid else 0,
            alpha=4.0,
            menu_hash=menu.menu_hash,
            valid=valid,
        )
        for direction in ("0to1", "1to0")
    }


def test_directional_model_enforces_strict_h_q_e_and_has_no_residual_variance() -> None:
    rows = _donor_rows()
    model = fit_directional_logit(rows, heldout_h="0", family="R")
    prediction = predict_direction(
        model,
        candidate_source="2",
        feature_names=FEATURES,
        values=(4.0, 0.25),
    )
    assert 0.0 < prediction.probability < 1.0
    assert prediction.parameter_variance >= 0.0
    assert model.candidate_sources == ("1", "2", "3", "4")
    assert tuple(model.source_effects) == model.candidate_sources
    assert "residual_variance" not in model.to_payload()
    with pytest.raises(ProtocolError, match="outside trained topology"):
        predict_direction(
            model,
            candidate_source="9",
            feature_names=FEATURES,
            values=(4.0, 0.25),
        )
    with pytest.raises(ProtocolError, match="penalties are frozen"):
        fit_directional_logit(
            rows,
            heldout_h="0",
            family="R",
            source_alpha=SOURCE_ALPHA / 2.0,
        )
    with pytest.raises(ProtocolError, match="strict H/q/e"):
        rows[0].__class__(
            **{
                **rows[0].__dict__,
                "candidate_source": rows[0].query_center,
            }
        )


def test_prediction_applies_candidate_source_effect() -> None:
    model = _model("0to1", coefficient=0.0)
    source_one = predict_direction(
        model,
        candidate_source="1",
        feature_names=("signal",),
        values=(0.0,),
    )
    source_two = predict_direction(
        model,
        candidate_source="2",
        feature_names=("signal",),
        values=(0.0,),
    )
    assert source_one.probability > source_two.probability
    assert source_one.design != source_two.design


@pytest.mark.parametrize("family", ("G", "R", "P"))
def test_every_model_family_contains_frozen_source_effects(family: str) -> None:
    model = fit_directional_logit(_donor_rows(), heldout_h="0", family=family)
    feature_dimension = 0 if family == "G" else len(FEATURES)
    assert model.dimension == (
        1
        + feature_dimension
        + len(model.candidate_sources)
        + len(model.query_centers)
    )
    assert model.source_alpha == SOURCE_ALPHA
    assert tuple(model.source_effects) == ("1", "2", "3", "4")


def test_complete_case_permutation_deranges_features_without_moving_responses() -> None:
    rows = _donor_rows()
    permuted = permute_complete_case_feature_blocks(rows, seed=90_902_026)
    original = {
        (row.query_center, row.case_id, row.action_id, row.direction): row
        for row in rows
    }
    assert len(permuted) == len(rows)
    for row in permuted:
        prior = original[(row.query_center, row.case_id, row.action_id, row.direction)]
        assert (row.success_count, row.trial_count) == (
            prior.success_count,
            prior.trial_count,
        )
        assert row.feature_case_id != row.case_id


def test_support_menu_keeps_all_rank_evidence_but_routes_only_top_three() -> None:
    targets = {}
    for ordinal in range(8):
        action = f"A1::source={ordinal + 1}"
        targets[action] = tuple(
            ContributionTarget(
                case_id=f"case-{case}",
                action_id=action,
                delta_tp=ordinal - 2,
                delta_tn=0,
                n_positive=20,
                n_negative=20,
            )
            for case in range(5)
        )
    menu = build_candidate_menu(targets)
    assert menu.action_ids == (
        "B",
        "A1::source=8",
        "A1::source=7",
        "A1::source=6",
    )
    assert menu.anchor_action_id == "A1::source=8"
    assert len(menu.ranked_support_actions) == 8


def test_decision_uses_paired_joint_covariance_and_anchor_fallback() -> None:
    menu = _menu()
    models = {direction: _model(direction) for direction in ("0to1", "1to0")}
    calibrations = _calibrations(menu)
    baseline = baseline_action_score(models=models)

    def scored(action: str, signal: float, flips: int) -> ActionScore:
        predictions = {
            direction: predict_direction(
                models[direction],
                candidate_source=action.rsplit("=", 1)[-1],
                feature_names=("signal",),
                values=(signal,),
            )
            for direction in ("0to1", "1to0")
        }
        return score_action_against_baseline(
            action_id=action,
            predictions=predictions,
            models=models,
            calibrations=calibrations,
            flip_counts={"0to1": flips, "1to0": 0},
            n_positive=100,
            n_negative=100,
        )

    anchor = scored("A1::source=1", 0.5, 4)
    winner = scored("A1::source=2", 2.0, 30)
    third = scored("A1::source=3", -1.0, 2)
    decision = select_action_with_margin(
        case_id="case-1",
        method_id="R_multi",
        menu=menu,
        scores=(baseline, anchor, winner, third),
        models=models,
        calibrations=calibrations,
    )
    assert decision.selected_action_id == "A1::source=2"
    assert decision.reason == "positive_winner_runner_up_margin_lcb"
    naive = math.sqrt(
        winner.epistemic_variance
        + winner.calibration_variance
        + anchor.epistemic_variance
        + anchor.calibration_variance
    )
    assert decision.margin_standard_error < naive

    invalid = select_action_with_margin(
        case_id="case-1",
        method_id="R_multi",
        menu=menu,
        scores=(baseline, anchor, winner, third),
        models=models,
        calibrations=_calibrations(menu, valid=False),
    )
    assert invalid.selected_action_id == menu.anchor_action_id
    assert invalid.reason == "invalid_calibration_anchor_fallback"


def test_fitted_numeric_fingerprint_ignores_subprecision_fit_noise_only() -> None:
    left = {"coefficient": 0.123456789012345, "selection": "A"}
    right = {"coefficient": 0.123456789012346, "selection": "A"}
    assert fitted_numeric_fingerprint(left) == fitted_numeric_fingerprint(right)
    assert canonical_hash(left) != canonical_hash(right)
    assert fitted_numeric_fingerprint(left) != fitted_numeric_fingerprint(
        {"coefficient": left["coefficient"], "selection": "B"}
    )


def test_hierarchical_calibration_retains_prior_uncertainty_when_sparse() -> None:
    menu = _menu()
    prior_only = fit_direction_calibration(
        (), direction="0to1", menu_hash=menu.menu_hash
    )
    assert prior_only.valid is True
    assert prior_only.case_count == 0
    assert prior_only.offset == 0.0
    assert prior_only.offset_variance == pytest.approx(0.25)

    prediction = predict_direction(
        _model("0to1"),
        candidate_source="1",
        feature_names=("signal",),
        values=(0.5,),
    )
    one_case = fit_direction_calibration(
        (
            build_calibration_observation(
                case_id="support-case",
                action_id="A1::source=1",
                direction="0to1",
                success_count=2,
                trial_count=3,
                prediction=prediction,
            ),
        ),
        direction="0to1",
        menu_hash=menu.menu_hash,
    )
    assert one_case.valid is True
    assert one_case.case_count == 1
    assert 0.0 < one_case.offset_variance < prior_only.offset_variance
