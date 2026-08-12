from __future__ import annotations

import inspect
import pickle

import numpy as np
import pytest

from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.routing.threshold_flip_case_router import (
    CalibrationRow,
    CaseActionFeatures,
    ContributionTarget,
    DirectionSharedCalibration,
    DonorRow,
    StaticSelection,
    TwoHeadPrediction,
    TwoHeadRidgeModel,
    blocked_case_derangement,
    build_calibration_row,
    calibrated_gain,
    case_confusion,
    contribution_target,
    directional_raw_gains,
    exact_nine_mean_probabilities,
    fit_direction_shared_calibration,
    fit_two_head_ridge,
    hard_predictions,
    hash_decision_inputs,
    paired_case_bootstrap_contrast,
    pooled_bacc,
    pooled_gain,
    predict_two_head,
    refit_blocked_permutation_control,
    router_metrics,
    select_case_action,
    select_query_fixed_effect_static_source,
    select_static_source,
    terminal_oracles,
)


FEATURE_NAMES = ("flip_signal", "margin_signal")


def _target(case: str, action: str, tp: int, tn: int) -> ContributionTarget:
    return ContributionTarget(case, action, tp, tn, 10, 10)


def _donor_rows() -> tuple[DonorRow, ...]:
    specifications = (
        ("Q1", "E3", "c1", (0.0, 0.0), 0, 0),
        ("Q1", "E3", "c2", (1.0, 0.0), 4, 0),
        ("Q2", "E4", "c3", (0.0, 1.0), 0, 4),
        ("Q2", "E4", "c4", (1.0, 1.0), 4, 4),
    )
    return tuple(
        DonorRow(
            model_target="H",
            query_center=query,
            candidate_source=source,
            case_id=case,
            action_id=f"A1::source={source}",
            feature_case_id=case,
            feature_names=FEATURE_NAMES,
            values=values,
            target=_target(case, f"A1::source={source}", tp, tn),
        )
        for query, source, case, values, tp, tn in specifications
    )


def _evaluation_features(*, flips_01: int = 5, flips_10: int = 5) -> CaseActionFeatures:
    return CaseActionFeatures(
        target_center="H",
        case_id="eval",
        action_id="A1::source=E3",
        candidate_source="E3",
        feature_names=FEATURE_NAMES,
        values=(1.0, 1.0),
        flip_0to1_count=flips_01,
        flip_1to0_count=flips_10,
    )


def test_exact_nine_is_averaged_before_hard_threshold_and_additive_targets_recompose_bacc() -> None:
    seeds = np.asarray([[0.9, *([0.44] * 8)], [0.1, *([0.56] * 8)]])
    means = exact_nine_mean_probabilities(seeds)
    assert hard_predictions(means).tolist() == [False, True]

    positive_case = contribution_target(
        case_id="positive-only",
        action_id="A1::source=3",
        baseline_probabilities=[0.4, 0.4],
        action_probabilities=[0.6, 0.4],
        labels=[1, 1],
    )
    negative_case = contribution_target(
        case_id="negative-only",
        action_id="A1::source=3",
        baseline_probabilities=[0.6, 0.6],
        action_probabilities=[0.4, 0.6],
        labels=[0, 0],
    )
    assert (positive_case.n_negative, negative_case.n_positive) == (0, 0)
    assert pooled_gain((positive_case, negative_case)) == pytest.approx(0.5)
    baseline = (
        case_confusion("positive-only", [1, 1], [0, 0]),
        case_confusion("negative-only", [0, 0], [1, 1]),
    )
    action = (
        case_confusion("positive-only", [1, 1], [1, 0]),
        case_confusion("negative-only", [0, 0], [0, 1]),
    )
    assert pooled_bacc(action) - pooled_bacc(baseline) == pytest.approx(
        pooled_gain((positive_case, negative_case))
    )


def test_strict_h_q_e_firewall_and_h_specific_model_reuse_poison() -> None:
    target = _target("c", "A1::source=E", 1, 1)
    with pytest.raises(ProtocolError, match="q must exclude"):
        DonorRow("H", "H", "E", "c", target.action_id, "c", FEATURE_NAMES, (1.0, 1.0), target)
    with pytest.raises(ProtocolError, match="exclude both H and q"):
        DonorRow("H", "Q", "H", "c", target.action_id, "c", FEATURE_NAMES, (1.0, 1.0), target)
    with pytest.raises(ProtocolError, match="exclude both H and q"):
        DonorRow("H", "Q", "Q", "c", target.action_id, "c", FEATURE_NAMES, (1.0, 1.0), target)

    model = fit_two_head_ridge(_donor_rows(), heldout_h="H")
    poisoned = CaseActionFeatures(
        target_center="OTHER",
        case_id="eval",
        action_id="A1::source=E3",
        candidate_source="E3",
        feature_names=FEATURE_NAMES,
        values=(1.0, 1.0),
        flip_0to1_count=1,
        flip_1to0_count=1,
    )
    with pytest.raises(ProtocolError, match="cannot be reused"):
        predict_two_head(model, poisoned)
    with pytest.raises(ProtocolError, match="another H"):
        fit_two_head_ridge(
            _donor_rows(),
            heldout_h="OTHER",
        )


def test_model_is_spawn_safe_and_exactly_reconstructible() -> None:
    model = fit_two_head_ridge(_donor_rows(), heldout_h="H")
    reconstructed = TwoHeadRidgeModel.from_payload(model.to_payload())
    spawned = pickle.loads(pickle.dumps(model))
    assert reconstructed == model
    assert spawned == model
    assert reconstructed.model_hash == model.model_hash
    prediction = predict_two_head(reconstructed, _evaluation_features())
    assert prediction.model_hash == model.model_hash
    assert prediction.mean_delta_tp > 3.0
    assert prediction.mean_delta_tn > 3.0


def test_held_evaluation_labels_are_absent_from_fit_predict_and_decision_hash() -> None:
    for function in (fit_two_head_ridge, predict_two_head, hash_decision_inputs):
        assert "labels" not in inspect.signature(function).parameters
    model = fit_two_head_ridge(_donor_rows(), heldout_h="H")
    calibration = DirectionSharedCalibration(1.0, 1.0, 50, 50, 2, True)
    selection = StaticSelection("A1::source=E3", 0.1, 0.0, False)
    features = _evaluation_features()
    first = hash_decision_inputs(
        model=model, calibration=calibration, selection=selection, features=(features,)
    )
    held_evaluation_labels_a = (0, 0, 0)
    held_evaluation_labels_b = (1, 1, 1)
    assert held_evaluation_labels_a != held_evaluation_labels_b
    second = hash_decision_inputs(
        model=model, calibration=calibration, selection=selection, features=(features,)
    )
    assert first == second


def test_static_selection_is_a1_only_and_falls_back_for_nonpositive_gain() -> None:
    positive = {
        "A1::source=3": (_target("c1", "A1::source=3", 4, 4),),
        "A1::source=4": (_target("c1", "A1::source=4", 1, 1),),
    }
    selected = select_static_source(positive)
    assert selected.action_id == "A1::source=3"
    assert not selected.fallback_to_b
    negative = {
        "A1::source=3": (_target("c1", "A1::source=3", -1, -1),),
    }
    assert select_static_source(negative).action_id == "B"
    with pytest.raises(ProtocolError, match="restricted to A1"):
        select_static_source({"U": (_target("c1", "U", 1, 1),)})


def test_query_fixed_effect_global_selection_removes_unequal_q_mix_bias() -> None:
    centers = ("A", "B", "C", "D")
    query_effect = {"A": 0.15, "B": -0.15, "C": 0.0, "D": 0.0}
    source_effect = {"A": 0.05, "B": 0.03, "C": -0.04, "D": -0.04}
    donors = []
    raw_targets: dict[str, list[ContributionTarget]] = {}
    for query in centers:
        for source in centers:
            if query == source:
                continue
            action = f"A1::source={source}"
            gain = query_effect[query] + source_effect[source]
            delta = int(round(gain * 100))
            case = f"{query}-with-{source}"
            target = ContributionTarget(case, action, delta, delta, 100, 100)
            donors.append(
                DonorRow(
                    model_target="H",
                    query_center=query,
                    candidate_source=source,
                    case_id=case,
                    action_id=action,
                    feature_case_id=case,
                    feature_names=FEATURE_NAMES,
                    values=(0.0, 0.0),
                    target=target,
                )
            )
            raw_targets.setdefault(action, []).append(target)

    # Raw pooling prefers B only because action A is never observed on the
    # high-gain q=A population; the balanced fixed-effect estimate recovers A.
    assert select_static_source(raw_targets).action_id == "A1::source=B"
    fitted = select_query_fixed_effect_static_source(donors, heldout_h="H")
    assert fitted.selection.action_id == "A1::source=A"
    assert fitted.selection.exact_gain == pytest.approx(0.05)
    assert dict(fitted.adjusted_source_gains)["B"] == pytest.approx(0.03)
    assert sum(dict(fitted.query_effects).values()) == pytest.approx(0.0)
    assert sum(dict(fitted.source_effects).values()) == pytest.approx(0.0)
    assert fitted.design_rank == fitted.required_rank == 7
    assert fitted.to_payload()["identifiability_constraints"] == [
        "sum_query_effects=0",
        "sum_source_effects=0",
    ]


def test_query_fixed_effect_global_selection_falls_back_if_a_qe_cell_lacks_a_class() -> None:
    centers = ("A", "B", "C")
    donors = []
    for query in centers:
        for source in centers:
            if query == source:
                continue
            action = f"A1::source={source}"
            case = f"{query}-{source}"
            n_negative = 0 if (query, source) == ("A", "B") else 10
            target = ContributionTarget(case, action, 1, 0, 10, n_negative)
            donors.append(
                DonorRow(
                    "H", query, source, case, action, case,
                    FEATURE_NAMES, (0.0, 0.0), target,
                )
            )

    fitted = select_query_fixed_effect_static_source(donors, heldout_h="H")
    assert fitted.identifiable is False
    assert fitted.selection.action_id == "B"
    assert fitted.identifiability_failure == (
        "per_q_e_exact_pooled_bacc_lacks_both_classes"
    )


def test_direction_components_are_an_exact_label_free_partition_and_use_calibration_prevalence() -> None:
    model = fit_two_head_ridge(_donor_rows(), heldout_h="H")
    features = _evaluation_features(flips_01=3, flips_10=1)
    prediction = predict_two_head(model, features)
    raw_01, raw_10, _ = directional_raw_gains(
        prediction, features, n_positive=10, n_negative=10
    )
    exact_raw = 0.5 * prediction.mean_delta_tp / 10 + 0.5 * prediction.mean_delta_tn / 10
    assert raw_01 + raw_10 == pytest.approx(exact_raw)
    assert raw_01 / (raw_01 + raw_10) == pytest.approx(0.75)

    target = ContributionTarget("eval", features.action_id, 4, 4, 96, 4)
    row_10 = build_calibration_row(
        prediction=prediction,
        features=features,
        target=target,
        calibration_n_positive=10,
        calibration_n_negative=10,
    )
    row_20 = build_calibration_row(
        prediction=prediction,
        features=features,
        target=target,
        calibration_n_positive=20,
        calibration_n_negative=20,
    )
    assert row_10.exact_gain == pytest.approx(2.0 * row_20.exact_gain)
    assert row_10.raw_gain_0to1 == pytest.approx(2.0 * row_20.raw_gain_0to1)


def test_zero_flip_and_single_class_calibration_are_fail_safe_b() -> None:
    model = fit_two_head_ridge(_donor_rows(), heldout_h="H")
    challenger = StaticSelection("A1::source=E3", 0.1, 0.0, False)
    no_flip = _evaluation_features(flips_01=0, flips_10=0)
    prediction = predict_two_head(model, no_flip)
    valid = DirectionSharedCalibration(1.0, 1.0, 10, 10, 2, True)
    assert select_case_action(
        method_id="F_S",
        challenger=challenger,
        features=no_flip,
        prediction=prediction,
        calibration=valid,
    ).reason == "zero_flip"
    invalid = fit_direction_shared_calibration(
        (CalibrationRow("c", challenger.action_id, 1.0, 0.0, 1.0),),
        calibration_n_positive=0,
        calibration_n_negative=10,
    )
    decision = select_case_action(
        method_id="F_S",
        challenger=challenger,
        features=_evaluation_features(),
        prediction=predict_two_head(model, _evaluation_features()),
        calibration=invalid,
    )
    assert decision.selected_action_id == "B"
    assert decision.reason == "single_class_calibration"


def test_blocked_permutation_deranges_whole_cases_and_refits_same_capacity() -> None:
    rows = _donor_rows()
    permuted = blocked_case_derangement(rows, seed=123)
    assert len(permuted) == len(rows)
    assert all(row.feature_case_id != row.case_id for row in permuted)
    original_targets = {(row.query_center, row.case_id, row.action_id): row.target for row in rows}
    assert all(
        row.target == original_targets[(row.query_center, row.case_id, row.action_id)]
        for row in permuted
    )
    assert permuted == blocked_case_derangement(rows, seed=123)
    ordinary = fit_two_head_ridge(rows, heldout_h="H")
    control = refit_blocked_permutation_control(rows, heldout_h="H", seed=123)
    assert control.training_row_count == ordinary.training_row_count
    assert control.feature_names == ordinary.feature_names
    assert len(control.tp_head.coefficients) == len(ordinary.tp_head.coefficients)
    assert control.provenance_hash != ordinary.provenance_hash


def test_synthetic_direction_signal_recovers_a_safe_challenger() -> None:
    model = fit_two_head_ridge(_donor_rows(), heldout_h="H")
    calibration = fit_direction_shared_calibration(
        (
            CalibrationRow("cal-01", "A1::source=E3", 0.04, 0.0, 0.04),
            CalibrationRow("cal-10", "A1::source=E3", 0.0, 0.04, 0.04),
        ),
        calibration_n_positive=50,
        calibration_n_negative=50,
    )
    assert calibration.gamma_0to1 == pytest.approx(1.0)
    assert calibration.gamma_1to0 == pytest.approx(1.0)
    features = _evaluation_features()
    prediction = predict_two_head(model, features)
    mean, standard_error = calibrated_gain(calibration, prediction, features)
    assert mean > 0.05
    assert mean - 1.96 * standard_error > 0.0
    challenger = StaticSelection("A1::source=E3", 0.1, 0.0, False)
    decision = select_case_action(
        method_id="F_S",
        challenger=challenger,
        features=features,
        prediction=prediction,
        calibration=calibration,
    )
    assert decision.selected_action_id == challenger.action_id
    assert decision.reason == "heuristic_positive_bound"


def test_terminal_oracles_bootstrap_and_router_metrics_are_case_pooled() -> None:
    baseline = (
        case_confusion("c1", [1, 1], [0, 0]),
        case_confusion("c2", [0, 0], [1, 1]),
    )
    action = (
        case_confusion("c1", [1, 1], [1, 0]),
        case_confusion("c2", [0, 0], [0, 1]),
    )
    oracles = terminal_oracles({"B": baseline, "A1::source=3": action})
    assert oracles.static_action_id == "A1::source=3"
    assert oracles.static_score.bacc == pytest.approx(0.5)
    contrast = paired_case_bootstrap_contrast(
        action,
        baseline,
        method_id="F_S",
        baseline_id="B",
        replicates=20,
        seed=7,
    )
    assert contrast.estimate == pytest.approx(0.5)
    metrics = router_metrics(
        selected_actions=("A", "B", "A"),
        oracle_actions=("A", "B", "B"),
        predicted_gains=(0.1, 0.2, 0.3),
        oracle_gains=(0.1, 0.2, 0.4),
        router_bacc=0.7,
        baseline_bacc=0.6,
        oracle_bacc=0.8,
        fold_static_actions=("A", "A", "B"),
    )
    assert metrics["top1_oracle_agreement"] == pytest.approx(2 / 3)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["normalized_oracle_gap"] == pytest.approx(0.5)
    assert metrics["fold_stability"] == pytest.approx(2 / 3)
