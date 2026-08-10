from __future__ import annotations

import math

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_actionability_recoverability import (
    B_ACTION_ID,
    GEOMETRY_IDS,
    U_ACTION_ID,
    ActionScoreRow,
    BinaryLabelRow,
    BinaryPredictionRow,
    CaseActionFeatureRow,
    CaseConfusionCounts,
    DiagnosticEvaluationResult,
    GeometryEvaluationResult,
    MethodEvaluationResult,
    PooledBacc,
    SeedProbabilityRow,
    UtilityTargetRow,
    actions_for_target,
    aggregate_exact_nine_probabilities,
    build_action_library,
    build_label_free_case_action_features,
    build_class_balanced_proper_loss_targets,
    build_pre_support_decisions,
    build_support_static_decisions,
    candidate_sources,
    complementarity_metrics,
    fit_fixed_alpha_ridge_models,
    geometry_action_id,
    matched_blocked_feature_permutation,
    normalized_oracle_gap,
    predict_action_scores,
    rank_stability,
    terminal_oracles,
)
from midogpp_thesis.cvae.protocol import ProtocolError


STORE_HASH = "a" * 64


def _seed_rows(target: str = "0") -> tuple[SeedProbabilityRow, ...]:
    output: list[SeedProbabilityRow] = []
    actions = actions_for_target(target)
    for case_id, sample_id, sample_shift in (
        ("case-a", "sample-0", 0.00),
        ("case-a", "sample-1", 0.01),
        ("case-b", "sample-2", 0.02),
        ("case-b", "sample-3", 0.03),
    ):
        for action_index, action in enumerate(actions):
            for seed in range(9):
                output.append(
                    SeedProbabilityRow(
                        target_center=target,
                        case_id=case_id,
                        sample_id=sample_id,
                        action_id=action.action_id,
                        seed_pair_ordinal=seed,
                        probability=0.20 + sample_shift + 0.01 * action_index + 0.001 * seed,
                        probability_store_hash=STORE_HASH,
                    )
                )
    return tuple(output)


def test_action_library_freezes_b_u_a0_a1_and_never_selects_seeds() -> None:
    target_actions = actions_for_target("0")
    assert len(target_actions) == 18
    assert len(build_action_library()) == 9 * 18
    assert tuple(action.action_id for action in target_actions[:2]) == ("B", "U")
    assert tuple(action.geometry_id for action in target_actions[2:10]) == ("A0",) * 8
    assert tuple(action.geometry_id for action in target_actions[10:]) == ("A1",) * 8
    assert all(action.selected_source != "0" for action in target_actions)
    assert len({action.action_id for action in target_actions}) == len(target_actions)

    baseline, uniform = target_actions[:2]
    assert baseline.physical_fit_required is True
    assert uniform.physical_fit_required is True
    assert sum(baseline.counts_by_class[0].values()) == 1024
    assert sum(uniform.counts_by_class[0].values()) == 1152

    a0 = next(action for action in target_actions if action.action_id == geometry_action_id("A0", "1"))
    a1 = next(action for action in target_actions if action.action_id == geometry_action_id("A1", "1"))
    assert dict(a0.counts_by_class[0]) == dict(a1.counts_by_class[0])
    assert a0.counts_by_class[0]["1"] == 256
    assert all(a0.counts_by_class[0][source] == 128 for source in candidate_sources("0") if source != "1")
    assert a1.sample_weight_by_source["1"] == 23 / 16
    assert all(a1.sample_weight_by_source[source] == 7 / 8 for source in candidate_sources("0") if source != "1")
    assert math.isclose(
        sum(
            a1.counts_by_class[0][source] * a1.sample_weight_by_source[source]
            for source in candidate_sources("0")
        ),
        1152.0,
    )


def test_exact_nine_aggregation_requires_complete_actions_and_averages_seeds() -> None:
    seed_rows = _seed_rows()
    surface = aggregate_exact_nine_probabilities(seed_rows)
    assert len(surface.rows) == 4 * 18
    first = next(
        row
        for row in surface.rows
        if row.case_id == "case-a" and row.sample_id == "sample-0" and row.action_id == B_ACTION_ID
    )
    assert first.seed_pair_count == 9
    assert first.probability_mean == pytest.approx(0.204)
    assert surface.predictions_sealed_before_labels is True

    with pytest.raises(ProtocolError, match="ordinals"):
        aggregate_exact_nine_probabilities(seed_rows[:-1])

    missing_action = tuple(
        row
        for row in seed_rows
        if not (
            row.sample_id == "sample-0"
            and row.action_id == actions_for_target("0")[-1].action_id
        )
    )
    with pytest.raises(ProtocolError, match="Every sample must cover"):
        aggregate_exact_nine_probabilities(missing_action)


def test_label_free_features_and_blocked_permutation_preserve_case_blocks() -> None:
    features = build_label_free_case_action_features(
        aggregate_exact_nine_probabilities(_seed_rows())
    )
    assert len(features) == 2 * 2 * 8
    assert all(row.values[0] == 1.0 for row in features)
    assert all(row.selected_source != row.query_center for row in features)

    permuted = matched_blocked_feature_permutation(
        features, excluded_candidate_centers=("0",)
    )
    assert tuple(row.row_key for row in permuted) == tuple(row.row_key for row in features)
    assert all(row.feature_origin_source != row.selected_source for row in permuted)
    for case_id in ("case-a", "case-b"):
        for geometry in GEOMETRY_IDS:
            aligned_block = {
                row.values
                for row in features
                if row.case_id == case_id and row.geometry_id == geometry
            }
            permuted_block = {
                row.values
                for row in permuted
                if row.case_id == case_id and row.geometry_id == geometry
            }
            assert aligned_block == permuted_block


def test_dense_targets_are_additive_class_balanced_proper_loss_gain_vs_u() -> None:
    surface = aggregate_exact_nine_probabilities(_seed_rows())
    labels = (
        BinaryLabelRow("0", "case-a", "sample-0", 0),
        BinaryLabelRow("0", "case-a", "sample-1", 1),
        BinaryLabelRow("0", "case-b", "sample-2", 0),
        BinaryLabelRow("0", "case-b", "sample-3", 1),
    )
    targets = build_class_balanced_proper_loss_targets(surface, labels)
    assert len(targets) == 2 * 2 * 8
    assert {row.response_kind for row in targets} == {
        "class_balanced_proper_loss_gain_vs_u"
    }
    assert all(math.isfinite(row.response) for row in targets)


def _model_surface() -> tuple[tuple[CaseActionFeatureRow, ...], tuple[UtilityTargetRow, ...]]:
    features: list[CaseActionFeatureRow] = []
    targets: list[UtilityTargetRow] = []
    centers = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
    for query_index, query in enumerate(centers):
        for case_index in range(2):
            for source in candidate_sources(query):
                source_index = centers.index(source)
                values = (
                    1.0,
                    0.2 + 0.01 * query_index,
                    0.01,
                    0.02,
                    0.02,
                    0.01 * source_index,
                    0.01 * source_index,
                    0.005,
                    0.01 * source_index,
                    0.02 * source_index,
                    0.5,
                    float(case_index),
                    0.001,
                )
                features.append(
                    CaseActionFeatureRow(
                        query_center=query,
                        case_id=f"{query}-case-{case_index}",
                        geometry_id="A0",
                        selected_source=source,
                        values=values,
                    )
                )
                targets.append(
                    UtilityTargetRow(
                        query_center=query,
                        case_id=f"{query}-case-{case_index}",
                        geometry_id="A0",
                        selected_source=source,
                        response=0.02 * source_index + 0.001 * case_index,
                    )
                )
    return tuple(features), tuple(targets)


@pytest.mark.parametrize("family", ("G", "R", "P"))
def test_fixed_alpha_models_enforce_outer_candidate_and_nested_donor_exclusions(family: str) -> None:
    features, targets = _model_surface()
    models = fit_fixed_alpha_ridge_models(
        features,
        targets,
        outer_target_center="0",
        geometry_id="A0",
        family=family,
    )
    assert tuple(model.selected_source for model in models) == candidate_sources("0")
    for model in models:
        assert "0" not in model.training_query_centers
        assert model.selected_source not in model.training_query_centers
        assert model.ridge_alpha == 1.0
        assert model.feature_names == (("intercept",) if family == "G" else tuple(row for row in models[0].feature_names))

    scores = predict_action_scores(models, features)
    assert len(scores) == 2 * 8
    assert {row.target_center for row in scores} == {"0"}

    nested = fit_fixed_alpha_ridge_models(
        features,
        targets,
        outer_target_center="0",
        heldout_donor_center="1",
        geometry_id="A0",
        family=family,
    )
    assert len(nested) == 7
    assert all(model.selected_source != "1" for model in nested)
    assert all("1" not in model.training_query_centers for model in nested)
    with pytest.raises(ProtocolError, match="fixed at 1.0"):
        fit_fixed_alpha_ridge_models(
            features,
            targets,
            outer_target_center="0",
            geometry_id="A0",
            family=family,
            ridge_alpha=0.1,
        )


def _decision_scores() -> tuple[ActionScoreRow, ...]:
    rows: list[ActionScoreRow] = []
    for case_id in ("eval-a", "eval-b"):
        for geometry in GEOMETRY_IDS:
            for family in ("G", "R", "P"):
                for source in candidate_sources("0"):
                    gain = 0.2 if source == "1" and family != "P" else -0.1
                    rows.append(
                        ActionScoreRow(
                            target_center="0",
                            case_id=case_id,
                            geometry_id=geometry,
                            selected_source=source,
                            family=family,
                            predicted_gain=gain,
                            model_hash="b" * 64,
                        )
                    )
    return tuple(rows)


def test_pre_support_decisions_keep_b_global_and_fall_back_to_shared_u() -> None:
    decisions = build_pre_support_decisions(
        target_center="0",
        case_ids=("eval-a", "eval-b"),
        action_scores=_decision_scores(),
    )
    baseline = tuple(row for row in decisions if row.method_id == "B")
    assert len(baseline) == 2
    assert all(row.geometry_id is None and row.action_id == B_ACTION_ID for row in baseline)
    assert all(
        row.action_id == U_ACTION_ID
        for row in decisions
        if row.method_id == "P"
    )
    assert all(
        row.action_id == geometry_action_id(row.geometry_id, "1")
        for row in decisions
        if row.method_id in ("G", "R")
    )


def _support_counts() -> tuple[CaseConfusionCounts, ...]:
    rows: list[CaseConfusionCounts] = []
    actions = (
        U_ACTION_ID,
        *(geometry_action_id("A0", source) for source in candidate_sources("0")),
    )
    for case_id in ("support-a", "support-b"):
        for action in actions:
            correct = 10 if action == geometry_action_id("A0", "1") else 7
            rows.append(CaseConfusionCounts("0", case_id, action, 10, correct, 10, correct))
    return tuple(rows)


def test_support_static_selector_uses_u_plus_sources_and_never_eval_labels() -> None:
    decisions = build_support_static_decisions(
        target_center="0",
        geometry_id="A0",
        support_counts=_support_counts(),
        evaluation_case_ids=("eval-a", "eval-b"),
    )
    assert {row.action_id for row in decisions} == {geometry_action_id("A0", "1")}
    assert all(row.predicted_gain == pytest.approx(0.3) for row in decisions)
    assert all(row.evaluation_labels_used is False for row in decisions)
    with pytest.raises(ProtocolError, match="disjoint"):
        build_support_static_decisions(
            target_center="0",
            geometry_id="A0",
            support_counts=_support_counts(),
            evaluation_case_ids=("support-a",),
        )


def _oracle_counts() -> tuple[CaseConfusionCounts, ...]:
    rows: list[CaseConfusionCounts] = []
    action_order = (
        U_ACTION_ID,
        *(geometry_action_id("A0", source) for source in candidate_sources("0")),
    )
    first = geometry_action_id("A0", "1")
    second = geometry_action_id("A0", "2")
    for case_id in ("case-a", "case-b"):
        for action in action_order:
            correct = 5
            if (case_id, action) in (("case-a", first), ("case-b", second)):
                correct = 10
            elif action in (first, second):
                correct = 0
            rows.append(CaseConfusionCounts("0", case_id, action, 10, correct, 10, correct))
    return tuple(rows)


def test_terminal_oracles_use_additive_pooled_bacc_not_case_bacc() -> None:
    static, case = terminal_oracles(
        _oracle_counts(), target_center="0", geometry_id="A0"
    )
    assert static.oracle_method == "O_static"
    assert {action for _case, action in static.selected_action_by_case} == {U_ACTION_ID}
    assert static.pooled_bacc.exact_bacc == pytest.approx(0.5)
    assert case.oracle_method == "O_case"
    assert case.pooled_bacc.exact_bacc == pytest.approx(1.0)
    assert dict(case.selected_action_by_case) == {
        "case-a": geometry_action_id("A0", "1"),
        "case-b": geometry_action_id("A0", "2"),
    }


def test_complementarity_rank_stability_and_normalized_gap() -> None:
    labels = (
        BinaryLabelRow("0", "c", "s0", 0),
        BinaryLabelRow("0", "c", "s1", 1),
    )
    predictions = (
        BinaryPredictionRow("0", "c", "s0", "left", 0),
        BinaryPredictionRow("0", "c", "s1", "left", 0),
        BinaryPredictionRow("0", "c", "s0", "right", 1),
        BinaryPredictionRow("0", "c", "s1", "right", 1),
    )
    result = complementarity_metrics(predictions, labels, target_center="0")
    assert len(result) == 1
    assert result[0].disagreement_rate == 1.0
    assert result[0].left_only_correct_rate == 0.5
    assert result[0].right_only_correct_rate == 0.5

    stability = rank_stability(
        {"U": 0.0, "e1": 2.0, "e2": 1.0},
        {"U": 0.0, "e1": 3.0, "e2": 1.0},
    )
    assert stability.identifiable is True
    assert stability.spearman == pytest.approx(1.0)
    assert normalized_oracle_gap(selected=0.7, baseline=0.5, oracle=0.9) == pytest.approx(0.5)
    assert normalized_oracle_gap(selected=0.95, baseline=0.5, oracle=0.9) == pytest.approx(-0.125)


def test_evaluation_contract_hard_codes_consumed_test_claim_boundary() -> None:
    pooled = PooledBacc("method", 2, 20, 15, 20, 15, 0.75, 0.75, 0.75)
    stability = rank_stability({"U": 0.0, "e": 1.0}, {"U": 0.0, "e": 1.0})

    def geometry_result(geometry: str) -> GeometryEvaluationResult:
        methods = tuple(
            MethodEvaluationResult(
                geometry_id=geometry,
                method_id=method,
                pooled_bacc=pooled,
                delta_vs_b=0.0,
                delta_vs_u=0.0,
                decision_count=2,
                terminal_diagnostic_only=method.startswith("O_"),
            )
            for method in ("U", "G", "R", "P", "S_y", "O_static", "O_case")
        )
        return GeometryEvaluationResult(
            geometry_id=geometry,
            methods=methods,
            rank_stability=stability,
            complementarity=(),
            normalized_oracle_gaps={method: 0.0 for method in ("G", "R", "P", "S_y")},
        )

    result = DiagnosticEvaluationResult(
        global_b=pooled,
        geometries=(geometry_result("A0"), geometry_result("A1")),
    )
    assert result.claim_status == "EXPLORATORY_CONSUMED_DATA_ONLY"
    assert result.routing_or_promotion_authorized is False
    assert result.another_experiment_authorized is False
    with pytest.raises(ProtocolError, match="cannot authorize"):
        DiagnosticEvaluationResult(
            global_b=pooled,
            geometries=(geometry_result("A0"), geometry_result("A1")),
            routing_or_promotion_authorized=True,
        )
