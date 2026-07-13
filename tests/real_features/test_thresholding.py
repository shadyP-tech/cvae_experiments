from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.thresholding import (
    ThresholdPredictionSet,
    apply_threshold,
    fixed_threshold_spec,
    select_threshold_source_inner_lodo,
    threshold_policy_group_id,
)


def test_apply_threshold_uses_frozen_numeric_cutoff() -> None:
    assert apply_threshold([0.2, 0.5, 0.8], 0.5) == [0, 1, 1]
    assert apply_threshold([0.2, 0.5, 0.8], 0.7) == [0, 0, 1]


def test_source_inner_threshold_selection_excludes_outer_target() -> None:
    try:
        select_threshold_source_inner_lodo(
            outer_target_center="0",
            prediction_sets=[
                ThresholdPredictionSet(
                    pseudo_target_center="0",
                    y_true=(0, 1),
                    prob_pos=(0.2, 0.8),
                )
            ],
            threshold_policy_group_payload={"surface": "test"},
        )
    except ProtocolError:
        pass
    else:
        raise AssertionError("outer target center was accepted as threshold pseudo-target")


def test_source_inner_threshold_falls_back_when_too_few_valid_centers() -> None:
    result = select_threshold_source_inner_lodo(
        outer_target_center="0",
        prediction_sets=[
            ThresholdPredictionSet("1", (0, 1), (0.2, 0.8)),
            ThresholdPredictionSet("2", (0, 0), (0.2, 0.3)),
        ],
        threshold_policy_group_payload={"surface": "test"},
    )

    assert result.selected_threshold == 0.5
    assert result.decision.fallback_reason == "insufficient_valid_pseudo_targets"
    assert result.decision.n_valid_pseudo_targets == 1


def test_source_inner_threshold_is_invariant_to_target_metrics() -> None:
    prediction_sets = [
        ThresholdPredictionSet("1", (0, 1, 1), (0.2, 0.45, 0.8), scoring_unit_id="a"),
        ThresholdPredictionSet("2", (0, 1, 1), (0.1, 0.4, 0.9), scoring_unit_id="a"),
        ThresholdPredictionSet("3", (0, 1, 1), (0.3, 0.42, 0.7), scoring_unit_id="a"),
    ]
    payload = {"surface": "test", "target_metric_that_must_not_matter": 0.1}
    first = select_threshold_source_inner_lodo(
        outer_target_center="0",
        prediction_sets=prediction_sets,
        threshold_policy_group_payload=payload,
    )
    second = select_threshold_source_inner_lodo(
        outer_target_center="0",
        prediction_sets=prediction_sets,
        threshold_policy_group_payload=payload | {"target_metric_that_must_not_matter": 0.99},
    )

    assert first.selected_threshold == second.selected_threshold
    assert first.decision.threshold_source_score_table_hash == second.decision.threshold_source_score_table_hash


def test_fixed_threshold_spec_has_stable_group_identity() -> None:
    group_id = threshold_policy_group_id({"surface": "test", "seed": 42})
    spec = fixed_threshold_spec(0.5, threshold_policy_group_id=group_id)

    assert spec.threshold_policy == "fixed_0_5"
    assert spec.threshold_value == 0.5
    assert spec.threshold_policy_group_id == group_id
