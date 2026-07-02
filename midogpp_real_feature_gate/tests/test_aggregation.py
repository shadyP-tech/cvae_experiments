from __future__ import annotations

import pytest

from midogpp_real_feature_gate.aggregation import (
    WeightPolicy,
    aggregate_positive_probabilities,
    largest_remainder_allocation,
    source_inner_softmax_weights,
)
from midogpp_real_feature_gate.validation import ValidationError


def test_source_inner_softmax_weights_are_convex_and_do_not_resurrect_missing_scores() -> None:
    result = source_inner_softmax_weights(
        ("0", "1", "2"),
        {"0": 1.0, "1": 0.0},
        policy=WeightPolicy(tau=1.0, cap_min=0.05, shrinkage=0.1),
    )

    assert set(result.weights) == {"0", "1", "2"}
    assert result.weights["2"] == 0.0
    assert result.weights["0"] > result.weights["1"] > result.weights["2"]
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert result.fallback_reason == ""


def test_source_inner_softmax_weights_falls_back_to_uniform_for_ties() -> None:
    result = source_inner_softmax_weights(("0", "1"), {"0": 0.0, "1": 0.0}, policy=WeightPolicy())

    assert result.weights == {"0": 0.5, "1": 0.5}
    assert result.fallback_reason == "all_scores_tied_uniform"


def test_probability_aggregation_requires_aligned_members_and_convex_weights() -> None:
    probs = aggregate_positive_probabilities(
        {"0": [0.2, 0.8], "1": [0.6, 0.4]},
        {"0": 0.25, "1": 0.75},
    )

    assert probs == pytest.approx([0.5, 0.5])

    with pytest.raises(ValidationError):
        aggregate_positive_probabilities({"0": [0.2], "1": [0.6]}, {"0": 0.1, "1": 0.1})


def test_largest_remainder_allocation_is_deterministic() -> None:
    allocation = largest_remainder_allocation({"0": 0.5, "1": 0.3, "2": 0.2}, total_budget=7)

    assert allocation == {"0": 4, "1": 2, "2": 1}
    assert sum(allocation.values()) == 7
