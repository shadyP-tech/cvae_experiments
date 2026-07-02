from __future__ import annotations

import pytest

from midogpp_real_feature_gate.reliability import UtilityObservation, source_inner_scores
from midogpp_real_feature_gate.validation import ValidationError


def test_source_inner_scores_exclude_pseudo_target_expert_and_zscore_by_fold() -> None:
    scores = source_inner_scores(
        heldout_center="9",
        candidates=("0", "1", "2"),
        pseudo_targets=("0", "1"),
        utility_family="balanced_accuracy",
        observations=(
            UtilityObservation("9", "0", "1", "balanced_accuracy", 0.8),
            UtilityObservation("9", "0", "2", "balanced_accuracy", 0.4),
            UtilityObservation("9", "1", "0", "balanced_accuracy", 0.7),
            UtilityObservation("9", "1", "2", "balanced_accuracy", 0.3),
        ),
    )

    excluded = [
        row
        for row in scores.reliability_rows
        if row["pseudo_target_center"] == row["expert_center"]
    ]
    assert excluded
    assert all(row["eligible"] is False for row in excluded)
    assert all(row["fallback_reason"] == "pseudo_target_expert_excluded" for row in excluded)
    assert scores.scores["0"] == pytest.approx(1.0)
    assert scores.scores["1"] == pytest.approx(1.0)
    assert scores.scores["2"] == pytest.approx(-1.0)


def test_source_inner_scores_reject_mixed_utility_families() -> None:
    with pytest.raises(ValidationError):
        source_inner_scores(
            heldout_center="9",
            candidates=("0", "1"),
            pseudo_targets=("0",),
            utility_family="balanced_accuracy",
            observations=(UtilityObservation("9", "0", "1", "macro_f1", 0.8),),
        )
