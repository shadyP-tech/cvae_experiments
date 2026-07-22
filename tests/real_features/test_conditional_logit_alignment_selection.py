from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.config import (
    EXPECTED_VERSIONS,
    GAMMA_GRID,
    load_conditional_logit_alignment_config,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.selection import (
    GammaFoldScore,
    plan_outer_evaluation,
    summarize_gamma_scores,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError


CONFIG_PATH = Path(
    "experiments/midogpp/stages/10_real_feature_reference/configs/"
    "conditional_logit_alignment_v1.yaml"
)


def test_equal_inner_center_arithmetic_mean_selects_gamma() -> None:
    scores = _scores(
        {
            0.0: (0.9, 0.1),
            1.0e-4: (0.6, 0.6),
        },
        default=(0.2, 0.2),
    )

    selection = summarize_gamma_scores(
        outer_target_center="0",
        fold_scores=scores,
        expected_inner_centers=("1", "2"),
    )

    by_gamma = {summary.gamma: summary for summary in selection.gamma_summaries}
    assert by_gamma[0.0].equal_center_mean_bacc == pytest.approx(0.5)
    assert by_gamma[1.0e-4].equal_center_mean_bacc == pytest.approx(0.6)
    assert selection.selected_gamma == 1.0e-4
    assert selection.selected_summary.gamma == 1.0e-4


def test_smallest_gamma_wins_absolute_1e_12_tie() -> None:
    within = _scores(
        {
            0.0: (0.8, 0.8),
            1.0e-4: (0.8000000000005, 0.8000000000005),
        },
        default=(0.1, 0.1),
    )
    outside = _scores(
        {
            0.0: (0.8, 0.8),
            1.0e-4: (0.800000000002, 0.800000000002),
        },
        default=(0.1, 0.1),
    )

    tied = summarize_gamma_scores(
        outer_target_center="0", fold_scores=within
    )
    distinct = summarize_gamma_scores(
        outer_target_center="0", fold_scores=outside
    )

    assert tied.selected_gamma == 0.0
    assert distinct.selected_gamma == 1.0e-4
    assert tied.tie_atol == 1.0e-12
    assert tied.tie_rtol == 0.0


def test_nonconverged_gamma_is_ineligible_on_every_center_rule() -> None:
    scores = list(
        _scores(
            {
                0.0: (0.6, 0.6),
                1.0e-4: (0.9, 0.9),
            },
            default=(0.2, 0.2),
        )
    )
    for index, score in enumerate(scores):
        if score.gamma == 1.0e-4 and score.inner_pseudo_target_center == "2":
            scores[index] = GammaFoldScore(
                outer_target_center=score.outer_target_center,
                inner_pseudo_target_center=score.inner_pseudo_target_center,
                gamma=score.gamma,
                bacc=score.bacc,
                macro_f1=score.macro_f1,
                converged=False,
                status="optimizer_nonconverged",
            )

    selection = summarize_gamma_scores(
        outer_target_center="0", fold_scores=scores
    )

    assert selection.selected_gamma == 0.0
    summary = next(row for row in selection.gamma_summaries if row.gamma == 1.0e-4)
    assert summary.eligible is False


def test_selected_gamma_zero_reuses_one_physical_outer_fit() -> None:
    plan = plan_outer_evaluation(0.0)

    assert plan.shared_fit is True
    assert plan.unique_fit_gammas == (0.0,)
    assert plan.gamma_for_role("selected") == 0.0
    assert plan.gamma_for_role("gamma0") == 0.0

    nonzero = plan_outer_evaluation(0.1)
    assert nonzero.shared_fit is False
    assert nonzero.unique_fit_gammas == (0.1, 0.0)


def test_summary_rejects_incomplete_or_wrong_outer_score_matrix() -> None:
    scores = list(_scores({}, default=(0.5, 0.5)))
    with pytest.raises(ProtocolError, match="incomplete"):
        summarize_gamma_scores(outer_target_center="0", fold_scores=scores[:-1])

    wrong = scores.copy()
    score = wrong[0]
    wrong[0] = GammaFoldScore(
        outer_target_center="3",
        inner_pseudo_target_center=score.inner_pseudo_target_center,
        gamma=score.gamma,
        bacc=score.bacc,
        macro_f1=score.macro_f1,
    )
    with pytest.raises(ProtocolError, match="wrong outer"):
        summarize_gamma_scores(outer_target_center="0", fold_scores=wrong)


def test_frozen_config_loader_accepts_exact_yaml_and_rejects_drift(
    tmp_path: Path,
) -> None:
    config = load_conditional_logit_alignment_config(CONFIG_PATH)

    assert config.gamma_grid == GAMMA_GRID
    assert config.classifier_spec.C == 0.01
    assert config.classifier_spec.class_weight is None
    assert dict(config.expected_versions) == dict(EXPECTED_VERSIONS)
    assert config.optimizer.require_single_thread is True

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["selection"]["tie_rtol"] = 1.0e-9
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProtocolError, match="selection values drifted"):
        load_conditional_logit_alignment_config(drifted)


def _scores(
    overrides: dict[float, tuple[float, float]],
    *,
    default: tuple[float, float],
) -> tuple[GammaFoldScore, ...]:
    rows = []
    for inner_index, inner in enumerate(("1", "2")):
        for gamma in GAMMA_GRID:
            values = overrides.get(gamma, default)
            rows.append(
                GammaFoldScore(
                    outer_target_center="0",
                    inner_pseudo_target_center=inner,
                    gamma=gamma,
                    bacc=values[inner_index],
                    macro_f1=values[inner_index],
                )
            )
    return tuple(rows)
