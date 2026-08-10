from __future__ import annotations

from dataclasses import replace
import math

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_core import (
    BOOTSTRAP_SEED,
    BinaryLabel,
    CaseClassWeights,
    CaseConfusionCounts,
    DonorResponseRow,
    SampleActionProbability,
    baseline_predictions,
    build_donor_responses,
    calibrated_baseline_predictions,
    compose_probabilities,
    compute_case_features,
    fit_baseline_intercept,
    logit_clip,
    paired_whole_case_cluster_lcb,
    permute_case_features,
    pooled_exact_bacc,
    score_case_confusions,
    sigmoid,
    soft_class_residual,
    top2_sparse_simplex,
    whole_case_bootstrap,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.scientific_constants import (
    DESIGN_TERMS,
    PHI_NAMES,
    PROBABILITY_EPSILON,
    RIDGE_GRID,
    VARIANCE_FLOOR,
    candidate_sources,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_hierarchical_residual_stacker.config_payloads import (
    LOCAL_RESIDUAL_FEATURE_NAMES,
    LOGIT_CLIP_EPSILON,
    MODEL_FEATURE_NAMES,
    RIDGE_ALPHA_GRID,
    VARIANCE_FLOOR as CONFIG_VARIANCE_FLOOR,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _probabilities(
    *,
    target: str = "0",
    cases: tuple[str, ...] = ("a", "b"),
    candidate_shift: float = 0.0,
) -> tuple[SampleActionProbability, ...]:
    rows: list[SampleActionProbability] = []
    for case_index, case in enumerate(cases):
        for sample_index in range(2):
            sample = f"{case}-{sample_index}"
            baseline = (0.2, 0.8)[sample_index]
            rows.append(SampleActionProbability(target, case, sample, "B", baseline))
            for source_index, source in enumerate(candidate_sources(target)):
                candidate = min(
                    max(baseline + candidate_shift + 0.01 * (source_index - 3), 0.001),
                    0.999,
                )
                rows.append(
                    SampleActionProbability(target, case, sample, source, candidate)
                )
    return tuple(rows)


def _weights(target: str = "0") -> tuple[CaseClassWeights, ...]:
    gains = tuple((source, 1.0 - index * 0.01) for index, source in enumerate(candidate_sources(target)))
    selected = tuple(sorted(((candidate_sources(target)[0], 0.7), (candidate_sources(target)[1], 0.3))))
    return tuple(
        CaseClassWeights(target, case, side, selected, gains)
        for case in ("a", "b")
        for side in (0, 1)
    )


def _labels(scope: str = "target_support") -> tuple[BinaryLabel, ...]:
    return tuple(
        BinaryLabel("0", case, f"{case}-{sample}", sample, scope)  # type: ignore[arg-type]
        for case in ("a", "b")
        for sample in (0, 1)
    )


def test_scientific_constants_are_directly_bound_to_canonical_config() -> None:
    assert PROBABILITY_EPSILON == LOGIT_CLIP_EPSILON
    assert RIDGE_GRID == RIDGE_ALPHA_GRID
    assert VARIANCE_FLOOR == CONFIG_VARIANCE_FLOOR == 1.0e-6
    assert PHI_NAMES == LOCAL_RESIDUAL_FEATURE_NAMES
    assert DESIGN_TERMS == MODEL_FEATURE_NAMES


def test_logit_sigmoid_are_bounded_and_stable_at_probability_limits() -> None:
    assert math.isfinite(logit_clip(0.0))
    assert math.isfinite(logit_clip(1.0))
    assert sigmoid(logit_clip(0.0)) == pytest.approx(1.0e-4)
    assert sigmoid(logit_clip(1.0)) == pytest.approx(1.0 - 1.0e-4)
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(1000.0) == 1.0


def test_case_features_are_per_case_and_phi_only_permutation_is_deterministic() -> None:
    features = compute_case_features(_probabilities())
    assert len(features) == 2 * 8
    assert {(row.target_center, row.case_id) for row in features} == {("0", "a"), ("0", "b")}
    first = permute_case_features(features)
    second = permute_case_features(tuple(reversed(features)))
    assert first == second
    for case in ("a", "b"):
        original = sorted(row.phi for row in features if row.case_id == case)
        permuted = sorted(row.phi for row in first if row.case_id == case)
        assert original == permuted
        assert all(row.feature_origin_source_id != row.source_id for row in first if row.case_id == case)


def test_positive_only_top2_simplex_has_source_id_ties_and_no_gain_fallback() -> None:
    weights = top2_sparse_simplex({"3": 0.2, "2": 0.2, "1": 0.1})
    assert tuple(source for source, _ in weights) == ("2", "3")
    assert sum(value for _, value in weights) == pytest.approx(1.0)
    assert dict(weights)["2"] == pytest.approx(0.5)
    assert top2_sparse_simplex({"1": -0.1, "2": 0.0, "3": -2.0}) == ()


def test_soft_class_mixing_has_exact_limits_and_midpoint() -> None:
    assert soft_class_residual(-2.0, 4.0, 0.0) == -2.0
    assert soft_class_residual(-2.0, 4.0, 1.0) == 4.0
    assert soft_class_residual(-2.0, 4.0, 0.5) == 1.0


def test_lambda_zero_is_bit_exact_bcal_and_candidate_values_cannot_change_bcal() -> None:
    probabilities = _probabilities()
    labels = _labels()
    calibration = fit_baseline_intercept(probabilities, labels)
    poisoned = _probabilities(candidate_shift=0.15)
    assert fit_baseline_intercept(poisoned, labels).intercept == calibration.intercept
    baseline_calibrated = calibrated_baseline_predictions(
        probabilities, intercept=calibration.intercept
    )
    composed = compose_probabilities(
        probabilities,
        _weights(),
        intercept=calibration.intercept,
        residual_scale=0.0,
        method_id="R",
    )
    assert tuple(row.probability for row in composed) == tuple(
        row.probability for row in baseline_calibrated
    )


def test_single_class_cases_are_retained_and_only_pooled_scope_requires_both() -> None:
    negative = CaseConfusionCounts("R", "0", "negative", 0, 0, 2, 1)
    positive = CaseConfusionCounts("R", "0", "positive", 3, 2, 0, 0)
    with pytest.raises(ProtocolError, match="both classes"):
        pooled_exact_bacc((negative,))
    metric = pooled_exact_bacc((negative, positive))
    assert metric.case_count == 2
    assert metric.exact_bacc == pytest.approx(0.5 * (2 / 3 + 1 / 2))
    assert metric.to_payload()["per_case_bacc_used"] is False


def test_whole_case_uncertainty_and_permutation_are_deterministic() -> None:
    challenger = tuple(
        CaseConfusionCounts("R", center, case, 2, tp, 2, tn)
        for center in ("0", "1")
        for case, tp, tn in (("a", 2, 1), ("b", 1, 2), ("c", 2, 2))
    )
    reference = tuple(
        CaseConfusionCounts("B_cal", center, case, 2, 1, 2, 1)
        for center in ("0", "1")
        for case in ("a", "b", "c")
    )
    contrast = paired_whole_case_cluster_lcb(challenger, reference)
    assert contrast.case_count == 6
    first = whole_case_bootstrap(challenger, reference, replicates=100, seed=BOOTSTRAP_SEED)
    second = whole_case_bootstrap(challenger, reference, replicates=100, seed=BOOTSTRAP_SEED)
    assert first == second
    assert first.bootstrap_hash == second.bootstrap_hash


def test_smooth_response_poison_cannot_change_exact_terminal_endpoint() -> None:
    probabilities = _probabilities()
    donor_labels = tuple(replace(row, label_scope="loco_donor") for row in _labels())
    responses = build_donor_responses(probabilities, donor_labels)
    poisoned = tuple(replace(row, smooth_response=-row.smooth_response) for row in responses)
    terminal_labels = tuple(replace(row, label_scope="terminal_evaluation") for row in _labels())
    predictions = baseline_predictions(probabilities)
    before = pooled_exact_bacc(score_case_confusions(predictions, terminal_labels))
    # The exact endpoint accepts no donor-response input; poison is deliberately unused.
    assert any(left.response_hash != right.response_hash for left, right in zip(responses, poisoned))
    after = pooled_exact_bacc(score_case_confusions(predictions, terminal_labels))
    assert before == after
    assert before.to_payload()["smooth_response_used"] is False
