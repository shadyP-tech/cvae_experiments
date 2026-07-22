from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.real_features.classifier_reference.classifiers import (
    ClassifierSpec,
    _fit_standardized_logistic_classifier,
    standardize_fit_eval,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.config import (
    canonical_classifier_spec,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.estimator import (
    conditional_logit_objective_and_gradient,
    conditional_logit_objective_terms,
    fit_prepared_conditional_logit,
    prepare_conditional_logit,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.folds import (
    make_outer_fold,
)
from midogpp_thesis.real_features.classifier_reference.conditional_logit_alignment.penalty import (
    build_conditional_penalty,
)
from midogpp_thesis.real_features.classifier_reference.protocol import ProtocolError
from midogpp_thesis.real_features.classifier_reference.real_feature_frame import (
    RealFeatureFrame,
    RealFeatureRow,
)


def test_objective_analytic_gradient_matches_finite_difference() -> None:
    x = np.asarray(
        [
            [-1.2, 0.4, 0.3],
            [0.2, 1.1, -0.8],
            [1.4, -0.5, 0.7],
            [-0.7, -1.0, 1.2],
            [0.9, 0.6, -0.2],
            [1.8, -0.1, 0.4],
        ],
        dtype=np.float64,
    )
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=int)
    domains = ("0", "0", "1", "1", "2", "2")
    operator = build_conditional_penalty(x, y, domains)
    theta = np.asarray([0.3, -0.2, 0.5, -0.1], dtype=np.float64)

    value, gradient = conditional_logit_objective_and_gradient(
        theta,
        x,
        y,
        operator,
        C=0.01,
        gamma=0.1,
    )
    epsilon = 1.0e-6
    numerical = np.empty_like(theta)
    for index in range(theta.size):
        left = theta.copy()
        right = theta.copy()
        left[index] -= epsilon
        right[index] += epsilon
        left_value, _ = conditional_logit_objective_and_gradient(
            left, x, y, operator, C=0.01, gamma=0.1
        )
        right_value, _ = conditional_logit_objective_and_gradient(
            right, x, y, operator, C=0.01, gamma=0.1
        )
        numerical[index] = (right_value - left_value) / (2.0 * epsilon)

    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)


def test_intercept_receives_neither_l2_nor_alignment_penalty() -> None:
    x = np.asarray(
        [[-1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [3.0, 2.0]], dtype=float
    )
    y = (0, 1, 0, 1)
    domains = ("0", "0", "1", "1")
    operator = build_conditional_penalty(x, y, domains)
    first = conditional_logit_objective_terms(
        (0.4, -0.7, -2.0), x, y, operator, C=0.01, gamma=10.0
    )
    second = conditional_logit_objective_terms(
        (0.4, -0.7, 3.0), x, y, operator, C=0.01, gamma=10.0
    )
    gamma_zero = conditional_logit_objective_terms(
        (0.4, -0.7, -2.0), x, y, operator, C=0.01, gamma=0.0
    )

    assert first.l2_penalty == second.l2_penalty
    assert first.alignment_penalty == second.alignment_penalty
    # Changing gamma changes only coefficient-gradient components.
    assert np.asarray(first.gradient)[-1] == pytest.approx(
        np.asarray(gamma_zero.gradient)[-1], rel=0.0, abs=0.0
    )


def test_gamma_zero_exactly_uses_shared_standardized_sklearn_path() -> None:
    fold = make_outer_fold(_frame(), "0")
    spec = canonical_classifier_spec()
    prepared = prepare_conditional_logit(fold, spec)
    result = fit_prepared_conditional_logit(prepared, 0.0)

    standardized = standardize_fit_eval(fold.fit_embeddings, fold.eval_embeddings)
    direct = _fit_standardized_logistic_classifier(
        standardized,
        fold.fit_labels,
        spec=spec,
        sample_weight=None,
    )

    assert result.backend == "sklearn_lbfgs"
    assert result.gamma == 0.0
    assert result.scaler_state_hash == direct.scaler_state_hash
    assert result.classifier_config_hash == direct.classifier_config_hash
    assert result.classes == direct.classes
    assert result.n_iter == direct.n_iter
    assert result.converged is direct.converged
    np.testing.assert_array_equal(result.predictions, direct.predictions)
    np.testing.assert_array_equal(result.probabilities, direct.probabilities)
    np.testing.assert_array_equal(
        result.coefficients, np.asarray(direct.coefficients).reshape(-1)
    )
    assert result.intercept == float(np.asarray(direct.intercept).reshape(-1)[0])


def test_positive_gamma_uses_scipy_and_shrinks_alignment_value() -> None:
    fold = make_outer_fold(_frame(), "0")
    prepared = prepare_conditional_logit(fold, canonical_classifier_spec())
    pooled = fit_prepared_conditional_logit(prepared, 0.0)
    aligned = fit_prepared_conditional_logit(prepared, 10.0)

    assert aligned.backend == "scipy_lbfgsb"
    assert aligned.optimizer_success is True
    assert aligned.converged is True
    assert aligned.gradient_inf_norm <= 1.0e-4
    assert aligned.unscaled_alignment_value <= (
        pooled.unscaled_alignment_value + 1.0e-12
    )
    assert aligned.alignment_penalty == pytest.approx(
        10.0 * aligned.unscaled_alignment_value
    )
    assert aligned.fit_identity != pooled.fit_identity


def test_estimator_rejects_noncanonical_classifier_spec() -> None:
    fold = make_outer_fold(_frame(), "0")
    with pytest.raises(ProtocolError, match="frozen C=0.01"):
        prepare_conditional_logit(fold, ClassifierSpec(C=0.1, random_state=23))


def _frame() -> RealFeatureFrame:
    rows: list[RealFeatureRow] = []
    embeddings: list[list[float]] = []
    for center in ("0", "1", "2", "3"):
        domain = float(center)
        for label in (0, 1):
            for replicate in range(4):
                sample_id = f"sample-{center}-{label}-{replicate}"
                rows.append(
                    RealFeatureRow(
                        row_index=len(rows),
                        sample_id=sample_id,
                        case_id=f"case-{sample_id}",
                        center=center,
                        label=label,
                        split="train",
                        image_path=f"/images/{sample_id}.png",
                    )
                )
                embeddings.append(
                    [
                        2.0 * label + (0.45 if label else -0.25) * domain + 0.03 * replicate,
                        (-0.35 if label else 0.55) * domain + 0.07 * replicate,
                        (1.0 if label else -1.0) + 0.12 * domain - 0.02 * replicate,
                    ]
                )
    return RealFeatureFrame(
        embeddings=np.asarray(embeddings, dtype=np.float64),
        rows=tuple(rows),
        feature_extractor={"name": "virchow2"},
        feature_cache_path=Path("/tmp/midogpp/virchow2/cache.pt"),
        feature_cache_hash="cache-hash",
        manifest_path=Path("/tmp/midogpp/manifest.csv"),
        manifest_hash="manifest-hash",
        expected_feature_dim=3,
    )
