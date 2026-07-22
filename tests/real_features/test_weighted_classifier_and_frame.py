from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from midogpp_thesis.real_features.classifier_reference import (
    midogpp_real_feature_classifier as compatibility_module,
)
from midogpp_thesis.real_features.classifier_reference.artifacts import stable_hash
from midogpp_thesis.real_features.classifier_reference.classifiers import (
    DEFAULT_LOCKED_CLASSIFIER_SPEC,
    ClassifierSpec,
    _fit_standardized_logistic_classifier,
    fit_logistic_classifier,
    standardize_fit_eval,
)
from midogpp_thesis.real_features.classifier_reference.real_feature_frame import (
    RealFeatureFrame,
    RealFeatureRow,
    load_midogpp_real_feature_frame,
)


X_TRAIN = np.asarray(
    [
        [-2.0, 1.0],
        [-1.0, 2.0],
        [0.5, -1.0],
        [1.5, -2.0],
        [2.0, 0.0],
        [3.0, 1.0],
    ],
    dtype=float,
)
Y_TRAIN = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
X_EVAL = np.asarray([[-0.5, 0.5], [2.5, -0.5]], dtype=float)


def test_legacy_real_feature_module_reexports_extracted_public_symbols() -> None:
    assert compatibility_module.RealFeatureRow is RealFeatureRow
    assert compatibility_module.RealFeatureFrame is RealFeatureFrame
    assert (
        compatibility_module.load_midogpp_real_feature_frame
        is load_midogpp_real_feature_frame
    )


@pytest.mark.parametrize(
    ("sample_weight", "message"),
    [
        ([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]], "1D"),
        ([1.0, 1.0], "align"),
        ([1.0, 1.0, np.nan, 1.0, 1.0, 1.0], "finite"),
        ([1.0, 1.0, np.inf, 1.0, 1.0, 1.0], "finite"),
        ([1.0, 1.0, -0.1, 1.0, 1.0, 1.0], "nonnegative"),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "positive total"),
        ([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], "both classes"),
        ([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], "both classes"),
    ],
)
def test_fit_logistic_classifier_rejects_invalid_sample_weight(
    sample_weight: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_logistic_classifier(
            X_TRAIN,
            Y_TRAIN,
            X_EVAL,
            spec=ClassifierSpec(),
            sample_weight=sample_weight,  # type: ignore[arg-type]
        )


def test_all_ones_sample_weight_is_exactly_equivalent_to_none() -> None:
    spec = ClassifierSpec(C=0.7, random_state=31)
    unweighted = fit_logistic_classifier(
        X_TRAIN,
        Y_TRAIN,
        X_EVAL,
        spec=spec,
    )
    all_ones = fit_logistic_classifier(
        X_TRAIN,
        Y_TRAIN,
        X_EVAL,
        spec=spec,
        sample_weight=np.ones(Y_TRAIN.shape[0], dtype=float),
    )

    np.testing.assert_array_equal(all_ones.predictions, unweighted.predictions)
    np.testing.assert_array_equal(all_ones.probabilities, unweighted.probabilities)
    assert all_ones.classes == unweighted.classes
    assert all_ones.n_iter == unweighted.n_iter
    assert all_ones.converged is unweighted.converged
    assert all_ones.classifier_config_hash == unweighted.classifier_config_hash
    assert all_ones.scaler_state_hash == unweighted.scaler_state_hash


def test_weighted_fit_matches_manual_unweighted_scaler_and_sklearn_fit() -> None:
    sample_weight = np.asarray([0.5, 2.0, 1.0, 3.0, 0.25, 4.0], dtype=float)
    spec = ClassifierSpec(
        C=0.7,
        max_iter=5000,
        class_weight="balanced",
        random_state=31,
    )

    result = fit_logistic_classifier(
        X_TRAIN,
        Y_TRAIN,
        X_EVAL,
        spec=spec,
        sample_weight=sample_weight,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_TRAIN)
    x_eval_scaled = scaler.transform(X_EVAL)
    classifier = LogisticRegression(**spec.to_sklearn_kwargs())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        classifier.fit(x_train_scaled, Y_TRAIN, sample_weight=sample_weight)

    np.testing.assert_array_equal(result.predictions, classifier.predict(x_eval_scaled))
    np.testing.assert_allclose(
        result.probabilities,
        classifier.predict_proba(x_eval_scaled),
        rtol=0.0,
        atol=0.0,
    )
    assert result.classes == tuple(int(value) for value in classifier.classes_)
    assert result.scaler_state_hash == stable_hash(
        {
            "mean_": np.asarray(scaler.mean_, dtype=float).tolist(),
            "var_": np.asarray(scaler.var_, dtype=float).tolist(),
            "scale_": np.asarray(scaler.scale_, dtype=float).tolist(),
            "n_features_in_": int(scaler.n_features_in_),
            "n_samples_seen_": np.asarray(scaler.n_samples_seen_).tolist(),
        }
    )


def test_classifier_and_scaler_hash_snapshots_remain_stable() -> None:
    assert DEFAULT_LOCKED_CLASSIFIER_SPEC.to_payload() == {
        "family": "sklearn_logistic_regression",
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": None,
        "random_state": 17,
        "l1_ratio": None,
        "threshold_policy": "predict",
        "scaler_fit": "synthetic_train_only",
    }
    assert DEFAULT_LOCKED_CLASSIFIER_SPEC.config_hash == "6d7da6a8f2852e5f"

    result = fit_logistic_classifier(
        X_TRAIN,
        Y_TRAIN,
        X_EVAL,
        spec=DEFAULT_LOCKED_CLASSIFIER_SPEC,
    )
    assert result.scaler_state_hash == "099c2df0f2908b88"


def test_shared_standardized_fit_preserves_public_wrapper_and_exposes_private_state() -> None:
    spec = ClassifierSpec(C=0.01, max_iter=5000, random_state=23)
    standardized = standardize_fit_eval(X_TRAIN, X_EVAL)
    private = _fit_standardized_logistic_classifier(
        standardized,
        Y_TRAIN,
        spec=spec,
    )
    public = fit_logistic_classifier(X_TRAIN, Y_TRAIN, X_EVAL, spec=spec)

    np.testing.assert_array_equal(private.predictions, public.predictions)
    np.testing.assert_array_equal(private.probabilities, public.probabilities)
    assert private.classes == public.classes
    assert private.n_iter == public.n_iter
    assert private.converged is public.converged
    assert private.scaler_state_hash == public.scaler_state_hash
    assert np.asarray(private.coefficients).shape == (1, X_TRAIN.shape[1])
    assert np.asarray(private.intercept).shape == (1,)
