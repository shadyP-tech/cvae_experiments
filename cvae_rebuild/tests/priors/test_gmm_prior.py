import numpy as np
import pytest

from priors.gmm import fit_class_conditional_gaussian_prior, fit_class_conditional_gmm_prior


def test_gmm_prior_sampling_is_class_conditioned_and_shape_stable() -> None:
    latents = np.array(
        [
            [-2.0, -2.0],
            [-2.2, -1.9],
            [-1.8, -2.1],
            [2.0, 2.0],
            [2.2, 1.9],
            [1.8, 2.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

    prior = fit_class_conditional_gmm_prior(
        latents,
        labels,
        n_components=2,
        covariance_type="diag",
        random_state=7,
        min_class_count=2,
    )
    samples = prior.sample(labels=[0, 1, 1, 0], random_state=11)

    assert prior.classes == (0, 1)
    assert prior.latent_dim == 2
    assert samples.shape == (4, 2)
    assert prior.models[0].n_components == 2
    assert prior.models[1].n_components == 2


def test_gmm_prior_sampling_uses_call_seed_reproducibly() -> None:
    latents = np.array(
        [
            [-2.0, -2.0],
            [-2.2, -1.9],
            [-1.8, -2.1],
            [2.0, 2.0],
            [2.2, 1.9],
            [1.8, 2.1],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    prior = fit_class_conditional_gmm_prior(
        latents,
        labels,
        n_components=2,
        covariance_type="diag",
        random_state=7,
        min_class_count=2,
    )

    first = prior.sample(labels=[0, 0, 1, 1], random_state=17)
    second = prior.sample(labels=[0, 0, 1, 1], random_state=17)
    third = prior.sample(labels=[0, 0, 1, 1], random_state=18)

    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, third)


def test_gmm_prior_caps_components_to_class_count() -> None:
    latents = np.array(
        [
            [-2.0, -2.0],
            [-2.2, -1.9],
            [-1.8, -2.1],
            [2.0, 2.0],
            [2.2, 1.9],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int64)

    prior = fit_class_conditional_gmm_prior(
        latents,
        labels,
        n_components=8,
        covariance_type="diag",
        random_state=7,
        min_class_count=2,
    )

    assert prior.models[0].n_components == 3
    assert prior.models[1].n_components == 2


def test_gmm_prior_rejects_missing_class_fit_rows_and_unknown_sample_labels() -> None:
    latents = np.array([[0.0, 0.0], [0.2, 0.1], [1.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="too few fit rows"):
        fit_class_conditional_gmm_prior(latents, labels, n_components=1, min_class_count=2)

    prior = fit_class_conditional_gmm_prior(
        latents[:2],
        labels[:2],
        n_components=1,
        covariance_type="diag",
        random_state=7,
        min_class_count=2,
    )
    with pytest.raises(ValueError, match="No latent prior fitted"):
        prior.sample(labels=[1], random_state=17)


def test_gaussian_prior_rejects_missing_class_fit_rows() -> None:
    latents = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)

    try:
        fit_class_conditional_gaussian_prior(latents, labels, min_class_count=2)
    except ValueError as exc:
        assert "too few fit rows" in str(exc)
    else:
        raise AssertionError("Expected class-count validation failure.")
