from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from midogpp_thesis.cvae.latent_mixture_prior import (
    ClassConditionalLowRankMixturePrior,
)
from midogpp_thesis.cvae.models import AggregateMatchedMixturePriorCVAE


def test_component_kl_and_log_prob_match_dense_gaussians() -> None:
    torch.manual_seed(7)
    prior = ClassConditionalLowRankMixturePrior(
        3,
        n_components=2,
        rank=2,
        weight_floor=0.02,
        variance_floor=1e-4,
    ).double()
    target_diagonal = torch.tensor(
        [
            [[0.7, 1.2, 0.5], [1.1, 0.4, 0.8]],
            [[0.6, 0.9, 1.3], [0.3, 1.4, 0.7]],
        ],
        dtype=torch.double,
    )
    with torch.no_grad():
        prior.mixture_logits.copy_(
            torch.tensor([[0.4, -0.2], [-0.3, 0.5]], dtype=torch.double)
        )
        prior.component_means.copy_(torch.randn(2, 2, 3, dtype=torch.double))
        prior.diag_rho.copy_(
            _inverse_softplus(target_diagonal - prior.variance_floor)
        )
        prior.low_rank.copy_(0.15 * torch.randn(2, 2, 3, 2, dtype=torch.double))

    posterior_mu = torch.randn(5, 3, dtype=torch.double)
    posterior_logvar = 0.2 * torch.randn(5, 3, dtype=torch.double)
    labels = torch.tensor([0, 1, 1, 0, 1])
    observed = prior.component_kl(posterior_mu, posterior_logvar, labels)
    expected = torch.empty_like(observed)
    for row in range(len(labels)):
        q = MultivariateNormal(
            posterior_mu[row],
            covariance_matrix=torch.diag(posterior_logvar[row].exp()),
        )
        for component in range(2):
            p = MultivariateNormal(
                prior.component_means[labels[row], component],
                covariance_matrix=prior.covariance()[labels[row], component],
            )
            expected[row, component] = kl_divergence(q, p)
    assert torch.allclose(observed, expected, atol=1e-9, rtol=1e-8)

    latent = torch.randn(5, 3, dtype=torch.double)
    dense_components = []
    for component in range(2):
        component_values = []
        for row in range(len(labels)):
            p = MultivariateNormal(
                prior.component_means[labels[row], component],
                covariance_matrix=prior.covariance()[labels[row], component],
            )
            component_values.append(p.log_prob(latent[row]))
        dense_components.append(torch.stack(component_values))
    dense = torch.stack(dense_components, dim=1)
    expected_log_prob = torch.logsumexp(
        prior.weights()[labels].log() + dense,
        dim=1,
    )
    assert torch.allclose(
        prior.log_prob(latent, labels),
        expected_log_prob,
        atol=1e-9,
        rtol=1e-8,
    )


def test_mixture_bound_is_formed_before_dimension_normalization() -> None:
    prior = ClassConditionalLowRankMixturePrior(4)
    posterior_mu = torch.tensor([[1.0, -0.5, 0.2, 0.7]])
    posterior_logvar = torch.tensor([[0.1, -0.2, 0.3, -0.1]])
    labels = torch.tensor([0])

    component = prior.component_kl(posterior_mu, posterior_logvar, labels)
    expected = -torch.logsumexp(
        prior.weights()[labels].log() - component,
        dim=-1,
    ) / 4.0
    assert torch.allclose(
        prior.kl_upper_bound(posterior_mu, posterior_logvar, labels),
        expected,
    )


def test_sampling_is_explicit_deterministic_and_has_mixture_moments() -> None:
    prior = ClassConditionalLowRankMixturePrior(
        2,
        n_components=2,
        rank=1,
        weight_floor=0.01,
    )
    target_diagonal = torch.full((2, 2, 2), 0.25)
    with torch.no_grad():
        prior.mixture_logits.copy_(torch.tensor([[0.8, -0.4], [0.0, 0.0]]))
        prior.component_means.copy_(
            torch.tensor(
                [[[-1.0, 0.0], [2.0, 1.0]], [[0.0, -1.0], [0.0, 1.0]]]
            )
        )
        prior.diag_rho.copy_(
            _inverse_softplus(target_diagonal - prior.variance_floor)
        )
        prior.low_rank.zero_()
    labels = torch.zeros(50_000, dtype=torch.long)
    first = prior.sample(
        labels,
        generator=torch.Generator().manual_seed(42),
    )
    second = prior.sample(
        labels,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(first, second)
    weights = prior.weights()[0]
    expected_mean = (
        weights[:, None] * prior.component_means[0]
    ).sum(dim=0)
    centered = prior.component_means[0] - expected_mean
    expected_covariance = (
        weights[:, None, None]
        * (
            prior.covariance()[0]
            + centered[:, :, None] * centered[:, None, :]
        )
    ).sum(dim=0)
    assert torch.allclose(first.mean(dim=0), expected_mean, atol=0.025)
    assert torch.allclose(
        torch.cov(first.T, correction=0),
        expected_covariance,
        atol=0.04,
        rtol=0.04,
    )


def test_source_aggregate_initialization_enforces_counts_and_health() -> None:
    rng = np.random.default_rng(9)
    labels = np.asarray([0] * 40 + [1] * 40)
    mode = np.asarray([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20)
    means = (
        rng.normal(scale=0.15, size=(80, 3))
        + mode[:, None] * np.asarray([3.0, -2.0, 1.0])
        + labels[:, None] * 0.4
    )
    logvar = np.full_like(means, -1.2)
    prior = ClassConditionalLowRankMixturePrior(3, weight_floor=0.05)
    initialized = prior.initialize_from_aggregate_posterior(
        torch.tensor(means, dtype=torch.float32),
        torch.tensor(logvar, dtype=torch.float32),
        torch.tensor(labels),
        case_ids=[f"case-{index}" for index in range(80)],
        random_state=17,
        minimum_component_rows=8,
        minimum_component_cases=2,
    )
    assert initialized.component_row_counts == ((20, 20), (20, 20))
    prior.assert_healthy(maximum_condition_number=1e6)
    assert prior.state_diagnostics().minimum_weight >= 0.05

    imbalanced = means.copy()
    imbalanced[:39] = 0.0
    imbalanced[39] = 20.0
    repaired = prior.initialize_from_aggregate_posterior(
        torch.tensor(imbalanced, dtype=torch.float32),
        torch.tensor(logvar, dtype=torch.float32),
        torch.tensor(labels),
        case_ids=[f"case-{index}" for index in range(80)],
        random_state=17,
        minimum_component_rows=1,
        minimum_component_cases=1,
    )
    assert repaired.assignment_fallbacks[0]
    assert min(repaired.component_row_counts[0]) > 0.05 * 40


def test_mixture_model_refuses_to_mislabel_bound_as_nelbo() -> None:
    model = AggregateMatchedMixturePriorCVAE(
        input_dim=4,
        hidden_dim=8,
        latent_dim=3,
    )
    with pytest.raises(RuntimeError, match="not an exact NELBO"):
        model.nelbo_for_class(
            torch.zeros(2, 4),
            torch.tensor([0, 1]),
        )
    diagnostics = model.latent_prior.state_diagnostics()
    assert diagnostics.finite
    assert math.isclose(diagnostics.minimum_eigenvalue, 1.0, rel_tol=1e-5)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))
