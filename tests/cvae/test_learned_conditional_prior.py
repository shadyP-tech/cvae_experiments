from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from midogpp_thesis.cvae.latent_priors import (
    PRIOR_LOGVAR_LIMIT,
    LearnedClassConditionalDiagonalPrior,
    conditional_prior_diagnostics,
    normalized_diagonal_gaussian_kl,
    normalized_symmetric_diagonal_kl,
    standardized_active_unit_scores,
)
from midogpp_thesis.cvae.models import (
    ClassConditionedCVAE,
    LearnedConditionalPriorCVAE,
)


def test_prior_parameters_have_locked_shape_and_zero_initialization() -> None:
    model = LearnedConditionalPriorCVAE(
        input_dim=5,
        hidden_dim=8,
        latent_dim=3,
    )

    assert model.prior_mu.shape == (2, 3)
    assert model.prior_rho.shape == (2, 3)
    assert torch.equal(model.prior_mu, torch.zeros(2, 3))
    assert torch.equal(model.prior_rho, torch.zeros(2, 3))
    assert torch.equal(model.prior_logvar, torch.zeros(2, 3))
    assert set(model.shared_parameters()).isdisjoint(
        set(model.learned_prior_parameters())
    )


def test_initial_conditional_kl_matches_standard_kl_and_encoder_gradients() -> None:
    torch.manual_seed(17)
    model = LearnedConditionalPriorCVAE(
        input_dim=4,
        hidden_dim=7,
        latent_dim=3,
    )
    x = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 1, 0])
    posterior_mu, posterior_logvar = model.encode(x, labels)

    learned = model.kl_to_prior(
        posterior_mu,
        posterior_logvar,
        labels,
    ).mean()
    standard = (
        -0.5
        * torch.sum(
            1.0
            + posterior_logvar
            - posterior_mu.pow(2)
            - posterior_logvar.exp(),
            dim=1,
        )
        / float(model.latent_dim)
    ).mean()
    assert torch.allclose(learned, standard, atol=1e-7, rtol=1e-6)

    encoder_parameters = tuple(model.encoder.parameters()) + tuple(
        model.fc_mu.parameters()
    ) + tuple(model.fc_logvar.parameters())
    learned_gradients = torch.autograd.grad(
        learned,
        encoder_parameters,
        retain_graph=True,
    )
    standard_gradients = torch.autograd.grad(standard, encoder_parameters)
    for learned_gradient, standard_gradient in zip(
        learned_gradients,
        standard_gradients,
        strict=True,
    ):
        assert torch.allclose(
            learned_gradient,
            standard_gradient,
            atol=1e-7,
            rtol=1e-6,
        )


def test_normalized_diagonal_kl_matches_closed_form_value() -> None:
    posterior_mu = torch.tensor([[1.0, -1.0]])
    posterior_logvar = torch.log(torch.tensor([[2.0, 0.5]]))
    prior_mu = torch.tensor([[0.5, -0.5]])
    prior_logvar = torch.log(torch.tensor([[1.5, 0.25]]))

    observed = normalized_diagonal_gaussian_kl(
        posterior_mu,
        posterior_logvar,
        prior_mu,
        prior_logvar,
    )
    expected_per_dimension = 0.5 * (
        prior_logvar
        - posterior_logvar
        + posterior_logvar.exp() / prior_logvar.exp()
        + (posterior_mu - prior_mu).pow(2) / prior_logvar.exp()
        - 1.0
    )
    expected = expected_per_dimension.mean(dim=1)
    assert torch.allclose(observed, expected)


def test_prior_gradients_are_label_indexed() -> None:
    prior = LearnedClassConditionalDiagonalPrior(latent_dim=2)
    posterior_mu = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    posterior_logvar = torch.tensor([[-0.7, 0.3], [0.4, -0.2]])
    labels = torch.tensor([0, 0])

    prior.kl_from_posterior(
        posterior_mu,
        posterior_logvar,
        labels,
    ).sum().backward()

    assert prior.prior_mu.grad is not None
    assert prior.prior_rho.grad is not None
    assert torch.count_nonzero(prior.prior_mu.grad[0]) > 0
    assert torch.count_nonzero(prior.prior_rho.grad[0]) > 0
    assert torch.equal(prior.prior_mu.grad[1], torch.zeros(2))
    assert torch.equal(prior.prior_rho.grad[1], torch.zeros(2))


def test_prior_sampling_is_deterministic_and_matches_empirical_moments() -> None:
    prior = LearnedClassConditionalDiagonalPrior(latent_dim=2)
    target_mu = torch.tensor([[-1.0, 0.5], [2.0, -0.25]])
    target_logvar = torch.log(torch.tensor([[0.5, 2.0], [1.5, 0.25]]))
    with torch.no_grad():
        prior.prior_mu.copy_(target_mu)
        prior.prior_rho.copy_(_rho_for_logvar(target_logvar))

    labels = torch.tensor([0, 1, 0, 1])
    first = prior.sample(
        labels,
        generator=torch.Generator().manual_seed(101),
    )
    second = prior.sample(
        labels,
        generator=torch.Generator().manual_seed(101),
    )
    assert torch.equal(first, second)

    epsilon = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [-1.0, 0.5], [0.5, -1.0]]
    )
    explicit = prior.sample(labels, epsilon=epsilon)
    expected_mu = target_mu[labels]
    expected_logvar = target_logvar[labels]
    assert torch.allclose(
        explicit,
        expected_mu + expected_logvar.mul(0.5).exp() * epsilon,
    )

    n_per_class = 30_000
    moment_labels = torch.tensor(
        [0] * n_per_class + [1] * n_per_class,
        dtype=torch.long,
    )
    samples = prior.sample(
        moment_labels,
        generator=torch.Generator().manual_seed(42),
    )
    for class_label in (0, 1):
        selected = samples[moment_labels == class_label]
        assert torch.allclose(
            selected.mean(dim=0),
            target_mu[class_label],
            atol=0.025,
            rtol=0.0,
        )
        assert torch.allclose(
            selected.var(dim=0, unbiased=False),
            target_logvar[class_label].exp(),
            atol=0.04,
            rtol=0.04,
        )


def test_diagnostics_use_conditional_prior_standardization_and_final_saturation() -> None:
    prior_mu = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    prior_logvar = torch.zeros(2, 2)
    posterior_mu = torch.tensor(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.2],
            [2.0, -0.2],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1])

    scores = standardized_active_unit_scores(
        posterior_mu,
        labels,
        prior_mu,
        prior_logvar,
    )
    assert torch.allclose(scores, torch.tensor([1.0, 0.02]))
    assert torch.allclose(
        normalized_symmetric_diagonal_kl(prior_mu, prior_logvar),
        torch.tensor(0.25),
    )

    diagnostics = conditional_prior_diagnostics(
        posterior_mu,
        labels,
        prior_mu,
        prior_logvar,
        prior_rho=torch.zeros_like(prior_mu),
    )
    assert diagnostics.active_unit_count == 2
    assert diagnostics.active_unit_mask == (True, True)
    assert math.isclose(diagnostics.normalized_symmetric_kl, 0.25)
    assert not diagnostics.near_class_independent
    assert not diagnostics.saturated
    assert diagnostics.finite

    saturated_logvar = prior_logvar.clone()
    saturated_logvar[0, 0] = 5.95
    saturated = conditional_prior_diagnostics(
        posterior_mu,
        labels,
        prior_mu,
        saturated_logvar,
        prior_rho=_rho_for_logvar(saturated_logvar),
    )
    assert saturated.saturated
    assert saturated.saturation_count == 1
    assert math.isclose(saturated.max_abs_logvar, 5.95, abs_tol=1e-5)

    independent = conditional_prior_diagnostics(
        posterior_mu,
        labels,
        torch.zeros_like(prior_mu),
        torch.zeros_like(prior_logvar),
        prior_rho=torch.zeros_like(prior_mu),
    )
    assert independent.near_class_independent


def test_reconstruction_path_does_not_reach_prior_parameters() -> None:
    torch.manual_seed(23)
    model = LearnedConditionalPriorCVAE(
        input_dim=4,
        hidden_dim=6,
        latent_dim=2,
    )
    x = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 0, 1, 0])
    reconstruction, posterior_mu, posterior_logvar = model(
        x,
        labels,
        epsilon=torch.zeros(5, 2),
    )
    reconstruction_loss = F.mse_loss(reconstruction, x)
    reconstruction_gradients = torch.autograd.grad(
        reconstruction_loss,
        (model.prior_mu, model.prior_rho),
        allow_unused=True,
        retain_graph=True,
    )
    assert reconstruction_gradients == (None, None)

    prior_gradients = torch.autograd.grad(
        model.kl_to_prior(posterior_mu, posterior_logvar, labels).mean(),
        (model.prior_mu, model.prior_rho),
    )
    assert all(gradient is not None for gradient in prior_gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in prior_gradients)


def test_loss_nelbo_marginal_and_prior_sampling_use_learned_prior() -> None:
    torch.manual_seed(7)
    model = LearnedConditionalPriorCVAE(
        input_dim=3,
        hidden_dim=5,
        latent_dim=2,
    )
    x = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])
    epsilon = torch.zeros(4, 2)

    loss = model.loss_for_batch(x, labels, beta=0.25, epsilon=epsilon)
    assert torch.allclose(loss.total, loss.reconstruction + 0.25 * loss.kl)
    assert loss.beta == 0.25
    assert model.nelbo_for_class(x, labels).shape == (4,)
    marginal = model.marginal_nelbo(x, class_prior=(0.4, 0.6))
    assert marginal.shape == (4,)
    assert torch.isfinite(marginal).all()

    with torch.no_grad():
        model.prior_mu[1].fill_(2.0)
    sampled = model.sample_prior(labels, epsilon=torch.zeros(4, 2))
    assert torch.equal(sampled[labels == 0], torch.zeros(2, 2))
    assert torch.equal(sampled[labels == 1], torch.full((2, 2), 2.0))


def test_initial_nelbo_matches_unchanged_standard_prior_model() -> None:
    torch.manual_seed(31)
    learned = LearnedConditionalPriorCVAE(
        input_dim=3,
        hidden_dim=5,
        latent_dim=2,
    )
    standard = ClassConditionedCVAE(
        input_dim=3,
        hidden_dim=5,
        latent_dim=2,
    )
    shared_state = {
        key: value
        for key, value in learned.state_dict().items()
        if not key.startswith("latent_prior.")
    }
    standard.load_state_dict(shared_state, strict=True)
    x = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 0, 1])

    assert torch.allclose(
        learned.nelbo_for_class(x, labels),
        standard.nelbo_for_class(x, labels),
        atol=1e-6,
        rtol=1e-6,
    )


def _rho_for_logvar(logvar: torch.Tensor) -> torch.Tensor:
    if bool((logvar.abs() >= PRIOR_LOGVAR_LIMIT).any()):
        raise ValueError("Test logvar must lie inside the smooth prior bounds.")
    return PRIOR_LOGVAR_LIMIT * torch.atanh(logvar / PRIOR_LOGVAR_LIMIT)
