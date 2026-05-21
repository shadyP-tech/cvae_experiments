from pathlib import Path
import math
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cvae_expert import (  # noqa: E402
    DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
    RECON_LOSS_GAUSSIAN_NLL_DIAG,
    REDUCTION_MEAN,
    CVAEExpert,
    elbo_components,
    gaussian_nll_diag_terms,
    kl_divergence_diag_gaussian,
)


def test_legacy_mse_cvae_api_remains_compatible() -> None:
    model = CVAEExpert(input_dim=4, hidden_dim=8, latent_dim=2)
    x = torch.randn(3, 4)

    recon, mu, logvar = model(x)

    assert recon.shape == x.shape
    assert mu.shape == (3, 2)
    assert logvar.shape == (3, 2)
    mean, recon_logvar = model.decode(torch.randn(3, 2), return_distribution=True)
    assert mean.shape == x.shape
    assert recon_logvar is None


def test_heteroscedastic_class_conditioned_forward_shapes_and_clamp() -> None:
    model = CVAEExpert(
        input_dim=4,
        hidden_dim=8,
        latent_dim=2,
        class_condition_dim=2,
        decoder_likelihood=DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
        decoder_logvar_min=-9.21,
        decoder_logvar_max=2.0,
        decoder_min_variance=1.0e-4,
    )
    assert model.dec_logvar is not None
    with torch.no_grad():
        model.dec_logvar.weight.zero_()
        model.dec_logvar.bias.fill_(10.0)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 0])

    (recon_mu, recon_logvar), mu, logvar = model(x, y=y, return_distribution=True)

    assert recon_mu.shape == x.shape
    assert recon_logvar is not None
    assert recon_logvar.shape == x.shape
    assert float(recon_logvar.max().item()) <= 2.0
    assert mu.shape == (3, 2)
    assert logvar.shape == (3, 2)

    with torch.no_grad():
        model.dec_logvar.bias.fill_(-20.0)
    mean, low_logvar = model.decode(torch.randn(3, 2), y=y, return_distribution=True)
    assert mean.shape == x.shape
    assert low_logvar is not None
    assert float(low_logvar.min().item()) >= -9.21 - 1.0e-6


def test_gaussian_nll_uses_dimension_mean_reduction() -> None:
    x = torch.zeros(2, 4)
    recon_mu = torch.zeros_like(x)
    recon_logvar = torch.zeros_like(x)
    mu = torch.zeros(2, 2)
    logvar = torch.zeros_like(mu)

    recon, kl = elbo_components(
        recon_mu,
        x,
        mu,
        logvar,
        recon_logvar_x=recon_logvar,
        reconstruction_loss=RECON_LOSS_GAUSSIAN_NLL_DIAG,
        recon_reduction=REDUCTION_MEAN,
        kl_reduction=REDUCTION_MEAN,
    )

    expected_nll = 0.5 * math.log(2.0 * math.pi)
    assert torch.allclose(recon, torch.full((2,), expected_nll))
    assert torch.allclose(kl, torch.zeros(2))
    terms = gaussian_nll_diag_terms(recon_mu, x, recon_logvar)
    assert torch.allclose(terms["logvar_term"], torch.zeros(2))
    assert torch.allclose(terms["squared_error_scaled"], torch.zeros(2))


def test_kl_latent_dimension_mean_reduction_is_explicit() -> None:
    mu = torch.ones(1, 2)
    logvar = torch.zeros_like(mu)

    kl_mean = kl_divergence_diag_gaussian(mu, logvar, reduction=REDUCTION_MEAN)

    assert torch.allclose(kl_mean, torch.tensor([0.5]))


def test_class_conditioning_rejects_out_of_range_labels() -> None:
    model = CVAEExpert(input_dim=4, hidden_dim=8, latent_dim=2, class_condition_dim=2)

    try:
        model.encode(torch.randn(2, 4), y=torch.tensor([0, 2]))
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("out-of-range class labels were not rejected")


def test_metadata_and_class_conditioning_cannot_be_mixed() -> None:
    try:
        CVAEExpert(input_dim=4, hidden_dim=8, latent_dim=2, metadata_dim=2, class_condition_dim=2)
    except ValueError as exc:
        assert "mixing metadata conditioning and class conditioning" in str(exc)
    else:
        raise AssertionError("mixed metadata/class conditioning was not rejected")
