from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.cvae_expert import CVAEExpert, elbo_components  # noqa: E402
from src.train.train_utils import _compute_training_objective, _prior_label_batch  # noqa: E402


def test_family_d_heads_disabled_by_default_preserves_forward_contract() -> None:
    torch = pytest.importorskip("torch")
    model = CVAEExpert(input_dim=4, hidden_dim=8, latent_dim=2, class_condition_dim=2)
    assert model.label_utility_enabled is False
    assert model.latent_label_head is None
    assert model.decoded_label_head is None
    x = torch.randn(3, 4)
    y = torch.eye(2)[torch.tensor([0, 1, 0])].float()
    out = model(x, y=y)
    assert len(out) == 3


def test_label_conditioned_family_d_still_requires_y() -> None:
    torch = pytest.importorskip("torch")
    model = CVAEExpert(
        input_dim=4,
        hidden_dim=8,
        latent_dim=2,
        class_condition_dim=2,
        label_utility_cfg={"enabled": True, "num_classes": 2},
    )
    with pytest.raises(ValueError, match="Class-condition tensor is required"):
        model(torch.randn(3, 4))


def test_family_d_total_loss_matches_hand_computed_terms() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    model = CVAEExpert(
        input_dim=4,
        hidden_dim=8,
        latent_dim=2,
        class_condition_dim=2,
        label_utility_cfg={"enabled": True, "num_classes": 2},
    )
    model.reparameterize = lambda mu, logvar: mu  # type: ignore[method-assign]
    x = torch.randn(5, 4)
    y = torch.eye(2)[torch.tensor([0, 1, 0, 1, 0])].float()
    cfg = {
        "lambda_latent_cls": 0.2,
        "lambda_recon_cls": 0.3,
        "lambda_prior_cls": 0.0,
        "prior_samples_per_batch": "same_batch_size",
    }
    loss, _ = _compute_training_objective(model, x=x, m=None, y=y, label_utility_cfg=cfg)
    recon, mu, logvar = model(x, y=y)
    recon_terms, kl_terms = elbo_components(recon, x, mu, logvar)
    nelbo = (recon_terms + kl_terms).mean()
    latent_loss = model.label_utility_loss(model.label_utility_latent_logits(mu), y)
    recon_loss = model.label_utility_loss(model.label_utility_decoded_logits(recon), y)
    expected = nelbo + (0.2 * latent_loss) + (0.3 * recon_loss)
    assert torch.allclose(loss, expected)


def test_prior_sample_loss_uses_labels_matching_decoder_conditions() -> None:
    torch = pytest.importorskip("torch")
    y = torch.eye(2)[torch.tensor([0, 1, 1])].float()
    repeated = _prior_label_batch(y, label_utility_cfg={"prior_samples_per_batch": 5})
    assert repeated.argmax(dim=1).tolist() == [0, 1, 1, 0, 1]
