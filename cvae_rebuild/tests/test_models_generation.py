import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from model.generation import generate_reference_posterior
from model.models import ClassConditionedCVAE, loss_for_batch


def test_class_conditioned_cvae_scores_every_class_label() -> None:
    model = ClassConditionedCVAE(input_dim=3, hidden_dim=8, latent_dim=2, n_classes=2)
    x = torch.randn(5, 3)
    y0 = torch.zeros(5, dtype=torch.long)
    y1 = torch.ones(5, dtype=torch.long)
    n0 = model.nelbo_for_class(x, y0)
    n1 = model.nelbo_for_class(x, y1)
    marginal = model.marginal_nelbo(x)
    assert n0.shape == (5,)
    assert n1.shape == (5,)
    assert marginal.shape == (5,)


def test_training_loss_runs_with_class_conditioning() -> None:
    model = ClassConditionedCVAE(input_dim=4, hidden_dim=8, latent_dim=2, n_classes=2)
    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    loss = loss_for_batch(model, x, y)
    assert torch.isfinite(loss.nelbo)


def test_reference_posterior_generation_is_class_stratified() -> None:
    model = ClassConditionedCVAE(input_dim=3, hidden_dim=8, latent_dim=2, n_classes=2)
    refs = {
        0: np.random.default_rng(0).normal(size=(4, 3)),
        1: np.random.default_rng(1).normal(size=(4, 3)),
    }
    batch = generate_reference_posterior(
        model=model,
        expert_id="1",
        source_embeddings_by_class=refs,
        budget_per_class=3,
        generation_seed=17,
    )
    assert batch.expert_id == "1"
    assert tuple(batch.labels) == (0, 0, 0, 1, 1, 1)
    assert batch.embeddings.shape == (6, 3)
