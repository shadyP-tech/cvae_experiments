from __future__ import annotations

import numpy as np
import torch

from midogpp_thesis.cvae.objectives import beta_objective, validate_trace_normalized_metric


def test_identity_metric_matches_isotropic_reconstruction() -> None:
    target = torch.tensor([[1.0, 2.0], [0.0, -1.0]])
    reconstruction = torch.tensor([[0.5, 3.0], [1.0, -1.0]])
    mu = torch.zeros((2, 1))
    logvar = torch.zeros((2, 1))
    isotropic = beta_objective(reconstruction, target, mu, logvar, beta=0.1)
    metric = beta_objective(reconstruction, target, mu, logvar, beta=0.1, metric=torch.eye(2))
    assert torch.allclose(isotropic.total, metric.total)
    assert torch.allclose(isotropic.reconstruction, metric.reconstruction)


def test_metric_validation_rejects_wrong_trace() -> None:
    validate_trace_normalized_metric(np.eye(3), input_dim=3)
    try:
        validate_trace_normalized_metric(np.eye(3) * 2.0, input_dim=3)
    except ValueError as exc:
        assert "trace" in str(exc)
    else:
        raise AssertionError("Non-normalized metric was accepted.")
