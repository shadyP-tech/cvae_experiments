from __future__ import annotations

import math

import pytest
import torch

from midogpp_thesis.cvae.geco import GECOController


def test_geco_dual_moves_in_constraint_direction_and_round_trips() -> None:
    controller = GECOController(
        target=0.5,
        ema_decay=0.0,
        dual_step_size=0.2,
        initial_multiplier=1.0,
    )
    increased = controller.update(torch.tensor(0.8))
    assert increased > 1.0
    decreased = controller.update(torch.tensor(0.1))
    assert decreased < increased
    restored = GECOController.from_state_payload(controller.state_payload())
    assert restored.state_payload() == controller.state_payload()


def test_geco_loss_uses_rate_plus_detached_multiplier_constraint() -> None:
    controller = GECOController(target=0.25, initial_multiplier=2.0)
    rate = torch.tensor(0.4, requires_grad=True)
    distortion = torch.tensor(0.5, requires_grad=True)
    loss = controller.loss(rate=rate, distortion=distortion)
    loss.backward()
    assert math.isclose(float(rate.grad), 1.0)
    assert math.isclose(float(distortion.grad), 2.0)
    assert not isinstance(controller.multiplier, torch.Tensor)


def test_geco_rejects_unbound_or_tampered_state() -> None:
    controller = GECOController(target=0.4)
    payload = controller.state_payload()
    payload["target_provenance"] = "inner_bacc"
    with pytest.raises(ValueError, match="source-only"):
        GECOController.from_state_payload(payload)
