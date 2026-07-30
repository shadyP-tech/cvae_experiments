"""Serializable GECO controller for reconstruction-constrained CVAE training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


@dataclass
class GECOController:
    """Exponentiated dual-ascent controller with an EMA constraint signal.

    The model minimizes ``rate + lambda * (distortion - target)``.  The dual
    update is detached from the model graph and increases ``lambda`` whenever
    the source-only reconstruction constraint is violated.
    """

    target: float
    ema_decay: float = 0.99
    dual_step_size: float = 1e-3
    initial_multiplier: float = 1.0
    minimum_multiplier: float = 1e-6
    maximum_multiplier: float = 1e6
    ema_constraint: float = 0.0
    log_multiplier: float | None = None
    update_count: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(self.target) or self.target <= 0.0:
            raise ValueError("GECO target must be finite and positive.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0,1).")
        if not math.isfinite(self.dual_step_size) or self.dual_step_size <= 0.0:
            raise ValueError("dual_step_size must be finite and positive.")
        if not (
            0.0
            < self.minimum_multiplier
            <= self.initial_multiplier
            <= self.maximum_multiplier
        ):
            raise ValueError("GECO multiplier bounds are inconsistent.")
        if self.log_multiplier is None:
            self.log_multiplier = math.log(self.initial_multiplier)
        self._clamp_log_multiplier()

    @property
    def multiplier(self) -> float:
        assert self.log_multiplier is not None
        return math.exp(self.log_multiplier)

    def constraint(self, distortion: torch.Tensor) -> torch.Tensor:
        return distortion - float(self.target)

    def loss(
        self,
        *,
        rate: torch.Tensor,
        distortion: torch.Tensor,
    ) -> torch.Tensor:
        if rate.ndim != 0 or distortion.ndim != 0:
            raise ValueError("GECO rate and distortion must be scalar tensors.")
        return rate + float(self.multiplier) * self.constraint(distortion)

    def update(self, distortion: torch.Tensor | float) -> float:
        value = (
            float(distortion.detach().cpu())
            if isinstance(distortion, torch.Tensor)
            else float(distortion)
        )
        if not math.isfinite(value):
            raise FloatingPointError("GECO distortion is nonfinite.")
        observed_constraint = value - self.target
        if self.update_count == 0:
            self.ema_constraint = observed_constraint
        else:
            self.ema_constraint = (
                self.ema_decay * self.ema_constraint
                + (1.0 - self.ema_decay) * observed_constraint
            )
        assert self.log_multiplier is not None
        self.log_multiplier += self.dual_step_size * self.ema_constraint
        self._clamp_log_multiplier()
        self.update_count += 1
        return self.multiplier

    def state_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_geco_controller_state_v1",
            "target": self.target,
            "target_provenance": "source_only_warmup_reconstruction",
            "ema_decay": self.ema_decay,
            "dual_step_size": self.dual_step_size,
            "initial_multiplier": self.initial_multiplier,
            "minimum_multiplier": self.minimum_multiplier,
            "maximum_multiplier": self.maximum_multiplier,
            "ema_constraint": self.ema_constraint,
            "log_multiplier": self.log_multiplier,
            "multiplier": self.multiplier,
            "update_count": self.update_count,
        }

    @classmethod
    def from_state_payload(cls, payload: Mapping[str, object]) -> "GECOController":
        if payload.get("schema_version") != "midogpp_geco_controller_state_v1":
            raise ValueError("Unsupported GECO state schema.")
        if payload.get("target_provenance") != "source_only_warmup_reconstruction":
            raise ValueError("GECO target provenance is not source-only.")
        controller = cls(
            target=float(payload["target"]),
            ema_decay=float(payload["ema_decay"]),
            dual_step_size=float(payload["dual_step_size"]),
            initial_multiplier=float(payload["initial_multiplier"]),
            minimum_multiplier=float(payload["minimum_multiplier"]),
            maximum_multiplier=float(payload["maximum_multiplier"]),
            ema_constraint=float(payload["ema_constraint"]),
            log_multiplier=float(payload["log_multiplier"]),
            update_count=int(payload["update_count"]),
        )
        if not math.isclose(
            controller.multiplier,
            float(payload["multiplier"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("GECO multiplier does not match log_multiplier.")
        return controller

    def _clamp_log_multiplier(self) -> None:
        assert self.log_multiplier is not None
        self.log_multiplier = min(
            math.log(self.maximum_multiplier),
            max(math.log(self.minimum_multiplier), self.log_multiplier),
        )
