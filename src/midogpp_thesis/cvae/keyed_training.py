"""Deterministic keyed CVAE training with optional frozen auxiliary objectives.

This module is deliberately experiment-neutral.  Existing preservation studies
keep their public trainers and checkpoint schemas; new studies can use this
kernel without copying minibatch, RNG, GECO, and branch-state mechanics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import io
import math
import os
from typing import Mapping, Protocol, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ..common.hashing import stable_hash
from .geco import GECOController
from .models import ClassConditionedCVAE
from .protocol import ProtocolError
from .schedules import BalancedSchedule


FIXED_BETA = "fixed_beta"
GECO = "geco"
OBJECTIVES = frozenset({FIXED_BETA, GECO})


@dataclass(frozen=True)
class AuxiliaryContext:
    """Read-only inputs supplied to one auxiliary objective call."""

    model: ClassConditionedCVAE
    prior_z: torch.Tensor
    requested_labels: torch.Tensor
    global_step: int
    stream_key: str


@dataclass(frozen=True)
class AuxiliaryResult:
    """One scalar auxiliary loss and finite scalar diagnostics."""

    loss: torch.Tensor
    diagnostics: Mapping[str, float] = field(default_factory=dict)


class AuxiliaryObjective(Protocol):
    """Frozen callback contract used by :func:`run_keyed_steps`.

    Implementations must not generate randomness, mutate the model/controller,
    write artifacts, or make protocol decisions.  The kernel supplies all prior
    randomness and records ``identity_hash`` in its stream identity.
    """

    @property
    def identity_hash(self) -> str: ...

    def __call__(self, context: AuxiliaryContext) -> AuxiliaryResult: ...


@dataclass(frozen=True)
class KeyedTrainingSpec:
    batch_size: int
    hidden_dim: int
    latent_dim: int
    learning_rate: float
    weight_decay: float
    beta_final: float
    gradient_clip_norm: float
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        if (
            self.batch_size <= 0
            or self.batch_size % 2
            or self.hidden_dim <= 0
            or self.latent_dim <= 0
            or self.cpu_threads <= 0
        ):
            raise ProtocolError("Keyed-training integer settings are invalid.")
        values = (
            self.learning_rate,
            self.weight_decay,
            self.beta_final,
            self.gradient_clip_norm,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ProtocolError("Keyed-training numeric settings must be finite.")
        if (
            self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.beta_final < 0.0
            or self.gradient_clip_norm <= 0.0
        ):
            raise ProtocolError("Keyed-training numeric settings are out of range.")

    @property
    def identity_hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_keyed_training_spec_v1",
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta_final": self.beta_final,
            "gradient_clip_norm": self.gradient_clip_norm,
            "cpu_threads": self.cpu_threads,
        }


@dataclass
class KeyedTrainingState:
    model: ClassConditionedCVAE
    optimizer: torch.optim.Optimizer
    controller: GECOController | None
    device: str
    completed_step: int
    initialization_hash: str
    stream_records: list[Mapping[str, object]] = field(default_factory=list)
    diagnostics: list[Mapping[str, object]] = field(default_factory=list)

    @property
    def state_hash(self) -> str:
        return training_state_hash(self)


def initialize_training_state(
    *,
    input_dim: int,
    spec: KeyedTrainingSpec,
    pairing_key: str,
    device: str,
) -> KeyedTrainingState:
    """Create a source-local standard-normal CVAE and AdamW state."""

    if input_dim <= 0 or not pairing_key:
        raise ProtocolError("Training initialization identity is incomplete.")
    _configure_determinism(spec.cpu_threads)
    resolved = _resolve_device(device)
    seed = derived_seed(pairing_key, "initialization")
    torch.manual_seed(seed)
    if resolved.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = ClassConditionedCVAE(
        input_dim=input_dim,
        hidden_dim=spec.hidden_dim,
        latent_dim=spec.latent_dim,
        num_hidden_layers=2,
    ).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    return KeyedTrainingState(
        model=model,
        optimizer=optimizer,
        controller=None,
        device=resolved,
        completed_step=0,
        initialization_hash=model_state_hash(model),
    )


def attach_geco(
    state: KeyedTrainingState,
    controller: GECOController,
) -> None:
    """Attach a fresh GECO state before a GECO phase begins."""

    if state.controller is not None:
        raise ProtocolError("GECO controller is already attached.")
    state.controller = deepcopy(controller)


def clone_training_state(state: KeyedTrainingState) -> KeyedTrainingState:
    """Clone model, optimizer, controller, cursor, and audit state exactly."""

    model = ClassConditionedCVAE(
        input_dim=state.model.input_dim,
        hidden_dim=state.model.hidden_dim,
        latent_dim=state.model.latent_dim,
        n_classes=state.model.n_classes,
        num_hidden_layers=2,
    ).to(state.device)
    model.load_state_dict(deepcopy(state.model.state_dict()), strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(state.optimizer.defaults["lr"]),
        weight_decay=float(state.optimizer.defaults.get("weight_decay", 0.0)),
    )
    optimizer.load_state_dict(deepcopy(state.optimizer.state_dict()))
    controller = (
        None
        if state.controller is None
        else GECOController.from_state_payload(state.controller.state_payload())
    )
    clone = KeyedTrainingState(
        model=model,
        optimizer=optimizer,
        controller=controller,
        device=state.device,
        completed_step=state.completed_step,
        initialization_hash=state.initialization_hash,
        stream_records=[dict(row) for row in state.stream_records],
        diagnostics=[dict(row) for row in state.diagnostics],
    )
    if clone.state_hash != state.state_hash:
        raise ProtocolError("Cloned keyed-training state changed identity.")
    return clone


def run_keyed_steps(
    state: KeyedTrainingState,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    schedule: BalancedSchedule,
    spec: KeyedTrainingSpec,
    end_step: int,
    stream_key: str,
    objective: str,
    auxiliary: AuxiliaryObjective | None = None,
    auxiliary_weight: float = 0.0,
) -> KeyedTrainingState:
    """Advance ``state`` to ``end_step`` using exact keyed stochastic streams."""

    import numpy as np

    if objective not in OBJECTIVES:
        raise ProtocolError(f"Unsupported keyed objective: {objective!r}")
    if objective == GECO and state.controller is None:
        raise ProtocolError("GECO phase requires an attached controller.")
    if objective == FIXED_BETA and state.controller is not None:
        raise ProtocolError("Fixed-beta phase cannot update an attached GECO controller.")
    if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0.0:
        raise ProtocolError("Auxiliary weight must be finite and nonnegative.")
    if auxiliary is None and auxiliary_weight != 0.0:
        raise ProtocolError("Nonzero auxiliary weight requires a callback.")
    if end_step <= state.completed_step:
        raise ProtocolError("Keyed phase must advance the training cursor.")
    batches = np.asarray(schedule.batches, dtype=np.int64)
    if end_step > len(batches):
        raise ProtocolError("Training schedule is shorter than the requested phase.")
    x_np = np.asarray(embeddings, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    if (
        x_np.ndim != 2
        or len(x_np) != len(y_np)
        or x_np.shape[1] != state.model.input_dim
        or set(int(value) for value in y_np.tolist()) != {0, 1}
    ):
        raise ProtocolError("Keyed-training arrays violate the binary CVAE contract.")
    x_cpu = torch.from_numpy(x_np)
    y_cpu = torch.from_numpy(y_np)
    callback_hash = "none" if auxiliary is None else str(auxiliary.identity_hash)
    if auxiliary is not None and not callback_hash:
        raise ProtocolError("Auxiliary callback identity hash is empty.")

    state.model.train()
    for global_step in range(state.completed_step + 1, end_step + 1):
        indices = batches[global_step - 1]
        xb = x_cpu[indices].to(state.device)
        yb = y_cpu[indices].to(state.device)
        if int((yb == 0).sum()) != int((yb == 1).sum()):
            raise ProtocolError("Keyed-training batch is not class balanced.")
        state.optimizer.zero_grad(set_to_none=True)
        mu, logvar = state.model.encode(xb, yb)
        posterior_seed = derived_seed(stream_key, global_step, "posterior")
        posterior_generator = torch_generator(state.device, posterior_seed)
        epsilon = torch.randn(
            mu.shape,
            generator=posterior_generator,
            dtype=mu.dtype,
            device=mu.device,
        )
        decoded = state.model.decode(
            mu + epsilon * torch.exp(0.5 * logvar),
            yb,
        )
        distortion = F.mse_loss(decoded, xb, reduction="none").mean(dim=1).mean()
        rate = (
            -0.5
            * torch.sum(
                1 + logvar - mu.square() - logvar.exp(),
                dim=1,
            )
            / float(mu.shape[1])
        ).mean()
        if objective == GECO:
            assert state.controller is not None
            base_loss = state.controller.loss(rate=rate, distortion=distortion)
        else:
            base_loss = distortion + float(spec.beta_final) * rate

        prior_seed = derived_seed(stream_key, global_step, "prior")
        prior_generator = torch_generator(state.device, prior_seed)
        prior_z = torch.randn(
            (len(yb), state.model.latent_dim),
            generator=prior_generator,
            dtype=xb.dtype,
            device=state.device,
        )
        auxiliary_loss = torch.zeros((), dtype=xb.dtype, device=state.device)
        auxiliary_diagnostics: Mapping[str, float] = {}
        if auxiliary is not None:
            result = auxiliary(
                AuxiliaryContext(
                    model=state.model,
                    prior_z=prior_z,
                    requested_labels=yb,
                    global_step=global_step,
                    stream_key=stream_key,
                )
            )
            if result.loss.ndim != 0 or not torch.isfinite(result.loss):
                raise ProtocolError("Auxiliary objective returned a nonfinite scalar.")
            auxiliary_loss = result.loss
            auxiliary_diagnostics = _finite_diagnostics(result.diagnostics)
        total = base_loss + float(auxiliary_weight) * auxiliary_loss
        if not torch.isfinite(total):
            raise ProtocolError("Keyed training produced a nonfinite objective.")
        total.backward()
        gradient_norm = float(
            clip_grad_norm_(state.model.parameters(), spec.gradient_clip_norm)
        )
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in state.model.parameters()
        ):
            raise ProtocolError("Keyed training produced a nonfinite gradient.")
        state.optimizer.step()
        multiplier = None
        constraint = None
        if objective == GECO:
            assert state.controller is not None
            constraint = float(state.controller.constraint(distortion).detach().cpu())
            multiplier = state.controller.update(distortion)
        record = {
            "step": global_step,
            "batch_hash": schedule.step_hashes[global_step - 1],
            "posterior_seed": posterior_seed,
            "prior_seed": prior_seed,
            "objective": objective,
            "auxiliary_identity_hash": callback_hash,
        }
        state.stream_records.append(record)
        if (
            global_step == 1
            or global_step == end_step
            or global_step % 100 == 0
        ):
            state.diagnostics.append(
                {
                    "schema_version": "midogpp_keyed_training_diagnostic_v1",
                    **record,
                    "total": float(total.detach().cpu()),
                    "distortion": float(distortion.detach().cpu()),
                    "rate": float(rate.detach().cpu()),
                    "auxiliary": float(auxiliary_loss.detach().cpu()),
                    "auxiliary_weight": float(auxiliary_weight),
                    "gradient_norm": gradient_norm,
                    "geco_constraint": constraint,
                    "geco_multiplier": multiplier,
                    **auxiliary_diagnostics,
                }
            )
        state.completed_step = global_step
    return state


def training_state_hash(state: KeyedTrainingState) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            "model": {
                key: value.detach().cpu()
                for key, value in state.model.state_dict().items()
            },
            # Optimizer tensors are normalized to CPU so the same numerical
            # state has one identity when moved between CUDA devices for
            # checkpointing or generation.
            "optimizer": _cpu_tree(state.optimizer.state_dict()),
            "controller": (
                None
                if state.controller is None
                else state.controller.state_payload()
            ),
            "completed_step": state.completed_step,
            "initialization_hash": state.initialization_hash,
            "stream_records": list(state.stream_records),
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _cpu_tree(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {
            key: _cpu_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def model_state_hash(model: ClassConditionedCVAE) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def stream_hash(state: KeyedTrainingState) -> str:
    return stable_hash(
        {
            "schema_version": "midogpp_keyed_training_stream_v1",
            "records": list(state.stream_records),
        }
    )


def optimizer_state_hash(optimizer: torch.optim.Optimizer) -> str:
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def derived_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def torch_generator(device: str, seed: int) -> torch.Generator:
    generator_device = device if str(device).startswith("cuda") else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _finite_diagnostics(values: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        numeric = float(value)
        if not math.isfinite(numeric) or abs(numeric) > 1e30:
            raise ProtocolError(f"Auxiliary diagnostic {key!r} is invalid.")
        result[str(key)] = numeric
    return result


def _resolve_device(device: str) -> str:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ProtocolError(f"Requested CUDA device is unavailable: {requested}")
    return requested


def _configure_determinism(cpu_threads: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(max(1, int(cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


__all__ = (
    "AuxiliaryContext",
    "AuxiliaryObjective",
    "AuxiliaryResult",
    "FIXED_BETA",
    "GECO",
    "KeyedTrainingSpec",
    "KeyedTrainingState",
    "attach_geco",
    "clone_training_state",
    "derived_seed",
    "initialize_training_state",
    "model_state_hash",
    "optimizer_state_hash",
    "run_keyed_steps",
    "stream_hash",
    "torch_generator",
    "training_state_hash",
)
