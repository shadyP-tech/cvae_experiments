"""Deterministic fixed-step training primitives for class-conditioned CVAEs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
import os
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ..common.hashing import stable_hash
from .models import ClassConditionedCVAE
from .protocol import ProtocolError
from .schedules import BalancedSchedule


EPSILON_TRACE_HASH_SCHEMA = "midogpp_explicit_epsilon_content_v1"


@dataclass(frozen=True)
class StepTrainingSpec:
    optimizer_steps: int = 1000
    batch_size: int = 128
    hidden_dim: int = 512
    latent_dim: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta_final: float = 1e-3
    kl_warmup_steps: int = 250
    gradient_clip_norm: float = 5.0

    def __post_init__(self) -> None:
        integer_fields = {
            "optimizer_steps": self.optimizer_steps,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "kl_warmup_steps": self.kl_warmup_steps,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_fields.values()
        ):
            raise ProtocolError(
                "Fixed-step integer hyperparameters must be positive integers."
            )
        if self.batch_size % 2:
            raise ProtocolError(
                "Fixed-step balanced training requires an even batch_size."
            )
        numeric_fields = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta_final": self.beta_final,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        try:
            finite = all(math.isfinite(float(value)) for value in numeric_fields.values())
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ProtocolError("Fixed-step numeric hyperparameters must be finite.")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.beta_final < 0
            or self.gradient_clip_norm <= 0
        ):
            raise ProtocolError("Fixed-step numeric hyperparameters are out of range.")

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        # Keep the recovered payload identity stable for existing artifacts.
        return {
            "schema_version": "midogpp_b_adaptation_step_training_spec_v1",
            "optimizer_steps": self.optimizer_steps,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_hidden_layers": 2,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta_final": self.beta_final,
            "kl_warmup_steps": self.kl_warmup_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "objective": "stochastic_isotropic_beta_objective_step_normalized_v1",
        }


@dataclass
class FixedStepTrainingRuntime:
    model: ClassConditionedCVAE
    device: str
    training_key_hash: str
    checkpoint_hash: str
    initialization_hash: str
    schedule_hash: str
    posterior_stream_hash: str
    diagnostics: tuple[Mapping[str, object], ...]
    peak_cuda_bytes: int
    optimizer_steps: int = 0
    decoder_forwards: int = 0
    epsilon_trace_hash: str = ""


# Compatibility for recovered pilot/probe callers and existing checkpoints.
StepTrainingRuntime = FixedStepTrainingRuntime
PilotRuntime = FixedStepTrainingRuntime


def train_fixed_steps(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    schedule: BalancedSchedule,
    spec: StepTrainingSpec,
    pairing_key: str,
    training_key_hash: str,
    device: str,
    cpu_threads: int = 1,
    posterior_estimator: str = "one_epsilon",
    posterior_stream_key: str | None = None,
    epsilon_trace: object | None = None,
    epsilon_trace_hash: str | None = None,
) -> FixedStepTrainingRuntime:
    """Train for an exact number of updates with an auditable epsilon trace.

    When ``epsilon_trace`` is omitted, the recovered per-step seeded stream is
    preserved exactly. An explicit trace must have shape
    ``[optimizer_steps, batch_size, latent_dim]``. Its canonical content hash is
    checked against ``epsilon_trace_hash`` when an expected hash is supplied.
    """

    import numpy as np

    x_np = np.asarray(embeddings, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    batches = np.asarray(schedule.batches, dtype=np.int64)
    expected_epsilon_shape = (
        int(spec.optimizer_steps),
        int(spec.batch_size),
        int(spec.latent_dim),
    )
    if (
        x_np.ndim != 2
        or x_np.shape[1] <= 0
        or len(x_np) != len(y_np)
        or batches.shape != (spec.optimizer_steps, spec.batch_size)
    ):
        raise ProtocolError("Fixed-step training inputs violate the training contract.")
    input_dim = int(x_np.shape[1])
    if posterior_estimator not in {"one_epsilon", "antithetic_epsilon"}:
        raise ProtocolError("Unknown posterior reconstruction estimator.")

    materialized_epsilon = None
    if epsilon_trace is not None:
        materialized_epsilon = _canonical_epsilon_trace(epsilon_trace)
        if materialized_epsilon.shape != expected_epsilon_shape:
            raise ProtocolError(
                "Explicit epsilon trace must have shape "
                f"{expected_epsilon_shape}, got {materialized_epsilon.shape}."
            )
        observed_epsilon_hash = epsilon_trace_content_hash(materialized_epsilon)
        if (
            epsilon_trace_hash is not None
            and str(epsilon_trace_hash) != observed_epsilon_hash
        ):
            raise ProtocolError("Explicit epsilon trace content hash mismatch.")
    else:
        observed_epsilon_hash = ""

    _configure_determinism(cpu_threads)
    resolved = _resolve_device(device)
    initialization_seed = _derived_seed(pairing_key, "initialization")
    torch.manual_seed(initialization_seed)
    if resolved.startswith("cuda"):
        torch.cuda.set_device(resolved)
        torch.cuda.manual_seed_all(initialization_seed)
        torch.cuda.reset_peak_memory_stats(resolved)
    model = ClassConditionedCVAE(
        input_dim=input_dim,
        hidden_dim=spec.hidden_dim,
        latent_dim=spec.latent_dim,
        num_hidden_layers=2,
    ).to(resolved)
    initialization_hash = model_state_hash(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    x_cpu = torch.from_numpy(x_np)
    y_cpu = torch.from_numpy(y_np)
    posterior_seeds: list[int] = []
    diagnostics: list[dict[str, object]] = []
    totals = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
    optimizer_steps = 0
    decoder_forwards = 0
    generated_trace_hasher = (
        _new_epsilon_trace_hasher(expected_epsilon_shape)
        if materialized_epsilon is None
        else None
    )
    model.train()
    for step_index, batch_indices in enumerate(batches, start=1):
        xb = x_cpu[batch_indices].to(resolved)
        yb = y_cpu[batch_indices].to(resolved)
        optimizer.zero_grad(set_to_none=True)
        mu, logvar = model.encode(xb, yb)
        posterior_seed = _derived_seed(
            posterior_stream_key or pairing_key, step_index, "posterior"
        )
        posterior_seeds.append(posterior_seed)
        if materialized_epsilon is None:
            generator = _generator(resolved, posterior_seed)
            epsilon = torch.randn(
                mu.shape,
                generator=generator,
                dtype=mu.dtype,
                device=mu.device,
            )
            assert generated_trace_hasher is not None
            generated_trace_hasher.update(
                _epsilon_tensor_bytes(epsilon)
            )
        else:
            epsilon = torch.as_tensor(
                materialized_epsilon[step_index - 1],
                dtype=mu.dtype,
                device=mu.device,
            )
        if tuple(epsilon.shape) != tuple(mu.shape) or not torch.isfinite(epsilon).all():
            raise ProtocolError(
                f"Epsilon trace violates the fixed-step contract at step {step_index}."
            )
        std = torch.exp(0.5 * logvar)
        decoded = model.decode(mu + epsilon * std, yb)
        decoder_forwards += 1
        reconstruction = F.mse_loss(decoded, xb, reduction="none").mean(dim=1).mean()
        if posterior_estimator == "antithetic_epsilon":
            decoded_minus = model.decode(mu - epsilon * std, yb)
            decoder_forwards += 1
            reconstruction_minus = F.mse_loss(
                decoded_minus, xb, reduction="none"
            ).mean(dim=1).mean()
            # One shared analytic KL; average losses, never decoded tensors.
            reconstruction = 0.5 * (reconstruction + reconstruction_minus)
        kl = (
            -0.5
            * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
            / float(mu.shape[1])
        ).mean()
        beta = beta_for_step(spec, step_index)
        loss = reconstruction + beta * kl
        if not torch.isfinite(loss):
            raise ProtocolError(
                f"Fixed-step training produced nonfinite loss at step {step_index}."
            )
        loss.backward()
        gradient_norm = float(
            clip_grad_norm_(model.parameters(), spec.gradient_clip_norm)
        )
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ProtocolError("Fixed-step training produced a nonfinite gradient.")
        optimizer.step()
        optimizer_steps += 1
        totals["loss"] += float(loss.detach().cpu())
        totals["reconstruction"] += float(reconstruction.detach().cpu())
        totals["kl"] += float(kl.detach().cpu())
        if (
            step_index == 1
            or step_index % 100 == 0
            or step_index == spec.optimizer_steps
        ):
            diagnostics.append(
                {
                    # Preserve the recovered diagnostic schema for replay.
                    "schema_version": "midogpp_b_adaptation_training_diagnostic_v1",
                    "step": step_index,
                    "beta": beta,
                    "mean_loss_to_step": totals["loss"] / step_index,
                    "mean_reconstruction_to_step": totals["reconstruction"]
                    / step_index,
                    "mean_kl_to_step": totals["kl"] / step_index,
                    "gradient_norm": gradient_norm,
                    "batch_sample_hash": schedule.step_hashes[step_index - 1],
                    "schedule_hash": schedule.stream_hash,
                    "training_key_hash": training_key_hash,
                    "optimizer_steps": optimizer_steps,
                    "decoder_forwards": decoder_forwards,
                }
            )
    checkpoint_hash = model_state_hash(model)
    posterior_stream_hash = stable_hash(
        {
            "posterior_stream_key": posterior_stream_key or pairing_key,
            "posterior_seeds": posterior_seeds,
            "schedule_hash": schedule.stream_hash,
        }
    )
    if generated_trace_hasher is not None:
        observed_epsilon_hash = generated_trace_hasher.hexdigest()
        if (
            epsilon_trace_hash is not None
            and str(epsilon_trace_hash) != observed_epsilon_hash
        ):
            raise ProtocolError("Generated epsilon trace content hash mismatch.")
    diagnostics_with_trace = tuple(
        dict(row, epsilon_trace_hash=observed_epsilon_hash)
        for row in diagnostics
    )
    peak = (
        int(torch.cuda.max_memory_allocated(resolved))
        if resolved.startswith("cuda")
        else 0
    )
    return FixedStepTrainingRuntime(
        model=model,
        device=resolved,
        training_key_hash=str(training_key_hash),
        checkpoint_hash=checkpoint_hash,
        initialization_hash=initialization_hash,
        schedule_hash=schedule.stream_hash,
        posterior_stream_hash=posterior_stream_hash,
        diagnostics=diagnostics_with_trace,
        peak_cuda_bytes=peak,
        optimizer_steps=optimizer_steps,
        decoder_forwards=decoder_forwards,
        epsilon_trace_hash=observed_epsilon_hash,
    )


def epsilon_trace_content_hash(epsilon_trace: object) -> str:
    """Hash one canonical contiguous float32 epsilon tensor."""

    trace = _canonical_epsilon_trace(epsilon_trace)
    hasher = _new_epsilon_trace_hasher(tuple(int(value) for value in trace.shape))
    hasher.update(trace.tobytes(order="C"))
    return hasher.hexdigest()


def beta_for_step(spec: StepTrainingSpec, step: int) -> float:
    if step <= 0:
        raise ProtocolError("KL warmup step must be positive.")
    return spec.beta_final * min(
        1.0, float(step) / float(spec.kl_warmup_steps)
    )


def checkpoint_payload(
    runtime: FixedStepTrainingRuntime,
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Serialize a fixed-step runtime while preserving the pilot schema."""

    return {
        "schema_version": "midogpp_b_adaptation_checkpoint_v1",
        "training_key_hash": runtime.training_key_hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "initialization_hash": runtime.initialization_hash,
        "schedule_hash": runtime.schedule_hash,
        "posterior_stream_hash": runtime.posterior_stream_hash,
        "epsilon_trace_hash": runtime.epsilon_trace_hash,
        "optimizer_steps": runtime.optimizer_steps,
        "peak_cuda_bytes": runtime.peak_cuda_bytes,
        "decoder_forwards": runtime.decoder_forwards,
        "metadata": dict(metadata),
        "diagnostics": [dict(row) for row in runtime.diagnostics],
        "state_dict": {
            key: value.detach().cpu()
            for key, value in runtime.model.state_dict().items()
        },
    }


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


def _canonical_epsilon_trace(epsilon_trace: object) -> object:
    import numpy as np

    try:
        trace = np.asarray(epsilon_trace, dtype="<f4")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("Epsilon trace must be numeric and array-like.") from exc
    if trace.ndim != 3 or not np.isfinite(trace).all():
        raise ProtocolError("Epsilon trace must be a finite rank-three array.")
    return np.array(trace, dtype="<f4", order="C", copy=True)


def _new_epsilon_trace_hasher(shape: tuple[int, ...]) -> object:
    hasher = hashlib.sha256()
    hasher.update(EPSILON_TRACE_HASH_SCHEMA.encode("ascii"))
    hasher.update(b"\nfloat32\n")
    hasher.update(",".join(str(int(value)) for value in shape).encode("ascii"))
    hasher.update(b"\n")
    return hasher


def _epsilon_tensor_bytes(epsilon: torch.Tensor) -> bytes:
    import numpy as np

    return (
        epsilon.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
        .astype(np.dtype("<f4"), copy=False)
        .tobytes(order="C")
    )


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _resolve_device(device: str) -> str:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ProtocolError(f"Requested CUDA device is unavailable: {requested}")
    return requested


def _generator(device: str, seed: int) -> torch.Generator:
    generator_device = device if device.startswith("cuda") else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


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
    "EPSILON_TRACE_HASH_SCHEMA",
    "FixedStepTrainingRuntime",
    "PilotRuntime",
    "StepTrainingRuntime",
    "StepTrainingSpec",
    "beta_for_step",
    "checkpoint_payload",
    "epsilon_trace_content_hash",
    "model_state_hash",
    "train_fixed_steps",
)
