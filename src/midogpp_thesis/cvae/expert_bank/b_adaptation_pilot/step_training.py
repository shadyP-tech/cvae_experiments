"""Paired fixed-step CVAE training for the diagnostic pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ...models import ClassConditionedCVAE
from .case_balanced_sampler import BalancedSchedule


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

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
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
class PilotRuntime:
    model: ClassConditionedCVAE
    device: str
    training_key_hash: str
    checkpoint_hash: str
    initialization_hash: str
    schedule_hash: str
    posterior_stream_hash: str
    diagnostics: tuple[Mapping[str, object], ...]
    peak_cuda_bytes: int


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
) -> PilotRuntime:
    import numpy as np

    x_np = np.asarray(embeddings, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    batches = np.asarray(schedule.batches, dtype=np.int64)
    if (
        x_np.ndim != 2
        or x_np.shape[1] != 128
        or len(x_np) != len(y_np)
        or batches.shape != (spec.optimizer_steps, spec.batch_size)
    ):
        raise ProtocolError("Fixed-step training inputs violate the pilot contract.")
    _configure_determinism(cpu_threads)
    resolved = _resolve_device(device)
    initialization_seed = _derived_seed(pairing_key, "initialization")
    torch.manual_seed(initialization_seed)
    if resolved.startswith("cuda"):
        torch.cuda.set_device(resolved)
        torch.cuda.manual_seed_all(initialization_seed)
        torch.cuda.reset_peak_memory_stats(resolved)
    model = ClassConditionedCVAE(
        input_dim=128,
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
    model.train()
    for step_index, batch_indices in enumerate(batches, start=1):
        xb = x_cpu[batch_indices].to(resolved)
        yb = y_cpu[batch_indices].to(resolved)
        optimizer.zero_grad(set_to_none=True)
        mu, logvar = model.encode(xb, yb)
        posterior_seed = _derived_seed(pairing_key, step_index, "posterior")
        posterior_seeds.append(posterior_seed)
        generator = _generator(resolved, posterior_seed)
        epsilon = torch.randn(
            mu.shape,
            generator=generator,
            dtype=mu.dtype,
            device=mu.device,
        )
        decoded = model.decode(mu + epsilon * torch.exp(0.5 * logvar), yb)
        reconstruction = F.mse_loss(decoded, xb, reduction="none").mean(dim=1).mean()
        kl = (
            -0.5
            * torch.sum(1 + logvar - mu.square() - logvar.exp(), dim=1)
            / float(mu.shape[1])
        ).mean()
        beta = beta_for_step(spec, step_index)
        loss = reconstruction + beta * kl
        if not torch.isfinite(loss):
            raise ProtocolError(f"Pilot training produced nonfinite loss at step {step_index}.")
        loss.backward()
        gradient_norm = float(clip_grad_norm_(model.parameters(), spec.gradient_clip_norm))
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ProtocolError("Pilot training produced a nonfinite gradient.")
        optimizer.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["reconstruction"] += float(reconstruction.detach().cpu())
        totals["kl"] += float(kl.detach().cpu())
        if step_index == 1 or step_index % 100 == 0 or step_index == spec.optimizer_steps:
            diagnostics.append(
                {
                    "schema_version": "midogpp_b_adaptation_training_diagnostic_v1",
                    "step": step_index,
                    "beta": beta,
                    "mean_loss_to_step": totals["loss"] / step_index,
                    "mean_reconstruction_to_step": totals["reconstruction"] / step_index,
                    "mean_kl_to_step": totals["kl"] / step_index,
                    "gradient_norm": gradient_norm,
                    "batch_sample_hash": schedule.step_hashes[step_index - 1],
                    "schedule_hash": schedule.stream_hash,
                    "training_key_hash": training_key_hash,
                }
            )
    checkpoint_hash = model_state_hash(model)
    posterior_stream_hash = stable_hash(
        {
            "pairing_key": pairing_key,
            "posterior_seeds": posterior_seeds,
            "schedule_hash": schedule.stream_hash,
        }
    )
    peak = (
        int(torch.cuda.max_memory_allocated(resolved))
        if resolved.startswith("cuda")
        else 0
    )
    return PilotRuntime(
        model=model,
        device=resolved,
        training_key_hash=str(training_key_hash),
        checkpoint_hash=checkpoint_hash,
        initialization_hash=initialization_hash,
        schedule_hash=schedule.stream_hash,
        posterior_stream_hash=posterior_stream_hash,
        diagnostics=tuple(diagnostics),
        peak_cuda_bytes=peak,
    )


def beta_for_step(spec: StepTrainingSpec, step: int) -> float:
    if step <= 0:
        raise ProtocolError("KL warmup step must be positive.")
    return spec.beta_final * min(1.0, float(step) / float(spec.kl_warmup_steps))


def checkpoint_payload(runtime: PilotRuntime, *, metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": "midogpp_b_adaptation_checkpoint_v1",
        "training_key_hash": runtime.training_key_hash,
        "checkpoint_hash": runtime.checkpoint_hash,
        "initialization_hash": runtime.initialization_hash,
        "schedule_hash": runtime.schedule_hash,
        "posterior_stream_hash": runtime.posterior_stream_hash,
        "peak_cuda_bytes": runtime.peak_cuda_bytes,
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
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _resolve_device(device: str) -> str:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ProtocolError(f"Requested CUDA device is unavailable: {requested}")
    return requested


def _generator(device: str, seed: int) -> torch.Generator:
    kind = "cuda" if device.startswith("cuda") else "cpu"
    return torch.Generator(device=kind).manual_seed(seed)


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
