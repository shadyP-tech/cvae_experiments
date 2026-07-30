"""Exact v2 replay with one fixed last-quarter Polyak parameter average."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ...models import ClassConditionedCVAE
from ..b_adaptation_pilot.case_balanced_sampler import BalancedSchedule
from ..b_adaptation_pilot.step_training import (
    StepTrainingSpec,
    _configure_determinism,
    _derived_seed,
    _generator,
    _resolve_device,
    beta_for_step,
    model_state_hash,
)


@dataclass
class TailAverageRuntime:
    endpoint_model: ClassConditionedCVAE
    averaged_model: ClassConditionedCVAE
    device: str
    training_key_hash: str
    endpoint_hash: str
    averaged_hash: str
    initialization_hash: str
    schedule_hash: str
    posterior_stream_hash: str
    averaging_derivation_hash: str
    tail_steps: tuple[int, ...]
    tail_state_count: int
    diagnostics: tuple[Mapping[str, object], ...]
    peak_cuda_bytes: int


def train_with_tail_average(
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    schedule: BalancedSchedule,
    spec: StepTrainingSpec,
    pairing_key: str,
    training_key_hash: str,
    device: str,
    tail_steps: Sequence[int],
    cpu_threads: int = 1,
) -> TailAverageRuntime:
    """Replay the exact endpoint while averaging post-update tail parameters."""

    import numpy as np

    x_np = np.asarray(embeddings, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    batches = np.asarray(schedule.batches, dtype=np.int64)
    ordered_tail = tuple(int(step) for step in tail_steps)
    if (
        x_np.ndim != 2
        or x_np.shape[1] != 128
        or len(x_np) != len(y_np)
        or batches.shape != (spec.optimizer_steps, spec.batch_size)
        or ordered_tail != tuple(range(751, 1001))
    ):
        raise ProtocolError("Tail-average training inputs violate the frozen contract.")

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
    if tuple(model.named_buffers()):
        raise ProtocolError("Tail averaging requires a buffer-free CVAE.")
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
    tail_mean: dict[str, torch.Tensor] = {}
    tail_count = 0
    tail_set = frozenset(ordered_tail)
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
            raise ProtocolError(
                f"Tail-average replay produced nonfinite loss at step {step_index}."
            )
        loss.backward()
        gradient_norm = float(
            clip_grad_norm_(model.parameters(), spec.gradient_clip_norm)
        )
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ProtocolError("Tail-average replay produced a nonfinite gradient.")
        optimizer.step()

        if step_index in tail_set:
            tail_count += 1
            with torch.no_grad():
                _update_online_parameter_mean(tail_mean, model, tail_count)

        totals["loss"] += float(loss.detach().cpu())
        totals["reconstruction"] += float(reconstruction.detach().cpu())
        totals["kl"] += float(kl.detach().cpu())
        if step_index == 1 or step_index % 100 == 0:
            diagnostics.append(
                {
                    "schema_version": "midogpp_b_tail_average_training_diagnostic_v1",
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
                    "tail_state_count": tail_count,
                }
            )

    if tail_count != 250 or set(tail_mean) != {
        name for name, _ in model.named_parameters()
    }:
        raise ProtocolError("Tail accumulator did not consume exactly 250 model states.")
    endpoint_hash = model_state_hash(model)
    averaged_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, parameter in averaged_model.named_parameters():
            parameter.copy_(tail_mean[name])
    averaged_hash = model_state_hash(averaged_model)
    posterior_stream_hash = stable_hash(
        {
            "pairing_key": pairing_key,
            "posterior_seeds": posterior_seeds,
            "schedule_hash": schedule.stream_hash,
        }
    )
    averaging_derivation_hash = stable_hash(
        {
            "schema_version": "midogpp_b_tail_average_derivation_v1",
            "method": "uniform_fp32_online_parameter_mean_v1",
            "update_timing": "after_optimizer_step",
            "tail_steps": list(ordered_tail),
            "tail_state_count": tail_count,
            "endpoint_hash": endpoint_hash,
            "averaged_hash": averaged_hash,
        }
    )
    peak = (
        int(torch.cuda.max_memory_allocated(resolved))
        if resolved.startswith("cuda")
        else 0
    )
    return TailAverageRuntime(
        endpoint_model=model,
        averaged_model=averaged_model,
        device=resolved,
        training_key_hash=str(training_key_hash),
        endpoint_hash=endpoint_hash,
        averaged_hash=averaged_hash,
        initialization_hash=initialization_hash,
        schedule_hash=schedule.stream_hash,
        posterior_stream_hash=posterior_stream_hash,
        averaging_derivation_hash=averaging_derivation_hash,
        tail_steps=ordered_tail,
        tail_state_count=tail_count,
        diagnostics=tuple(diagnostics),
        peak_cuda_bytes=peak,
    )


def checkpoint_payload(
    runtime: TailAverageRuntime, *, metadata: Mapping[str, object]
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_b_tail_average_checkpoint_v1",
        "training_key_hash": runtime.training_key_hash,
        "endpoint_hash": runtime.endpoint_hash,
        "averaged_hash": runtime.averaged_hash,
        "initialization_hash": runtime.initialization_hash,
        "schedule_hash": runtime.schedule_hash,
        "posterior_stream_hash": runtime.posterior_stream_hash,
        "averaging_derivation_hash": runtime.averaging_derivation_hash,
        "tail_steps": list(runtime.tail_steps),
        "tail_state_count": runtime.tail_state_count,
        "peak_cuda_bytes": runtime.peak_cuda_bytes,
        "metadata": dict(metadata),
        "diagnostics": [dict(row) for row in runtime.diagnostics],
        "endpoint_state_dict": {
            key: value.detach().cpu()
            for key, value in runtime.endpoint_model.state_dict().items()
        },
        "averaged_state_dict": {
            key: value.detach().cpu()
            for key, value in runtime.averaged_model.state_dict().items()
        },
    }


def _update_online_parameter_mean(
    accumulator: dict[str, torch.Tensor],
    model: ClassConditionedCVAE,
    count: int,
) -> None:
    """Update a uniform FP32 parameter mean without touching model state or RNG."""

    if count < 1:
        raise ProtocolError("Tail accumulator count must be positive.")
    for name, parameter in model.named_parameters():
        value = parameter.detach()
        if value.dtype != torch.float32:
            raise ProtocolError(f"Tail accumulator requires FP32 parameters: {name}")
        if count == 1:
            accumulator[name] = value.clone()
        elif name not in accumulator:
            raise ProtocolError(f"Tail accumulator is missing parameter: {name}")
        else:
            accumulator[name].add_(
                value - accumulator[name], alpha=1.0 / float(count)
            )


__all__ = (
    "TailAverageRuntime",
    "checkpoint_payload",
    "train_with_tail_average",
)
