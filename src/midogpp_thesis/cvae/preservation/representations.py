"""Decode, posterior-sample, and generation-sampler representation builders."""

from __future__ import annotations

from typing import Sequence

from ..generation_samplers import AggregatePosteriorSampler, sample_latents
from ..training import TrainedCVAERuntime


def decode_means(
    runtime: TrainedCVAERuntime,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> tuple[object, object, object]:
    import numpy as np
    import torch

    device = torch.device(runtime.device)
    x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    runtime.model.eval()
    with torch.no_grad():
        mu, logvar = runtime.model.encode(x, y)
        decoded = runtime.model.decode(mu, y)
    return decoded.cpu().numpy(), mu.cpu().numpy(), logvar.cpu().numpy()


def posterior_samples(
    runtime: TrainedCVAERuntime,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    seed: int,
) -> tuple[object, object, object]:
    import numpy as np
    import torch

    device = torch.device(runtime.device)
    x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    runtime.model.eval()
    with torch.no_grad():
        mu, logvar = runtime.model.encode(x, y)
        noise = torch.randn(
            mu.shape,
            generator=generator,
            dtype=mu.dtype,
            device=mu.device,
        )
        decoded = runtime.model.decode(mu + noise * torch.exp(0.5 * logvar), y)
    return decoded.cpu().numpy(), mu.cpu().numpy(), logvar.cpu().numpy()


def sampler_decodes(
    runtime: TrainedCVAERuntime,
    sampler: AggregatePosteriorSampler,
    labels: Sequence[int],
    *,
    seed: int,
) -> object:
    import numpy as np
    import torch

    device = torch.device(runtime.device)
    z = torch.as_tensor(
        sample_latents(sampler, labels, seed=seed),
        dtype=torch.float32,
        device=device,
    )
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long, device=device)
    runtime.model.eval()
    with torch.no_grad():
        return runtime.model.decode(z, y).cpu().numpy()


def encode_posterior(
    runtime: TrainedCVAERuntime,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> tuple[object, object]:
    import numpy as np
    import torch

    device = torch.device(runtime.device)
    x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    runtime.model.eval()
    with torch.no_grad():
        mu, logvar = runtime.model.encode(x, y)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def source_budget_labels(labels: Sequence[int]) -> tuple[int, ...]:
    counts = {class_label: sum(int(value) == class_label for value in labels) for class_label in (0, 1)}
    if not all(counts.values()):
        raise ValueError("Source generation budget requires both classes.")
    return tuple([0] * counts[0] + [1] * counts[1])
