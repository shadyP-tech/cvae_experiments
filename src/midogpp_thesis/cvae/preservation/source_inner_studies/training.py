"""V2-only deterministic CVAE training for source-inner mechanism studies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ...models import ClassConditionedCVAE
from ...models.learned_conditional_prior import LearnedConditionalPriorCVAE
from ...latent_priors import PRIOR_SATURATION_THRESHOLD
from ...objectives import validate_trace_normalized_metric
from ...generation_samplers import AggregatePosteriorSampler, DIAGONAL_SAMPLER
from .contracts import (
    LEARNED_PRIOR_MODEL_FAMILY,
    STANDARD_MODEL_FAMILY,
    StudyTrainingKey,
    StudyTrainingVariant,
)


ALLOWED_MODEL_FAMILIES = frozenset(
    {STANDARD_MODEL_FAMILY, LEARNED_PRIOR_MODEL_FAMILY}
)


@dataclass
class StudyRuntime:
    model: ClassConditionedCVAE | LearnedConditionalPriorCVAE
    variant: StudyTrainingVariant
    training_key: StudyTrainingKey
    model_family: str
    checkpoint_hash: str
    diagnostics: tuple[Mapping[str, object], ...]
    device: str
    shared_initialization_hash: str
    prior_initialization_hash: str
    full_initialization_hash: str
    training_stream_hash: str
    resumed_from_checkpoint: bool = False


def train_study_cvae(
    train_embeddings: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    *,
    variant: StudyTrainingVariant,
    training_key: StudyTrainingKey,
    model_family: str,
    task_metric: object | None = None,
    device: str = "cpu",
) -> StudyRuntime:
    """Train one v2 study arm with an arm-neutral paired stochastic stream."""

    import numpy as np

    if model_family not in ALLOWED_MODEL_FAMILIES:
        raise ProtocolError(f"Unsupported source-inner study model family: {model_family!r}")
    if str(training_key.model_family) != model_family:
        raise ProtocolError("Study training key/model family mismatch.")
    if str(variant.model_family) != model_family:
        raise ProtocolError("Study training variant/model family mismatch.")
    if training_key.variant_hash != variant.hash:
        raise ProtocolError("Study training key/variant hash mismatch.")
    x_np = np.asarray(train_embeddings, dtype=np.float32)
    y_np = np.asarray(train_labels, dtype=np.int64)
    if x_np.ndim != 2 or len(x_np) != len(y_np) or len(x_np) == 0:
        raise ProtocolError("Study training embeddings/labels must be aligned and nonempty.")
    if sorted(set(int(value) for value in y_np.tolist())) != [0, 1]:
        raise ProtocolError("Source-inner study CVAE training requires both classes.")
    if float(variant.alpha) == 0.0 and task_metric is not None:
        raise ProtocolError("Literal alpha-zero study training must use metric=None.")
    if float(variant.alpha) > 0.0 and task_metric is None:
        raise ProtocolError("Nonzero Fisher study training requires its derived metric.")
    if task_metric is not None:
        validate_trace_normalized_metric(task_metric, input_dim=x_np.shape[1])

    _configure_determinism()
    resolved_device = _resolve_device(device)
    pairing_hash = training_key.arm_neutral_pairing_hash
    initialization_seed = _derived_seed(pairing_hash, "shared_initialization")
    torch.manual_seed(initialization_seed)
    if resolved_device.startswith("cuda"):
        torch.cuda.manual_seed_all(initialization_seed)
    model = _construct_model(
        model_family,
        input_dim=x_np.shape[1],
        variant=variant,
    ).to(resolved_device)
    shared_initialization_hash, prior_initialization_hash = initialization_partition_hashes(
        model
    )
    full_initialization_hash = model_state_hash(model)

    prior_parameters, shared_parameters = parameter_partitions(model)
    parameter_groups: list[dict[str, object]] = [
        {
            "params": shared_parameters,
            "weight_decay": float(variant.weight_decay),
        }
    ]
    if prior_parameters:
        parameter_groups.append(
            {
                "params": prior_parameters,
                "weight_decay": float(variant.prior_weight_decay),
                "lr": float(variant.learning_rate)
                * float(variant.prior_learning_rate_multiplier),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(variant.learning_rate),
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np)),
        batch_size=int(variant.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(_derived_seed(pairing_hash, "loader")),
    )
    metric_tensor = (
        None
        if task_metric is None
        else torch.as_tensor(task_metric, dtype=torch.float32, device=resolved_device)
    )
    diagnostics: list[dict[str, object]] = []
    posterior_seeds: list[int] = []
    for epoch in range(1, int(variant.train_epochs) + 1):
        model.train()
        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "prior_grad_norm": 0.0,
            "shared_grad_norm": 0.0,
            "n": 0,
        }
        beta = beta_for_epoch(variant, epoch)
        for batch_index, (xb_cpu, yb_cpu) in enumerate(loader):
            xb = xb_cpu.to(resolved_device)
            yb = yb_cpu.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            mu, logvar = model.encode(xb, yb)
            seed = _derived_seed(pairing_hash, epoch, batch_index, "training_posterior")
            posterior_seeds.append(seed)
            generator = _torch_generator(resolved_device, seed)
            epsilon = torch.randn(
                mu.shape,
                generator=generator,
                dtype=mu.dtype,
                device=mu.device,
            )
            decoded = model.decode(mu + epsilon * torch.exp(0.5 * logvar), yb)
            reconstruction = _reconstruction_loss(decoded, xb, metric_tensor)
            if model_family == LEARNED_PRIOR_MODEL_FAMILY:
                kl = model.kl_to_prior(mu, logvar, yb).mean()
            else:
                kl = (
                    -0.5
                    * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    / float(mu.shape[1])
                ).mean()
            loss = reconstruction + float(beta) * kl
            if not torch.isfinite(loss):
                raise ProtocolError("Study training produced a nonfinite loss.")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ProtocolError("Study training produced a nonfinite gradient.")
            shared_grad_norm = float(
                clip_grad_norm_(shared_parameters, float(variant.network_gradient_clip_norm))
            )
            prior_grad_norm = (
                float(
                    clip_grad_norm_(
                        prior_parameters,
                        float(variant.prior_gradient_clip_norm),
                    )
                )
                if prior_parameters
                else 0.0
            )
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ProtocolError("Study training produced nonfinite model state.")
            n_batch = int(xb.shape[0])
            totals["total"] += float(loss.detach().cpu()) * n_batch
            totals["reconstruction"] += float(reconstruction.detach().cpu()) * n_batch
            totals["kl"] += float(kl.detach().cpu()) * n_batch
            totals["shared_grad_norm"] += shared_grad_norm
            totals["prior_grad_norm"] += prior_grad_norm
            totals["n"] += n_batch
        n_rows = int(totals["n"])
        prior_epoch_diagnostics: dict[str, object] = {}
        if model_family == LEARNED_PRIOR_MODEL_FAMILY:
            with torch.no_grad():
                prior_mu = model.prior_mu.detach()
                prior_rho = model.prior_rho.detach()
                prior_logvar = model.prior_logvar.detach()
                prior_std = torch.exp(0.5 * prior_logvar)
            prior_epoch_diagnostics = {
                "prior_mu_min": float(prior_mu.min().cpu()),
                "prior_mu_max": float(prior_mu.max().cpu()),
                "prior_mu_l2_by_class": [
                    float(value)
                    for value in torch.linalg.vector_norm(prior_mu, dim=1)
                    .cpu()
                    .tolist()
                ],
                "prior_rho_min": float(prior_rho.min().cpu()),
                "prior_rho_max": float(prior_rho.max().cpu()),
                "effective_logvar_min": float(prior_logvar.min().cpu()),
                "effective_logvar_max": float(prior_logvar.max().cpu()),
                "prior_std_min": float(prior_std.min().cpu()),
                "prior_std_max": float(prior_std.max().cpu()),
                "prior_saturation_count": int(
                    (prior_logvar.abs() >= PRIOR_SATURATION_THRESHOLD)
                    .sum()
                    .cpu()
                ),
                "prior_saturated": bool(
                    (prior_logvar.abs() >= PRIOR_SATURATION_THRESHOLD).any()
                ),
            }
        diagnostics.append(
            {
                "schema_version": "midogpp_source_inner_study_training_diagnostic_v2",
                "epoch": epoch,
                "beta": beta,
                "mean_objective": totals["total"] / n_rows,
                "mean_reconstruction": totals["reconstruction"] / n_rows,
                "mean_kl": totals["kl"] / n_rows,
                "mean_shared_grad_norm": totals["shared_grad_norm"] / len(loader),
                "mean_prior_grad_norm": totals["prior_grad_norm"] / len(loader),
                "n_rows": n_rows,
                "training_key_hash": training_key.hash,
                "model_family": model_family,
                "alpha": float(variant.alpha),
                **prior_epoch_diagnostics,
            }
        )
    training_stream_hash = stable_hash(
        {
            "pairing_hash": pairing_hash,
            "initialization_seed": initialization_seed,
            "loader_seed": _derived_seed(pairing_hash, "loader"),
            "posterior_seeds": posterior_seeds,
        }
    )
    return StudyRuntime(
        model=model,
        variant=variant,
        training_key=training_key,
        model_family=model_family,
        checkpoint_hash=model_state_hash(model),
        diagnostics=tuple(diagnostics),
        device=resolved_device,
        shared_initialization_hash=shared_initialization_hash,
        prior_initialization_hash=prior_initialization_hash,
        full_initialization_hash=full_initialization_hash,
        training_stream_hash=training_stream_hash,
        resumed_from_checkpoint=False,
    )


def parameter_partitions(
    model: ClassConditionedCVAE | LearnedConditionalPriorCVAE,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    prior: list[torch.nn.Parameter] = []
    shared: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        (prior if _is_prior_key(name) else shared).append(parameter)
    if not shared:
        raise ProtocolError("Study model lacks shared encoder/decoder parameters.")
    return prior, shared


def state_key_partitions(model: torch.nn.Module) -> dict[str, list[str]]:
    keys = sorted(model.state_dict())
    return {
        "shared": [key for key in keys if not _is_prior_key(key)],
        "prior": [key for key in keys if _is_prior_key(key)],
    }


def initialization_partition_hashes(model: torch.nn.Module) -> tuple[str, str]:
    state = model.state_dict()
    shared = {
        _normalized_shared_key(key): value.detach().cpu()
        for key, value in state.items()
        if not _is_prior_key(key)
    }
    prior = {
        key: value.detach().cpu()
        for key, value in state.items()
        if _is_prior_key(key)
    }
    return _tensor_mapping_hash(shared), (
        _tensor_mapping_hash(prior) if prior else "none"
    )


def model_state_hash(model: torch.nn.Module) -> str:
    return _tensor_mapping_hash(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}
    )


def beta_for_epoch(variant: StudyTrainingVariant, epoch: int) -> float:
    return float(variant.beta_final) * min(
        1.0, float(epoch) / float(variant.kl_warmup_epochs)
    )


def _construct_model(
    model_family: str,
    *,
    input_dim: int,
    variant: StudyTrainingVariant,
) -> ClassConditionedCVAE | LearnedConditionalPriorCVAE:
    kwargs = {
        "input_dim": int(input_dim),
        "hidden_dim": int(variant.hidden_dim),
        "latent_dim": int(variant.latent_dim),
        "num_hidden_layers": int(variant.num_hidden_layers),
    }
    if model_family == STANDARD_MODEL_FAMILY:
        return ClassConditionedCVAE(**kwargs)
    if model_family == LEARNED_PRIOR_MODEL_FAMILY:
        return LearnedConditionalPriorCVAE(**kwargs)
    raise ProtocolError(f"Unsupported study model family: {model_family}")


def _reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    metric: torch.Tensor | None,
) -> torch.Tensor:
    if metric is None:
        return F.mse_loss(reconstruction, target, reduction="none").mean(dim=1).mean()
    residual = reconstruction - target
    return (
        torch.einsum("bi,ij,bj->b", residual, metric, residual)
        / float(target.shape[1])
    ).mean()


def _tensor_mapping_hash(mapping: Mapping[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(dict(sorted(mapping.items())), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _is_prior_key(name: str) -> bool:
    lowered = str(name).lower()
    return "prior_mu" in lowered or "prior_rho" in lowered or ".prior." in lowered


def _normalized_shared_key(key: str) -> str:
    return str(key)[5:] if str(key).startswith("base.") else str(key)


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def evaluation_seed(*parts: object) -> int:
    """Training-seed-neutral seed for paired generation/evaluation epsilon."""

    return _derived_seed("source_inner_study_evaluation_v2", *parts)


def paired_epsilon(
    *,
    study_id: str,
    outer_target_center: str,
    inner_pseudo_target_center: str,
    generation_seed: int,
    labels: Sequence[int],
    latent_dim: int,
    stream: str,
) -> tuple[object, str]:
    """Create row-aligned, class-independent noise shared across arms and t."""

    import numpy as np

    y = np.asarray(labels, dtype=np.int64)
    if not set(int(value) for value in y.tolist()).issubset({0, 1}):
        raise ProtocolError("Paired study epsilon supports binary labels only.")
    epsilon = np.empty((len(y), int(latent_dim)), dtype=np.float32)
    class_seeds: dict[str, int] = {}
    for class_label in (0, 1):
        indices = np.flatnonzero(y == class_label)
        seed = evaluation_seed(
            study_id,
            outer_target_center,
            inner_pseudo_target_center,
            int(generation_seed),
            int(class_label),
            str(stream),
        )
        class_seeds[str(class_label)] = seed
        epsilon[indices] = np.random.default_rng(seed).normal(
            size=(len(indices), int(latent_dim))
        )
    epsilon_hash = stable_hash(
        {
            "study_id": study_id,
            "outer": str(outer_target_center),
            "inner": str(inner_pseudo_target_center),
            "generation_seed": int(generation_seed),
            "stream": str(stream),
            "class_seeds": class_seeds,
            "label_vector_hash": hashlib.sha256(y.tobytes()).hexdigest(),
            "epsilon_sha256": hashlib.sha256(epsilon.tobytes()).hexdigest(),
        }
    )
    return epsilon, epsilon_hash


def encode_runtime(
    runtime: StudyRuntime,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> tuple[object, object]:
    import numpy as np

    device = torch.device(runtime.device)
    x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    runtime.model.eval()
    with torch.no_grad():
        mu, logvar = runtime.model.encode(x, y)
    return mu.cpu().numpy(), logvar.cpu().numpy()


def decode_means(
    runtime: StudyRuntime,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> object:
    import numpy as np

    device = torch.device(runtime.device)
    x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    runtime.model.eval()
    with torch.no_grad():
        mu, _ = runtime.model.encode(x, y)
        return runtime.model.decode(mu, y).cpu().numpy()


def posterior_decodes(
    runtime: StudyRuntime,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    epsilon: object,
) -> object:
    import numpy as np

    device = torch.device(runtime.device)
    x = torch.as_tensor(np.asarray(embeddings, dtype=np.float32), device=device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=device)
    noise = torch.as_tensor(np.asarray(epsilon, dtype=np.float32), device=device)
    runtime.model.eval()
    with torch.no_grad():
        mu, logvar = runtime.model.encode(x, y)
        if noise.shape != mu.shape:
            raise ProtocolError("Posterior evaluation epsilon shape mismatch.")
        z = mu + noise * torch.exp(0.5 * logvar)
        return runtime.model.decode(z, y).cpu().numpy()


def prior_decodes_from_epsilon(
    runtime: StudyRuntime,
    labels: Sequence[int],
    *,
    epsilon: object,
    sampler: AggregatePosteriorSampler | None = None,
) -> object:
    """Decode standard, learned, or realized diagonal ex-post latent draws."""

    import numpy as np

    y_np = np.asarray(labels, dtype=np.int64)
    eps_np = np.asarray(epsilon, dtype=np.float32)
    if eps_np.shape != (len(y_np), int(runtime.variant.latent_dim)):
        raise ProtocolError("Prior evaluation epsilon shape mismatch.")
    if sampler is not None:
        if (
            sampler.requested_family != DIAGONAL_SAMPLER
            or not sampler.requested_family_realized_for_both_classes
        ):
            raise ProtocolError("C-diag must realize the requested diagonal family for both classes.")
        z_np = np.empty_like(eps_np)
        for class_label in (0, 1):
            indices = np.flatnonzero(y_np == class_label)
            state = sampler.classes[class_label]
            diagonal_std = np.sqrt(np.diag(np.asarray(state.covariance, dtype=np.float64)))
            z_np[indices] = (
                np.asarray(state.mean, dtype=np.float64)
                + eps_np[indices] * diagonal_std
            ).astype(np.float32)
    elif runtime.model_family == LEARNED_PRIOR_MODEL_FAMILY:
        device = torch.device(runtime.device)
        y_tensor = torch.as_tensor(y_np, dtype=torch.long, device=device)
        eps_tensor = torch.as_tensor(eps_np, dtype=torch.float32, device=device)
        runtime.model.eval()
        with torch.no_grad():
            z_np = runtime.model.sample_prior(y_tensor, epsilon=eps_tensor).cpu().numpy()
    else:
        z_np = eps_np
    device = torch.device(runtime.device)
    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    runtime.model.eval()
    with torch.no_grad():
        return runtime.model.decode(z, y).cpu().numpy()


def _torch_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def _resolve_device(device: str) -> str:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return requested


def _configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
