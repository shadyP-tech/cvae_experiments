"""Paired source-local training for aggregate-posterior mixture and GECO arms."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import math
import os
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from ....real_features.classifier_reference.artifacts import stable_hash
from ....real_features.classifier_reference.protocol import ProtocolError
from ...geco import GECOController
from ...models import AggregateMatchedMixturePriorCVAE, ClassConditionedCVAE
from .config import AggregatePriorStudyConfig
from .contracts import (
    ARMS,
    GECO,
    MIXTURE_PRIOR,
    SourceExpertTrainingKey,
    objective_family,
    prior_family,
    rate_family,
)


@dataclass
class SourceExpertRuntime:
    model: ClassConditionedCVAE | AggregateMatchedMixturePriorCVAE
    arm: str
    training_key: SourceExpertTrainingKey
    device: str
    checkpoint_hash: str
    warmup_checkpoint_hash: str
    shared_initialization_hash: str
    training_stream_hash: str
    mixture_refit_records: tuple[Mapping[str, object], ...]
    geco_state: Mapping[str, object] | None
    geco_trajectory: tuple[Mapping[str, object], ...]
    epoch_diagnostics: tuple[Mapping[str, object], ...]


def train_source_expert_panel(
    train_embeddings: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    case_ids: Sequence[str],
    *,
    config: AggregatePriorStudyConfig,
    training_keys: Mapping[str, SourceExpertTrainingKey],
) -> Mapping[str, SourceExpertRuntime]:
    """Train the exact four-arm panel from one shared source-only warm start."""

    x_np = np.asarray(train_embeddings, dtype=np.float32)
    y_np = np.asarray(train_labels, dtype=np.int64)
    cases = tuple(str(value) for value in case_ids)
    if (
        x_np.ndim != 2
        or len(x_np) == 0
        or len(x_np) != len(y_np)
        or len(cases) != len(y_np)
        or set(int(value) for value in y_np.tolist()) != {0, 1}
    ):
        raise ProtocolError("Source expert training arrays are malformed.")
    if tuple(training_keys) != ARMS:
        raise ProtocolError("Source expert panel requires exact ordered four-arm keys.")
    neutral_hashes = {
        training_keys[arm].arm_neutral_hash for arm in ARMS
    }
    if len(neutral_hashes) != 1:
        raise ProtocolError("Four source-expert arms do not share one neutral identity.")
    neutral_hash = next(iter(neutral_hashes))
    if any(training_keys[arm].arm != arm for arm in ARMS):
        raise ProtocolError("Source expert training-key arm mismatch.")

    _configure_determinism()
    device = _resolve_device(config.device)
    initialization_seed = _derived_seed(neutral_hash, "shared_initialization")
    torch.manual_seed(initialization_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(initialization_seed)
    warm_model = ClassConditionedCVAE(
        input_dim=x_np.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_hidden_layers=config.num_hidden_layers,
    ).to(device)
    shared_initialization_hash = model_state_hash(warm_model)
    warm_optimizer = torch.optim.AdamW(
        warm_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warm_diagnostics: list[dict[str, object]] = []
    warm_stream_records: list[dict[str, object]] = []
    for epoch in range(1, config.warmup_epochs + 1):
        diagnostic, stream = _train_epoch(
            model=warm_model,
            optimizer=warm_optimizer,
            x_np=x_np,
            y_np=y_np,
            epoch=epoch,
            stream_scope="warmup",
            neutral_hash=neutral_hash,
            batch_size=config.batch_size,
            gradient_clip_norm=config.gradient_clip_norm,
            objective="fixed_beta",
            beta=config.beta_final
            * min(1.0, epoch / float(config.kl_warmup_epochs)),
            controller=None,
        )
        warm_diagnostics.append(diagnostic)
        warm_stream_records.append(stream)
    warmup_checkpoint_hash = model_state_hash(warm_model)
    warmup_distortion = _mean_source_distortion(
        warm_model,
        x_np,
        y_np,
        device=device,
    )
    geco_target = warmup_distortion * config.geco_target_slack
    if not math.isfinite(geco_target) or geco_target <= 0.0:
        raise ProtocolError("Source-only warmup produced an invalid GECO target.")

    runtimes: dict[str, SourceExpertRuntime] = {}
    for arm in ARMS:
        if prior_family(arm) == MIXTURE_PRIOR:
            model: ClassConditionedCVAE | AggregateMatchedMixturePriorCVAE
            model = AggregateMatchedMixturePriorCVAE(
                input_dim=x_np.shape[1],
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
                num_hidden_layers=config.num_hidden_layers,
                n_components=config.n_components,
                mixture_rank=config.mixture_rank,
                weight_floor=config.weight_floor,
                variance_floor=config.variance_floor,
            ).to(device)
            missing, unexpected = model.load_state_dict(
                deepcopy(warm_model.state_dict()),
                strict=False,
            )
            expected_missing = {
                "latent_prior.mixture_logits",
                "latent_prior.component_means",
                "latent_prior.diag_rho",
                "latent_prior.low_rank",
            }
            if set(missing) != expected_missing or unexpected:
                raise ProtocolError("Mixture arm warm-start state did not align.")
            mixture_records = [
                _refit_mixture(
                    model,
                    x_np=x_np,
                    y_np=y_np,
                    case_ids=cases,
                    config=config,
                    device=device,
                    random_state=_derived_seed(neutral_hash, "mixture_initial"),
                    refit_index=0,
                    after_continuation_epoch=0,
                )
            ]
            _freeze_prior(model)
        else:
            model = deepcopy(warm_model)
            mixture_records = []
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            tuple(_shared_parameters(model)),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        controller = (
            GECOController(
                target=geco_target,
                ema_decay=config.geco_ema_decay,
                dual_step_size=config.geco_dual_step_size,
                initial_multiplier=config.geco_initial_multiplier,
                minimum_multiplier=config.geco_minimum_multiplier,
                maximum_multiplier=config.geco_maximum_multiplier,
            )
            if objective_family(arm) == GECO
            else None
        )
        diagnostics: list[Mapping[str, object]] = [
            {
                **row,
                "phase": "warmup",
                "arm": arm,
                "training_key_hash": training_keys[arm].hash,
            }
            for row in warm_diagnostics
        ]
        geco_trajectory: list[Mapping[str, object]] = []
        continuation_stream: list[Mapping[str, object]] = []
        refit_cutoff = (
            config.continuation_epochs - config.final_stabilization_epochs
        )
        for continuation_epoch in range(1, config.continuation_epochs + 1):
            diagnostic, stream = _train_epoch(
                model=model,
                optimizer=optimizer,
                x_np=x_np,
                y_np=y_np,
                epoch=continuation_epoch,
                stream_scope="continuation",
                neutral_hash=neutral_hash,
                batch_size=config.batch_size,
                gradient_clip_norm=config.gradient_clip_norm,
                objective=(
                    "geco" if controller is not None else "fixed_beta"
                ),
                beta=config.beta_final,
                controller=controller,
                controller_trace=geco_trajectory,
                arm=arm,
                training_key_hash=training_keys[arm].hash,
            )
            diagnostics.append(
                {
                    **diagnostic,
                    "phase": "continuation",
                    "arm": arm,
                    "training_key_hash": training_keys[arm].hash,
                }
            )
            continuation_stream.append(stream)
            should_refit = (
                isinstance(model, AggregateMatchedMixturePriorCVAE)
                and continuation_epoch <= refit_cutoff
                and continuation_epoch % config.refit_interval_epochs == 0
            )
            if should_refit:
                mixture_records.append(
                    _refit_mixture(
                        model,
                        x_np=x_np,
                        y_np=y_np,
                        case_ids=cases,
                        config=config,
                        device=device,
                        random_state=_derived_seed(
                            neutral_hash,
                            "mixture_refit",
                            len(mixture_records),
                        ),
                        refit_index=len(mixture_records),
                        after_continuation_epoch=continuation_epoch,
                    )
                )
                _freeze_prior(model)
        if isinstance(model, AggregateMatchedMixturePriorCVAE):
            if (
                not mixture_records
                or int(mixture_records[-1]["after_continuation_epoch"])
                != refit_cutoff
            ):
                raise ProtocolError(
                    "Mixture arm lacks its final pre-stabilization aggregate refit."
                )
            model.latent_prior.assert_healthy(
                maximum_condition_number=config.maximum_condition_number
            )
            if any(parameter.grad is not None for parameter in model.latent_prior.parameters()):
                raise ProtocolError("Frozen mixture prior accumulated gradients.")
        stream_hash = stable_hash(
            {
                "schema_version": "midogpp_source_expert_training_stream_v3",
                "neutral_hash": neutral_hash,
                "warmup": warm_stream_records,
                "continuation": continuation_stream,
            }
        )
        runtimes[arm] = SourceExpertRuntime(
            model=model,
            arm=arm,
            training_key=training_keys[arm],
            device=device,
            checkpoint_hash=model_state_hash(model),
            warmup_checkpoint_hash=warmup_checkpoint_hash,
            shared_initialization_hash=shared_initialization_hash,
            training_stream_hash=stream_hash,
            mixture_refit_records=tuple(mixture_records),
            geco_state=(
                None if controller is None else controller.state_payload()
            ),
            geco_trajectory=tuple(geco_trajectory),
            epoch_diagnostics=tuple(diagnostics),
        )
    if len({runtime.warmup_checkpoint_hash for runtime in runtimes.values()}) != 1:
        raise ProtocolError("Four arms do not share one warmup checkpoint.")
    if len({runtime.training_stream_hash for runtime in runtimes.values()}) != 1:
        raise ProtocolError("Four arms do not share one paired training stream.")
    return runtimes


def generate_projected(
    runtime: SourceExpertRuntime,
    labels: Sequence[int],
    *,
    epsilon: np.ndarray,
    component_uniform: np.ndarray,
) -> np.ndarray:
    y_np = np.asarray(labels, dtype=np.int64)
    eps_np = np.asarray(epsilon, dtype=np.float32)
    uniform_np = np.asarray(component_uniform, dtype=np.float32)
    if eps_np.shape != (len(y_np), runtime.model.latent_dim):
        raise ProtocolError("Generation epsilon shape mismatch.")
    device = torch.device(runtime.device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    eps = torch.as_tensor(eps_np, dtype=torch.float32, device=device)
    runtime.model.eval()
    with torch.no_grad():
        if isinstance(runtime.model, AggregateMatchedMixturePriorCVAE):
            uniform = torch.as_tensor(
                uniform_np,
                dtype=torch.float32,
                device=device,
            )
            latent = runtime.model.sample_prior(
                y,
                epsilon=eps,
                component_uniform=uniform,
            )
        else:
            latent = eps
        return runtime.model.decode(latent, y).cpu().numpy()


def posterior_projected(
    runtime: SourceExpertRuntime,
    embeddings: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    epsilon: np.ndarray,
) -> np.ndarray:
    x_np = np.asarray(embeddings, dtype=np.float32)
    y_np = np.asarray(labels, dtype=np.int64)
    eps_np = np.asarray(epsilon, dtype=np.float32)
    if (
        x_np.ndim != 2
        or len(x_np) != len(y_np)
        or eps_np.shape != (len(y_np), runtime.model.latent_dim)
    ):
        raise ProtocolError("Posterior-generation arrays are not aligned.")
    device = torch.device(runtime.device)
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    eps = torch.as_tensor(eps_np, dtype=torch.float32, device=device)
    runtime.model.eval()
    with torch.no_grad():
        posterior_mu, posterior_logvar = runtime.model.encode(x, y)
        latent = posterior_mu + torch.exp(0.5 * posterior_logvar) * eps
        return runtime.model.decode(latent, y).cpu().numpy()


def paired_generation_noise(
    *,
    neutral_evaluation_hash: str,
    labels: Sequence[int],
    latent_dim: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    y = np.asarray(labels, dtype=np.int64)
    seed = _derived_seed(neutral_evaluation_hash, "prior_generation")
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(size=(len(y), int(latent_dim))).astype(np.float32)
    component_uniform = rng.uniform(size=len(y)).astype(np.float32)
    noise_hash = stable_hash(
        {
            "schema_version": "midogpp_v3_paired_generation_noise_v1",
            "neutral_evaluation_hash": neutral_evaluation_hash,
            "seed": seed,
            "label_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
            "epsilon_sha256": hashlib.sha256(epsilon.tobytes()).hexdigest(),
            "component_uniform_sha256": hashlib.sha256(
                component_uniform.tobytes()
            ).hexdigest(),
        }
    )
    return epsilon, component_uniform, noise_hash


def model_state_hash(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(
        {
            key: value.detach().cpu()
            for key, value in sorted(model.state_dict().items())
        },
        buffer,
    )
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _train_epoch(
    *,
    model: ClassConditionedCVAE | AggregateMatchedMixturePriorCVAE,
    optimizer: torch.optim.Optimizer,
    x_np: np.ndarray,
    y_np: np.ndarray,
    epoch: int,
    stream_scope: str,
    neutral_hash: str,
    batch_size: int,
    gradient_clip_norm: float,
    objective: str,
    beta: float,
    controller: GECOController | None,
    controller_trace: list[Mapping[str, object]] | None = None,
    arm: str = "warmup",
    training_key_hash: str = "warmup",
) -> tuple[dict[str, object], dict[str, object]]:
    device = next(model.parameters()).device
    permutation_seed = _derived_seed(
        neutral_hash,
        stream_scope,
        epoch,
        "permutation",
    )
    permutation = np.random.default_rng(permutation_seed).permutation(len(x_np))
    posterior_seeds: list[int] = []
    totals = {"loss": 0.0, "distortion": 0.0, "rate": 0.0, "n": 0}
    model.train()
    for batch_index, start in enumerate(range(0, len(x_np), int(batch_size))):
        indices = permutation[start : start + int(batch_size)]
        x = torch.as_tensor(x_np[indices], dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np[indices], dtype=torch.long, device=device)
        posterior_seed = _derived_seed(
            neutral_hash,
            stream_scope,
            epoch,
            batch_index,
            "posterior",
        )
        posterior_seeds.append(posterior_seed)
        generator = torch.Generator(
            device="cuda" if str(device).startswith("cuda") else "cpu"
        ).manual_seed(posterior_seed)
        optimizer.zero_grad(set_to_none=True)
        posterior_mu, posterior_logvar = model.encode(x, y)
        epsilon = torch.randn(
            posterior_mu.shape,
            dtype=posterior_mu.dtype,
            device=device,
            generator=generator,
        )
        reconstruction = model.decode(
            posterior_mu + torch.exp(0.5 * posterior_logvar) * epsilon,
            y,
        )
        distortion = F.mse_loss(
            reconstruction,
            x,
            reduction="none",
        ).mean(dim=-1).mean()
        if isinstance(model, AggregateMatchedMixturePriorCVAE):
            rate = model.latent_prior.kl_upper_bound(
                posterior_mu,
                posterior_logvar,
                y,
            ).mean()
        else:
            rate = (
                -0.5
                * torch.sum(
                    1.0
                    + posterior_logvar
                    - posterior_mu.square()
                    - posterior_logvar.exp(),
                    dim=-1,
                )
                / float(model.latent_dim)
            ).mean()
        if objective == "geco":
            if controller is None:
                raise ProtocolError("GECO epoch lacks a controller.")
            loss = controller.loss(rate=rate, distortion=distortion)
        elif objective == "fixed_beta":
            if controller is not None:
                raise ProtocolError("Fixed-beta epoch received a GECO controller.")
            loss = distortion + float(beta) * rate
        else:
            raise ProtocolError(f"Unknown training objective: {objective!r}.")
        if not bool(torch.isfinite(loss)):
            raise ProtocolError("Source expert produced a nonfinite objective.")
        loss.backward()
        parameters = tuple(_shared_parameters(model))
        if any(
            parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
            for parameter in parameters
        ):
            raise ProtocolError("Source expert produced a nonfinite gradient.")
        clip_grad_norm_(parameters, float(gradient_clip_norm))
        optimizer.step()
        if any(
            not bool(torch.isfinite(parameter).all())
            for parameter in model.parameters()
        ):
            raise ProtocolError("Source expert produced nonfinite model state.")
        if isinstance(model, AggregateMatchedMixturePriorCVAE):
            model.latent_prior.assert_healthy(maximum_condition_number=1e12)
        if controller is not None:
            before = controller.multiplier
            after = controller.update(distortion)
            if controller_trace is not None:
                controller_trace.append(
                    {
                        "schema_version": "midogpp_geco_trajectory_v1",
                        "arm": arm,
                        "training_key_hash": training_key_hash,
                        "continuation_epoch": epoch,
                        "batch_index": batch_index,
                        "distortion": float(distortion.detach().cpu()),
                        "target": controller.target,
                        "constraint": (
                            float(distortion.detach().cpu()) - controller.target
                        ),
                        "ema_constraint": controller.ema_constraint,
                        "multiplier_before": before,
                        "multiplier_after": after,
                        "update_count": controller.update_count,
                        "target_provenance": (
                            "source_only_warmup_reconstruction"
                        ),
                    }
                )
        n_batch = len(indices)
        totals["loss"] += float(loss.detach().cpu()) * n_batch
        totals["distortion"] += float(distortion.detach().cpu()) * n_batch
        totals["rate"] += float(rate.detach().cpu()) * n_batch
        totals["n"] += n_batch
    diagnostic = {
        "schema_version": "midogpp_source_expert_training_epoch_v3",
        "epoch": int(epoch),
        "objective": objective,
        "beta": float(beta) if objective == "fixed_beta" else None,
        "mean_objective": totals["loss"] / totals["n"],
        "mean_distortion": totals["distortion"] / totals["n"],
        "mean_rate": totals["rate"] / totals["n"],
        "rate_semantics": (
            "mixture_KL_upper_bound"
            if isinstance(model, AggregateMatchedMixturePriorCVAE)
            else "analytic_standard_normal_KL"
        ),
        "not_exact_nelbo": True,
        "n_rows": totals["n"],
    }
    stream = {
        "epoch": int(epoch),
        "scope": stream_scope,
        "permutation_seed": permutation_seed,
        "permutation_sha256": hashlib.sha256(
            permutation.astype(np.int64).tobytes()
        ).hexdigest(),
        "posterior_seeds": posterior_seeds,
    }
    return diagnostic, stream


def _refit_mixture(
    model: AggregateMatchedMixturePriorCVAE,
    *,
    x_np: np.ndarray,
    y_np: np.ndarray,
    case_ids: Sequence[str],
    config: AggregatePriorStudyConfig,
    device: str,
    random_state: int,
    refit_index: int,
    after_continuation_epoch: int,
) -> Mapping[str, object]:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=device)
        posterior_mu, posterior_logvar = model.encode(x, y)
    initialization = model.latent_prior.initialize_from_aggregate_posterior(
        posterior_mu,
        posterior_logvar,
        y,
        case_ids=case_ids,
        random_state=random_state,
        shrinkage=config.covariance_shrinkage,
        minimum_component_rows=config.minimum_component_rows,
        minimum_component_cases=config.minimum_component_cases,
    )
    model.latent_prior.assert_healthy(
        maximum_condition_number=config.maximum_condition_number
    )
    state = dict(model.latent_prior.state_payload())
    diagnostics = model.latent_prior.state_diagnostics().to_payload()
    payload: dict[str, object] = {
        "schema_version": "midogpp_aggregate_posterior_refit_record_v1",
        "refit_index": int(refit_index),
        "after_continuation_epoch": int(after_continuation_epoch),
        "random_state": int(random_state),
        "fit_scope": "source_center_only_all_rows",
        "optimizer_updates_prior_parameters": False,
        "coordinate_update": True,
        "state": state,
        "state_hash": stable_hash(state),
        "diagnostics": diagnostics,
        "initialization": initialization.to_payload(),
    }
    payload["record_hash"] = stable_hash(payload)
    return payload


def _freeze_prior(model: AggregateMatchedMixturePriorCVAE) -> None:
    for parameter in model.latent_prior.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def _shared_parameters(
    model: ClassConditionedCVAE | AggregateMatchedMixturePriorCVAE,
) -> Sequence[torch.nn.Parameter]:
    if isinstance(model, AggregateMatchedMixturePriorCVAE):
        parameters = tuple(model.shared_parameters())
    else:
        parameters = tuple(model.parameters())
    if not parameters:
        raise ProtocolError("Source expert model lacks shared parameters.")
    return parameters


def _mean_source_distortion(
    model: ClassConditionedCVAE,
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    device: str,
) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        y = torch.as_tensor(y_np, dtype=torch.long, device=device)
        posterior_mu, _ = model.encode(x, y)
        reconstruction = model.decode(posterior_mu, y)
        value = F.mse_loss(reconstruction, x, reduction="mean")
    return float(value.cpu())


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


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
