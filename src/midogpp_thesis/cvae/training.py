"""Deterministic stochastic training for preservation-only CVAE variants."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from typing import Mapping, Sequence

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from ..real_features.classifier_reference.artifacts import stable_hash
from .models import ClassConditionedCVAE
from .objectives import (
    ISOTROPIC_OBJECTIVE,
    TASK_FISHER_OBJECTIVE,
    beta_objective,
    validate_trace_normalized_metric,
)


@dataclass(frozen=True)
class TrainingVariant:
    objective_id: str = ISOTROPIC_OBJECTIVE
    hidden_dim: int = 512
    latent_dim: int = 32
    num_hidden_layers: int = 2
    train_epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    beta_final: float = 1e-3
    kl_warmup_epochs: int = 20
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.objective_id not in {ISOTROPIC_OBJECTIVE, TASK_FISHER_OBJECTIVE}:
            raise ValueError(f"Unsupported objective_id: {self.objective_id!r}")
        if self.train_epochs <= 0 or self.batch_size <= 0 or self.latent_dim <= 0:
            raise ValueError("Training epochs, batch size, and latent dimension must be positive.")
        if self.beta_final < 0.0 or self.kl_warmup_epochs <= 0:
            raise ValueError("beta_final must be nonnegative and kl_warmup_epochs positive.")

    def to_payload(self) -> dict[str, object]:
        return {
            "objective_id": self.objective_id,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "train_epochs": self.train_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta_final": self.beta_final,
            "kl_warmup_epochs": self.kl_warmup_epochs,
            "alpha": self.alpha,
        }

    def stochastic_pairing_payload(self) -> dict[str, object]:
        """Return architecture/optimizer fields shared by paired A/B arms."""

        payload = self.to_payload()
        payload.pop("objective_id")
        payload.pop("alpha")
        return payload


@dataclass(frozen=True)
class TrainingKey:
    fit_centers: tuple[str, ...]
    fit_row_hash: str
    objective_id: str
    training_seed: int
    frame_hash: str
    dataset_contract_hash: str
    feature_cache_hash: str
    backbone_output_frame_id: str
    protocol_hash: str
    code_version: str
    variant_hash: str
    stochastic_pairing_hash: str
    objective_context_hash: str

    @property
    def hash(self) -> str:
        return stable_hash(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "fit_centers": list(self.fit_centers),
            "fit_row_hash": self.fit_row_hash,
            "objective_id": self.objective_id,
            "training_seed": self.training_seed,
            "frame_hash": self.frame_hash,
            "dataset_contract_hash": self.dataset_contract_hash,
            "feature_cache_hash": self.feature_cache_hash,
            "backbone_output_frame_id": self.backbone_output_frame_id,
            "protocol_hash": self.protocol_hash,
            "code_version": self.code_version,
            "variant_hash": self.variant_hash,
            "stochastic_pairing_hash": self.stochastic_pairing_hash,
            "objective_context_hash": self.objective_context_hash,
        }


@dataclass
class TrainedCVAERuntime:
    model: ClassConditionedCVAE
    variant: TrainingVariant
    training_key: TrainingKey
    checkpoint_hash: str
    diagnostics: tuple[Mapping[str, object], ...]
    device: str
    initialization_hash: str
    stochastic_stream_hash: str
    reproducibility_policy: str


def train_cvae(
    train_embeddings: Sequence[Sequence[float]],
    train_labels: Sequence[int],
    *,
    variant: TrainingVariant,
    training_key: TrainingKey,
    task_metric: object | None = None,
    device: str = "cpu",
) -> TrainedCVAERuntime:
    """Train with one keyed reparameterized draw per minibatch."""

    import numpy as np

    x_np = np.asarray(train_embeddings, dtype=np.float32)
    y_np = np.asarray(train_labels, dtype=np.int64)
    if x_np.ndim != 2 or len(x_np) != len(y_np) or len(x_np) == 0:
        raise ValueError("Training embeddings/labels must be aligned nonempty arrays.")
    if sorted(set(int(value) for value in y_np.tolist())) != [0, 1]:
        raise ValueError("CVAE training requires both classes 0 and 1.")
    if variant.objective_id == TASK_FISHER_OBJECTIVE and task_metric is None:
        raise ValueError("Task-Fisher training requires a fitted task metric.")
    if variant.objective_id == ISOTROPIC_OBJECTIVE and task_metric is not None:
        raise ValueError("Isotropic training must not receive a task metric.")
    if task_metric is not None:
        validate_trace_normalized_metric(task_metric, input_dim=x_np.shape[1])
    if training_key.objective_id != variant.objective_id:
        raise ValueError("TrainingKey objective_id does not match the training variant.")
    if training_key.variant_hash != training_variant_hash(variant):
        raise ValueError("TrainingKey variant_hash does not match the training variant.")
    if variant.objective_id == ISOTROPIC_OBJECTIVE and training_key.objective_context_hash != "none":
        raise ValueError("Isotropic training must use objective_context_hash='none'.")
    if variant.objective_id == TASK_FISHER_OBJECTIVE and training_key.objective_context_hash == "none":
        raise ValueError("Task-Fisher training requires a fitted objective context hash.")
    _configure_determinism()
    resolved_device = _resolve_device(device)
    initialization_seed = _derived_seed(training_key.stochastic_pairing_hash, "initialization")
    torch.manual_seed(initialization_seed)
    if resolved_device.startswith("cuda"):
        torch.cuda.manual_seed_all(initialization_seed)
    model = ClassConditionedCVAE(
        input_dim=int(x_np.shape[1]),
        hidden_dim=int(variant.hidden_dim),
        latent_dim=int(variant.latent_dim),
        num_hidden_layers=int(variant.num_hidden_layers),
    ).to(resolved_device)
    initialization_hash = checkpoint_hash(model)
    dataset = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    loader = DataLoader(
        dataset,
        batch_size=int(variant.batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(
            _derived_seed(training_key.stochastic_pairing_hash, "loader")
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(variant.learning_rate),
        weight_decay=float(variant.weight_decay),
    )
    metric_tensor = None if task_metric is None else torch.as_tensor(task_metric, dtype=torch.float32, device=resolved_device)
    diagnostics: list[dict[str, object]] = []
    posterior_seeds: list[int] = []
    for epoch in range(1, int(variant.train_epochs) + 1):
        model.train()
        totals = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "n": 0}
        beta = beta_for_epoch(variant, epoch)
        for batch_index, (xb_cpu, yb_cpu) in enumerate(loader):
            xb = xb_cpu.to(resolved_device)
            yb = yb_cpu.to(resolved_device)
            optimizer.zero_grad()
            mu, logvar = model.encode(xb, yb)
            posterior_seed = _derived_seed(
                training_key.stochastic_pairing_hash,
                epoch,
                batch_index,
                "posterior",
            )
            posterior_seeds.append(posterior_seed)
            generator = _torch_generator(resolved_device, posterior_seed)
            noise = torch.randn(mu.shape, generator=generator, dtype=mu.dtype, device=mu.device)
            decoded = model.decode(mu + noise * torch.exp(0.5 * logvar), yb)
            loss = beta_objective(decoded, xb, mu, logvar, beta=beta, metric=metric_tensor)
            loss.total.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            n_batch = int(xb.shape[0])
            totals["total"] += float(loss.total.detach().cpu()) * n_batch
            totals["reconstruction"] += float(loss.reconstruction.detach().cpu()) * n_batch
            totals["kl"] += float(loss.kl.detach().cpu()) * n_batch
            totals["n"] += n_batch
        n_epoch = int(totals["n"])
        diagnostics.append(
            {
                "epoch": epoch,
                "beta": beta,
                "mean_beta_objective": totals["total"] / n_epoch,
                "mean_reconstruction": totals["reconstruction"] / n_epoch,
                "mean_kl": totals["kl"] / n_epoch,
                "n_rows": n_epoch,
                "training_key_hash": training_key.hash,
                "objective_id": variant.objective_id,
                "stochastic_pairing_hash": training_key.stochastic_pairing_hash,
            }
        )
    stochastic_stream_hash = stable_hash(
        {
            "pairing_hash": training_key.stochastic_pairing_hash,
            "initialization_seed": initialization_seed,
            "loader_seed": _derived_seed(training_key.stochastic_pairing_hash, "loader"),
            "posterior_seeds": posterior_seeds,
        }
    )
    final_checkpoint_hash = checkpoint_hash(model)
    return TrainedCVAERuntime(
        model=model,
        variant=variant,
        training_key=training_key,
        checkpoint_hash=final_checkpoint_hash,
        diagnostics=tuple(diagnostics),
        device=resolved_device,
        initialization_hash=initialization_hash,
        stochastic_stream_hash=stochastic_stream_hash,
        reproducibility_policy="torch_deterministic_algorithms_v1",
    )


def beta_for_epoch(variant: TrainingVariant, epoch: int) -> float:
    return float(variant.beta_final) * min(1.0, float(epoch) / float(variant.kl_warmup_epochs))


def checkpoint_hash(model: ClassConditionedCVAE) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def training_variant_hash(variant: TrainingVariant) -> str:
    return stable_hash(variant.to_payload())


def _derived_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _resolve_device(device: str) -> str:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return requested


def _configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _torch_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.Generator(device=generator_device).manual_seed(int(seed))
