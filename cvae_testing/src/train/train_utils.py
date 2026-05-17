from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cvae_expert import CVAEExpert, elbo_components
from src.train.checkpoint_utils import load_resume_state, save_resume_state, training_state_path
from src.train.checkpoint_provenance import (
    load_model_checkpoint,
    wrap_model_state_dict,
)

try:
    tqdm = getattr(importlib.import_module("tqdm"), "tqdm")
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


@dataclass
class TrainResult:
    checkpoint_path: Path
    history: Dict[str, List[float]]


def _make_loader(
    embeddings: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    metadata_vectors: torch.Tensor | None = None,
    class_condition_vectors: torch.Tensor | None = None,
) -> DataLoader:
    tensors = [embeddings]
    if metadata_vectors is not None:
        tensors.append(metadata_vectors)
    if class_condition_vectors is not None:
        tensors.append(class_condition_vectors)
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_training(
    train_embeddings: torch.Tensor,
    val_embeddings: torch.Tensor,
    out_dir: Path,
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    resume_from: Path | None = None,
    train_metadata_vectors: torch.Tensor | None = None,
    val_metadata_vectors: torch.Tensor | None = None,
    train_class_condition_vectors: torch.Tensor | None = None,
    val_class_condition_vectors: torch.Tensor | None = None,
    metadata_dim: int = 0,
    class_condition_dim: int = 0,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    label_utility_cfg: Dict[str, Any] | None = None,
    checkpoint_metadata: Dict[str, Any] | None = None,
) -> TrainResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{model_name}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = CVAEExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        metadata_dim=int(metadata_dim),
        metadata_constraint_cfg=metadata_constraint_cfg,
        aux_metadata_dim=int(metadata_dim),
        class_condition_dim=int(class_condition_dim),
        label_utility_cfg=label_utility_cfg,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state_ckpt = training_state_path(ckpt)
    effective_checkpoint_metadata = dict(checkpoint_metadata or {})

    if train_metadata_vectors is not None and int(train_metadata_vectors.shape[0]) != int(train_embeddings.shape[0]):
        raise ValueError("train_metadata_vectors must have the same number of rows as train_embeddings")
    if val_metadata_vectors is not None and int(val_metadata_vectors.shape[0]) != int(val_embeddings.shape[0]):
        raise ValueError("val_metadata_vectors must have the same number of rows as val_embeddings")
    if train_class_condition_vectors is not None and int(train_class_condition_vectors.shape[0]) != int(train_embeddings.shape[0]):
        raise ValueError("train_class_condition_vectors must have the same number of rows as train_embeddings")
    if val_class_condition_vectors is not None and int(val_class_condition_vectors.shape[0]) != int(val_embeddings.shape[0]):
        raise ValueError("val_class_condition_vectors must have the same number of rows as val_embeddings")
    if int(class_condition_dim) > 0 and (train_class_condition_vectors is None or val_class_condition_vectors is None):
        raise ValueError("class_condition_vectors are required when class_condition_dim > 0")
    if train_class_condition_vectors is not None and int(train_class_condition_vectors.shape[1]) != int(class_condition_dim):
        raise ValueError("train_class_condition_vectors width must match class_condition_dim")
    if val_class_condition_vectors is not None and int(val_class_condition_vectors.shape[1]) != int(class_condition_dim):
        raise ValueError("val_class_condition_vectors width must match class_condition_dim")

    train_loader = _make_loader(
        train_embeddings,
        batch_size=batch_size,
        shuffle=True,
        metadata_vectors=train_metadata_vectors,
        class_condition_vectors=train_class_condition_vectors,
    )
    val_loader = _make_loader(
        val_embeddings,
        batch_size=batch_size,
        shuffle=False,
        metadata_vectors=val_metadata_vectors,
        class_condition_vectors=val_class_condition_vectors,
    )

    best_val = float("inf")
    bad_epochs = 0
    history = {"train": [], "val": []}
    start_epoch = 0
    best_epoch = -1

    resume_state_path = None
    if resume_from is not None:
        resume_state_path = resume_from if resume_from.name.endswith(".training.pt") else training_state_path(resume_from)
    elif state_ckpt.exists():
        resume_state_path = state_ckpt

    if resume_state_path is not None and resume_state_path.exists():
        state = load_resume_state(resume_state_path)
        model.load_state_dict(state["model_payload"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = state.get("history", history)
        start_epoch = int(state.get("epoch", -1)) + 1
        best_val = float(state.get("best_metric", best_val))
        bad_epochs = int(state.get("bad_epochs", bad_epochs))
    elif ckpt.exists():
        # Backward compatibility: plain model checkpoint without optimizer state.
        model.load_state_dict(load_model_checkpoint(ckpt, map_location=device).model_state_dict)

    epoch_iter = range(start_epoch, epochs)
    epoch_bar = tqdm(epoch_iter, desc=f"train:{model_name}", unit="epoch") if tqdm is not None else None

    for epoch in epoch_iter:
        model.train()
        train_metrics = _empty_metric_sums(model)
        for batch in train_loader:
            offset = 1
            x = batch[0]
            m = batch[offset].to(device) if train_metadata_vectors is not None else None
            if train_metadata_vectors is not None:
                offset += 1
            y = batch[offset].to(device) if train_class_condition_vectors is not None else None
            x = x.to(device)
            loss, metrics = _compute_training_objective(
                model,
                x=x,
                m=m,
                y=y,
                label_utility_cfg=label_utility_cfg,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _accumulate_metric_sums(train_metrics, metrics, batch_size=int(x.size(0)))

        model.eval()
        val_metrics = _empty_metric_sums(model)
        with torch.no_grad():
            for batch in val_loader:
                offset = 1
                x = batch[0]
                m = batch[offset].to(device) if val_metadata_vectors is not None else None
                if val_metadata_vectors is not None:
                    offset += 1
                y = batch[offset].to(device) if val_class_condition_vectors is not None else None
                x = x.to(device)
                loss, metrics = _compute_training_objective(
                    model,
                    x=x,
                    m=m,
                    y=y,
                    label_utility_cfg=label_utility_cfg,
                )
                _ = loss
                _accumulate_metric_sums(val_metrics, metrics, batch_size=int(x.size(0)))

        train_epoch_metrics = _finalize_metric_sums(train_metrics, n_rows=max(len(train_embeddings), 1))
        val_epoch_metrics = _finalize_metric_sums(val_metrics, n_rows=max(len(val_embeddings), 1))
        train_epoch = train_epoch_metrics["total_loss"]
        val_epoch = val_epoch_metrics["total_loss"]
        history["train"].append(train_epoch)
        history["val"].append(val_epoch)
        _append_epoch_metrics(history, "train", train_epoch_metrics)
        _append_epoch_metrics(history, "val", val_epoch_metrics)

        if epoch_bar is not None:
            postfix = {
                "train": f"{train_epoch:.4f}",
                "val": f"{val_epoch:.4f}",
                "best": f"{best_val:.4f}",
                "bad": f"{bad_epochs}/{patience}",
            }
            if model.metadata_constraint_enabled and model.metadata_constraint_variant == "aux_head":
                postfix["aux"] = f"{history['val_metadata_aux_loss'][-1]:.4f}"
            if model.label_utility_enabled:
                postfix["nelbo"] = f"{history['val_nelbo'][-1]:.4f}"
                postfix["prior_acc"] = f"{history['val_prior_cls_acc'][-1]:.4f}"
            epoch_bar.set_postfix(**postfix)
            epoch_bar.update(1)

        if val_epoch < best_val:
            best_val = val_epoch
            bad_epochs = 0
            best_epoch = epoch
            _record_best_diagnostics(history, epoch=epoch, val_metrics=val_epoch_metrics)
            torch.save(wrap_model_state_dict(model.state_dict(), effective_checkpoint_metadata), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                save_resume_state(
                    state_ckpt,
                    model_payload=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    history=history,
                    epoch=epoch,
                    best_metric=best_val,
                    bad_epochs=bad_epochs,
                    meta={"model_name": model_name},
                )
                break

        save_resume_state(
            state_ckpt,
            model_payload=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            history=history,
            epoch=epoch,
            best_metric=best_val,
            bad_epochs=bad_epochs,
            meta={"model_name": model_name},
        )

    if epoch_bar is not None:
        epoch_bar.close()

    if best_epoch < 0 and history["val"]:
        best_epoch = int(min(range(len(history["val"])), key=lambda idx: history["val"][idx]))
        val_metrics = {key[4:]: values[best_epoch] for key, values in history.items() if key.startswith("val_") and values}
        _record_best_diagnostics(history, epoch=best_epoch, val_metrics=val_metrics)

    return TrainResult(checkpoint_path=ckpt, history=history)


def _empty_metric_sums(model: CVAEExpert) -> dict[str, float]:
    keys = [
        "total_loss",
        "nelbo",
        "reconstruction_mse",
        "kl",
        "metadata_aux_loss",
        "latent_cls_loss",
        "recon_cls_loss",
        "prior_cls_loss",
        "latent_cls_acc",
        "recon_cls_acc",
        "prior_cls_acc",
    ]
    return {key: 0.0 for key in keys}


def _compute_training_objective(
    model: CVAEExpert,
    *,
    x: torch.Tensor,
    m: torch.Tensor | None,
    y: torch.Tensor | None,
    label_utility_cfg: Dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon, mu, logvar, aux_logits = model(x, m=m, y=y, return_aux=True)
    prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=m)
    recon_terms, kl_terms = elbo_components(
        recon,
        x,
        mu,
        logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        kl_weight=kl_weight,
    )
    nelbo = (recon_terms + kl_terms).mean()
    loss = nelbo
    metrics = {
        "total_loss": float(nelbo.detach().item()),
        "nelbo": float(nelbo.detach().item()),
        "reconstruction_mse": float(recon_terms.detach().mean().item()),
        "kl": float(kl_terms.detach().mean().item()),
        "metadata_aux_loss": 0.0,
        "latent_cls_loss": 0.0,
        "recon_cls_loss": 0.0,
        "prior_cls_loss": 0.0,
        "latent_cls_acc": 0.0,
        "recon_cls_acc": 0.0,
        "prior_cls_acc": 0.0,
    }

    if model.metadata_constraint_enabled and model.metadata_constraint_variant == "aux_head":
        aux_loss = model.metadata_constraint_loss(aux_logits=aux_logits, metadata_targets=m)
        loss = loss + (model.metadata_constraint_weight * aux_loss)
        metrics["metadata_aux_loss"] = float(aux_loss.detach().item())

    if model.label_utility_enabled:
        if y is None:
            raise ValueError("Family D label-utility training requires class-condition vectors")
        utility_cfg = label_utility_cfg or {}
        lambda_latent = float(utility_cfg.get("lambda_latent_cls", 0.0))
        lambda_recon = float(utility_cfg.get("lambda_recon_cls", 0.0))
        lambda_prior = float(utility_cfg.get("lambda_prior_cls", 0.0))

        latent_logits = model.label_utility_latent_logits(mu=mu)
        recon_logits = model.label_utility_decoded_logits(recon)
        latent_loss = model.label_utility_loss(latent_logits, y)
        recon_loss = model.label_utility_loss(recon_logits, y)
        prior_y = _prior_label_batch(y, label_utility_cfg=utility_cfg)
        z_prior = torch.randn((int(prior_y.shape[0]), int(model.latent_dim)), dtype=x.dtype, device=x.device)
        prior_decoded = model.decode(z_prior, y=prior_y)
        prior_logits = model.label_utility_decoded_logits(prior_decoded)
        prior_loss = model.label_utility_loss(prior_logits, prior_y)

        loss = loss + (lambda_latent * latent_loss) + (lambda_recon * recon_loss) + (lambda_prior * prior_loss)
        metrics.update(
            {
                "latent_cls_loss": float(latent_loss.detach().item()),
                "recon_cls_loss": float(recon_loss.detach().item()),
                "prior_cls_loss": float(prior_loss.detach().item()),
                "latent_cls_acc": float(model.label_utility_accuracy(latent_logits, y).detach().item()),
                "recon_cls_acc": float(model.label_utility_accuracy(recon_logits, y).detach().item()),
                "prior_cls_acc": float(model.label_utility_accuracy(prior_logits, prior_y).detach().item()),
            }
        )
        metrics["total_loss"] = float(loss.detach().item())
    else:
        metrics["total_loss"] = float(loss.detach().item())

    return loss, metrics


def _prior_label_batch(y: torch.Tensor, *, label_utility_cfg: Dict[str, Any]) -> torch.Tensor:
    requested = label_utility_cfg.get("prior_samples_per_batch", "same_batch_size")
    if requested in {None, "same_batch_size", "same_batch", "batch"}:
        return y
    n_requested = int(requested)
    if n_requested <= 0:
        return y
    idx = torch.arange(n_requested, device=y.device) % int(y.shape[0])
    return y.index_select(0, idx)


def _accumulate_metric_sums(
    sums: dict[str, float],
    metrics: dict[str, float],
    *,
    batch_size: int,
) -> None:
    for key, value in metrics.items():
        sums[key] = sums.get(key, 0.0) + (float(value) * float(batch_size))


def _finalize_metric_sums(sums: dict[str, float], *, n_rows: int) -> dict[str, float]:
    denom = float(max(int(n_rows), 1))
    return {key: float(value) / denom for key, value in sums.items()}


def _append_epoch_metrics(history: Dict[str, List[float]], split: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        history.setdefault(f"{split}_{key}", []).append(float(value))
    if "total_loss" in metrics:
        history.setdefault(f"{split}_total_loss_alias", []).append(float(metrics["total_loss"]))


def _record_best_diagnostics(history: Dict[str, List[float]], *, epoch: int, val_metrics: dict[str, float]) -> None:
    history["best_epoch"] = [float(epoch)]
    history["best_source_val_total_loss"] = [float(val_metrics.get("total_loss", float("nan")))]
    history["source_val_nelbo_at_best_epoch"] = [float(val_metrics.get("nelbo", float("nan")))]
    val_nelbo = history.get("val_nelbo", [])
    history["source_val_nelbo_best_epoch"] = [
        float(min(range(len(val_nelbo)), key=lambda idx: val_nelbo[idx])) if val_nelbo else float("nan")
    ]
    if val_nelbo:
        nelbo_best_idx = int(min(range(len(val_nelbo)), key=lambda idx: val_nelbo[idx]))
        total_at_nelbo_best = history.get("val_total_loss", [float("nan")])[nelbo_best_idx]
    else:
        total_at_nelbo_best = float("nan")
    history["source_val_total_loss_at_nelbo_best_epoch"] = [float(total_at_nelbo_best)]
