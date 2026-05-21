from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cvae_expert import (
    DECODER_LIKELIHOOD_GAUSSIAN_DIAG,
    DECODER_LIKELIHOOD_MSE,
    RECON_LOSS_GAUSSIAN_NLL_DIAG,
    RECON_LOSS_MSE,
    REDUCTION_MEAN,
    REDUCTION_SUM,
    CVAEExpert,
    elbo_loss_terms,
    negative_elbo,
)
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
    class_labels: torch.Tensor | None = None,
) -> DataLoader:
    tensors = [embeddings]
    if metadata_vectors is not None:
        tensors.append(metadata_vectors)
    if class_labels is not None:
        tensors.append(class_labels.long())
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _unpack_batch(
    batch: tuple[torch.Tensor, ...],
    *,
    has_metadata: bool,
    has_class_labels: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x = batch[0].to(device)
    index = 1
    m = None
    y = None
    if has_metadata:
        m = batch[index].to(device)
        index += 1
    if has_class_labels:
        y = batch[index].to(device)
    return x, m, y


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
    metadata_dim: int = 0,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    checkpoint_metadata: Dict[str, Any] | None = None,
    train_class_labels: torch.Tensor | None = None,
    val_class_labels: torch.Tensor | None = None,
    class_condition_dim: int = 0,
    decoder_likelihood: str = DECODER_LIKELIHOOD_MSE,
    decoder_logvar_min: float = -9.21,
    decoder_logvar_max: float = 2.0,
    decoder_min_variance: float = 1.0e-4,
    reconstruction_loss: str | None = None,
    recon_reduction: str = REDUCTION_SUM,
    kl_reduction: str = REDUCTION_SUM,
    beta: float = 1.0,
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
        decoder_likelihood=decoder_likelihood,
        decoder_logvar_min=decoder_logvar_min,
        decoder_logvar_max=decoder_logvar_max,
        decoder_min_variance=decoder_min_variance,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state_ckpt = training_state_path(ckpt)
    effective_checkpoint_metadata = dict(checkpoint_metadata or {})
    effective_checkpoint_metadata.update(
        {
            "class_condition_dim": int(class_condition_dim),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "latent_dim": int(latent_dim),
            "decoder_likelihood": str(decoder_likelihood).strip().lower(),
            "decoder_logvar_min": float(decoder_logvar_min),
            "decoder_logvar_max": float(decoder_logvar_max),
            "decoder_min_variance": float(decoder_min_variance),
            "recon_reduction": str(recon_reduction).strip().lower(),
            "kl_reduction": str(kl_reduction).strip().lower(),
            "beta_effective": float(beta),
        }
    )

    if train_metadata_vectors is not None and int(train_metadata_vectors.shape[0]) != int(train_embeddings.shape[0]):
        raise ValueError("train_metadata_vectors must have the same number of rows as train_embeddings")
    if val_metadata_vectors is not None and int(val_metadata_vectors.shape[0]) != int(val_embeddings.shape[0]):
        raise ValueError("val_metadata_vectors must have the same number of rows as val_embeddings")
    if train_class_labels is not None and int(train_class_labels.shape[0]) != int(train_embeddings.shape[0]):
        raise ValueError("train_class_labels must have the same number of rows as train_embeddings")
    if val_class_labels is not None and int(val_class_labels.shape[0]) != int(val_embeddings.shape[0]):
        raise ValueError("val_class_labels must have the same number of rows as val_embeddings")
    if int(metadata_dim) > 0 and int(class_condition_dim) > 0:
        raise ValueError("run_training does not support metadata conditioning and class conditioning together")
    if int(class_condition_dim) > 0 and (train_class_labels is None or val_class_labels is None):
        raise ValueError("class_condition_dim > 0 requires train_class_labels and val_class_labels")

    decoder_likelihood_norm = str(decoder_likelihood).strip().lower()
    if reconstruction_loss is None:
        reconstruction_loss = (
            RECON_LOSS_GAUSSIAN_NLL_DIAG
            if decoder_likelihood_norm == DECODER_LIKELIHOOD_GAUSSIAN_DIAG
            else RECON_LOSS_MSE
        )
    reconstruction_loss_norm = str(reconstruction_loss).strip().lower()
    effective_checkpoint_metadata["reconstruction_loss"] = reconstruction_loss_norm
    if decoder_likelihood_norm == DECODER_LIKELIHOOD_GAUSSIAN_DIAG:
        if reconstruction_loss_norm != RECON_LOSS_GAUSSIAN_NLL_DIAG:
            raise ValueError("decoder_likelihood='gaussian_diag' requires reconstruction_loss='gaussian_nll_diag'")
        if str(recon_reduction).strip().lower() != REDUCTION_MEAN:
            raise ValueError("C4.1 gaussian_diag training requires recon_reduction='mean'")
        if str(kl_reduction).strip().lower() != REDUCTION_MEAN:
            raise ValueError("C4.1 gaussian_diag training requires kl_reduction='mean'")
    if float(beta) <= 0:
        raise ValueError("beta must be > 0")

    train_loader = _make_loader(
        train_embeddings,
        batch_size=batch_size,
        shuffle=True,
        metadata_vectors=train_metadata_vectors,
        class_labels=train_class_labels,
    )
    val_loader = _make_loader(
        val_embeddings,
        batch_size=batch_size,
        shuffle=False,
        metadata_vectors=val_metadata_vectors,
        class_labels=val_class_labels,
    )

    best_val = float("inf")
    bad_epochs = 0
    history = {"train": [], "val": []}
    start_epoch = 0

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
        train_loss = 0.0
        train_aux_loss = 0.0
        train_diag_sums: dict[str, float] = {}
        train_count = 0
        for batch in train_loader:
            x, m, y = _unpack_batch(
                batch,
                has_metadata=train_metadata_vectors is not None,
                has_class_labels=train_class_labels is not None,
                device=device,
            )
            return_distribution = decoder_likelihood_norm == DECODER_LIKELIHOOD_GAUSSIAN_DIAG
            recon_payload, mu, logvar, aux_logits = model(
                x,
                m=m,
                y=y,
                return_aux=True,
                return_distribution=return_distribution,
            )
            prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=m)
            effective_kl_weight = float(kl_weight) * float(beta)
            if return_distribution:
                recon, recon_logvar = recon_payload
                terms = elbo_loss_terms(
                    recon,
                    x,
                    mu,
                    logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    kl_weight=effective_kl_weight,
                    recon_logvar_x=recon_logvar,
                    reconstruction_loss=reconstruction_loss_norm,
                    recon_reduction=recon_reduction,
                    kl_reduction=kl_reduction,
                )
                nelbo = terms["loss"].mean()
                batch_n = int(x.size(0))
                train_diag_sums["recon_nll_mean"] = train_diag_sums.get("recon_nll_mean", 0.0) + (
                    float(terms["recon_nll"].mean().item()) * batch_n
                )
                train_diag_sums["logvar_term_mean"] = train_diag_sums.get("logvar_term_mean", 0.0) + (
                    float(terms["logvar_term"].mean().item()) * batch_n
                )
                train_diag_sums["squared_error_scaled_mean"] = train_diag_sums.get(
                    "squared_error_scaled_mean", 0.0
                ) + (float(terms["squared_error_scaled"].mean().item()) * batch_n)
                train_diag_sums["kl_mean"] = train_diag_sums.get("kl_mean", 0.0) + (
                    float(terms["kl"].mean().item()) * batch_n
                )
                train_count += batch_n
            else:
                recon = recon_payload
                nelbo = negative_elbo(
                    recon,
                    x,
                    mu,
                    logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    kl_weight=effective_kl_weight,
                    reconstruction_loss=reconstruction_loss_norm,
                    recon_reduction=recon_reduction,
                    kl_reduction=kl_reduction,
                )
            loss = nelbo
            if model.metadata_constraint_enabled and model.metadata_constraint_variant == "aux_head":
                aux_loss = model.metadata_constraint_loss(aux_logits=aux_logits, metadata_targets=m)
                loss = nelbo + (model.metadata_constraint_weight * aux_loss)
                train_aux_loss += aux_loss.item() * x.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        val_aux_loss = 0.0
        val_diag_sums: dict[str, float] = {}
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x, m, y = _unpack_batch(
                    batch,
                    has_metadata=val_metadata_vectors is not None,
                    has_class_labels=val_class_labels is not None,
                    device=device,
                )
                return_distribution = decoder_likelihood_norm == DECODER_LIKELIHOOD_GAUSSIAN_DIAG
                recon_payload, mu, logvar, aux_logits = model(
                    x,
                    m=m,
                    y=y,
                    return_aux=True,
                    return_distribution=return_distribution,
                )
                prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=m)
                effective_kl_weight = float(kl_weight) * float(beta)
                if return_distribution:
                    recon, recon_logvar = recon_payload
                    terms = elbo_loss_terms(
                        recon,
                        x,
                        mu,
                        logvar,
                        prior_mu=prior_mu,
                        prior_logvar=prior_logvar,
                        kl_weight=effective_kl_weight,
                        recon_logvar_x=recon_logvar,
                        reconstruction_loss=reconstruction_loss_norm,
                        recon_reduction=recon_reduction,
                        kl_reduction=kl_reduction,
                    )
                    nelbo = terms["loss"].mean()
                    batch_n = int(x.size(0))
                    val_diag_sums["recon_nll_mean"] = val_diag_sums.get("recon_nll_mean", 0.0) + (
                        float(terms["recon_nll"].mean().item()) * batch_n
                    )
                    val_diag_sums["logvar_term_mean"] = val_diag_sums.get("logvar_term_mean", 0.0) + (
                        float(terms["logvar_term"].mean().item()) * batch_n
                    )
                    val_diag_sums["squared_error_scaled_mean"] = val_diag_sums.get(
                        "squared_error_scaled_mean", 0.0
                    ) + (float(terms["squared_error_scaled"].mean().item()) * batch_n)
                    val_diag_sums["kl_mean"] = val_diag_sums.get("kl_mean", 0.0) + (
                        float(terms["kl"].mean().item()) * batch_n
                    )
                    val_count += batch_n
                else:
                    recon = recon_payload
                    nelbo = negative_elbo(
                        recon,
                        x,
                        mu,
                        logvar,
                        prior_mu=prior_mu,
                        prior_logvar=prior_logvar,
                        kl_weight=effective_kl_weight,
                        reconstruction_loss=reconstruction_loss_norm,
                        recon_reduction=recon_reduction,
                        kl_reduction=kl_reduction,
                    )
                loss = nelbo
                if model.metadata_constraint_enabled and model.metadata_constraint_variant == "aux_head":
                    aux_loss = model.metadata_constraint_loss(aux_logits=aux_logits, metadata_targets=m)
                    loss = nelbo + (model.metadata_constraint_weight * aux_loss)
                    val_aux_loss += aux_loss.item() * x.size(0)
                val_loss += loss.item() * x.size(0)

        train_epoch = train_loss / max(len(train_embeddings), 1)
        val_epoch = val_loss / max(len(val_embeddings), 1)
        history["train"].append(train_epoch)
        history["val"].append(val_epoch)
        if decoder_likelihood_norm == DECODER_LIKELIHOOD_GAUSSIAN_DIAG:
            for key, value in sorted(train_diag_sums.items()):
                history.setdefault(f"train_{key}", []).append(value / float(max(train_count, 1)))
            for key, value in sorted(val_diag_sums.items()):
                history.setdefault(f"val_{key}", []).append(value / float(max(val_count, 1)))
            train_recon = history["train_recon_nll_mean"][-1]
            val_recon = history["val_recon_nll_mean"][-1]
            history.setdefault("train_posterior_kl_to_recon_ratio", []).append(
                history["train_kl_mean"][-1] / train_recon if abs(train_recon) > 1.0e-12 else float("inf")
            )
            history.setdefault("val_posterior_kl_to_recon_ratio", []).append(
                history["val_kl_mean"][-1] / val_recon if abs(val_recon) > 1.0e-12 else float("inf")
            )
        if model.metadata_constraint_enabled and model.metadata_constraint_variant == "aux_head":
            train_aux_epoch = train_aux_loss / max(len(train_embeddings), 1)
            val_aux_epoch = val_aux_loss / max(len(val_embeddings), 1)
            history.setdefault("train_aux", []).append(train_aux_epoch)
            history.setdefault("val_aux", []).append(val_aux_epoch)

        if epoch_bar is not None:
            postfix = {
                "train": f"{train_epoch:.4f}",
                "val": f"{val_epoch:.4f}",
                "best": f"{best_val:.4f}",
                "bad": f"{bad_epochs}/{patience}",
            }
            if model.metadata_constraint_enabled and model.metadata_constraint_variant == "aux_head":
                postfix["aux"] = f"{history['val_aux'][-1]:.4f}"
            epoch_bar.set_postfix(**postfix)
            epoch_bar.update(1)

        if val_epoch < best_val:
            best_val = val_epoch
            bad_epochs = 0
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

    return TrainResult(checkpoint_path=ckpt, history=history)
