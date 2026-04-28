from __future__ import annotations

from typing import Any
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.cvae_expert import CVAEExpert, negative_elbo
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
) -> DataLoader:
    dataset = TensorDataset(embeddings) if metadata_vectors is None else TensorDataset(embeddings, metadata_vectors)
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
    metadata_dim: int = 0,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
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
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state_ckpt = training_state_path(ckpt)
    effective_checkpoint_metadata = dict(checkpoint_metadata or {})

    if train_metadata_vectors is not None and int(train_metadata_vectors.shape[0]) != int(train_embeddings.shape[0]):
        raise ValueError("train_metadata_vectors must have the same number of rows as train_embeddings")
    if val_metadata_vectors is not None and int(val_metadata_vectors.shape[0]) != int(val_embeddings.shape[0]):
        raise ValueError("val_metadata_vectors must have the same number of rows as val_embeddings")

    train_loader = _make_loader(
        train_embeddings,
        batch_size=batch_size,
        shuffle=True,
        metadata_vectors=train_metadata_vectors,
    )
    val_loader = _make_loader(
        val_embeddings,
        batch_size=batch_size,
        shuffle=False,
        metadata_vectors=val_metadata_vectors,
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
        for batch in train_loader:
            if len(batch) == 1:
                (x,) = batch
                m = None
            else:
                x, m = batch
                m = m.to(device)
            x = x.to(device)
            recon, mu, logvar, aux_logits = model(x, m=m, return_aux=True)
            prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=m)
            nelbo = negative_elbo(
                recon,
                x,
                mu,
                logvar,
                prior_mu=prior_mu,
                prior_logvar=prior_logvar,
                kl_weight=kl_weight,
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
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 1:
                    (x,) = batch
                    m = None
                else:
                    x, m = batch
                    m = m.to(device)
                x = x.to(device)
                recon, mu, logvar, aux_logits = model(x, m=m, return_aux=True)
                prior_mu, prior_logvar, kl_weight = model.metadata_constraint_prior(metadata_targets=m)
                nelbo = negative_elbo(
                    recon,
                    x,
                    mu,
                    logvar,
                    prior_mu=prior_mu,
                    prior_logvar=prior_logvar,
                    kl_weight=kl_weight,
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
