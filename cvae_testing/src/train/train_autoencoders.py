from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.feature_autoencoder import FeatureAutoencoder, reconstruction_mse_per_sample
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import (
    build_checkpoint_metadata_from_cache,
    load_model_checkpoint,
    wrap_model_state_dict,
)


AUTOENCODER_PROXY_PROTOCOL = "support_ae_reconstruction_proxy_v1"


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    hidden_dim: int
    latent_dim: int
    learning_rate: float
    epochs: int
    patience: int
    batch_size: int
    score_normalization: str
    score_normalization_eps: float


def _indices_by_domain(payload: Mapping[str, object], domain: int) -> list[int]:
    metadata = payload["metadata"]
    return [i for i, m in enumerate(metadata) if int(m["magnification"]) == int(domain)]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def _loader(x: torch.Tensor, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(x),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
    )


def _validation_stats(
    *,
    model: FeatureAutoencoder,
    val_x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    eps: float,
) -> Dict[str, float | int]:
    vals: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in _loader(val_x, batch_size=int(batch_size), shuffle=False, seed=0):
            vals.append(reconstruction_mse_per_sample(model, xb.to(device)).cpu())
    if vals:
        scores = torch.cat(vals, dim=0).numpy().astype(np.float64, copy=False)
    else:
        scores = np.asarray([], dtype=np.float64)
    mean = float(np.mean(scores)) if scores.size else float("nan")
    std = float(np.std(scores)) if scores.size else float("nan")
    return {
        "source_val_mean_recon_mse": mean,
        "source_val_std_recon_mse": std,
        "score_normalization_eps": float(eps),
        "zero_or_near_zero_std_flag": int((not np.isfinite(std)) or std <= float(eps)),
    }


def _train_one_autoencoder(
    *,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    domain: int,
    out_path: Path,
    resume_from: Path | None,
    cfg: AutoencoderTrainingConfig,
    checkpoint_metadata: Dict[str, Any],
    seed: int,
) -> tuple[FeatureAutoencoder, Dict[str, list[float]]]:
    device = _device()
    torch.manual_seed(int(seed) + int(domain) * 1009)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + int(domain) * 1009)

    model = FeatureAutoencoder(
        input_dim=int(train_x.shape[1]),
        hidden_dim=int(cfg.hidden_dim),
        latent_dim=int(cfg.latent_dim),
    ).to(device)
    if resume_from is not None and resume_from.exists():
        model.load_state_dict(load_model_checkpoint(resume_from, map_location=device).model_state_dict)
        model.eval()
        return model, {"train": [], "val": []}
    if out_path.exists():
        model.load_state_dict(load_model_checkpoint(out_path, map_location=device).model_state_dict)
        model.eval()
        return model, {"train": [], "val": []}

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    train_loader = _loader(
        train_x,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        seed=int(seed) + int(domain) * 2003,
    )
    val_loader = _loader(val_x, batch_size=int(cfg.batch_size), shuffle=False, seed=0)
    best_val = float("inf")
    bad_epochs = 0
    history: Dict[str, list[float]] = {"train": [], "val": []}
    best_state: Dict[str, Any] | None = None

    for _epoch in range(int(cfg.epochs)):
        model.train()
        train_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            loss = reconstruction_mse_per_sample(model, xb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(xb.shape[0])

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                loss = reconstruction_mse_per_sample(model, xb).mean()
                val_loss += float(loss.item()) * int(xb.shape[0])

        train_epoch = train_loss / max(int(train_x.shape[0]), 1)
        val_epoch = val_loss / max(int(val_x.shape[0]), 1)
        history["train"].append(float(train_epoch))
        history["val"].append(float(val_epoch))

        if val_epoch < best_val:
            best_val = float(val_epoch)
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(wrap_model_state_dict(model.state_dict(), checkpoint_metadata), out_path)
    model.eval()
    return model, history


def train_domain_autoencoders(
    *,
    train_cache: Path,
    val_cache: Path,
    out_dir: Path,
    domains: Sequence[int],
    hidden_dim: int,
    latent_dim: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    score_normalization: str,
    score_normalization_eps: float,
    seed: int,
    resume_from_dir: Path | None = None,
) -> Dict[str, Any]:
    if str(score_normalization).strip().lower() != "source_val_zscore":
        raise ValueError("autoencoder_proxy.score_normalization must be 'source_val_zscore'")
    if float(score_normalization_eps) <= 0.0:
        raise ValueError("autoencoder_proxy.score_normalization_eps must be > 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")
    input_dim = int(train_payload["embeddings"].shape[1])
    cfg = AutoencoderTrainingConfig(
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        learning_rate=float(learning_rate),
        epochs=int(epochs),
        patience=int(patience),
        batch_size=int(batch_size),
        score_normalization=str(score_normalization).strip().lower(),
        score_normalization_eps=float(score_normalization_eps),
    )

    checkpoints: Dict[str, str] = {}
    domain_entries: Dict[str, Dict[str, Any]] = {}
    for domain_raw in domains:
        domain = int(domain_raw)
        train_idxs = _indices_by_domain(train_payload, domain)
        val_idxs = _indices_by_domain(val_payload, domain)
        if not train_idxs or not val_idxs:
            continue
        train_x = train_payload["embeddings"][train_idxs].to(dtype=torch.float32)
        val_x = val_payload["embeddings"][val_idxs].to(dtype=torch.float32)
        ckpt = out_dir / f"autoencoder_{domain}x.pt"
        checkpoint_metadata = build_checkpoint_metadata_from_cache(
            train_payload,
            extra={
                "model_name": f"autoencoder_{domain}x",
                "artifact_role": "source_domain_feature_autoencoder_proxy",
                "protocol_version": AUTOENCODER_PROXY_PROTOCOL,
                "source_domain": int(domain),
                "train_cache": str(train_cache),
                "val_cache": str(val_cache),
                "train_size": int(train_x.shape[0]),
                "val_size": int(val_x.shape[0]),
                "autoencoder_config": asdict(cfg),
            },
        )
        resume_from = (resume_from_dir / ckpt.name) if resume_from_dir is not None else None
        model, history = _train_one_autoencoder(
            train_x=train_x,
            val_x=val_x,
            domain=int(domain),
            out_path=ckpt,
            resume_from=resume_from,
            cfg=cfg,
            checkpoint_metadata=checkpoint_metadata,
            seed=int(seed),
        )
        stats = _validation_stats(
            model=model,
            val_x=val_x,
            batch_size=int(batch_size),
            device=next(model.parameters()).device,
            eps=float(score_normalization_eps),
        )
        key = f"{domain}x"
        checkpoints[key] = str(ckpt)
        domain_entries[str(domain)] = {
            "source_domain": int(domain),
            "checkpoint": str(ckpt),
            "train_cache": str(train_cache),
            "val_cache": str(val_cache),
            "train_cache_indices": [int(i) for i in train_idxs],
            "val_cache_indices": [int(i) for i in val_idxs],
            "train_size": int(train_x.shape[0]),
            "val_size": int(val_x.shape[0]),
            "input_dim": int(input_dim),
            "autoencoder_config": asdict(cfg),
            "history": history,
            **stats,
            "feature_extractor": dict(train_payload.get("feature_extractor", {}) or {}),
        }

    checkpoints_path = out_dir / "autoencoder_checkpoints.json"
    provenance_path = out_dir / "autoencoder_provenance.json"
    provenance = {
        "protocol_version": AUTOENCODER_PROXY_PROTOCOL,
        "score_definition": "mean_feature_reconstruction_mse_lower_is_more_compatible_proxy",
        "score_normalization": str(score_normalization).strip().lower(),
        "score_normalization_eps": float(score_normalization_eps),
        "seed": int(seed),
        "train_cache": str(train_cache),
        "val_cache": str(val_cache),
        "input_dim": int(input_dim),
        "config": asdict(cfg),
        "domains": domain_entries,
    }
    checkpoints_path.write_text(json.dumps(checkpoints, indent=2) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        "checkpoints": checkpoints,
        "checkpoints_path": str(checkpoints_path),
        "provenance": provenance,
        "provenance_path": str(provenance_path),
    }
