from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
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


AUTOENCODER_PROXY_PROTOCOL = "support_free_ae_proxy_v1"


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
    domain_field: str


def _cache_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_rows(payload: Mapping[str, object]) -> list[Mapping[str, Any]]:
    metadata = payload.get("metadata", [])
    if not isinstance(metadata, list):
        raise ValueError("Feature cache metadata must be a list")
    return [m if isinstance(m, Mapping) else {} for m in metadata]


def _metadata_domain(meta: Mapping[str, Any], *, domain_field: str) -> int:
    field = str(domain_field)
    if field in meta:
        return int(meta[field])
    if field != "magnification" and "magnification" in meta:
        return int(meta["magnification"])
    if "domain_id" in meta:
        return int(meta["domain_id"])
    raise KeyError(f"Metadata row has no domain field {field!r}, magnification, or domain_id")


def _indices_by_domain(payload: Mapping[str, object], domain: int, *, domain_field: str) -> list[int]:
    return [
        i
        for i, meta in enumerate(_metadata_rows(payload))
        if _metadata_domain(meta, domain_field=domain_field) == int(domain)
    ]


def _sample_key(meta: Mapping[str, Any], index: int) -> str:
    for key in (
        "sample_id",
        "id",
        "image_id",
        "patient_id",
        "case_id",
        "slide",
        "path",
        "image_path",
        "filepath",
        "file_path",
        "filename",
    ):
        value = meta.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{str(value)}"
    return "metadata:" + json.dumps(dict(meta), sort_keys=True, default=str) + f":idx:{int(index)}"


def _sample_keys(payload: Mapping[str, object]) -> set[str]:
    return {_sample_key(meta, i) for i, meta in enumerate(_metadata_rows(payload))}


def build_support_free_ae_overlap_audit(
    *,
    train_cache: Path,
    val_cache: Path,
    routing_cache: Path,
) -> Dict[str, Any]:
    train_payload = safe_torch_load(train_cache, map_location="cpu")
    val_payload = safe_torch_load(val_cache, map_location="cpu")
    routing_payload = safe_torch_load(routing_cache, map_location="cpu")
    train_keys = _sample_keys(train_payload)
    val_keys = _sample_keys(val_payload)
    routing_keys = _sample_keys(routing_payload)
    return {
        "ae_train_cache": str(train_cache),
        "ae_val_cache": str(val_cache),
        "routing_query_cache": str(routing_cache),
        "routing_eval_cache": str(routing_cache),
        "ae_train_cache_hash": _cache_hash(Path(train_cache)),
        "ae_val_cache_hash": _cache_hash(Path(val_cache)),
        "routing_query_cache_hash": _cache_hash(Path(routing_cache)),
        "routing_eval_cache_hash": _cache_hash(Path(routing_cache)),
        "ae_train_query_overlap_count": int(len(train_keys.intersection(routing_keys))),
        "ae_val_query_overlap_count": int(len(val_keys.intersection(routing_keys))),
    }


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(x: torch.Tensor, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(x),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
    )


def _validation_scores(
    *,
    model: FeatureAutoencoder,
    val_x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    vals: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in _loader(val_x, batch_size=int(batch_size), shuffle=False, seed=0):
            vals.append(reconstruction_mse_per_sample(model, xb.to(device)).cpu())
    if not vals:
        return np.asarray([], dtype=np.float64)
    return torch.cat(vals, dim=0).numpy().astype(np.float64, copy=False)


def _validation_stats(
    *,
    model: FeatureAutoencoder,
    val_x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    eps: float,
) -> Dict[str, float | int]:
    scores = _validation_scores(model=model, val_x=val_x, batch_size=batch_size, device=device)
    mean = float(np.mean(scores)) if scores.size else float("nan")
    std = float(np.std(scores)) if scores.size else float("nan")
    return {
        "source_val_reconstruction_mse": mean,
        "source_val_reconstruction_std": std,
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
) -> tuple[FeatureAutoencoder, Dict[str, Any]]:
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
        return model, {
            "history": {"train": [], "val": []},
            "ae_training_converged": 1,
            "ae_best_epoch": -1,
            "ae_val_loss": float("nan"),
            "loaded_from": str(resume_from),
        }
    if out_path.exists():
        model.load_state_dict(load_model_checkpoint(out_path, map_location=device).model_state_dict)
        model.eval()
        return model, {
            "history": {"train": [], "val": []},
            "ae_training_converged": 1,
            "ae_best_epoch": -1,
            "ae_val_loss": float("nan"),
            "loaded_from": str(out_path),
        }

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    train_loader = _loader(
        train_x,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        seed=int(seed) + int(domain) * 2003,
    )
    val_loader = _loader(val_x, batch_size=int(cfg.batch_size), shuffle=False, seed=0)
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: Dict[str, list[float]] = {"train": [], "val": []}
    best_state: Dict[str, Any] | None = None

    for epoch in range(int(cfg.epochs)):
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
            best_epoch = int(epoch)
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
    return model, {
        "history": history,
        "ae_training_converged": int(best_state is not None),
        "ae_best_epoch": int(best_epoch),
        "ae_val_loss": float(best_val),
        "loaded_from": "",
    }


def train_domain_autoencoders(
    *,
    train_cache: Path,
    val_cache: Path,
    routing_cache: Path,
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
    domain_field: str,
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
        domain_field=str(domain_field),
    )
    overlap_audit = build_support_free_ae_overlap_audit(
        train_cache=Path(train_cache),
        val_cache=Path(val_cache),
        routing_cache=Path(routing_cache),
    )

    checkpoints: Dict[str, str] = {}
    domain_entries: Dict[str, Dict[str, Any]] = {}
    for domain_raw in domains:
        domain = int(domain_raw)
        train_idxs = _indices_by_domain(train_payload, domain, domain_field=str(domain_field))
        val_idxs = _indices_by_domain(val_payload, domain, domain_field=str(domain_field))
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
                "domain_field": str(domain_field),
                "train_cache": str(train_cache),
                "val_cache": str(val_cache),
                "routing_cache": str(routing_cache),
                "train_size": int(train_x.shape[0]),
                "val_size": int(val_x.shape[0]),
                "autoencoder_config": asdict(cfg),
            },
        )
        resume_from = (resume_from_dir / ckpt.name) if resume_from_dir is not None else None
        model, training_summary = _train_one_autoencoder(
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
        key = str(domain)
        checkpoints[key] = str(ckpt)
        checkpoints[f"{domain}x"] = str(ckpt)
        domain_entries[key] = {
            "source_domain": int(domain),
            "checkpoint": str(ckpt),
            "domain_field": str(domain_field),
            "train_cache": str(train_cache),
            "val_cache": str(val_cache),
            "routing_cache": str(routing_cache),
            "train_cache_indices": [int(i) for i in train_idxs],
            "val_cache_indices": [int(i) for i in val_idxs],
            "train_size": int(train_x.shape[0]),
            "val_size": int(val_x.shape[0]),
            "input_dim": int(input_dim),
            "autoencoder_config": asdict(cfg),
            "feature_extractor": dict(train_payload.get("feature_extractor", {}) or {}),
            **training_summary,
            **stats,
        }

    checkpoints_path = out_dir / "support_free_ae_checkpoints.json"
    provenance_path = out_dir / "support_free_ae_provenance.json"
    quality_path = out_dir / "support_free_ae_quality_diagnostics.json"
    provenance = {
        "protocol_version": AUTOENCODER_PROXY_PROTOCOL,
        "thesis_wording": "target-support-free AE proxy routing",
        "score_definition": "mean_feature_reconstruction_mse_lower_is_source_manifold_fit_proxy",
        "compatibility_claim_boundary": (
            "AE reconstruction fit is a proxy and is not treated as CVAE utility or compatibility."
        ),
        "score_normalization": str(score_normalization).strip().lower(),
        "score_normalization_eps": float(score_normalization_eps),
        "seed": int(seed),
        "train_cache": str(train_cache),
        "val_cache": str(val_cache),
        "routing_cache": str(routing_cache),
        "input_dim": int(input_dim),
        "config": asdict(cfg),
        "domains": domain_entries,
        "overlap_audit": overlap_audit,
    }
    quality_rows = [
        {
            "source_domain": int(domain),
            "source_val_reconstruction_mse_by_domain": float(entry["source_val_reconstruction_mse"]),
            "source_val_reconstruction_std_by_domain": float(entry["source_val_reconstruction_std"]),
            "ae_training_converged": int(entry["ae_training_converged"]),
            "ae_best_epoch": int(entry["ae_best_epoch"]),
            "ae_val_loss": float(entry["ae_val_loss"]),
            "train_size": int(entry["train_size"]),
            "val_size": int(entry["val_size"]),
        }
        for domain, entry in sorted(domain_entries.items(), key=lambda kv: int(kv[0]))
    ]
    checkpoints_path.write_text(json.dumps(checkpoints, indent=2) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    quality_path.write_text(json.dumps(quality_rows, indent=2) + "\n", encoding="utf-8")
    return {
        "checkpoints": checkpoints,
        "checkpoints_path": str(checkpoints_path),
        "provenance": provenance,
        "provenance_path": str(provenance_path),
        "quality_rows": quality_rows,
        "quality_path": str(quality_path),
        "overlap_audit": overlap_audit,
    }
