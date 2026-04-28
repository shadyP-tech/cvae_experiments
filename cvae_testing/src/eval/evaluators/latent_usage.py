from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.data.metadata_conditioning import build_domain_one_hot, resolve_domain_order
from src.models.cvae_expert import CVAEExpert
from src.torch_utils import safe_torch_load


def _parse_expert_domain(name: str) -> int:
    text = str(name)
    if text.startswith("expert_"):
        text = text[len("expert_") :]
    return int(text.replace("x", ""))


def _load_model(
    checkpoint: Path,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    device: torch.device,
    metadata_dim: int,
    metadata_constraint_cfg: Dict[str, Any] | None,
) -> CVAEExpert:
    model = CVAEExpert(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        metadata_dim=int(metadata_dim),
        metadata_constraint_cfg=metadata_constraint_cfg,
        aux_metadata_dim=int(metadata_dim),
    ).to(device)
    model.load_state_dict(safe_torch_load(checkpoint, map_location=device))
    model.eval()
    return model


def _stratified_split_indices(
    labels: np.ndarray,
    train_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train_idx: list[int] = []
    eval_idx: list[int] = []

    classes = sorted(set(int(v) for v in labels.tolist()))
    for cls in classes:
        cls_idx = np.where(labels == int(cls))[0]
        if cls_idx.size == 0:
            continue
        shuffled = rng.permutation(cls_idx)
        if shuffled.size == 1:
            train_idx.append(int(shuffled[0]))
            continue
        split_at = int(round(float(shuffled.size) * float(train_fraction)))
        split_at = max(1, min(split_at, int(shuffled.size) - 1))
        train_idx.extend(int(v) for v in shuffled[:split_at].tolist())
        eval_idx.extend(int(v) for v in shuffled[split_at:].tolist())

    if not eval_idx:
        eval_idx = list(train_idx)

    return np.asarray(train_idx, dtype=np.int64), np.asarray(eval_idx, dtype=np.int64)


def _build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    conf = np.zeros((int(n_classes), int(n_classes)), dtype=np.float64)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        conf[int(yt), int(yp)] += 1.0
    return conf


def _mutual_information_bits(confusion: np.ndarray) -> float:
    total = float(confusion.sum())
    if total <= 0.0:
        return 0.0

    p_xy = confusion / total
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)

    mi = 0.0
    eps = 1e-12
    for i in range(p_xy.shape[0]):
        for j in range(p_xy.shape[1]):
            p = float(p_xy[i, j])
            if p <= 0.0:
                continue
            px = float(p_x[i, 0])
            py = float(p_y[0, j])
            mi += p * np.log2((p + eps) / max(px * py, eps))
    return float(mi)


def _entropy_bits(labels: np.ndarray, n_classes: int) -> float:
    counts = np.zeros((int(n_classes),), dtype=np.float64)
    for v in labels.tolist():
        counts[int(v)] += 1.0
    probs = counts / max(float(counts.sum()), 1.0)
    nz = probs[probs > 0.0]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log2(nz)))


def _separability_ratio(latents: np.ndarray, labels: np.ndarray, n_classes: int) -> Dict[str, float]:
    if latents.size == 0:
        return {
            "between_scatter": 0.0,
            "within_scatter": 0.0,
            "between_within_ratio": 0.0,
        }

    global_mu = latents.mean(axis=0)
    within = 0.0
    between = 0.0
    total_n = float(latents.shape[0])

    for cls in range(int(n_classes)):
        cls_idx = np.where(labels == int(cls))[0]
        if cls_idx.size == 0:
            continue
        cls_latents = latents[cls_idx]
        cls_mu = cls_latents.mean(axis=0)
        within += float(np.mean(np.sum((cls_latents - cls_mu) ** 2, axis=1)))
        between += float(cls_idx.size / total_n) * float(np.sum((cls_mu - global_mu) ** 2))

    ratio = float(between / max(within, 1e-12))
    return {
        "between_scatter": float(between),
        "within_scatter": float(within),
        "between_within_ratio": float(ratio),
    }


def _linear_probe_metrics(
    latents: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    split_seed: int,
    train_fraction: float,
    max_probe_samples: int,
) -> Dict[str, Any]:
    if latents.shape[0] != labels.shape[0]:
        raise ValueError("latents and labels must have the same number of rows")

    if int(max_probe_samples) > 0 and int(latents.shape[0]) > int(max_probe_samples):
        rng = np.random.default_rng(int(split_seed) + 123)
        keep = rng.choice(latents.shape[0], size=int(max_probe_samples), replace=False)
        keep = np.sort(keep)
        latents = latents[keep]
        labels = labels[keep]

    train_idx, eval_idx = _stratified_split_indices(labels=labels, train_fraction=train_fraction, seed=split_seed)
    x_train = latents[train_idx]
    y_train = labels[train_idx]
    x_eval = latents[eval_idx]
    y_eval = labels[eval_idx]

    if x_train.shape[0] == 0 or x_eval.shape[0] == 0:
        return {
            "train_size": int(x_train.shape[0]),
            "eval_size": int(x_eval.shape[0]),
            "accuracy": 0.0,
            "mi_bits": 0.0,
            "nmi": 0.0,
        }

    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    x_train_n = ((x_train - mu) / sigma).astype(np.float32, copy=False)
    x_eval_n = ((x_eval - mu) / sigma).astype(np.float32, copy=False)

    train_x = torch.from_numpy(x_train_n)
    train_y = torch.from_numpy(y_train.astype(np.int64, copy=False))
    eval_x = torch.from_numpy(x_eval_n)
    eval_y = torch.from_numpy(y_eval.astype(np.int64, copy=False))

    torch.manual_seed(int(split_seed))
    model = torch.nn.Linear(int(train_x.shape[1]), int(n_classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

    for _ in range(200):
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eval_logits = model(eval_x)
        pred = torch.argmax(eval_logits, dim=1)

    pred_np = pred.cpu().numpy().astype(np.int64, copy=False)
    y_eval_np = eval_y.cpu().numpy().astype(np.int64, copy=False)
    accuracy = float(np.mean(pred_np == y_eval_np)) if pred_np.size > 0 else 0.0

    conf = _build_confusion_matrix(y_true=y_eval_np, y_pred=pred_np, n_classes=n_classes)
    mi_bits = _mutual_information_bits(conf)
    h_bits = _entropy_bits(y_eval_np, n_classes=n_classes)
    nmi = float(mi_bits / max(h_bits, 1e-12))

    return {
        "train_size": int(x_train.shape[0]),
        "eval_size": int(x_eval.shape[0]),
        "accuracy": float(accuracy),
        "mi_bits": float(mi_bits),
        "nmi": float(nmi),
        "eval_confusion": conf.astype(np.int64).tolist(),
    }


def evaluate_expert_latent_usage(
    test_cache: Path,
    expert_checkpoints: Dict[str, str],
    hidden_dim: int,
    latent_dim: int,
    conditioning_cfg: Dict[str, Any] | None = None,
    configured_domains: Sequence[int] | None = None,
    metadata_constraint_cfg: Dict[str, Any] | None = None,
    split_seed: int = 17,
    train_fraction: float = 0.7,
    max_probe_samples: int = 4000,
    batch_size: int = 2048,
) -> Dict[str, Any]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]
    metadata = payload["metadata"]
    input_dim = int(x_cpu.shape[1])

    cond_cfg = conditioning_cfg or {}
    conditioning_enabled = bool(cond_cfg.get("enabled", False))

    observed_domains = sorted(set(int(m["magnification"]) for m in metadata))
    domain_order = resolve_domain_order(configured_domains or observed_domains)
    domain_to_idx = {int(d): i for i, d in enumerate(domain_order)}
    labels = np.asarray([domain_to_idx[int(m["magnification"])] for m in metadata], dtype=np.int64)

    metadata_vectors = None
    metadata_dim = 0
    if conditioning_enabled:
        metadata_vectors = build_domain_one_hot(metadata, domain_order)
        metadata_dim = int(len(domain_order))

    n_classes = int(len(domain_order))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    by_expert: list[Dict[str, Any]] = []

    for expert_name in sorted(expert_checkpoints.keys(), key=lambda x: _parse_expert_domain(str(x))):
        ckpt = Path(expert_checkpoints[expert_name])
        model = _load_model(
            checkpoint=ckpt,
            input_dim=input_dim,
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            device=device,
            metadata_dim=int(metadata_dim),
            metadata_constraint_cfg=metadata_constraint_cfg,
        )

        latents_parts: list[torch.Tensor] = []
        aux_logits_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, int(x_cpu.shape[0]), int(batch_size)):
                xb = x_cpu[i : i + int(batch_size)].to(device)
                mb = metadata_vectors[i : i + int(batch_size)].to(device) if metadata_vectors is not None else None
                _recon, mu, _logvar, aux_logits = model(xb, m=mb, return_aux=True)
                latents_parts.append(mu.detach().cpu())
                if aux_logits is not None:
                    aux_logits_parts.append(aux_logits.detach().cpu())

        if not latents_parts:
            continue

        latents = torch.cat(latents_parts, dim=0).numpy().astype(np.float64, copy=False)

        probe = _linear_probe_metrics(
            latents=latents,
            labels=labels,
            n_classes=n_classes,
            split_seed=int(split_seed),
            train_fraction=float(train_fraction),
            max_probe_samples=int(max_probe_samples),
        )

        sep = _separability_ratio(latents=latents, labels=labels, n_classes=n_classes)

        aux_head_accuracy = None
        if aux_logits_parts:
            aux_logits = torch.cat(aux_logits_parts, dim=0)
            aux_pred = torch.argmax(aux_logits, dim=1).numpy().astype(np.int64, copy=False)
            aux_head_accuracy = float(np.mean(aux_pred == labels)) if aux_pred.size > 0 else 0.0

        by_expert.append(
            {
                "expert_name": str(expert_name),
                "expert_domain": int(_parse_expert_domain(str(expert_name))),
                "n_samples": int(latents.shape[0]),
                "probe_accuracy": float(probe.get("accuracy", 0.0)),
                "probe_mi_bits": float(probe.get("mi_bits", 0.0)),
                "probe_nmi": float(probe.get("nmi", 0.0)),
                "probe_train_size": int(probe.get("train_size", 0)),
                "probe_eval_size": int(probe.get("eval_size", 0)),
                "separability_between_within_ratio": float(sep["between_within_ratio"]),
                "separability_between_scatter": float(sep["between_scatter"]),
                "separability_within_scatter": float(sep["within_scatter"]),
                "aux_head_accuracy": float(aux_head_accuracy) if aux_head_accuracy is not None else None,
            }
        )

    if by_expert:
        probe_acc_mean = float(np.mean([float(r["probe_accuracy"]) for r in by_expert]))
        probe_nmi_mean = float(np.mean([float(r["probe_nmi"]) for r in by_expert]))
        sep_ratio_mean = float(np.mean([float(r["separability_between_within_ratio"]) for r in by_expert]))
        aux_vals = [float(r["aux_head_accuracy"]) for r in by_expert if r.get("aux_head_accuracy") is not None]
        aux_mean = float(np.mean(aux_vals)) if aux_vals else None
    else:
        probe_acc_mean = 0.0
        probe_nmi_mean = 0.0
        sep_ratio_mean = 0.0
        aux_mean = None

    return {
        "settings": {
            "split_seed": int(split_seed),
            "train_fraction": float(train_fraction),
            "max_probe_samples": int(max_probe_samples),
            "batch_size": int(batch_size),
            "conditioning_enabled": bool(conditioning_enabled),
            "metadata_constraint_enabled": bool((metadata_constraint_cfg or {}).get("enabled", False)),
            "n_domains": int(n_classes),
            "domain_order": [int(d) for d in domain_order],
        },
        "aggregate": {
            "n_experts": int(len(by_expert)),
            "probe_accuracy_mean": float(probe_acc_mean),
            "probe_nmi_mean": float(probe_nmi_mean),
            "separability_ratio_mean": float(sep_ratio_mean),
            "aux_head_accuracy_mean": float(aux_mean) if aux_mean is not None else None,
        },
        "by_expert": by_expert,
    }
