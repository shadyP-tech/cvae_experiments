from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.bootstrap import (  # noqa: E402
    build_run_context,
    set_global_determinism,
    write_run_metadata,
    write_split_manifest,
)
from src.config.load_config import load_config  # noqa: E402
from src.data.datasets.breakhis import BreakHisRecord, write_manifest  # noqa: E402
from src.data.registry import prepare_dataset_records  # noqa: E402
from src.eval.metrics import pearson_corr, spearman_corr  # noqa: E402
from src.features.extract_embeddings import extract_and_cache_embeddings, validate_embedding_cache  # noqa: E402
from src.models.cvae_expert import CVAEExpert, elbo_components  # noqa: E402
from src.routing.strategies import compute_similarity  # noqa: E402
from src.train.train_experts import train_domain_experts  # noqa: E402


STATIC_MODES = {"static_metadata", "static_embedding", "static_combined"}
RESPONSE_INDIRECT_COLUMNS = [
    "posterior_mu_norm",
    "posterior_mu_mean",
    "posterior_mu_std",
    "posterior_logvar_mean",
    "posterior_logvar_std",
    "posterior_entropy_proxy",
    "decode_repeat_var_mean",
    "decode_repeat_var_max",
    "decode_repeat_var_q75",
    "recon_repeat_var",
    "recon_repeat_var_q75",
    "kl_repeat_var",
]
TARGET_ADJACENT_COLUMNS = [
    "recon_mean",
    "kl_mean",
    "recon_plus_kl_mean",
]
ORACLE_DIAGNOSTIC_COLUMNS = [
    "nelbo_mean",
    "nelbo_var",
    "nelbo_std",
    "nelbo_q25",
    "nelbo_q50",
    "nelbo_q75",
]
PROHIBITED_ADOPTION_MARKERS = {
    "target_adjacent",
    "oracle_diagnostic",
    "nelbo",
    "recon_mean",
    "kl_mean",
}


def _stable_seed(*parts: object) -> int:
    raw = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else 0.0


def _std(xs: Iterable[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))


def _q(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float().flatten(), float(q)).item())


def _domain(rec_or_meta: Any) -> int:
    if isinstance(rec_or_meta, dict):
        return int(rec_or_meta["magnification"])
    return int(getattr(rec_or_meta, "magnification"))


def _label(rec: BreakHisRecord) -> int:
    return int(getattr(rec, "label"))


def _record_key(rec: BreakHisRecord) -> str:
    return f"{rec.split}:{rec.sample_id}:{rec.image_path}"


def _cap_records(
    records: List[BreakHisRecord],
    caps: Dict[str, int],
    seed: int,
    class_balance: bool,
) -> Tuple[List[BreakHisRecord], Dict[str, Any]]:
    rng = random.Random(seed)
    selected: List[BreakHisRecord] = []
    report: Dict[str, Any] = {
        "query_sampling_seed": seed,
        "class_balance": bool(class_balance),
        "caps": caps,
        "actual_counts": {},
    }

    by_split_domain: Dict[Tuple[str, int], List[BreakHisRecord]] = {}
    for rec in records:
        by_split_domain.setdefault((str(rec.split), _domain(rec)), []).append(rec)

    for split in ["train", "val", "test"]:
        cap = int(caps.get(f"{split}_per_domain", 0))
        for domain in sorted({d for s, d in by_split_domain if s == split}):
            group = list(by_split_domain.get((split, domain), []))
            if cap <= 0 or len(group) <= cap:
                chosen = group
            elif class_balance:
                by_label: Dict[int, List[BreakHisRecord]] = {}
                for rec in group:
                    by_label.setdefault(_label(rec), []).append(rec)
                labels = sorted(by_label)
                per_label = max(cap // max(len(labels), 1), 1)
                chosen = []
                leftovers: List[BreakHisRecord] = []
                for lbl in labels:
                    candidates = list(by_label[lbl])
                    rng.shuffle(candidates)
                    chosen.extend(candidates[: min(per_label, len(candidates))])
                    leftovers.extend(candidates[min(per_label, len(candidates)) :])
                if len(chosen) < cap:
                    rng.shuffle(leftovers)
                    chosen.extend(leftovers[: cap - len(chosen)])
                chosen = chosen[:cap]
            else:
                rng.shuffle(group)
                chosen = group[:cap]

            selected.extend(chosen)
            report["actual_counts"][f"{split}:{domain}"] = len(chosen)

    selected = sorted(selected, key=_record_key)
    return selected, report


def _scope_caps(cfg: Dict[str, Any]) -> Tuple[str, Dict[str, int], int, bool]:
    protocol = cfg.get("protocol", {})
    scope = str(protocol.get("dataset_scope", "development"))
    scopes = protocol.get("dataset_scopes", {})
    if scope not in scopes:
        raise ValueError(f"Unknown protocol.dataset_scope={scope!r}; available={sorted(scopes)}")
    raw_caps = scopes[scope]
    caps = {
        "train_per_domain": int(raw_caps.get("train_per_domain", 250)),
        "val_per_domain": int(raw_caps.get("val_per_domain", 100)),
        "test_per_domain": int(raw_caps.get("test_per_domain", 200)),
    }
    sampling_seed = int(protocol.get("query_sampling_seed", cfg.get("seed", 42)))
    class_balance = bool(protocol.get("class_balance", True))
    return scope, caps, sampling_seed, class_balance


class DomainExpertBank:
    def __init__(
        self,
        expert_checkpoints: Dict[str, str],
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.models: Dict[int, CVAEExpert] = {}
        for name, ckpt in expert_checkpoints.items():
            domain = int(str(name).replace("expert_", "").replace("x", ""))
            model = CVAEExpert(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            self.models[domain] = model

    @property
    def domains(self) -> List[int]:
        return sorted(self.models)

    def model(self, domain: int) -> CVAEExpert:
        return self.models[int(domain)]


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _score_response_stream(
    model: CVAEExpert,
    x: torch.Tensor,
    repeats: int,
    seed_parts: Tuple[object, ...],
    device: torch.device,
) -> Dict[str, Any]:
    xb = x.to(device).float().unsqueeze(0)
    recons: List[torch.Tensor] = []
    rec_values: List[float] = []
    kl_values: List[float] = []
    nelbo_values: List[float] = []

    with torch.no_grad():
        mu0, logvar0 = model.encode(xb)
        for repeat_id in range(max(int(repeats), 1)):
            _set_torch_seed(_stable_seed(*seed_parts, repeat_id))
            recon, mu, logvar = model(xb)
            rec, kl = elbo_components(recon, xb, mu, logvar)
            nelbo = rec + kl
            recons.append(recon.detach().cpu().squeeze(0))
            rec_values.append(float(rec.item()))
            kl_values.append(float(kl.item()))
            nelbo_values.append(float(nelbo.item()))

    rec_t = torch.tensor(rec_values, dtype=torch.float32)
    kl_t = torch.tensor(kl_values, dtype=torch.float32)
    nelbo_t = torch.tensor(nelbo_values, dtype=torch.float32)
    recon_t = torch.stack(recons, dim=0) if recons else torch.empty((0, int(x.shape[0])))
    decode_var = recon_t.var(dim=0, unbiased=False) if recon_t.shape[0] > 1 else torch.zeros_like(x.cpu())
    rec_centered_abs = torch.abs(rec_t - rec_t.mean()) if rec_t.numel() else torch.empty((0,))

    mu_cpu = mu0.detach().cpu().squeeze(0)
    logvar_cpu = logvar0.detach().cpu().squeeze(0)
    entropy_proxy = 0.5 * torch.sum(1.0 + math.log(2.0 * math.pi) + logvar_cpu)

    return {
        "posterior_mu_norm": float(torch.linalg.vector_norm(mu_cpu).item()),
        "posterior_mu_mean": float(mu_cpu.mean().item()),
        "posterior_mu_std": float(mu_cpu.std(unbiased=False).item()),
        "posterior_logvar_mean": float(logvar_cpu.mean().item()),
        "posterior_logvar_std": float(logvar_cpu.std(unbiased=False).item()),
        "posterior_entropy_proxy": float(entropy_proxy.item()),
        "decode_repeat_var_mean": float(decode_var.mean().item()),
        "decode_repeat_var_max": float(decode_var.max().item()),
        "decode_repeat_var_q75": _q(decode_var, 0.75),
        "recon_repeat_var": float(rec_t.var(unbiased=False).item()) if rec_t.numel() > 1 else 0.0,
        "recon_repeat_var_q75": _q(rec_centered_abs, 0.75),
        "kl_repeat_var": float(kl_t.var(unbiased=False).item()) if kl_t.numel() > 1 else 0.0,
        "recon_mean": float(rec_t.mean().item()) if rec_t.numel() else 0.0,
        "recon_var": float(rec_t.var(unbiased=False).item()) if rec_t.numel() > 1 else 0.0,
        "kl_mean": float(kl_t.mean().item()) if kl_t.numel() else 0.0,
        "kl_var": float(kl_t.var(unbiased=False).item()) if kl_t.numel() > 1 else 0.0,
        "recon_plus_kl_mean": float((rec_t + kl_t).mean().item()) if rec_t.numel() else 0.0,
        "nelbo_mean": float(nelbo_t.mean().item()) if nelbo_t.numel() else 0.0,
        "nelbo_var": float(nelbo_t.var(unbiased=False).item()) if nelbo_t.numel() > 1 else 0.0,
        "nelbo_std": float(nelbo_t.std(unbiased=False).item()) if nelbo_t.numel() > 1 else 0.0,
        "nelbo_q25": _q(nelbo_t, 0.25),
        "nelbo_q50": _q(nelbo_t, 0.50),
        "nelbo_q75": _q(nelbo_t, 0.75),
        "_nelbo_repeats": nelbo_values,
    }


def _load_payloads(cache_paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    return {split: torch.load(path, map_location="cpu") for split, path in cache_paths.items()}


def _domain_centroids(train_payload: Dict[str, Any], expert_domains: Sequence[int]) -> Dict[int, torch.Tensor]:
    embeddings = train_payload["embeddings"].float()
    metadata = train_payload["metadata"]
    centroids: Dict[int, torch.Tensor] = {}
    global_centroid = embeddings.mean(dim=0) if embeddings.numel() else torch.zeros((embeddings.shape[1],))
    for domain in expert_domains:
        idxs = [i for i, meta in enumerate(metadata) if _domain(meta) == int(domain)]
        centroids[int(domain)] = embeddings[idxs].mean(dim=0) if idxs else global_centroid
    return centroids


def _metadata_features(
    query_domain: int,
    expert_domain: int,
    routing_cfg: Dict[str, Any],
) -> Dict[str, float]:
    similarity = compute_similarity(
        {"magnification": int(query_domain)},
        {"magnification": int(expert_domain)},
        strategy=str(routing_cfg.get("strategy", "categorical_exact")),
        tau=float(routing_cfg.get("tau", 1.0)),
        similarity_matrix=routing_cfg.get("similarity_matrix"),
    )
    return {
        "query_domain_value": float(query_domain),
        "expert_domain_value": float(expert_domain),
        "metadata_exact_match": 1.0 if int(query_domain) == int(expert_domain) else 0.0,
        "metadata_abs_distance": float(abs(int(query_domain) - int(expert_domain))),
        "metadata_similarity": float(similarity),
    }


def _build_pair_rows(
    cfg: Dict[str, Any],
    payloads: Dict[str, Dict[str, Any]],
    bank: DomainExpertBank,
    response_repeats: int,
    target_repeats: int,
    dataset_scope: str,
    sampling_seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dataset = str(cfg["experiment"]["dataset_name"])
    seed = int(cfg["seed"])
    device = bank.device
    feature_protocol = cfg.get("features", {})

    for split in ["train", "val", "test"]:
        payload = payloads[split]
        embeddings = payload["embeddings"].float()
        metadata = payload["metadata"]
        for query_index, meta in enumerate(metadata):
            query_id = f"{split}:{meta.get('sample_id', 'sample')}:{query_index}"
            query_domain = _domain(meta)
            x = embeddings[query_index]
            for expert_domain in bank.domains:
                feature_stats = _score_response_stream(
                    model=bank.model(expert_domain),
                    x=x,
                    repeats=response_repeats,
                    seed_parts=(dataset, seed, query_id, expert_domain, "feature"),
                    device=device,
                )
                target_stats = _score_response_stream(
                    model=bank.model(expert_domain),
                    x=x,
                    repeats=target_repeats,
                    seed_parts=(dataset, seed, query_id, expert_domain, "target"),
                    device=device,
                )
                utility = -float(target_stats["nelbo_mean"])
                utility_var = float(target_stats["nelbo_var"])
                row: Dict[str, Any] = {
                    "dataset": dataset,
                    "seed": seed,
                    "split": split,
                    "query_index": query_index,
                    "query_id": query_id,
                    "query_domain": query_domain,
                    "heldout_domain": query_domain,
                    "expert_domain": expert_domain,
                    "method_key": "",
                    "split_policy": "loqdo_query_domain",
                    "normalization_policy": "within_query_minmax_utility",
                    "target_noise_mode": "deterministic_repeated_target_stream",
                    "feature_family": "",
                    "response_feature_mode": "",
                    "probe_mode": "expert_conditioned_cvae_response",
                    "interaction_mode": "none",
                    "feature_extractor_name": str(feature_protocol.get("feature_extractor_name", "dinov2_vitb14")),
                    "feature_extractor_checkpoint": str(
                        feature_protocol.get("feature_extractor_checkpoint", "facebook/dinov2-base")
                    ),
                    "feature_extractor_layer": str(feature_protocol.get("feature_extractor_layer", "final_norm_cls")),
                    "embedding_pooling": str(feature_protocol.get("embedding_pooling", "cls_token")),
                    "dataset_scope": dataset_scope,
                    "query_sampling_policy": "split_domain_class_stratified_cap",
                    "query_sampling_seed": sampling_seed,
                    "num_queries_per_domain": "",
                    "utility_mean": utility,
                    "utility_var": utility_var,
                    "utility_confidence_weight": 1.0 / max(utility_var, 1e-8),
                    "utility_normalized": 0.0,
                    "oracle_expert": "",
                    "oracle_utility": 0.0,
                    "selected_utility": "",
                    "oracle_gap": "",
                    "normalized_oracle_gap": "",
                }
                for key in RESPONSE_INDIRECT_COLUMNS + TARGET_ADJACENT_COLUMNS:
                    row[key] = float(feature_stats.get(key, 0.0))
                for key in ORACLE_DIAGNOSTIC_COLUMNS:
                    row[key] = float(target_stats.get(key, 0.0))
                row["_target_recon_plus_kl_mean"] = float(target_stats.get("recon_plus_kl_mean", 0.0))
                row["_target_nelbo_repeats"] = target_stats.get("_nelbo_repeats", [])
                rows.append(row)

    _annotate_query_oracles(rows)
    return rows


def _annotate_query_oracles(rows: List[Dict[str, Any]]) -> None:
    by_query: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), []).append(row)

    for group in by_query.values():
        utilities = [float(row["utility_mean"]) for row in group]
        min_u = min(utilities)
        max_u = max(utilities)
        denom = max(max_u - min_u, 1e-12)
        oracle_row = max(group, key=lambda r: float(r["utility_mean"]))
        for row in group:
            row["utility_normalized"] = (float(row["utility_mean"]) - min_u) / denom
            row["oracle_expert"] = int(oracle_row["expert_domain"])
            row["oracle_utility"] = float(oracle_row["utility_mean"])


def _row_csv_projection(row: Dict[str, Any]) -> Dict[str, Any]:
    exclude = {"_target_nelbo_repeats", "_target_recon_plus_kl_mean"}
    return {k: v for k, v in row.items() if k not in exclude}


def _embedding_features(
    row: Dict[str, Any],
    payloads: Dict[str, Dict[str, Any]],
    centroids: Dict[int, torch.Tensor],
) -> torch.Tensor:
    q = payloads[str(row["split"])]["embeddings"][int(row["query_index"])].float()
    c = centroids[int(row["expert_domain"])].float()
    diff = q - c
    abs_diff = torch.abs(diff)
    denom = max(float(torch.linalg.vector_norm(q).item() * torch.linalg.vector_norm(c).item()), 1e-12)
    dot = float(torch.dot(q, c).item())
    cos = dot / denom
    l2 = float(torch.linalg.vector_norm(diff).item())
    return torch.cat([q, c, diff, abs_diff, torch.tensor([cos, l2, dot], dtype=torch.float32)])


def _response_vector(row: Dict[str, Any], columns: Sequence[str]) -> torch.Tensor:
    return torch.tensor([float(row.get(col, 0.0)) for col in columns], dtype=torch.float32)


def _make_shuffled_response_maps(rows: List[Dict[str, Any]], seed: int) -> Dict[int, Dict[str, float]]:
    rng = random.Random(seed)
    by_query: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), []).append(row)

    shuffled: Dict[int, Dict[str, float]] = {}
    fallback_values = [
        {col: float(row.get(col, 0.0)) for col in RESPONSE_INDIRECT_COLUMNS}
        for row in rows
    ]
    for group in by_query.values():
        values = [{col: float(row.get(col, 0.0)) for col in RESPONSE_INDIRECT_COLUMNS} for row in group]
        if len(values) > 1:
            offset = rng.randrange(1, len(values))
            values = values[offset:] + values[:offset]
        elif fallback_values:
            values = [rng.choice(fallback_values)]
        for row, value in zip(group, values):
            shuffled[id(row)] = value
    return shuffled


def _feature_mode_kind(mode: str) -> Tuple[str, bool]:
    adoption_eligible = not any(marker in mode for marker in PROHIBITED_ADOPTION_MARKERS)
    if mode.startswith("static_response"):
        return "static_response", adoption_eligible
    if mode.startswith("response_indirect_shuffled"):
        return "response_control", adoption_eligible
    if mode.startswith("response_indirect"):
        return "response", adoption_eligible
    if mode.startswith("response_target_adjacent"):
        return "target_adjacent_diagnostic", False
    if mode.startswith("response_oracle"):
        return "oracle_diagnostic", False
    return "static", adoption_eligible


def _build_feature_matrix(
    rows: List[Dict[str, Any]],
    mode: str,
    payloads: Dict[str, Dict[str, Any]],
    centroids: Dict[int, torch.Tensor],
    routing_cfg: Dict[str, Any],
    shuffled_map: Dict[int, Dict[str, float]] | None,
) -> torch.Tensor:
    vectors: List[torch.Tensor] = []
    for row in rows:
        parts: List[torch.Tensor] = []
        if mode in {"static_metadata", "static_combined", "static_response_indirect"}:
            meta = _metadata_features(int(row["query_domain"]), int(row["expert_domain"]), routing_cfg)
            parts.append(torch.tensor(list(meta.values()), dtype=torch.float32))
        if mode in {"static_embedding", "static_combined", "static_response_indirect"}:
            parts.append(_embedding_features(row, payloads, centroids))
        if mode in {"response_indirect", "static_response_indirect"}:
            parts.append(_response_vector(row, RESPONSE_INDIRECT_COLUMNS))
        if mode == "response_indirect_shuffled":
            source = shuffled_map.get(id(row), {}) if shuffled_map is not None else {}
            parts.append(torch.tensor([float(source.get(col, 0.0)) for col in RESPONSE_INDIRECT_COLUMNS]))
        if mode == "response_target_adjacent_diagnostic":
            parts.append(_response_vector(row, TARGET_ADJACENT_COLUMNS))
        if mode == "response_oracle_diagnostic":
            parts.append(_response_vector(row, ORACLE_DIAGNOSTIC_COLUMNS))
        if not parts:
            raise ValueError(f"Unsupported feature mode: {mode}")
        vectors.append(torch.cat(parts).float())
    return torch.stack(vectors, dim=0) if vectors else torch.empty((0, 0), dtype=torch.float32)


def _standardize(
    x_train: torch.Tensor,
    others: Sequence[torch.Tensor],
    variance_floor: float,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    if x_train.numel() == 0:
        return x_train, list(others), torch.ones((x_train.shape[1],), dtype=torch.bool)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False)
    keep = std > float(variance_floor)
    if not bool(keep.any()):
        keep = torch.ones_like(std, dtype=torch.bool)
    std = torch.where(std > float(variance_floor), std, torch.ones_like(std))
    x_train_s = (x_train[:, keep] - mean[keep]) / std[keep]
    out = [((x[:, keep] - mean[keep]) / std[keep]) if x.numel() else x[:, keep] for x in others]
    return x_train_s.float(), [x.float() for x in out], keep


def _fit_linear(x_train: torch.Tensor, y_train: torch.Tensor, ridge: float) -> Dict[str, torch.Tensor]:
    x = torch.cat([x_train.double(), torch.ones((x_train.shape[0], 1), dtype=torch.float64)], dim=1)
    y = y_train.double().unsqueeze(1)
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    eye[-1, -1] = 0.0
    lhs = x.T @ x + float(ridge) * eye
    rhs = x.T @ y
    try:
        weights = torch.linalg.solve(lhs, rhs).squeeze(1)
    except RuntimeError:
        weights = torch.linalg.lstsq(lhs, rhs).solution.squeeze(1)
    return {"weights": weights.float()}


def _predict_linear(model: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    weights = model["weights"]
    xb = torch.cat([x.float(), torch.ones((x.shape[0], 1), dtype=torch.float32)], dim=1)
    return xb @ weights


class CompactRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden = min(max(16, input_dim // 2), 128)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _fit_mlp(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    cfg: Dict[str, Any],
    objective: str,
) -> CompactRegressor:
    seed = int(cfg.get("seed", 42))
    _set_torch_seed(_stable_seed(seed, objective, "compat_mlp"))
    learned_cfg = cfg.get("learned_compatibility", {})
    epochs = int(learned_cfg.get("epochs", 120))
    lr = float(learned_cfg.get("learning_rate", 1e-3))
    patience = int(learned_cfg.get("patience", 20))
    batch_size = int(learned_cfg.get("batch_size", 256))
    model = CompactRegressor(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad = 0
    n = x_train.shape[0]

    for _ in range(epochs):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            pred = model(x_train[idx])
            loss = F.smooth_l1_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if x_val.numel():
                val_loss = float(F.mse_loss(model(x_val), y_val).item())
            else:
                val_loss = float(F.mse_loss(model(x_train), y_train).item())
        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _query_groups(rows: List[Dict[str, Any]]) -> List[List[int]]:
    by_query: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        by_query.setdefault(str(row["query_id"]), []).append(idx)
    return list(by_query.values())


def _fit_listwise(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_rows: List[Dict[str, Any]],
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    val_rows: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> CompactRegressor:
    _ = y_val
    seed = int(cfg.get("seed", 42))
    _set_torch_seed(_stable_seed(seed, "listwise_ranker"))
    learned_cfg = cfg.get("learned_compatibility", {})
    epochs = int(learned_cfg.get("listwise_epochs", learned_cfg.get("epochs", 120)))
    lr = float(learned_cfg.get("learning_rate", 1e-3))
    patience = int(learned_cfg.get("patience", 20))
    utility_temperature = float(learned_cfg.get("listwise_target_temperature", 5.0))
    model = CompactRegressor(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_groups = _query_groups(train_rows)
    val_groups = _query_groups(val_rows)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    bad = 0

    for _ in range(epochs):
        model.train()
        random.shuffle(train_groups)
        for group in train_groups:
            idx = torch.tensor(group, dtype=torch.long)
            scores = model(x_train[idx])
            target = torch.softmax(y_train[idx] * utility_temperature, dim=0)
            loss = -(target * F.log_softmax(scores, dim=0)).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            losses = []
            eval_groups = val_groups or train_groups
            eval_x = x_val if val_groups else x_train
            eval_y = y_val if val_groups else y_train
            for group in eval_groups:
                idx = torch.tensor(group, dtype=torch.long)
                scores = model(eval_x[idx])
                target = torch.softmax(eval_y[idx] * utility_temperature, dim=0)
                losses.append(float((-(target * F.log_softmax(scores, dim=0)).sum()).item()))
            val_loss = _mean(losses)
        if val_loss < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _fit_calibration(pred: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    if pred.numel() < 2:
        return 1.0, 0.0
    x = torch.cat([pred.double().unsqueeze(1), torch.ones((pred.numel(), 1), dtype=torch.float64)], dim=1)
    target = y.double().unsqueeze(1)
    try:
        coef = torch.linalg.lstsq(x, target).solution.squeeze(1)
        return float(coef[0].item()), float(coef[1].item())
    except RuntimeError:
        return 1.0, 0.0


def _pairwise_accuracy(preds: List[float], utils: List[float], tie_eps: float = 1e-8) -> float:
    total = 0
    correct = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            diff_u = utils[i] - utils[j]
            if abs(diff_u) <= tie_eps:
                continue
            diff_p = preds[i] - preds[j]
            if diff_p == 0:
                continue
            total += 1
            if diff_u * diff_p > 0:
                correct += 1
    return correct / total if total else 0.0


def _evaluate_predictions(
    rows: List[Dict[str, Any]],
    pred: torch.Tensor,
    pred_calibrated: torch.Tensor,
) -> Dict[str, float]:
    by_query: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        by_query.setdefault(str(row["query_id"]), []).append(idx)

    top1 = []
    spearman = []
    gaps = []
    norm_gaps = []
    ranks = []
    top2 = []
    pairwise = []
    selected_utils = []
    oracle_utils = []

    for group in by_query.values():
        utils = [float(rows[i]["utility_mean"]) for i in group]
        preds = [float(pred[i].item()) for i in group]
        oracle_local = max(range(len(group)), key=lambda j: utils[j])
        selected_local = max(range(len(group)), key=lambda j: preds[j])
        sorted_pred = sorted(range(len(group)), key=lambda j: preds[j], reverse=True)
        rank_oracle = sorted_pred.index(oracle_local) + 1
        oracle_u = utils[oracle_local]
        selected_u = utils[selected_local]
        denom = max(max(utils) - min(utils), 1e-12)

        top1.append(1.0 if oracle_local == selected_local else 0.0)
        spearman.append(spearman_corr(preds, utils))
        gaps.append(oracle_u - selected_u)
        norm_gaps.append((oracle_u - selected_u) / denom)
        ranks.append(float(rank_oracle))
        top2.append(1.0 if oracle_local in sorted_pred[: min(2, len(sorted_pred))] else 0.0)
        pairwise.append(_pairwise_accuracy(preds, utils))
        selected_utils.append(selected_u)
        oracle_utils.append(oracle_u)

    y = torch.tensor([float(row["utility_mean"]) for row in rows], dtype=torch.float32)
    calibration_error = float(torch.mean(torch.abs(pred_calibrated.float() - y)).item()) if y.numel() else 0.0

    return {
        "top1": _mean(top1),
        "spearman": _mean(spearman),
        "oracle_gap": _mean(gaps),
        "normalized_oracle_gap": _mean(norm_gaps),
        "calibration_error": calibration_error,
        "pairwise_accuracy": _mean(pairwise),
        "rank_of_oracle_expert": _mean(ranks),
        "top_k_contains_oracle": _mean(top2),
        "selected_utility": _mean(selected_utils),
        "oracle_utility": _mean(oracle_utils),
        "num_queries": float(len(by_query)),
        "num_pairs": float(len(rows)),
    }


def _train_and_eval_fold(
    cfg: Dict[str, Any],
    all_rows: List[Dict[str, Any]],
    heldout_domain: int,
    mode: str,
    objective: str,
    payloads: Dict[str, Dict[str, Any]],
    centroids: Dict[int, torch.Tensor],
    shuffled_map: Dict[int, Dict[str, float]],
) -> Dict[str, Any] | None:
    train_rows = [
        row for row in all_rows if row["split"] == "train" and int(row["query_domain"]) != int(heldout_domain)
    ]
    val_rows = [row for row in all_rows if row["split"] == "val" and int(row["query_domain"]) != int(heldout_domain)]
    test_rows = [row for row in all_rows if row["split"] == "test" and int(row["query_domain"]) == int(heldout_domain)]
    if not train_rows or not test_rows:
        return None

    routing_cfg = cfg.get("routing", {})
    variance_floor = float(cfg.get("learned_compatibility", {}).get("variance_floor", 1e-10))
    x_train = _build_feature_matrix(train_rows, mode, payloads, centroids, routing_cfg, shuffled_map)
    x_val = _build_feature_matrix(val_rows, mode, payloads, centroids, routing_cfg, shuffled_map) if val_rows else torch.empty((0, x_train.shape[1]))
    x_test = _build_feature_matrix(test_rows, mode, payloads, centroids, routing_cfg, shuffled_map)
    y_train = torch.tensor([float(row["utility_mean"]) for row in train_rows], dtype=torch.float32)
    y_val = torch.tensor([float(row["utility_mean"]) for row in val_rows], dtype=torch.float32)
    y_test = torch.tensor([float(row["utility_mean"]) for row in test_rows], dtype=torch.float32)

    x_train, (x_val, x_test), keep = _standardize(x_train, [x_val, x_test], variance_floor=variance_floor)

    if objective == "linear_regression":
        model = _fit_linear(x_train, y_train, ridge=float(cfg.get("learned_compatibility", {}).get("ridge", 1e-4)))
        pred_train = _predict_linear(model, x_train)
        pred_val = _predict_linear(model, x_val) if x_val.numel() else torch.empty((0,))
        pred_test = _predict_linear(model, x_test)
    elif objective == "compact_mlp":
        model_mlp = _fit_mlp(x_train, y_train, x_val, y_val, cfg, objective=f"{objective}:{mode}:{heldout_domain}")
        with torch.no_grad():
            pred_train = model_mlp(x_train)
            pred_val = model_mlp(x_val) if x_val.numel() else torch.empty((0,))
            pred_test = model_mlp(x_test)
    elif objective == "listwise_ranker":
        model_rank = _fit_listwise(x_train, y_train, train_rows, x_val, y_val, val_rows, cfg)
        with torch.no_grad():
            pred_train = model_rank(x_train)
            pred_val = model_rank(x_val) if x_val.numel() else torch.empty((0,))
            pred_test = model_rank(x_test)
    else:
        raise ValueError(f"Unsupported learned objective: {objective}")

    cal_source_pred = pred_val if pred_val.numel() else pred_train
    cal_source_y = y_val if y_val.numel() else y_train
    cal_a, cal_b = _fit_calibration(cal_source_pred, cal_source_y)
    pred_test_cal = pred_test * cal_a + cal_b
    metrics = _evaluate_predictions(test_rows, pred_test, pred_test_cal)
    feature_family, adoption_eligible = _feature_mode_kind(mode)
    method_key = f"{objective}__{mode}"

    return {
        "dataset": str(cfg["experiment"]["dataset_name"]),
        "seed": int(cfg["seed"]),
        "heldout_domain": int(heldout_domain),
        "method_key": method_key,
        "objective": objective,
        "feature_family": feature_family,
        "feature_mode": mode,
        "response_feature_mode": mode if mode.startswith("response") or "response" in mode else "none",
        "adoption_eligible": adoption_eligible,
        "n_features_raw": int(keep.numel()),
        "n_features_retained": int(keep.sum().item()),
        **metrics,
    }


def _run_learned_compatibility(
    cfg: Dict[str, Any],
    rows: List[Dict[str, Any]],
    payloads: Dict[str, Dict[str, Any]],
    centroids: Dict[int, torch.Tensor],
) -> List[Dict[str, Any]]:
    learned_cfg = cfg.get("learned_compatibility", {})
    feature_modes = [str(x) for x in learned_cfg.get("feature_modes", ["static_combined", "response_indirect", "static_response_indirect", "response_indirect_shuffled"])]
    objectives = [str(x) for x in learned_cfg.get("objectives", ["linear_regression", "compact_mlp", "listwise_ranker"])]
    heldout_domains = sorted({int(row["query_domain"]) for row in rows})
    shuffled_map = _make_shuffled_response_maps(rows, seed=_stable_seed(cfg.get("seed", 42), "response_indirect_shuffled"))
    decision_rows: List[Dict[str, Any]] = []

    for heldout_domain in heldout_domains:
        for mode in feature_modes:
            for objective in objectives:
                result = _train_and_eval_fold(
                    cfg=cfg,
                    all_rows=rows,
                    heldout_domain=heldout_domain,
                    mode=mode,
                    objective=objective,
                    payloads=payloads,
                    centroids=centroids,
                    shuffled_map=shuffled_map,
                )
                if result is not None:
                    decision_rows.append(result)
    return decision_rows


def _target_adjacency_audit(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    utility = [float(row["utility_mean"]) for row in rows]
    columns = RESPONSE_INDIRECT_COLUMNS + TARGET_ADJACENT_COLUMNS + ORACLE_DIAGNOSTIC_COLUMNS
    audit_rows = []
    for col in columns:
        vals = [float(row.get(col, 0.0)) for row in rows]
        corr = pearson_corr(vals, utility)
        audit_rows.append(
            {
                "feature": col,
                "corr_with_utility": corr,
                "abs_corr_with_utility": abs(corr),
                "near_perfect_target_adjacency": abs(corr) >= 0.98,
                "adoption_allowed": col in RESPONSE_INDIRECT_COLUMNS and abs(corr) < 0.98,
            }
        )

    algebraic_errors = [
        abs(float(row["nelbo_mean"]) - float(row.get("_target_recon_plus_kl_mean", 0.0)))
        for row in rows
        if math.isfinite(float(row.get("nelbo_mean", 0.0)))
    ]
    leakage = {
        "nelbo_equals_recon_plus_kl_mean_abs_error": _mean(algebraic_errors),
        "nelbo_equals_recon_plus_kl_max_abs_error": max(algebraic_errors) if algebraic_errors else 0.0,
        "algebraic_components_target_adjacent": (_mean(algebraic_errors) < 1e-4 if algebraic_errors else False),
    }
    return audit_rows, leakage


def _budget_reliability(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_domain: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_domain.setdefault(int(row["query_domain"]), []).append(row)

    out = []
    for domain, domain_rows in sorted(by_domain.items()):
        first_half = []
        second_half = []
        for row in domain_rows:
            repeats = [float(v) for v in row.get("_target_nelbo_repeats", [])]
            if len(repeats) >= 2:
                mid = len(repeats) // 2
                first_half.append(-_mean(repeats[:mid]))
                second_half.append(-_mean(repeats[mid:]))
        out.append(
            {
                "dataset": domain_rows[0]["dataset"] if domain_rows else "",
                "seed": domain_rows[0]["seed"] if domain_rows else "",
                "query_domain": domain,
                "repeat_consistency_spearman": spearman_corr(first_half, second_half),
                "response_feature_variance": _mean(
                    [
                        _std([float(row.get(col, 0.0)) for row in domain_rows])
                        for col in RESPONSE_INDIRECT_COLUMNS
                    ]
                ),
                "num_pairs": len(domain_rows),
            }
        )
    return out


def _aggregate_by_method(decision_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in decision_rows:
        by_method.setdefault(str(row["method_key"]), []).append(row)

    out = []
    for method, rows in sorted(by_method.items()):
        out.append(
            {
                "method_key": method,
                "feature_mode": rows[0]["feature_mode"],
                "feature_family": rows[0]["feature_family"],
                "adoption_eligible": bool(rows[0]["adoption_eligible"]),
                "top1": _mean(row["top1"] for row in rows),
                "spearman": _mean(row["spearman"] for row in rows),
                "oracle_gap": _mean(row["oracle_gap"] for row in rows),
                "normalized_oracle_gap": _mean(row["normalized_oracle_gap"] for row in rows),
                "calibration_error": _mean(row["calibration_error"] for row in rows),
                "domain_group_std_normalized_oracle_gap": _std(row["normalized_oracle_gap"] for row in rows),
                "num_folds": len(rows),
            }
        )
    return out


def _decision_summary(decision_rows: List[Dict[str, Any]], leakage: Dict[str, Any]) -> Dict[str, Any]:
    aggregates = _aggregate_by_method(decision_rows)
    eligible = [row for row in aggregates if row["adoption_eligible"]]
    best = None
    if eligible:
        best = sorted(
            eligible,
            key=lambda row: (
                -float(row["top1"]),
                float(row["normalized_oracle_gap"]),
                -float(row["spearman"]),
            ),
        )[0]

    by_mode_best: Dict[str, Dict[str, Any]] = {}
    for row in eligible:
        mode = str(row["feature_mode"])
        current = by_mode_best.get(mode)
        if current is None or (
            float(row["top1"]),
            -float(row["normalized_oracle_gap"]),
            float(row["spearman"]),
        ) > (
            float(current["top1"]),
            -float(current["normalized_oracle_gap"]),
            float(current["spearman"]),
        ):
            by_mode_best[mode] = row

    static = by_mode_best.get("static_combined")
    response = by_mode_best.get("response_indirect")
    static_response = by_mode_best.get("static_response_indirect")
    shuffled = by_mode_best.get("response_indirect_shuffled")

    def beats(a: Dict[str, Any] | None, b: Dict[str, Any] | None) -> bool:
        if a is None or b is None:
            return False
        return (
            float(a["top1"]) >= float(b["top1"])
            and float(a["spearman"]) >= float(b["spearman"])
            and float(a["normalized_oracle_gap"]) < float(b["normalized_oracle_gap"])
        )

    return {
        "best_adoption_eligible_method": best,
        "static_response_indirect_beats_static_combined": beats(static_response, static),
        "response_indirect_beats_shuffled_control": beats(response, shuffled),
        "target_algebraic_leakage": leakage,
        "adoption_guardrails": {
            "excluded_markers": sorted(PROHIBITED_ADOPTION_MARKERS),
            "diagnostic_modes_excluded_from_claims": True,
        },
        "method_aggregates": aggregates,
    }


def _failure_mode_summary(summary: Dict[str, Any]) -> str:
    sr_beats_static = bool(summary.get("static_response_indirect_beats_static_combined", False))
    response_beats_shuffled = bool(summary.get("response_indirect_beats_shuffled_control", False))
    if sr_beats_static and response_beats_shuffled:
        verdict = "response_signal_recoverable"
        explanation = "Indirect expert response statistics improve routing and beat the shuffled response control."
    elif sr_beats_static:
        verdict = "response_signal_ambiguous"
        explanation = "Static+response improves over static, but aligned response-only does not clearly beat shuffled control."
    else:
        verdict = "response_signal_not_recoverable"
        explanation = "Indirect response features do not improve over the DINOv2 static baseline under the adoption gate."

    return (
        "# Failure Mode Summary\n\n"
        f"- verdict: {verdict}\n"
        f"- interpretation: {explanation}\n"
        "- diagnostic modes containing NELBO, reconstruction means, or KL means are excluded from adoption claims.\n"
    )


def _write_artifacts(
    reports_dir: Path,
    rows: List[Dict[str, Any]],
    decision_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
    leakage: Dict[str, Any],
) -> None:
    response_fields = [
        "dataset",
        "seed",
        "split",
        "query_id",
        "query_domain",
        "heldout_domain",
        "expert_domain",
        "utility_mean",
        "utility_var",
        "utility_confidence_weight",
        "utility_normalized",
        "oracle_expert",
        "oracle_utility",
        *RESPONSE_INDIRECT_COLUMNS,
        *TARGET_ADJACENT_COLUMNS,
        *ORACLE_DIAGNOSTIC_COLUMNS,
        "feature_extractor_name",
        "feature_extractor_checkpoint",
        "feature_extractor_layer",
        "embedding_pooling",
        "dataset_scope",
        "query_sampling_policy",
        "query_sampling_seed",
    ]
    _write_csv(reports_dir / "loqdo_response_feature_table.csv", [_row_csv_projection(r) for r in rows], response_fields)
    _write_csv(reports_dir / "target_adjacency_audit.csv", audit_rows)
    _write_csv(reports_dir / "response_budget_reliability_table.csv", _budget_reliability(rows))

    _write_csv(
        reports_dir / "baseline_static_dinov2_reproduction.csv",
        [r for r in decision_rows if r["feature_mode"] in STATIC_MODES],
    )
    _write_csv(
        reports_dir / "response_indirect_decision_table.csv",
        [r for r in decision_rows if r["feature_mode"] in {"response_indirect", "static_response_indirect"}],
    )
    _write_csv(
        reports_dir / "response_indirect_shuffled_control_table.csv",
        [r for r in decision_rows if r["feature_mode"] == "response_indirect_shuffled"],
    )
    _write_csv(
        reports_dir / "response_target_adjacent_diagnostic_table.csv",
        [r for r in decision_rows if r["feature_mode"] == "response_target_adjacent_diagnostic"],
    )
    _write_csv(
        reports_dir / "response_oracle_diagnostic_table.csv",
        [r for r in decision_rows if r["feature_mode"] == "response_oracle_diagnostic"],
    )

    summary = _decision_summary(decision_rows, leakage)
    with (reports_dir / "decision_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    (reports_dir / "failure_mode_summary.md").write_text(_failure_mode_summary(summary), encoding="utf-8")

    cross_dataset_row = {
        "dataset": rows[0]["dataset"] if rows else "",
        "seed": rows[0]["seed"] if rows else "",
        "response_signal_recoverable": bool(summary.get("static_response_indirect_beats_static_combined", False))
        and bool(summary.get("response_indirect_beats_shuffled_control", False)),
        "response_signal_not_recoverable": not bool(summary.get("static_response_indirect_beats_static_combined", False)),
        "response_signal_dataset_specific": "requires_multi_dataset_aggregation",
    }
    _write_csv(reports_dir / "cross_dataset_response_assessment.csv", [cross_dataset_row])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned compatibility LOQDO response-statistics experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dataset-scope", type=str, default=None)
    parser.add_argument("--skip-training", action="store_true", help="Reuse existing checkpoints in the run directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    cfg = load_config(config_path)
    if args.dataset_scope:
        cfg.setdefault("protocol", {})["dataset_scope"] = args.dataset_scope

    set_global_determinism(seed=int(cfg["seed"]))
    run_ctx = build_run_context(PROJECT_ROOT, cfg, run_id_override=args.run_id)
    write_run_metadata(cfg, run_ctx)

    records, leakage_report = prepare_dataset_records(PROJECT_ROOT, cfg)
    dataset_scope, caps, sampling_seed, class_balance = _scope_caps(cfg)
    capped_records, sampling_report = _cap_records(records, caps=caps, seed=sampling_seed, class_balance=class_balance)
    write_manifest(capped_records, run_ctx.manifests_dir / "samples.csv")
    write_split_manifest(capped_records, run_ctx.reports_dir / "split_manifest.json")
    with (run_ctx.reports_dir / "leakage_report.json").open("w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)
    with (run_ctx.reports_dir / "query_sampling_report.json").open("w", encoding="utf-8") as f:
        json.dump(sampling_report, f, indent=2)

    cache_paths = extract_and_cache_embeddings(
        records=capped_records,
        cache_dir=run_ctx.embeddings_dir,
        image_size=int(cfg["features"]["image_size"]),
        batch_size=int(cfg["training"]["batch_size"]),
        feature_config=cfg.get("features", {}),
    )
    cache_report = validate_embedding_cache(cache_paths, expected_dim=int(cfg["features"]["embedding_dim"]))
    with (run_ctx.reports_dir / "cache_report.json").open("w", encoding="utf-8") as f:
        json.dump(cache_report, f, indent=2)

    checkpoint_json = run_ctx.checkpoints_dir / "expert_checkpoints.json"
    if args.skip_training and checkpoint_json.exists():
        expert_checkpoints = json.loads(checkpoint_json.read_text(encoding="utf-8"))
    else:
        expert_checkpoints = train_domain_experts(
            train_cache=cache_paths["train"],
            val_cache=cache_paths["val"],
            out_dir=run_ctx.checkpoints_dir,
            domains=[int(d) for d in cfg["data"]["magnifications"]],
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            lr=float(cfg["training"]["learning_rate"]),
            epochs=int(cfg["training"]["epochs"]),
            patience=int(cfg["training"]["patience"]),
            batch_size=int(cfg["training"]["batch_size"]),
            resume_from_dir=None,
        )

    payloads = _load_payloads(cache_paths)
    input_dim = int(payloads["train"]["embeddings"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bank = DomainExpertBank(
        expert_checkpoints=expert_checkpoints,
        input_dim=input_dim,
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        latent_dim=int(cfg["model"]["latent_dim"]),
        device=device,
    )
    centroids = _domain_centroids(payloads["train"], expert_domains=bank.domains)

    response_cfg = cfg.get("response_features", {})
    repeat_budgets = response_cfg.get("repeat_budgets", {"medium": 10})
    budget_name = str(response_cfg.get("repeat_budget", "medium"))
    target_budget_name = str(response_cfg.get("target_repeat_budget", budget_name))
    response_repeats = int(repeat_budgets.get(budget_name, budget_name if str(budget_name).isdigit() else 10))
    target_repeats = int(repeat_budgets.get(target_budget_name, target_budget_name if str(target_budget_name).isdigit() else response_repeats))

    rows = _build_pair_rows(
        cfg=cfg,
        payloads=payloads,
        bank=bank,
        response_repeats=response_repeats,
        target_repeats=target_repeats,
        dataset_scope=dataset_scope,
        sampling_seed=sampling_seed,
    )
    audit_rows, leakage = _target_adjacency_audit(rows)
    decision_rows = _run_learned_compatibility(cfg, rows, payloads, centroids)
    _write_artifacts(run_ctx.reports_dir, rows, decision_rows, audit_rows, leakage)

    print("Learned compatibility LOQDO experiment complete.")
    print("Run directory:", run_ctx.run_root)
    print("Reports:", run_ctx.reports_dir)


if __name__ == "__main__":
    main()
