from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from src.eval.metrics import spearman_corr
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.routing.strategies import ordinal_magnification_similarity, softmax
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import load_model_checkpoint


EPS = 1e-12


@dataclass(frozen=True)
class SupportSetRunMeta:
    dataset_name: str
    seed: int
    backbone_type: str
    run_id: str
    variant: str
    run_dir: str = ""


@dataclass(frozen=True)
class SplitResult:
    target_domain: int
    support_size_requested: int
    support_size_actual: int
    eval_size: int
    support_indices: List[int]
    eval_indices: List[int]
    sampling_policy_requested: str
    sampling_policy_effective: str
    support_labels_used: int
    split_status: str
    no_data_reason: str
    support_eval_split_id: str


@dataclass(frozen=True)
class ResponseProxyBaseline:
    selected_expert: int
    oracle_expert: int
    selected_eval_nelbo: float
    oracle_eval_nelbo: float
    worst_eval_nelbo: float
    normalized_oracle_gap: float
    oracle_gap: float
    top1_oracle_hit: float
    spearman_support_vs_eval_utility: float
    adoption_eligible: int
    baseline_available: int = 1
    source_method: str = ""


class DirectCVAEExpertBank:
    """Minimal expert-bank adapter for direct per-domain CVAE checkpoints."""

    def __init__(
        self,
        *,
        expert_checkpoints: Mapping[int, Path],
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        metadata_constraint_cfg: Mapping[str, object] | None,
        device: torch.device,
    ) -> None:
        if not expert_checkpoints:
            raise RuntimeError("No expert checkpoints were provided.")
        if bool((metadata_constraint_cfg or {}).get("enabled", False)):
            raise RuntimeError(
                "Direct CVAE expert-bank scoring does not support metadata-constraint checkpoints. "
                "Use a hybrid checkpoint for this run."
            )

        self.domains = sorted(int(d) for d in expert_checkpoints)
        self.device = device
        self.cvaes: Dict[int, CVAEExpert] = {}

        for domain, checkpoint in sorted(expert_checkpoints.items()):
            loaded = load_model_checkpoint(Path(checkpoint), map_location=device)
            model = CVAEExpert(
                int(input_dim),
                int(hidden_dim),
                int(latent_dim),
                metadata_constraint_cfg=dict(metadata_constraint_cfg or {}),
            ).to(device)
            model.load_state_dict(loaded.model_state_dict)
            model.eval()
            self.cvaes[int(domain)] = model

    def domain_cvae(self, domain: int) -> CVAEExpert:
        return self.cvaes[int(domain)]

    def project(self, domain: int, x: torch.Tensor) -> torch.Tensor:
        _ = domain
        return x

    def score_domain_nelbo(self, expert_domain: int, x: torch.Tensor) -> torch.Tensor:
        cvae = self.domain_cvae(int(expert_domain))
        recon, mu, logvar = cvae(x)
        rec, kl = elbo_components(recon, x, mu, logvar)
        return rec + kl


def parse_expert_domain(raw: object) -> int:
    text = str(raw).strip().lower()
    if text.startswith("expert_"):
        text = text[len("expert_") :]
    match = re.match(r"^(\d+)", text.replace("x", ""))
    if match is None:
        raise ValueError(f"Cannot parse expert domain from {raw!r}")
    return int(match.group(1))


def load_expert_manifest(path: Path) -> Dict[int, Path]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expert checkpoint manifest must be a dictionary: {path}")

    out: Dict[int, Path] = {}
    for key, value in payload.items():
        domain = parse_expert_domain(key)
        ckpt = Path(str(value))
        if not ckpt.is_absolute():
            ckpt = path.parent / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"Expert checkpoint for domain {domain} not found: {ckpt}")
        out[int(domain)] = ckpt
    if not out:
        raise RuntimeError(f"Expert checkpoint manifest contains no usable checkpoints: {path}")
    return out


def make_expert_bank(
    *,
    variant_checkpoint: Path | None,
    expert_manifest: Path | None,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    metadata_constraint_cfg: Mapping[str, object] | None,
    device: torch.device,
):
    if variant_checkpoint is not None and Path(variant_checkpoint).exists():
        from src.eval.evaluators.hybrid import HybridExpertBank

        return HybridExpertBank(Path(variant_checkpoint), device=device)
    if expert_manifest is None:
        raise RuntimeError("Expected either a hybrid variant checkpoint or expert_checkpoints.json.")
    return DirectCVAEExpertBank(
        expert_checkpoints=load_expert_manifest(Path(expert_manifest)),
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        metadata_constraint_cfg=metadata_constraint_cfg or {},
        device=device,
    )


def score_expert_nelbo_matrix(
    *,
    bank: Any,
    embeddings: torch.Tensor,
    expert_domains: Sequence[int],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n_samples = int(embeddings.shape[0])
    n_experts = len(expert_domains)
    out = np.zeros((n_samples, n_experts), dtype=np.float64)

    with torch.no_grad():
        for e_idx, expert_domain in enumerate(expert_domains):
            chunks: List[torch.Tensor] = []
            for start in range(0, n_samples, int(batch_size)):
                xb = embeddings[start : start + int(batch_size)].to(device)
                chunks.append(bank.score_domain_nelbo(int(expert_domain), xb).cpu())
            if chunks:
                out[:, e_idx] = torch.cat(chunks, dim=0).numpy().astype(np.float64, copy=False)
    return out


def _as_domain(value: object) -> int:
    return int(str(value).replace("x", ""))


def _as_label(meta: Mapping[str, object]) -> int:
    return int(meta.get("label", 0))


def _json_mapping(expert_domains: Sequence[int], values: Sequence[float]) -> str:
    payload = {str(int(e)): float(v) for e, v in zip(expert_domains, values)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _rank_mapping(expert_domains: Sequence[int], values: Sequence[float], *, lower_is_better: bool) -> str:
    direction = 1.0 if lower_is_better else -1.0
    order = sorted(range(len(values)), key=lambda i: (direction * float(values[i]), int(expert_domains[i])))
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = int(rank)
    return json.dumps({str(int(e)): int(r) for e, r in zip(expert_domains, ranks)}, sort_keys=True, separators=(",", ":"))


def _normalize_vector(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / span


def calibration_mae_from_vectors(support_mean_nelbo: Sequence[float], eval_mean_nelbo: Sequence[float]) -> float:
    support_compat = -np.asarray(support_mean_nelbo, dtype=np.float64)
    eval_utility = -np.asarray(eval_mean_nelbo, dtype=np.float64)
    support_norm = _normalize_vector(support_compat)
    eval_norm = _normalize_vector(eval_utility)
    if support_norm.size == 0:
        return 0.0
    return float(np.mean(np.abs(support_norm - eval_norm)))


def global_calibration_error_bin10(rows: Sequence[Mapping[str, object]], *, method: str | None = None) -> float:
    pred: List[float] = []
    true: List[float] = []
    seen_keys: set[Tuple[str, str, str, str, str, str, str, str]] = set()
    for row in rows:
        if str(row.get("split_status", "")) != "ok":
            continue
        if method is not None and str(row.get("method", "")) != str(method):
            continue
        dedupe_key = (
            str(row.get("dataset_name", "")),
            str(row.get("run_id", "")),
            str(row.get("target_domain", "")),
            str(row.get("support_seed", "")),
            str(row.get("support_size_requested", "")),
            str(row.get("sampling_policy", "")),
            str(row.get("sampling_policy_effective", "")),
            str(row.get("support_eval_split_id", "")),
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        try:
            support_map = json.loads(str(row.get("support_nelbo_by_expert_json", "{}")))
            eval_map = json.loads(str(row.get("eval_nelbo_by_expert_json", "{}")))
        except Exception:
            continue
        experts = sorted(set(support_map).intersection(eval_map), key=lambda x: int(x))
        if not experts:
            continue
        support_norm = _normalize_vector([-float(support_map[e]) for e in experts])
        eval_norm = _normalize_vector([-float(eval_map[e]) for e in experts])
        pred.extend(float(v) for v in support_norm.tolist())
        true.extend(float(v) for v in eval_norm.tolist())

    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    if pred_arr.size == 0:
        return 0.0

    order = np.argsort(pred_arr)
    pred_sorted = pred_arr[order]
    true_sorted = true_arr[order]
    n_bins = int(min(10, max(1, pred_sorted.size)))
    edges = np.linspace(0, pred_sorted.size, n_bins + 1, dtype=int)
    gaps: List[float] = []
    for i in range(n_bins):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            continue
        gaps.append(abs(float(np.mean(pred_sorted[start:end])) - float(np.mean(true_sorted[start:end]))))
    return float(np.mean(gaps)) if gaps else 0.0


def normalized_oracle_gap(selected_eval_nelbo: float, oracle_eval_nelbo: float, worst_eval_nelbo: float) -> float:
    denom = max(float(worst_eval_nelbo) - float(oracle_eval_nelbo), EPS)
    return float(max(float(selected_eval_nelbo) - float(oracle_eval_nelbo), 0.0) / denom)


def _stable_argmax(values: Sequence[float], expert_domains: Sequence[int]) -> int:
    return int(max(range(len(values)), key=lambda i: (float(values[i]), -int(expert_domains[i]))))


def _stable_argmin(values: Sequence[float], expert_domains: Sequence[int]) -> int:
    return int(min(range(len(values)), key=lambda i: (float(values[i]), int(expert_domains[i]))))


def _selection_margin(scores: Sequence[float]) -> float:
    arr = np.sort(np.asarray(scores, dtype=np.float64))
    if arr.size < 2:
        return 0.0
    return float(arr[-1] - arr[-2])


def _random_order(indices: Sequence[int], seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    arr = np.asarray(sorted(int(i) for i in indices), dtype=np.int64)
    return [int(i) for i in rng.permutation(arr).tolist()]


def _balanced_order(indices: Sequence[int], labels_by_index: Mapping[int, int], seed: int) -> List[int]:
    rng = np.random.default_rng(int(seed))
    by_label: Dict[int, List[int]] = {}
    for idx in sorted(int(i) for i in indices):
        by_label.setdefault(int(labels_by_index[int(idx)]), []).append(int(idx))

    shuffled: Dict[int, List[int]] = {}
    for label, vals in by_label.items():
        arr = np.asarray(vals, dtype=np.int64)
        shuffled[int(label)] = [int(i) for i in rng.permutation(arr).tolist()]

    labels = sorted(shuffled)
    positions = {label: 0 for label in labels}
    order: List[int] = []
    while True:
        added = False
        for label in labels:
            pos = positions[label]
            if pos < len(shuffled[label]):
                order.append(int(shuffled[label][pos]))
                positions[label] = pos + 1
                added = True
        if not added:
            break
    return order


def _balanced_possible(indices: Sequence[int], labels_by_index: Mapping[int, int], support_size: int) -> bool:
    labels = sorted({int(labels_by_index[int(i)]) for i in indices})
    if len(labels) < 2:
        return False
    available = {label: 0 for label in labels}
    for idx in indices:
        available[int(labels_by_index[int(idx)])] += 1
    needed = {label: 0 for label in labels}
    for pos in range(int(support_size)):
        needed[labels[pos % len(labels)]] += 1
    return all(available[label] >= needed[label] for label in labels)


def make_support_eval_split(
    *,
    target_domain: int,
    target_indices: Sequence[int],
    labels_by_index: Mapping[int, int],
    support_size: int,
    sampling_policy: str,
    support_seed: int,
) -> SplitResult:
    indices = sorted(int(i) for i in target_indices)
    requested = int(support_size)
    policy = str(sampling_policy).strip().lower()
    if len(indices) < requested + 1:
        split_id = f"target{int(target_domain)}_seed{int(support_seed)}_{policy}_k{requested}_skipped"
        return SplitResult(
            target_domain=int(target_domain),
            support_size_requested=requested,
            support_size_actual=0,
            eval_size=0,
            support_indices=[],
            eval_indices=[],
            sampling_policy_requested=policy,
            sampling_policy_effective=policy,
            support_labels_used=1 if policy == "class_balanced" else 0,
            split_status="skipped_insufficient_samples",
            no_data_reason="fewer_than_k_plus_one_target_samples",
            support_eval_split_id=split_id,
        )

    split_seed = int(support_seed) + int(target_domain) * 1009
    effective = policy
    support_labels_used = 0
    if policy == "class_balanced":
        if _balanced_possible(indices, labels_by_index, requested):
            order = _balanced_order(indices, labels_by_index, split_seed)
            support_labels_used = 1
        else:
            order = _random_order(indices, split_seed)
            effective = "random_fallback"
    elif policy == "random":
        order = _random_order(indices, split_seed)
    else:
        raise ValueError(f"Unknown sampling policy: {sampling_policy}")

    support = [int(i) for i in order[:requested]]
    support_set = set(support)
    evaluate = [int(i) for i in indices if int(i) not in support_set]
    if support_set.intersection(evaluate):
        raise RuntimeError("Support/eval overlap detected")

    split_id = f"target{int(target_domain)}_seed{int(support_seed)}_{effective}_k{requested}"
    return SplitResult(
        target_domain=int(target_domain),
        support_size_requested=requested,
        support_size_actual=len(support),
        eval_size=len(evaluate),
        support_indices=support,
        eval_indices=evaluate,
        sampling_policy_requested=policy,
        sampling_policy_effective=effective,
        support_labels_used=support_labels_used,
        split_status="ok",
        no_data_reason="",
        support_eval_split_id=split_id,
    )


def _domain_centroids(embeddings: np.ndarray, sample_domains: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for domain in sorted(set(int(v) for v in sample_domains.tolist())):
        idxs = np.where(sample_domains == int(domain))[0]
        if idxs.size:
            out[int(domain)] = embeddings[idxs].mean(axis=0)
    return out


def _base_row(
    *,
    meta: SupportSetRunMeta,
    split: SplitResult,
    candidate_experts: Sequence[int],
    target_domain: int,
    support_nelbo: np.ndarray,
    eval_nelbo: np.ndarray,
    oracle_idx: int,
    worst_idx: int,
) -> Dict[str, Any]:
    support_json = _json_mapping(candidate_experts, support_nelbo)
    eval_json = _json_mapping(candidate_experts, eval_nelbo)
    return {
        "dataset_name": meta.dataset_name,
        "seed": int(meta.seed),
        "backbone_type": meta.backbone_type,
        "run_id": meta.run_id,
        "variant": meta.variant,
        "run_dir": meta.run_dir,
        "target_domain": int(target_domain),
        "support_seed": int(split.support_eval_split_id.split("_seed", 1)[1].split("_", 1)[0]),
        "support_size_requested": int(split.support_size_requested),
        "support_size_actual": int(split.support_size_actual),
        "eval_size": int(split.eval_size),
        "sampling_policy": split.sampling_policy_requested,
        "sampling_policy_effective": split.sampling_policy_effective,
        "split_status": split.split_status,
        "no_data_reason": split.no_data_reason,
        "candidate_experts": "|".join(str(int(e)) for e in candidate_experts),
        "excluded_experts": str(int(target_domain)),
        "target_expert_excluded": 1,
        "support_eval_disjoint": int(set(split.support_indices).isdisjoint(set(split.eval_indices))),
        "support_labels_used": int(split.support_labels_used),
        "support_is_target_local": 1,
        "support_eval_split_id": split.support_eval_split_id,
        "oracle_expert": int(candidate_experts[int(oracle_idx)]),
        "oracle_eval_nelbo": float(eval_nelbo[int(oracle_idx)]),
        "worst_eval_nelbo": float(eval_nelbo[int(worst_idx)]),
        "support_nelbo_by_expert_json": support_json,
        "eval_nelbo_by_expert_json": eval_json,
        "support_rank_by_expert_json": _rank_mapping(candidate_experts, support_nelbo, lower_is_better=True),
        "eval_rank_by_expert_json": _rank_mapping(candidate_experts, eval_nelbo, lower_is_better=True),
        "calibration_mae": calibration_mae_from_vectors(support_nelbo, eval_nelbo),
        "spearman_support_vs_eval_utility": float(spearman_corr((-support_nelbo).tolist(), (-eval_nelbo).tolist())),
    }


def _method_row(
    *,
    base: Mapping[str, Any],
    method: str,
    scores: Sequence[float],
    candidate_experts: Sequence[int],
    support_nelbo: np.ndarray,
    eval_nelbo: np.ndarray,
    selected_idx: int,
    adoption_eligible: int,
    diagnostic_only: int,
    exploratory_only: int,
    routing_uses_eval_nelbo: int = 0,
    routing_uses_eval_indices: int = 0,
    baseline_available: int = 1,
    source_method: str = "",
) -> Dict[str, Any]:
    oracle_eval = float(base["oracle_eval_nelbo"])
    worst_eval = float(base["worst_eval_nelbo"])
    selected_eval = float(eval_nelbo[int(selected_idx)])
    oracle_expert = int(base["oracle_expert"])
    selected_expert = int(candidate_experts[int(selected_idx)])
    return {
        **dict(base),
        "method": str(method),
        "adoption_eligible": int(adoption_eligible),
        "diagnostic_only": int(diagnostic_only),
        "exploratory_only": int(exploratory_only),
        "baseline_available": int(baseline_available),
        "source_method": str(source_method),
        "routing_uses_eval_nelbo": int(routing_uses_eval_nelbo),
        "routing_uses_eval_indices": int(routing_uses_eval_indices),
        "selected_expert": selected_expert,
        "support_mean_nelbo": float(support_nelbo[int(selected_idx)]),
        "selected_eval_nelbo": selected_eval,
        "oracle_gap": float(selected_eval - oracle_eval),
        "normalized_oracle_gap": normalized_oracle_gap(selected_eval, oracle_eval, worst_eval),
        "top1_oracle_hit": 1.0 if selected_expert == oracle_expert else 0.0,
        "selection_margin": _selection_margin(scores),
    }


def _per_query_oracle_row(
    *,
    base: Mapping[str, Any],
    candidate_experts: Sequence[int],
    support_nelbo: np.ndarray,
    eval_sample_nelbo: np.ndarray,
) -> Dict[str, Any]:
    per_query_best = np.min(eval_sample_nelbo, axis=1)
    selected_eval = float(np.mean(per_query_best)) if per_query_best.size else 0.0
    return {
        **dict(base),
        "method": "per_query_oracle",
        "adoption_eligible": 0,
        "diagnostic_only": 1,
        "exploratory_only": 0,
        "baseline_available": 1,
        "source_method": "",
        "routing_uses_eval_nelbo": 1,
        "routing_uses_eval_indices": 1,
        "selected_expert": -1,
        "support_mean_nelbo": float(np.mean(support_nelbo)) if support_nelbo.size else 0.0,
        "selected_eval_nelbo": selected_eval,
        "oracle_gap": 0.0,
        "normalized_oracle_gap": 0.0,
        "top1_oracle_hit": 1.0,
        "selection_margin": 0.0,
        "candidate_experts": "|".join(str(int(e)) for e in candidate_experts),
    }


def evaluate_support_set_calibration_from_arrays(
    *,
    embeddings: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
    nelbo_matrix: np.ndarray,
    expert_domains: Sequence[int],
    run_meta: SupportSetRunMeta,
    support_sizes: Sequence[int],
    support_seeds: Sequence[int],
    sampling_policies: Sequence[str],
    metadata_tau: float = 100.0,
    topk_values: Sequence[int] = (),
    softmax_temperatures: Sequence[float] = (),
    response_proxy_lookup: Mapping[Tuple[str, int, str, str, int], ResponseProxyBaseline] | None = None,
) -> List[Dict[str, Any]]:
    if len(metadata) != int(embeddings.shape[0]):
        raise ValueError("Embedding and metadata lengths do not match")
    if nelbo_matrix.shape != (int(embeddings.shape[0]), len(expert_domains)):
        raise ValueError("NELBO matrix shape must be n_samples x n_experts")

    expert_domains_int = [int(d) for d in expert_domains]
    sample_domains = np.asarray([_as_domain(m["magnification"]) for m in metadata], dtype=np.int64)
    labels_by_index = {idx: _as_label(m) for idx, m in enumerate(metadata)}
    centroids = _domain_centroids(np.asarray(embeddings, dtype=np.float64), sample_domains)
    rows: List[Dict[str, Any]] = []
    response_proxy_lookup = response_proxy_lookup or {}

    for target_domain in sorted(set(int(v) for v in sample_domains.tolist())):
        target_indices = [int(i) for i, d in enumerate(sample_domains.tolist()) if int(d) == int(target_domain)]
        candidate_col_idxs = [i for i, e in enumerate(expert_domains_int) if int(e) != int(target_domain)]
        candidate_experts = [expert_domains_int[i] for i in candidate_col_idxs]
        if not candidate_experts:
            continue

        for support_seed in support_seeds:
            for policy in sampling_policies:
                for support_size in support_sizes:
                    split = make_support_eval_split(
                        target_domain=int(target_domain),
                        target_indices=target_indices,
                        labels_by_index=labels_by_index,
                        support_size=int(support_size),
                        sampling_policy=str(policy),
                        support_seed=int(support_seed),
                    )
                    if split.split_status != "ok":
                        rows.append(
                            {
                                "dataset_name": run_meta.dataset_name,
                                "seed": int(run_meta.seed),
                                "backbone_type": run_meta.backbone_type,
                                "run_id": run_meta.run_id,
                                "variant": run_meta.variant,
                                "run_dir": run_meta.run_dir,
                                "target_domain": int(target_domain),
                                "support_seed": int(support_seed),
                                "support_size_requested": int(support_size),
                                "support_size_actual": 0,
                                "eval_size": 0,
                                "sampling_policy": str(policy),
                                "sampling_policy_effective": split.sampling_policy_effective,
                                "split_status": split.split_status,
                                "no_data_reason": split.no_data_reason,
                                "method": "split_skipped",
                                "candidate_experts": "|".join(str(int(e)) for e in candidate_experts),
                                "excluded_experts": str(int(target_domain)),
                                "target_expert_excluded": 1,
                                "support_eval_disjoint": 1,
                                "support_labels_used": int(split.support_labels_used),
                                "support_is_target_local": 1,
                                "adoption_eligible": 0,
                                "diagnostic_only": 0,
                                "exploratory_only": 0,
                                "routing_uses_eval_nelbo": 0,
                                "routing_uses_eval_indices": 0,
                                "support_eval_split_id": split.support_eval_split_id,
                            }
                        )
                        continue

                    support_scores = nelbo_matrix[np.asarray(split.support_indices, dtype=np.int64)[:, None], candidate_col_idxs]
                    eval_scores = nelbo_matrix[np.asarray(split.eval_indices, dtype=np.int64)[:, None], candidate_col_idxs]
                    support_mean = np.mean(support_scores, axis=0)
                    eval_mean = np.mean(eval_scores, axis=0)
                    oracle_idx = _stable_argmin(eval_mean, candidate_experts)
                    worst_idx = int(max(range(len(eval_mean)), key=lambda i: (float(eval_mean[i]), -int(candidate_experts[i]))))
                    base = _base_row(
                        meta=run_meta,
                        split=split,
                        candidate_experts=candidate_experts,
                        target_domain=int(target_domain),
                        support_nelbo=support_mean,
                        eval_nelbo=eval_mean,
                        oracle_idx=oracle_idx,
                        worst_idx=worst_idx,
                    )

                    metadata_scores = [
                        ordinal_magnification_similarity(int(target_domain), int(e), float(metadata_tau))
                        for e in candidate_experts
                    ]
                    metadata_idx = _stable_argmax(metadata_scores, candidate_experts)
                    rows.append(
                        _method_row(
                            base=base,
                            method="metadata_ordinal_baseline",
                            scores=metadata_scores,
                            candidate_experts=candidate_experts,
                            support_nelbo=support_mean,
                            eval_nelbo=eval_mean,
                            selected_idx=metadata_idx,
                            adoption_eligible=1,
                            diagnostic_only=0,
                            exploratory_only=0,
                        )
                    )

                    support_centroid = np.asarray(embeddings[split.support_indices], dtype=np.float64).mean(axis=0)
                    embedding_scores: List[float] = []
                    for e in candidate_experts:
                        centroid = centroids.get(int(e))
                        if centroid is None:
                            embedding_scores.append(float("-inf"))
                        else:
                            embedding_scores.append(-float(np.linalg.norm(support_centroid - centroid, ord=2)))
                    embedding_idx = _stable_argmax(embedding_scores, candidate_experts)
                    rows.append(
                        _method_row(
                            base=base,
                            method="static_embedding_baseline",
                            scores=embedding_scores,
                            candidate_experts=candidate_experts,
                            support_nelbo=support_mean,
                            eval_nelbo=eval_mean,
                            selected_idx=embedding_idx,
                            adoption_eligible=1,
                            diagnostic_only=0,
                            exploratory_only=0,
                        )
                    )

                    support_compat = -support_mean
                    support_idx = _stable_argmax(support_compat, candidate_experts)
                    rows.append(
                        _method_row(
                            base=base,
                            method="support_set_calibration_top1",
                            scores=support_compat,
                            candidate_experts=candidate_experts,
                            support_nelbo=support_mean,
                            eval_nelbo=eval_mean,
                            selected_idx=support_idx,
                            adoption_eligible=1,
                            diagnostic_only=0,
                            exploratory_only=0,
                        )
                    )

                    oracle_scores = -eval_mean
                    rows.append(
                        _method_row(
                            base=base,
                            method="domain_oracle",
                            scores=oracle_scores,
                            candidate_experts=candidate_experts,
                            support_nelbo=support_mean,
                            eval_nelbo=eval_mean,
                            selected_idx=oracle_idx,
                            adoption_eligible=0,
                            diagnostic_only=1,
                            exploratory_only=0,
                            routing_uses_eval_nelbo=1,
                            routing_uses_eval_indices=1,
                        )
                    )
                    rows.append(
                        _per_query_oracle_row(
                            base=base,
                            candidate_experts=candidate_experts,
                            support_nelbo=support_mean,
                            eval_sample_nelbo=eval_scores,
                        )
                    )

                    response_key = (
                        run_meta.dataset_name,
                        int(run_meta.seed),
                        run_meta.backbone_type,
                        run_meta.run_id,
                        int(target_domain),
                    )
                    response_proxy = response_proxy_lookup.get(response_key)
                    if response_proxy is not None:
                        rows.append(
                            {
                                **dict(base),
                                "method": "response_proxy_baseline",
                                "adoption_eligible": int(response_proxy.adoption_eligible),
                                "diagnostic_only": 0,
                                "exploratory_only": 0,
                                "baseline_available": int(response_proxy.baseline_available),
                                "source_method": response_proxy.source_method,
                                "routing_uses_eval_nelbo": 0,
                                "routing_uses_eval_indices": 0,
                                "selected_expert": int(response_proxy.selected_expert),
                                "oracle_expert": int(response_proxy.oracle_expert),
                                "support_mean_nelbo": float(support_mean[support_idx]),
                                "selected_eval_nelbo": float(response_proxy.selected_eval_nelbo),
                                "oracle_eval_nelbo": float(response_proxy.oracle_eval_nelbo),
                                "worst_eval_nelbo": float(response_proxy.worst_eval_nelbo),
                                "oracle_gap": float(response_proxy.oracle_gap),
                                "normalized_oracle_gap": float(response_proxy.normalized_oracle_gap),
                                "top1_oracle_hit": float(response_proxy.top1_oracle_hit),
                                "spearman_support_vs_eval_utility": float(
                                    response_proxy.spearman_support_vs_eval_utility
                                ),
                                "selection_margin": 0.0,
                            }
                        )

                    order = sorted(range(len(support_compat)), key=lambda i: (-float(support_compat[i]), int(candidate_experts[i])))
                    for topk in topk_values:
                        k_eff = min(int(topk), len(order))
                        if k_eff <= 0:
                            continue
                        selected = order[:k_eff]
                        weighted_eval = float(np.mean(eval_mean[selected]))
                        pseudo_eval = np.asarray(eval_mean, dtype=np.float64).copy()
                        pseudo_idx = selected[0]
                        pseudo_eval[pseudo_idx] = weighted_eval
                        rows.append(
                            _method_row(
                                base=base,
                                method=f"support_set_topk{k_eff}_uniform",
                                scores=support_compat,
                                candidate_experts=candidate_experts,
                                support_nelbo=support_mean,
                                eval_nelbo=pseudo_eval,
                                selected_idx=pseudo_idx,
                                adoption_eligible=0,
                                diagnostic_only=0,
                                exploratory_only=1,
                            )
                        )

                    for temperature in softmax_temperatures:
                        weights = np.asarray(softmax(support_compat.tolist(), temperature=float(temperature)), dtype=np.float64)
                        weighted_eval = float(np.sum(weights * eval_mean))
                        pseudo_eval = np.asarray(eval_mean, dtype=np.float64).copy()
                        pseudo_idx = support_idx
                        pseudo_eval[pseudo_idx] = weighted_eval
                        row = _method_row(
                            base=base,
                            method=f"support_set_softmax_tau{float(temperature):g}",
                            scores=support_compat,
                            candidate_experts=candidate_experts,
                            support_nelbo=support_mean,
                            eval_nelbo=pseudo_eval,
                            selected_idx=pseudo_idx,
                            adoption_eligible=0,
                            diagnostic_only=0,
                            exploratory_only=1,
                        )
                        row["softmax_weight_by_expert_json"] = _json_mapping(candidate_experts, weights)
                        rows.append(row)

    return rows


def aggregate_support_set_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, Any]]:
    metrics = [
        "normalized_oracle_gap",
        "oracle_gap",
        "top1_oracle_hit",
        "calibration_mae",
        "spearman_support_vs_eval_utility",
        "selected_eval_nelbo",
        "oracle_eval_nelbo",
    ]
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        if str(row.get("split_status", "")) != "ok":
            continue
        key = (
            str(row.get("dataset_name", "")),
            str(row.get("backbone_type", "")),
            str(row.get("variant", "")),
            str(row.get("method", "")),
        )
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key, vals in sorted(groups.items()):
        dataset_name, backbone_type, variant, method = key
        row: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "backbone_type": backbone_type,
            "variant": variant,
            "method": method,
            "n_rows": int(len(vals)),
            "n_runs": int(len(set(str(v.get("run_id", "")) for v in vals))),
            "n_support_seeds": int(len(set(str(v.get("support_seed", "")) for v in vals))),
            "adoption_eligible": int(max(int(float(v.get("adoption_eligible", 0) or 0)) for v in vals)),
            "diagnostic_only": int(max(int(float(v.get("diagnostic_only", 0) or 0)) for v in vals)),
            "exploratory_only": int(max(int(float(v.get("exploratory_only", 0) or 0)) for v in vals)),
            "global_calibration_error_bin10": global_calibration_error_bin10(vals),
        }
        for metric in metrics:
            arr = np.asarray([float(v.get(metric, 0.0) or 0.0) for v in vals], dtype=np.float64)
            row[f"{metric}_mean"] = float(np.mean(arr)) if arr.size else 0.0
            row[f"{metric}_std"] = float(np.std(arr)) if arr.size else 0.0
        out.append(row)
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def read_embedding_cache(path: Path) -> Tuple[torch.Tensor, List[Mapping[str, object]]]:
    payload = safe_torch_load(path, map_location="cpu")
    embeddings = payload["embeddings"]
    metadata = list(payload["metadata"])
    return embeddings, metadata


def evaluate_support_set_calibration_for_run(
    *,
    test_cache: Path,
    variant_checkpoint: Path | None,
    expert_manifest: Path | None,
    hidden_dim: int,
    latent_dim: int,
    metadata_constraint_cfg: Mapping[str, object] | None,
    run_meta: SupportSetRunMeta,
    support_sizes: Sequence[int],
    support_seeds: Sequence[int],
    sampling_policies: Sequence[str],
    batch_size: int = 2048,
    metadata_tau: float = 100.0,
    topk_values: Sequence[int] = (),
    softmax_temperatures: Sequence[float] = (),
    response_proxy_lookup: Mapping[Tuple[str, int, str, str, int], ResponseProxyBaseline] | None = None,
) -> List[Dict[str, Any]]:
    x_cpu, metadata = read_embedding_cache(Path(test_cache))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bank = make_expert_bank(
        variant_checkpoint=variant_checkpoint,
        expert_manifest=expert_manifest,
        input_dim=int(x_cpu.shape[1]),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        metadata_constraint_cfg=metadata_constraint_cfg or {},
        device=device,
    )
    expert_domains = sorted(int(d) for d in bank.domains)
    nelbo = score_expert_nelbo_matrix(
        bank=bank,
        embeddings=x_cpu,
        expert_domains=expert_domains,
        batch_size=int(batch_size),
        device=device,
    )
    return evaluate_support_set_calibration_from_arrays(
        embeddings=x_cpu.detach().cpu().numpy().astype(np.float64, copy=False),
        metadata=metadata,
        nelbo_matrix=nelbo,
        expert_domains=expert_domains,
        run_meta=run_meta,
        support_sizes=support_sizes,
        support_seeds=support_seeds,
        sampling_policies=sampling_policies,
        metadata_tau=float(metadata_tau),
        topk_values=topk_values,
        softmax_temperatures=softmax_temperatures,
        response_proxy_lookup=response_proxy_lookup,
    )
