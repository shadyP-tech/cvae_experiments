from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.eval.metrics import spearman_corr
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.models.projection_head import ProjectionHead
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load


AGGREGATION_TOP1_HARD = "top1_hard"
AGGREGATION_TOPK2_UNIFORM = "topk2_uniform"
AGGREGATION_TOPK3_UNIFORM = "topk3_uniform"
AGGREGATION_SOFT_ALL_SOFTMAX = "soft_all_softmax"
ALLOWED_AGGREGATION_MODES = {
    AGGREGATION_TOP1_HARD,
    AGGREGATION_TOPK2_UNIFORM,
    AGGREGATION_TOPK3_UNIFORM,
    AGGREGATION_SOFT_ALL_SOFTMAX,
}


@dataclass
class RoutingStats:
    spearman_similarity_vs_neg_nelbo: float
    top1_agreement_with_best_expert: float
    mean_rank_of_metadata_selected_expert: float


def _safe_normalize_weights(weights: List[float]) -> List[float]:
    if not weights:
        return []
    total = float(sum(float(w) for w in weights))
    if total <= 0.0:
        uniform = 1.0 / float(len(weights))
        return [uniform for _ in weights]
    return [float(w) / total for w in weights]


def _resolve_topk(mode: str, configured_topk: int, n_experts: int) -> int:
    if mode == AGGREGATION_TOPK2_UNIFORM:
        return min(2, max(n_experts, 1))
    if mode == AGGREGATION_TOPK3_UNIFORM:
        return min(3, max(n_experts, 1))
    return min(max(int(configured_topk), 1), max(n_experts, 1))


def _select_indices_and_weights(
    similarities: List[float],
    mode: str,
    topk: int,
    temperature: float,
) -> Tuple[List[int], List[float]]:
    n = len(similarities)
    if n == 0:
        return [], []
    if mode not in ALLOWED_AGGREGATION_MODES:
        raise ValueError(f"Unsupported aggregation_mode '{mode}'. Allowed: {sorted(ALLOWED_AGGREGATION_MODES)}")

    order_desc = sorted(range(n), key=lambda idx: (-float(similarities[idx]), idx))

    if mode == AGGREGATION_TOP1_HARD:
        return [order_desc[0]], [1.0]

    if mode in {AGGREGATION_TOPK2_UNIFORM, AGGREGATION_TOPK3_UNIFORM}:
        k = _resolve_topk(mode=mode, configured_topk=topk, n_experts=n)
        selected = sorted(order_desc[:k])
        uniform = 1.0 / float(len(selected))
        return selected, [uniform for _ in selected]

    # soft_all_softmax
    temp = max(float(temperature), 1e-8)
    scaled = [float(s) / temp for s in similarities]
    m = max(scaled)
    exps = [float(np.exp(v - m)) for v in scaled]
    weights = _safe_normalize_weights(exps)
    selected = list(range(n))
    return selected, [weights[idx] for idx in selected]


def _allocate_counts_deterministic(
    total: int,
    weights: List[float],
    tie_break_ids: List[int],
) -> Tuple[List[int], List[float]]:
    if total <= 0 or not weights:
        return [0 for _ in weights], [0.0 for _ in weights]

    if len(weights) != len(tie_break_ids):
        raise ValueError("weights and tie_break_ids must have equal length")

    norm_w = _safe_normalize_weights(weights)
    ideals = [float(total) * float(w) for w in norm_w]
    base = [int(np.floor(v)) for v in ideals]
    remaining = int(total - sum(base))

    if remaining > 0:
        remainders = [ideals[i] - float(base[i]) for i in range(len(base))]
        order = sorted(range(len(base)), key=lambda i: (-float(remainders[i]), int(tie_break_ids[i])))
        for pos in order[:remaining]:
            base[pos] += 1

    return base, ideals


class HybridExpertBank:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        payload = safe_torch_load(checkpoint, map_location=device)
        self.variant = str(payload["variant"])
        self.domains = [int(d) for d in payload["domains"]]
        self.input_dim = int(payload["input_dim"])
        self.projection_dim = int(payload["projection_dim"])
        self.head_hidden_dim = int(payload["head_hidden_dim"])
        self.cvae_hidden_dim = int(payload["cvae_hidden_dim"])
        self.latent_dim = int(payload["latent_dim"])
        self.metadata_constraint_cfg = dict(payload.get("metadata_constraint_cfg", {}))
        self.aux_metadata_dim = int(payload.get("aux_metadata_dim", len(self.domains)))
        self.device = device

        self.shared_head: ProjectionHead | None = None
        self.shared_cvae: CVAEExpert | None = None
        self.heads: Dict[int, ProjectionHead] = {}
        self.cvaes: Dict[int, CVAEExpert] = {}

        if payload.get("shared_head") is not None:
            self.shared_head = ProjectionHead(self.input_dim, self.projection_dim, self.head_hidden_dim).to(device)
            self.shared_head.load_state_dict(payload["shared_head"])
            self.shared_head.eval()

        if payload.get("shared_cvae") is not None:
            self.shared_cvae = CVAEExpert(
                self.projection_dim,
                self.cvae_hidden_dim,
                self.latent_dim,
                metadata_constraint_cfg=self.metadata_constraint_cfg,
                aux_metadata_dim=self.aux_metadata_dim,
            ).to(device)
            self.shared_cvae.load_state_dict(payload["shared_cvae"])
            self.shared_cvae.eval()

        for k, state in payload.get("heads", {}).items():
            d = int(k)
            m = ProjectionHead(self.input_dim, self.projection_dim, self.head_hidden_dim).to(device)
            m.load_state_dict(state)
            m.eval()
            self.heads[d] = m

        for k, state in payload.get("cvaes", {}).items():
            d = int(k)
            m = CVAEExpert(
                self.projection_dim,
                self.cvae_hidden_dim,
                self.latent_dim,
                metadata_constraint_cfg=self.metadata_constraint_cfg,
                aux_metadata_dim=self.aux_metadata_dim,
            ).to(device)
            m.load_state_dict(state)
            m.eval()
            self.cvaes[d] = m

    def _head_for_domain(self, domain: int) -> ProjectionHead:
        if self.shared_head is not None:
            return self.shared_head
        return self.heads[int(domain)]

    def _cvae_for_domain(self, domain: int) -> CVAEExpert:
        if self.shared_cvae is not None:
            return self.shared_cvae
        return self.cvaes[int(domain)]

    def domain_cvae(self, domain: int) -> CVAEExpert:
        return self._cvae_for_domain(domain)

    def score_domain_nelbo(self, expert_domain: int, x: torch.Tensor) -> torch.Tensor:
        head = self._head_for_domain(expert_domain)
        cvae = self._cvae_for_domain(expert_domain)
        proj = head(x)
        recon, mu, logvar = cvae(proj)
        rec, kl = elbo_components(recon, proj, mu, logvar)
        return rec + kl

    def score_domain_recon(self, expert_domain: int, x: torch.Tensor) -> torch.Tensor:
        head = self._head_for_domain(expert_domain)
        cvae = self._cvae_for_domain(expert_domain)
        proj = head(x)
        recon, mu, logvar = cvae(proj)
        rec, _ = elbo_components(recon, proj, mu, logvar)
        return rec

    def project(self, domain: int, x: torch.Tensor) -> torch.Tensor:
        head = self._head_for_domain(domain)
        return head(x)

    def generate_from_reference(self, domain: int, x_ref: torch.Tensor, n_samples: int, seed: int) -> torch.Tensor:
        head = self._head_for_domain(domain)
        cvae = self._cvae_for_domain(domain)
        if x_ref.shape[0] == 0 or n_samples <= 0:
            return torch.empty((0, self.projection_dim), device=self.device)
        rng = random.Random(seed)
        with torch.no_grad():
            proj_ref = head(x_ref)
            mu, logvar = cvae.encode(proj_ref)
            idxs = [rng.randrange(int(mu.shape[0])) for _ in range(int(n_samples))]
            mu_s = mu[idxs]
            logvar_s = logvar[idxs]
            z = cvae.reparameterize(mu_s, logvar_s)
            return cvae.decode(z)


def _payload_by_domain(payload: Dict[str, object]) -> Dict[int, List[int]]:
    meta = payload["metadata"]
    domains = sorted(set(int(m["magnification"]) for m in meta))
    return {d: [i for i, m in enumerate(meta) if int(m["magnification"]) == d] for d in domains}


def _hard_domain_route(
    target_domain: int,
    expert_domains: List[int],
    strategy: str,
    tau: float,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[int, List[float]]:
    sims = [
        compute_similarity(
            {"magnification": int(target_domain)},
            {"magnification": int(d)},
            strategy=strategy,
            tau=float(tau),
            similarity_matrix=similarity_matrix,
        )
        for d in expert_domains
    ]
    best_idx = max(range(len(expert_domains)), key=lambda i: sims[i])
    return best_idx, sims


def _rank_desc(values: List[float]) -> List[int]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    rank = [0] * len(values)
    for r, idx in enumerate(order, start=1):
        rank[idx] = r
    return rank


def _score_domains_batched(
    bank: HybridExpertBank,
    expert_domains: List[int],
    x_cpu: torch.Tensor,
    device: torch.device,
    batch_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_scores: List[torch.Tensor] = []
    all_recon: List[torch.Tensor] = []

    with torch.no_grad():
        for ed in expert_domains:
            score_batches: List[torch.Tensor] = []
            recon_batches: List[torch.Tensor] = []
            for i in range(0, int(x_cpu.shape[0]), int(batch_size)):
                xb = x_cpu[i : i + int(batch_size)].to(device)
                score_batches.append(bank.score_domain_nelbo(ed, xb).cpu())
                recon_batches.append(bank.score_domain_recon(ed, xb).cpu())
            all_scores.append(torch.cat(score_batches, dim=0) if score_batches else torch.empty((0,), dtype=torch.float32))
            all_recon.append(torch.cat(recon_batches, dim=0) if recon_batches else torch.empty((0,), dtype=torch.float32))

    return torch.stack(all_scores, dim=0), torch.stack(all_recon, dim=0)


def _compatibility_sharpness(
    compatibility_nelbo: Dict[str, Dict[str, float]],
    expert_domains: List[int],
    target_domains: List[int],
) -> Dict[str, float]:
    if not expert_domains or not target_domains:
        return {
            "diagonal_mean": 0.0,
            "offdiagonal_mean": 0.0,
            "offdiagonal_std": 0.0,
            "diagonal_offdiagonal_gap": 0.0,
            "diagonal_gap_ratio": 0.0,
            "diagonal_margin_to_best_offdiagonal": 0.0,
        }

    mat = np.full((len(expert_domains), len(target_domains)), np.nan, dtype=np.float64)
    for i, ed in enumerate(expert_domains):
        row = compatibility_nelbo.get(f"{ed}x", {})
        for j, td in enumerate(target_domains):
            v = row.get(f"{td}x")
            if v is not None:
                mat[i, j] = float(v)

    if not np.isfinite(mat).any():
        return {
            "diagonal_mean": 0.0,
            "offdiagonal_mean": 0.0,
            "offdiagonal_std": 0.0,
            "diagonal_offdiagonal_gap": 0.0,
            "diagonal_gap_ratio": 0.0,
            "diagonal_margin_to_best_offdiagonal": 0.0,
        }

    diag_vals: List[float] = []
    offdiag_vals: List[float] = []
    diag_margins: List[float] = []
    domain_to_expert_idx = {int(d): i for i, d in enumerate(expert_domains)}

    for j, td in enumerate(target_domains):
        col = mat[:, j]
        finite_idxs = [idx for idx, v in enumerate(col.tolist()) if np.isfinite(v)]
        if not finite_idxs:
            continue

        best_offdiag = min(
            [float(col[idx]) for idx in finite_idxs],
            default=float("inf"),
        )
        if int(td) in domain_to_expert_idx:
            diag_idx = domain_to_expert_idx[int(td)]
            diag_v = float(col[diag_idx])
            if np.isfinite(diag_v):
                diag_vals.append(diag_v)
                offdiag_only = [float(col[idx]) for idx in finite_idxs if idx != diag_idx]
                if offdiag_only:
                    offdiag_vals.extend(offdiag_only)
                    diag_margins.append(diag_v - min(offdiag_only))
                else:
                    diag_margins.append(diag_v - best_offdiag)
        else:
            offdiag_vals.extend([float(col[idx]) for idx in finite_idxs])

    if not diag_vals:
        diag_vals = [float(v) for v in mat[np.isfinite(mat)].tolist()]

    diag_mean = float(np.mean(diag_vals)) if diag_vals else 0.0
    offdiag_mean = float(np.mean(offdiag_vals)) if offdiag_vals else diag_mean
    offdiag_std = float(np.std(offdiag_vals)) if offdiag_vals else 0.0
    gap = offdiag_mean - diag_mean
    gap_ratio = gap / (abs(offdiag_mean) + 1e-8)
    margin_mean = float(np.mean(diag_margins)) if diag_margins else 0.0

    return {
        "diagonal_mean": diag_mean,
        "offdiagonal_mean": offdiag_mean,
        "offdiagonal_std": offdiag_std,
        "diagonal_offdiagonal_gap": gap,
        "diagonal_gap_ratio": gap_ratio,
        "diagonal_margin_to_best_offdiagonal": margin_mean,
    }


def compute_hybrid_matrices_and_routing(
    test_cache: Path,
    hybrid_checkpoint: Path,
    strategy: str,
    tau: float,
    temperature: float,
    seed: int,
    similarity_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]
    by_domain = _payload_by_domain(payload)
    domains = sorted(by_domain.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    bank = HybridExpertBank(hybrid_checkpoint, device=device)
    expert_domains = [d for d in domains if d in bank.domains]
    if not expert_domains:
        raise RuntimeError("No overlapping expert domains found for hybrid routing evaluation.")

    compatibility_nelbo: Dict[str, Dict[str, float]] = {}
    compatibility_recon: Dict[str, Dict[str, float]] = {}

    # Dense scoring matrix [expert_idx, sample_idx], computed in batches to cap peak GPU memory.
    score_tensor, recon_tensor = _score_domains_batched(
        bank=bank,
        expert_domains=expert_domains,
        x_cpu=x_cpu,
        device=device,
        batch_size=2048,
    )

    for e_i, ed in enumerate(expert_domains):
        row_n = {}
        row_r = {}
        for td in domains:
            idxs = by_domain[td]
            if not idxs:
                continue
            row_n[f"{td}x"] = float(score_tensor[e_i, idxs].mean().item())
            row_r[f"{td}x"] = float(recon_tensor[e_i, idxs].mean().item())
        compatibility_nelbo[f"{ed}x"] = row_n
        compatibility_recon[f"{ed}x"] = row_r

    routing_matrix_counts = {f"{ed}x": {f"{td}x": 0 for td in domains} for ed in expert_domains}

    metadata_scores = []
    oracle_scores = []
    random_scores = []
    uniform_scores = []
    true_domains = []
    metadata_selected_domains = []
    oracle_selected_domains = []

    rng = random.Random(seed)
    fixed_random_idx = rng.randrange(len(expert_domains))

    domain_stats: List[RoutingStats] = []

    for td in domains:
        idxs = by_domain[td]
        if not idxs:
            continue

        hard_idx, sims = _hard_domain_route(
            target_domain=td,
            expert_domains=expert_domains,
            strategy=strategy,
            tau=tau,
            similarity_matrix=similarity_matrix,
        )
        utilities = []
        for e_i in range(len(expert_domains)):
            nelbo_mean = float(score_tensor[e_i, idxs].mean().item())
            utilities.append(-nelbo_mean)

        best_idx = max(range(len(expert_domains)), key=lambda i: utilities[i])
        top1_agree = 1.0 if best_idx == hard_idx else 0.0
        ranks = _rank_desc(utilities)
        selected_rank = float(ranks[hard_idx])
        rho = spearman_corr(sims, utilities)
        domain_stats.append(
            RoutingStats(
                spearman_similarity_vs_neg_nelbo=rho,
                top1_agreement_with_best_expert=top1_agree,
                mean_rank_of_metadata_selected_expert=selected_rank,
            )
        )

        for idx in idxs:
            sample_scores = [float(score_tensor[e_i, idx].item()) for e_i in range(len(expert_domains))]
            oracle_idx = min(range(len(expert_domains)), key=lambda i: sample_scores[i])
            sampled_idx = rng.randrange(len(expert_domains))

            metadata_scores.append(sample_scores[hard_idx])
            oracle_scores.append(sample_scores[oracle_idx])
            random_scores.append(sample_scores[fixed_random_idx])
            uniform_scores.append(sample_scores[sampled_idx])

            true_domains.append(f"{td}x")
            metadata_selected_domains.append(f"{expert_domains[hard_idx]}x")
            oracle_selected_domains.append(f"{expert_domains[oracle_idx]}x")
            routing_matrix_counts[f"{expert_domains[hard_idx]}x"][f"{td}x"] += 1

    routing_matrix = {}
    for e in expert_domains:
        row = {}
        for td in domains:
            denom = max(len(by_domain[td]), 1)
            row[f"{td}x"] = routing_matrix_counts[f"{e}x"][f"{td}x"] / denom
        routing_matrix[f"{e}x"] = row

    if domain_stats:
        stats = {
            "spearman_similarity_vs_neg_nelbo": float(
                sum(s.spearman_similarity_vs_neg_nelbo for s in domain_stats) / len(domain_stats)
            ),
            "top1_agreement_with_best_expert": float(
                sum(s.top1_agreement_with_best_expert for s in domain_stats) / len(domain_stats)
            ),
            "mean_rank_of_metadata_selected_expert": float(
                sum(s.mean_rank_of_metadata_selected_expert for s in domain_stats) / len(domain_stats)
            ),
            "compatibility_sharpness_nelbo": _compatibility_sharpness(
                compatibility_nelbo=compatibility_nelbo,
                expert_domains=expert_domains,
                target_domains=domains,
            ),
        }
    else:
        stats = {
            "spearman_similarity_vs_neg_nelbo": 0.0,
            "top1_agreement_with_best_expert": 0.0,
            "mean_rank_of_metadata_selected_expert": 0.0,
            "compatibility_sharpness_nelbo": _compatibility_sharpness(
                compatibility_nelbo=compatibility_nelbo,
                expert_domains=expert_domains,
                target_domains=domains,
            ),
        }

    metadata_routing_nelbo = float(torch.tensor(metadata_scores).mean().item()) if metadata_scores else 0.0
    oracle_routing_nelbo = float(torch.tensor(oracle_scores).mean().item()) if oracle_scores else 0.0
    metadata_to_oracle_gap = metadata_routing_nelbo - oracle_routing_nelbo
    metadata_to_oracle_gap_pct = (
        (100.0 * metadata_to_oracle_gap / max(abs(oracle_routing_nelbo), 1e-8))
        if metadata_scores and oracle_scores
        else 0.0
    )

    return {
        "compatibility_matrix": {
            "nelbo": compatibility_nelbo,
            "reconstruction": compatibility_recon,
        },
        "routing_matrix": routing_matrix,
        "routing_statistics": stats,
        "routing_metrics": {
            "metadata_routing_nelbo": metadata_routing_nelbo,
            "oracle_routing_nelbo": oracle_routing_nelbo,
            "random_routing_nelbo": float(torch.tensor(random_scores).mean().item()) if random_scores else 0.0,
            "uniform_routing_nelbo": float(torch.tensor(uniform_scores).mean().item()) if uniform_scores else 0.0,
            "metadata_to_oracle_gap": metadata_to_oracle_gap,
            "metadata_to_oracle_gap_pct": metadata_to_oracle_gap_pct,
        },
        "routing_assignments": {
            "true_domains": true_domains,
            "metadata_selected_domains": metadata_selected_domains,
            "oracle_selected_domains": oracle_selected_domains,
        },
    }


def _domain_labels(payload: Dict[str, object], idxs: List[int]) -> torch.Tensor:
    return torch.tensor([int(payload["metadata"][i]["label"]) for i in idxs], dtype=torch.long)


def _allocate_class_counts(labels: torch.Tensor, total: int) -> Dict[int, int]:
    classes = [0, 1]
    if labels.numel() == 0 or total <= 0:
        return {0: 0, 1: 0}
    counts = {c: int((labels == c).sum().item()) for c in classes}
    n = int(labels.numel())
    target = {c: int(round(total * counts[c] / max(n, 1))) for c in classes}
    diff = total - sum(target.values())
    # Assign remainder to majority class.
    major = max(classes, key=lambda c: counts[c])
    target[major] += diff
    return target


def _train_eval_logreg(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
) -> Dict[str, float]:
    try:
        import importlib

        linear_model = importlib.import_module("sklearn.linear_model")
        sk_metrics = importlib.import_module("sklearn.metrics")
        LogisticRegression = getattr(linear_model, "LogisticRegression")
        balanced_accuracy_score = getattr(sk_metrics, "balanced_accuracy_score")
        accuracy_score = getattr(sk_metrics, "accuracy_score")
        roc_auc_score = getattr(sk_metrics, "roc_auc_score")
    except Exception:
        return {"auroc": 0.0, "balanced_accuracy": 0.0}

    if x_train.shape[0] < 2 or x_test.shape[0] < 2:
        return {"auroc": 0.0, "balanced_accuracy": 0.0}

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train.cpu().numpy(), y_train.cpu().numpy())
    xte_np = x_test.cpu().numpy()
    yte_np = y_test.cpu().numpy()
    probs = clf.predict_proba(xte_np)[:, 1]
    preds = clf.predict(xte_np)

    # AUROC is undefined with single-class targets; use neutral 0.5 for reporting stability.
    if len(set(yte_np.tolist())) < 2:
        auroc = 0.5
        bacc = float(accuracy_score(yte_np, preds))
    else:
        auroc = float(roc_auc_score(yte_np, probs))
        bacc = float(balanced_accuracy_score(yte_np, preds))

    return {
        "auroc": auroc,
        "balanced_accuracy": bacc,
    }


def evaluate_downstream_utility(
    train_cache: Path,
    test_cache: Path,
    hybrid_checkpoint: Path,
    pooled_checkpoint: Path,
    strategy: str,
    tau: float,
    temperature: float,
    seed: int,
    budget_multipliers: List[float],
    aggregation_mode: str = AGGREGATION_TOP1_HARD,
    aggregation_topk: int = 2,
    aggregation_temperature: float = 1.0,
) -> Dict[str, object]:
    _ = temperature
    if aggregation_mode not in ALLOWED_AGGREGATION_MODES:
        raise ValueError(f"Unsupported aggregation_mode '{aggregation_mode}'. Allowed: {sorted(ALLOWED_AGGREGATION_MODES)}")
    if int(aggregation_topk) <= 0:
        raise ValueError("aggregation_topk must be > 0")
    if float(aggregation_temperature) <= 0.0:
        raise ValueError("aggregation_temperature must be > 0")

    train_payload = safe_torch_load(train_cache, map_location="cpu")
    test_payload = safe_torch_load(test_cache, map_location="cpu")
    train_by_domain = _payload_by_domain(train_payload)
    test_by_domain = _payload_by_domain(test_payload)

    domains = sorted(train_by_domain.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    routed_bank = HybridExpertBank(hybrid_checkpoint, device=device)
    pooled_bank = HybridExpertBank(pooled_checkpoint, device=device)

    x_train_cpu = train_payload["embeddings"]
    x_test_cpu = test_payload["embeddings"]
    expert_domains = [ed for ed in routed_bank.domains if ed in domains]
    if not expert_domains:
        raise RuntimeError("No overlapping expert domains found for downstream utility evaluation.")

    out: Dict[str, object] = {}

    for budget in budget_multipliers:
        budget_key = f"budget_{budget:.1f}x"
        out[budget_key] = {}
        rng = random.Random(seed + int(budget * 1000))

        for d in domains:
            train_idxs = train_by_domain.get(d, [])
            test_idxs = test_by_domain.get(d, [])
            if not train_idxs or not test_idxs:
                continue

            # Domain-level compatibility scores are fixed across modes; only aggregation policy changes.
            hard_idx, sims = _hard_domain_route(
                target_domain=d,
                expert_domains=expert_domains,
                strategy=strategy,
                tau=tau,
                similarity_matrix=None,
            )
            selected_indices, selected_weights = _select_indices_and_weights(
                similarities=sims,
                mode=str(aggregation_mode),
                topk=int(aggregation_topk),
                temperature=float(aggregation_temperature),
            )
            selected_domains = [int(expert_domains[idx]) for idx in selected_indices]

            routed_domain = expert_domains[hard_idx]
            random_domain = expert_domains[rng.randrange(len(expert_domains))]

            xtr_local = x_train_cpu[train_idxs].to(device)
            ytr_local = _domain_labels(train_payload, train_idxs).to(device)
            xte_local = x_test_cpu[test_idxs].to(device)
            yte_local = _domain_labels(test_payload, test_idxs).to(device)

            # Use routed projection head as the common representation frame for this domain.
            with torch.no_grad():
                xtr_real_proj = routed_bank.project(routed_domain, xtr_local)
                xte_real_proj = routed_bank.project(routed_domain, xte_local)

            n_syn = int(round(float(budget) * int(xtr_local.shape[0])))
            class_targets = _allocate_class_counts(ytr_local.cpu(), n_syn)

            def _make_condition(bank: HybridExpertBank, gen_domain: int, cond_seed: int):
                synth_chunks = []
                label_chunks = []
                for cls, n_cls in class_targets.items():
                    if n_cls <= 0:
                        continue
                    class_mask = ytr_local == int(cls)
                    if not torch.any(class_mask):
                        continue
                    ref = xtr_local[class_mask]
                    x_syn = bank.generate_from_reference(gen_domain, ref, n_cls, seed=cond_seed + int(cls))
                    y_syn = torch.full((int(x_syn.shape[0]),), int(cls), dtype=torch.long, device=device)
                    synth_chunks.append(x_syn)
                    label_chunks.append(y_syn)
                if not synth_chunks:
                    return torch.empty((0, xtr_real_proj.shape[1]), device=device), torch.empty((0,), dtype=torch.long, device=device)
                return torch.cat(synth_chunks, dim=0), torch.cat(label_chunks, dim=0)

            def _make_condition_aggregated(
                bank: HybridExpertBank,
                gen_domains: List[int],
                gen_weights: List[float],
                cond_seed: int,
                mode: str,
            ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
                if not gen_domains:
                    empty_x = torch.empty((0, xtr_real_proj.shape[1]), device=device)
                    empty_y = torch.empty((0,), dtype=torch.long, device=device)
                    return empty_x, empty_y, {"per_class": {}, "allocation_checksum": 0}

                # Preserve legacy top1 behavior exactly.
                if len(gen_domains) == 1 and mode == AGGREGATION_TOP1_HARD:
                    x_single, y_single = _make_condition(bank, int(gen_domains[0]), int(cond_seed))
                    per_class = {
                        str(cls): {
                            "experts": [
                                {
                                    "expert_domain": int(gen_domains[0]),
                                    "weight": 1.0,
                                    "ideal_count": float(n_cls),
                                    "allocated_count": int(n_cls),
                                }
                            ]
                        }
                        for cls, n_cls in class_targets.items()
                    }
                    return x_single, y_single, {
                        "per_class": per_class,
                        "allocation_checksum": int(sum(int(v) for v in class_targets.values())),
                    }

                synth_chunks: List[torch.Tensor] = []
                label_chunks: List[torch.Tensor] = []
                allocation_meta: Dict[str, Any] = {"per_class": {}}
                total_allocated = 0

                for cls, n_cls in class_targets.items():
                    if int(n_cls) <= 0:
                        continue
                    class_mask = ytr_local == int(cls)
                    if not torch.any(class_mask):
                        continue

                    ref = xtr_local[class_mask]
                    counts, ideals = _allocate_counts_deterministic(
                        total=int(n_cls),
                        weights=gen_weights,
                        tie_break_ids=[int(v) for v in gen_domains],
                    )

                    experts_meta: List[Dict[str, Any]] = []
                    for pos, gen_domain in enumerate(gen_domains):
                        n_gen = int(counts[pos])
                        experts_meta.append(
                            {
                                "expert_domain": int(gen_domain),
                                "weight": float(gen_weights[pos]),
                                "ideal_count": float(ideals[pos]),
                                "allocated_count": int(n_gen),
                            }
                        )
                        if n_gen <= 0:
                            continue
                        x_syn = bank.generate_from_reference(
                            int(gen_domain),
                            ref,
                            n_gen,
                            seed=int(cond_seed + (1000 * int(cls)) + int(gen_domain)),
                        )
                        y_syn = torch.full((int(x_syn.shape[0]),), int(cls), dtype=torch.long, device=device)
                        synth_chunks.append(x_syn)
                        label_chunks.append(y_syn)
                        total_allocated += int(x_syn.shape[0])

                    allocation_meta["per_class"][str(cls)] = {"experts": experts_meta}

                if not synth_chunks:
                    empty_x = torch.empty((0, xtr_real_proj.shape[1]), device=device)
                    empty_y = torch.empty((0,), dtype=torch.long, device=device)
                    allocation_meta["allocation_checksum"] = int(total_allocated)
                    return empty_x, empty_y, allocation_meta

                allocation_meta["allocation_checksum"] = int(total_allocated)
                return torch.cat(synth_chunks, dim=0), torch.cat(label_chunks, dim=0), allocation_meta

            x_syn_routed, y_syn_routed, routed_allocation = _make_condition_aggregated(
                routed_bank,
                gen_domains=selected_domains,
                gen_weights=selected_weights,
                cond_seed=seed + 11,
                mode=str(aggregation_mode),
            )
            x_syn_random, y_syn_random = _make_condition(routed_bank, random_domain, seed + 23)
            x_syn_pooled, y_syn_pooled = _make_condition(pooled_bank, d, seed + 37)

            conditions = {
                "real_only": (xtr_real_proj, ytr_local),
                "real_plus_routed_synthetic": (
                    torch.cat([xtr_real_proj, x_syn_routed], dim=0),
                    torch.cat([ytr_local, y_syn_routed], dim=0),
                ),
                "real_plus_random_synthetic": (
                    torch.cat([xtr_real_proj, x_syn_random], dim=0),
                    torch.cat([ytr_local, y_syn_random], dim=0),
                ),
                "real_plus_pooled_synthetic": (
                    torch.cat([xtr_real_proj, x_syn_pooled], dim=0),
                    torch.cat([ytr_local, y_syn_pooled], dim=0),
                ),
            }

            domain_res = {
                "synthetic_budget": n_syn,
                "class_targets": {str(k): int(v) for k, v in class_targets.items()},
                "routed_domain": f"{routed_domain}x",
                "random_domain": f"{random_domain}x",
                "aggregation": {
                    "mode": str(aggregation_mode),
                    "topk": int(aggregation_topk),
                    "temperature": float(aggregation_temperature),
                    "selected_domains": [f"{int(v)}x" for v in selected_domains],
                    "selected_weights": [float(v) for v in selected_weights],
                    "allocation": routed_allocation,
                },
                "metrics": {},
            }
            for name, (xtr_c, ytr_c) in conditions.items():
                domain_res["metrics"][name] = _train_eval_logreg(
                    x_train=xtr_c,
                    y_train=ytr_c,
                    x_test=xte_real_proj,
                    y_test=yte_local,
                    seed=seed,
                )
            out[budget_key][f"{d}x"] = domain_res

    return out


def evaluate_global_baselines(
    test_cache: Path,
    legacy_global_checkpoint: Path,
    legacy_hidden_dim: int,
    legacy_latent_dim: int,
    pooled_checkpoint: Path,
) -> Dict[str, float]:
    payload = safe_torch_load(test_cache, map_location="cpu")
    x_cpu = payload["embeddings"]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    x = x_cpu.to(device)

    # Legacy baseline: global CVAE directly on shared foundation embeddings.
    legacy = CVAEExpert(
        input_dim=int(x.shape[1]),
        hidden_dim=int(legacy_hidden_dim),
        latent_dim=int(legacy_latent_dim),
    ).to(device)
    legacy.load_state_dict(safe_torch_load(legacy_global_checkpoint, map_location=device))
    legacy.eval()

    with torch.no_grad():
        recon, mu, logvar = legacy(x)
        rec, kl = elbo_components(recon, x, mu, logvar)
        legacy_nelbo = float((rec + kl).mean().item())

    # Matched pooled hybrid baseline: shared projection head + shared CVAE.
    pooled_bank = HybridExpertBank(pooled_checkpoint, device=device)
    pooled_domain = int(pooled_bank.domains[0]) if pooled_bank.domains else 0
    with torch.no_grad():
        pooled_scores = pooled_bank.score_domain_nelbo(pooled_domain, x)
        pooled_nelbo = float(pooled_scores.mean().item())

    return {
        "legacy_global_nelbo": legacy_nelbo,
        "hybrid_pooled_global_nelbo": pooled_nelbo,
    }
