from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from src.eval.evaluators.learned_utility_scoring import _parse_expert_domain
from src.eval.evaluators.support_set_calibration import make_support_eval_split, normalized_oracle_gap
from src.eval.metrics import spearman_corr
from src.models.cvae_expert import CVAEExpert, elbo_components
from src.routing.strategies import compute_similarity
from src.torch_utils import safe_torch_load
from src.train.checkpoint_provenance import load_model_checkpoint


FAMILY_C_METHOD = "family_c_label_marginal"
FAMILY_C_SENSITIVITY_METHOD = "family_c_label_marginal_source_global_laplace"
FORBIDDEN_PRIORS = {"target_support_empirical", "target_eval_empirical", "target_true_label_prior"}
EPS = 1e-12


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(r) for r in rows)


def _as_domain(meta: Mapping[str, object]) -> int:
    return int(str(meta["magnification"]).replace("x", ""))


def _as_label(meta: Mapping[str, object], label_field: str) -> int:
    return int(meta.get(label_field, 0))


def _stable_argmin(values: Sequence[float], expert_domains: Sequence[int]) -> int:
    return int(min(range(len(values)), key=lambda i: (float(values[i]), int(expert_domains[i]))))


def _stable_random_expert(expert_domains: Sequence[int], *, seed: int, target_domain: int, support_seed: int, support_size: int) -> int:
    rng = np.random.default_rng(int(seed) + int(target_domain) * 1009 + int(support_seed) * 9173 + int(support_size) * 101)
    return int(rng.choice(np.asarray(sorted(int(e) for e in expert_domains), dtype=np.int64)))


def _json_mapping(keys: Sequence[int], values: Sequence[float]) -> str:
    return json.dumps({str(int(k)): float(v) for k, v in zip(keys, values)}, sort_keys=True, separators=(",", ":"))


def _label_one_hot(batch_size: int, class_idx: int, class_dim: int, device: torch.device) -> torch.Tensor:
    y = torch.zeros((int(batch_size), int(class_dim)), dtype=torch.float32, device=device)
    y[:, int(class_idx)] = 1.0
    return y


def label_marginal_nelbo_proxy(
    class_nelbo: torch.Tensor,
    class_prior: torch.Tensor,
) -> torch.Tensor:
    """ELBO-derived unlabeled score; lower is better."""

    if class_nelbo.ndim != 2:
        raise ValueError("class_nelbo must be 2D with shape n_samples x n_classes")
    if class_prior.ndim != 1 or class_prior.shape[0] != class_nelbo.shape[1]:
        raise ValueError("class_prior must be 1D with one value per class")
    if torch.any(class_prior <= 0):
        raise ValueError("class_prior must contain positive probabilities")
    log_prior = torch.log(class_prior.to(device=class_nelbo.device, dtype=class_nelbo.dtype))
    return -torch.logsumexp(log_prior.reshape(1, -1) - class_nelbo, dim=1)


def _score_model_label_marginal(
    model: CVAEExpert,
    x: torch.Tensor,
    *,
    class_prior: torch.Tensor,
) -> torch.Tensor:
    class_scores: list[torch.Tensor] = []
    for class_idx in range(int(class_prior.shape[0])):
        y = _label_one_hot(int(x.shape[0]), class_idx, int(class_prior.shape[0]), x.device)
        recon, mu, logvar = model(x, y=y)
        rec, kl = elbo_components(recon, x, mu, logvar)
        class_scores.append(rec + kl)
    return label_marginal_nelbo_proxy(torch.stack(class_scores, dim=1), class_prior.to(x.device))


def _load_label_conditioned_models(
    expert_checkpoints: Mapping[str, str],
    *,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    class_condition_dim: int,
    device: torch.device,
) -> tuple[list[int], list[CVAEExpert], list[dict[str, Any]]]:
    domains: list[int] = []
    models: list[CVAEExpert] = []
    provenance_rows: list[dict[str, Any]] = []
    for name in sorted(expert_checkpoints.keys()):
        domain = _parse_expert_domain(str(name))
        checkpoint = Path(str(expert_checkpoints[name]))
        loaded = load_model_checkpoint(checkpoint, map_location=device)
        model = CVAEExpert(
            input_dim=int(input_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            class_condition_dim=int(class_condition_dim),
        ).to(device)
        model.load_state_dict(loaded.model_state_dict)
        model.eval()
        domains.append(int(domain))
        models.append(model)
        meta = dict(loaded.checkpoint_metadata or {})
        provenance_rows.append(
            {
                "expert_domain": int(domain),
                "checkpoint": str(checkpoint),
                "expert_family": str(meta.get("expert_family", "")),
                "condition_type": str(meta.get("condition_type", "")),
                "label_field": str(meta.get("label_field", "")),
                "label_values_json": json.dumps(meta.get("label_values", []), sort_keys=True),
                "class_condition_dim": int(meta.get("class_condition_dim", class_condition_dim)),
                "feature_extractor_name": str(meta.get("feature_extractor_name", "")),
                "feature_extractor_checkpoint": str(meta.get("feature_extractor_checkpoint", "")),
                "feature_extractor_layer": str(meta.get("feature_extractor_layer", "")),
                "embedding_pooling": str(meta.get("embedding_pooling", "")),
                "embedding_dim": int(meta.get("embedding_dim", input_dim)),
                "latent_dim": int(latent_dim),
                "beta_kl_weight": float(meta.get("beta_kl_weight", 1.0)),
                "reconstruction_loss": str(meta.get("reconstruction_loss", "mse_sum")),
                "likelihood_variance_assumption": str(meta.get("likelihood_variance_assumption", "unit")),
            }
        )
    return domains, models, provenance_rows


def _score_matrix(
    *,
    x_cpu: torch.Tensor,
    models: Sequence[CVAEExpert],
    class_prior: Sequence[float],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    class_prior_t = torch.tensor([float(v) for v in class_prior], dtype=torch.float32, device=device)
    out = np.zeros((int(x_cpu.shape[0]), len(models)), dtype=np.float64)
    with torch.no_grad():
        for e_idx, model in enumerate(models):
            chunks: list[torch.Tensor] = []
            for start in range(0, int(x_cpu.shape[0]), int(batch_size)):
                xb = x_cpu[start : start + int(batch_size)].to(device)
                chunks.append(_score_model_label_marginal(model, xb, class_prior=class_prior_t).cpu())
            if chunks:
                out[:, e_idx] = torch.cat(chunks, dim=0).numpy().astype(np.float64, copy=False)
    return out


def _balanced_prior(label_values: Sequence[int]) -> list[float]:
    n = len(label_values)
    if n <= 0:
        raise ValueError("label_values must be non-empty")
    return [1.0 / float(n)] * n


def _source_global_laplace_prior(
    metadata: Sequence[Mapping[str, object]],
    *,
    heldout_domain: int,
    label_values: Sequence[int],
    label_field: str,
    alpha: float,
) -> list[float]:
    counts = {int(v): float(alpha) for v in label_values}
    for row in metadata:
        if _as_domain(row) == int(heldout_domain):
            continue
        label = _as_label(row, label_field)
        if label in counts:
            counts[int(label)] += 1.0
    total = sum(counts.values())
    return [float(counts[int(v)] / max(total, EPS)) for v in label_values]


def _metadata_selected(
    *,
    target_domain: int,
    candidate_experts: Sequence[int],
    strategy: str,
    tau: float,
) -> int:
    scores = [
        compute_similarity(
            {"magnification": int(target_domain)},
            {"magnification": int(e)},
            strategy=strategy,
            tau=float(tau),
        )
        for e in candidate_experts
    ]
    return int(max(range(len(scores)), key=lambda i: (float(scores[i]), -int(candidate_experts[i]))))


def _static_embedding_selected(
    *,
    train_embeddings: np.ndarray,
    train_domains: np.ndarray,
    support_embeddings: np.ndarray,
    candidate_experts: Sequence[int],
) -> int:
    target_stat = np.asarray(support_embeddings, dtype=np.float64).mean(axis=0)
    distances: list[float] = []
    for expert in candidate_experts:
        idx = np.where(train_domains == int(expert))[0]
        if idx.size == 0:
            distances.append(float("inf"))
            continue
        source_stat = np.asarray(train_embeddings[idx], dtype=np.float64).mean(axis=0)
        distances.append(float(np.linalg.norm(target_stat - source_stat, ord=2)))
    return _stable_argmin(distances, candidate_experts)


def _source_global_static_selected(
    *,
    train_score_matrix: np.ndarray,
    train_domains: np.ndarray,
    expert_domains: Sequence[int],
    candidate_col_idxs: Sequence[int],
    candidate_experts: Sequence[int],
    target_domain: int,
) -> int:
    scores: list[float] = []
    for col_idx, expert in zip(candidate_col_idxs, candidate_experts):
        idx = np.where((train_domains != int(target_domain)) & (train_domains != int(expert)))[0]
        if idx.size == 0:
            scores.append(float("inf"))
        else:
            scores.append(float(np.mean(train_score_matrix[idx, int(col_idx)])))
    _ = expert_domains
    return _stable_argmin(scores, candidate_experts)


def _method_row(
    *,
    base: Mapping[str, Any],
    method: str,
    selected_expert: int,
    candidate_experts: Sequence[int],
    support_scores: Sequence[float],
    eval_scores: Sequence[float],
    available: int = 1,
    selection_source: str = "",
) -> dict[str, Any]:
    candidate_to_idx = {int(e): i for i, e in enumerate(candidate_experts)}
    oracle_idx = _stable_argmin(eval_scores, candidate_experts)
    worst_idx = int(max(range(len(eval_scores)), key=lambda i: (float(eval_scores[i]), -int(candidate_experts[i]))))
    oracle_expert = int(candidate_experts[oracle_idx])
    if int(available) and int(selected_expert) in candidate_to_idx:
        selected_idx = int(candidate_to_idx[int(selected_expert)])
        selected_eval = float(eval_scores[selected_idx])
        selected_support = float(support_scores[selected_idx])
    else:
        selected_idx = -1
        selected_eval = float("nan")
        selected_support = float("nan")
    oracle_eval = float(eval_scores[oracle_idx])
    worst_eval = float(eval_scores[worst_idx])
    gap = float(selected_eval - oracle_eval) if selected_idx >= 0 else float("nan")
    norm_gap = normalized_oracle_gap(selected_eval, oracle_eval, worst_eval) if selected_idx >= 0 else float("nan")
    return {
        **dict(base),
        "method": str(method),
        "selected_expert": int(selected_expert) if selected_idx >= 0 else -1,
        "oracle_expert": int(oracle_expert),
        "available": int(available),
        "selection_source": str(selection_source),
        "selected_support_score": selected_support,
        "selected_eval_score": selected_eval,
        "oracle_eval_score": oracle_eval,
        "worst_eval_score": worst_eval,
        "oracle_gap": gap,
        "normalized_oracle_gap": norm_gap,
        "oracle_gap_pct": float(norm_gap * 100.0) if selected_idx >= 0 else float("nan"),
        "top1_oracle_hit": 1.0 if selected_idx >= 0 and int(selected_expert) == int(oracle_expert) else 0.0,
        "routing_uses_eval_score": 0,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method in sorted({str(r.get("method", "")) for r in rows}):
        vals = [r for r in rows if str(r.get("method", "")) == method and int(r.get("available", 1)) == 1]
        if not vals:
            out[method] = {"n_rows": 0.0}
            continue
        out[method] = {"n_rows": float(len(vals))}
        for metric in [
            "top1_oracle_hit",
            "spearman_support_eval_score",
            "normalized_oracle_gap",
            "oracle_gap_pct",
            "oracle_gap",
        ]:
            arr = np.asarray([float(v.get(metric, 0.0) or 0.0) for v in vals], dtype=np.float64)
            out[method][f"{metric}_mean"] = float(np.mean(arr)) if arr.size else 0.0
            out[method][f"{metric}_std"] = float(np.std(arr)) if arr.size else 0.0
    return out


def evaluate_label_marginal_support_nelbo(
    *,
    train_cache: Path,
    test_cache: Path,
    expert_checkpoints: Mapping[str, str],
    hidden_dim: int,
    latent_dim: int,
    strategy: str,
    tau: float,
    seed: int,
    learned_cfg: Mapping[str, Any],
    reports_dir: Path,
    batch_size: int = 2048,
    family_a_selection_path: Path | None = None,
) -> dict[str, Any]:
    cfg = dict((learned_cfg.get("label_marginal_support_nelbo", {}) or {}))
    label_cfg = dict(cfg.get("label_conditioning", {}) or {})
    label_values = [int(v) for v in label_cfg.get("label_values", cfg.get("label_values", [0, 1]))]
    label_field = str(label_cfg.get("label_field", cfg.get("label_field", "label")))
    if not label_values:
        raise ValueError("label_values must be non-empty for Family C")
    primary_prior_name = str(cfg.get("primary_prior", "balanced")).strip().lower()
    sensitivity_priors = [str(v).strip().lower() for v in cfg.get("sensitivity_priors", ["source_global_laplace"])]
    laplace_alpha = float(cfg.get("laplace_alpha", 1.0))
    all_priors = [primary_prior_name] + [p for p in sensitivity_priors if p != primary_prior_name]
    forbidden = sorted(set(all_priors).intersection(FORBIDDEN_PRIORS))
    if forbidden:
        raise ValueError(f"Forbidden Family C target-label prior configured: {forbidden}")
    if primary_prior_name != "balanced":
        raise ValueError("Family C primary_prior must be 'balanced' in v1")
    unsupported = sorted(set(all_priors).difference({"balanced", "source_global_laplace"}))
    if unsupported:
        raise ValueError(f"Unsupported Family C class prior(s): {unsupported}")

    train_payload = safe_torch_load(train_cache, map_location="cpu")
    test_payload = safe_torch_load(test_cache, map_location="cpu")
    train_x = train_payload["embeddings"]
    test_x = test_payload["embeddings"]
    train_metadata = list(train_payload["metadata"])
    test_metadata = list(test_payload["metadata"])
    train_domains = np.asarray([_as_domain(m) for m in train_metadata], dtype=np.int64)
    test_domains = np.asarray([_as_domain(m) for m in test_metadata], dtype=np.int64)
    labels_by_index = {idx: _as_label(m, label_field) for idx, m in enumerate(test_metadata)}

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    expert_domains, models, provenance_rows = _load_label_conditioned_models(
        expert_checkpoints,
        input_dim=int(test_x.shape[1]),
        hidden_dim=int(hidden_dim),
        latent_dim=int(latent_dim),
        class_condition_dim=int(len(label_values)),
        device=device,
    )
    _write_csv(reports_dir / "label_conditioned_checkpoint_provenance.csv", provenance_rows)

    comparability_keys = [
        "feature_extractor_name",
        "feature_extractor_checkpoint",
        "feature_extractor_layer",
        "embedding_pooling",
        "embedding_dim",
        "latent_dim",
        "beta_kl_weight",
        "reconstruction_loss",
        "likelihood_variance_assumption",
        "expert_family",
        "condition_type",
        "label_field",
        "label_values_json",
        "class_condition_dim",
    ]
    comparability_pass = 1
    for key in comparability_keys:
        observed = {str(row.get(key, "")) for row in provenance_rows}
        if len(observed) > 1:
            comparability_pass = 0

    primary_prior = _balanced_prior(label_values)
    test_primary_scores = _score_matrix(
        x_cpu=test_x,
        models=models,
        class_prior=primary_prior,
        batch_size=int(batch_size),
        device=device,
    )
    train_primary_scores = _score_matrix(
        x_cpu=train_x,
        models=models,
        class_prior=primary_prior,
        batch_size=int(batch_size),
        device=device,
    )

    family_a_rows: dict[tuple[int, int, int], int] = {}
    imported_rows: list[dict[str, Any]] = []
    if family_a_selection_path is not None and Path(family_a_selection_path).exists():
        with Path(family_a_selection_path).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                method = str(row.get("method", ""))
                if method and method not in {
                    "direct_support_nelbo",
                    "support_set_nelbo_top1",
                    "family_a_direct_support_nelbo_selection",
                }:
                    continue
                target = int(float(row.get("target_domain", row.get("heldout_center", -1))))
                support_seed = int(float(row.get("support_seed", -1)))
                support_size = int(float(row.get("support_size_requested", row.get("support_size", -1))))
                selected = int(float(row.get("selected_expert", -1)))
                family_a_rows[(target, support_seed, support_size)] = selected

    support_sizes = [int(v) for v in cfg.get("support_sizes", [4, 8, 16, 32])]
    support_seeds = [int(v) for v in cfg.get("support_seeds", [17, 23, 31])]
    sampling_policies = [str(v).strip().lower() for v in cfg.get("sampling_policies", ["random"])]
    if sampling_policies != ["random"]:
        raise ValueError("Family C requires random support sampling only; class-balanced support sampling is forbidden")

    support_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []

    for target_domain in sorted(set(int(v) for v in test_domains.tolist())):
        candidate_col_idxs = [i for i, e in enumerate(expert_domains) if int(e) != int(target_domain)]
        candidate_experts = [int(expert_domains[i]) for i in candidate_col_idxs]
        target_indices = [int(i) for i, d in enumerate(test_domains.tolist()) if int(d) == int(target_domain)]
        source_prior = _source_global_laplace_prior(
            train_metadata,
            heldout_domain=int(target_domain),
            label_values=label_values,
            label_field=label_field,
            alpha=laplace_alpha,
        )
        prior_rows.append(
            {
                "target_domain": int(target_domain),
                "label_prior_used_for_routing": "balanced",
                "label_values_json": json.dumps(label_values, sort_keys=True),
                "class_prior_json": json.dumps({str(k): v for k, v in zip(label_values, primary_prior)}, sort_keys=True),
                "allowed_for_primary_routing": 1,
            }
        )
        prior_rows.append(
            {
                "target_domain": int(target_domain),
                "label_prior_used_for_routing": "source_global_laplace",
                "label_values_json": json.dumps(label_values, sort_keys=True),
                "class_prior_json": json.dumps({str(k): v for k, v in zip(label_values, source_prior)}, sort_keys=True),
                "laplace_alpha": float(laplace_alpha),
                "allowed_for_primary_routing": 0,
            }
        )
        test_source_prior_scores = None
        if "source_global_laplace" in sensitivity_priors:
            test_source_prior_scores = _score_matrix(
                x_cpu=test_x,
                models=models,
                class_prior=source_prior,
                batch_size=int(batch_size),
                device=device,
            )

        for support_seed in support_seeds:
            for support_size in support_sizes:
                split = make_support_eval_split(
                    target_domain=int(target_domain),
                    target_indices=target_indices,
                    labels_by_index=labels_by_index,
                    support_size=int(support_size),
                    sampling_policy="random",
                    support_seed=int(support_seed),
                )
                if split.split_status != "ok":
                    continue

                support_scores = np.mean(
                    test_primary_scores[np.asarray(split.support_indices, dtype=np.int64)[:, None], candidate_col_idxs],
                    axis=0,
                )
                eval_scores = np.mean(
                    test_primary_scores[np.asarray(split.eval_indices, dtype=np.int64)[:, None], candidate_col_idxs],
                    axis=0,
                )
                sensitivity_support_scores = None
                sensitivity_eval_scores = None
                if test_source_prior_scores is not None:
                    sensitivity_support_scores = np.mean(
                        test_source_prior_scores[
                            np.asarray(split.support_indices, dtype=np.int64)[:, None],
                            candidate_col_idxs,
                        ],
                        axis=0,
                    )
                    sensitivity_eval_scores = np.mean(
                        test_source_prior_scores[
                            np.asarray(split.eval_indices, dtype=np.int64)[:, None],
                            candidate_col_idxs,
                        ],
                        axis=0,
                    )
                support_embeddings = test_x[split.support_indices].detach().cpu().numpy().astype(np.float64, copy=False)
                support_idx = _stable_argmin(support_scores, candidate_experts)
                sensitivity_idx = (
                    _stable_argmin(sensitivity_support_scores, candidate_experts)
                    if sensitivity_support_scores is not None
                    else -1
                )
                metadata_idx = _metadata_selected(
                    target_domain=int(target_domain),
                    candidate_experts=candidate_experts,
                    strategy=strategy,
                    tau=float(tau),
                )
                random_expert = _stable_random_expert(
                    candidate_experts,
                    seed=int(seed),
                    target_domain=int(target_domain),
                    support_seed=int(support_seed),
                    support_size=int(support_size),
                )
                source_global_idx = _source_global_static_selected(
                    train_score_matrix=train_primary_scores,
                    train_domains=train_domains,
                    expert_domains=expert_domains,
                    candidate_col_idxs=candidate_col_idxs,
                    candidate_experts=candidate_experts,
                    target_domain=int(target_domain),
                )
                embedding_idx = _static_embedding_selected(
                    train_embeddings=train_x.detach().cpu().numpy().astype(np.float64, copy=False),
                    train_domains=train_domains,
                    support_embeddings=support_embeddings,
                    candidate_experts=candidate_experts,
                )

                base = {
                    "target_domain": int(target_domain),
                    "support_seed": int(support_seed),
                    "support_size_requested": int(support_size),
                    "support_size_actual": int(split.support_size_actual),
                    "eval_size": int(split.eval_size),
                    "support_eval_split_id": split.support_eval_split_id,
                    "candidate_experts": "|".join(str(int(e)) for e in candidate_experts),
                    "excluded_experts": str(int(target_domain)),
                    "target_expert_excluded": 1,
                    "support_eval_disjoint": int(set(split.support_indices).isdisjoint(set(split.eval_indices))),
                    "support_labels_used_for_routing": 0,
                    "label_prior_used_for_routing": "balanced",
                    "routing_uses_eval_score": 0,
                    "nelbo_comparability_pass": int(comparability_pass),
                    "available": 1,
                    "selection_source": "not_applicable",
                    "score_name": "label_marginal_nelbo_proxy",
                    "score_direction": "lower_is_better",
                    "spearman_support_eval_score": float(spearman_corr(support_scores.tolist(), eval_scores.tolist())),
                    "support_score_by_expert_json": _json_mapping(candidate_experts, support_scores),
                    "eval_score_by_expert_json": _json_mapping(candidate_experts, eval_scores),
                }
                for expert, support_score, eval_score in zip(candidate_experts, support_scores, eval_scores):
                    support_rows.append(
                        {
                            **base,
                            "expert_domain": int(expert),
                            "support_score": float(support_score),
                            "eval_score": float(eval_score),
                        }
                    )
                if sensitivity_support_scores is not None and sensitivity_eval_scores is not None:
                    for expert, support_score, eval_score in zip(
                        candidate_experts,
                        sensitivity_support_scores,
                        sensitivity_eval_scores,
                    ):
                        support_rows.append(
                            {
                                **base,
                                "label_prior_used_for_routing": "source_global_laplace",
                                "spearman_support_eval_score": float(
                                    spearman_corr(
                                        sensitivity_support_scores.tolist(),
                                        sensitivity_eval_scores.tolist(),
                                    )
                                ),
                                "support_score_by_expert_json": _json_mapping(
                                    candidate_experts,
                                    sensitivity_support_scores,
                                ),
                                "eval_score_by_expert_json": _json_mapping(
                                    candidate_experts,
                                    sensitivity_eval_scores,
                                ),
                                "expert_domain": int(expert),
                                "support_score": float(support_score),
                                "eval_score": float(eval_score),
                            }
                        )
                decision_rows.extend(
                    [
                        _method_row(
                            base=base,
                            method=FAMILY_C_METHOD,
                            selected_expert=int(candidate_experts[support_idx]),
                            candidate_experts=candidate_experts,
                            support_scores=support_scores,
                            eval_scores=eval_scores,
                            selection_source="label_marginal_support_score",
                        ),
                        _method_row(
                            base=base,
                            method="metadata_routing",
                            selected_expert=int(candidate_experts[metadata_idx]),
                            candidate_experts=candidate_experts,
                            support_scores=support_scores,
                            eval_scores=eval_scores,
                            selection_source="metadata_similarity",
                        ),
                        _method_row(
                            base=base,
                            method="random_expert_floor",
                            selected_expert=int(random_expert),
                            candidate_experts=candidate_experts,
                            support_scores=support_scores,
                            eval_scores=eval_scores,
                            selection_source="deterministic_random_floor",
                        ),
                        _method_row(
                            base=base,
                            method="source_global_static_expert",
                            selected_expert=int(candidate_experts[source_global_idx]),
                            candidate_experts=candidate_experts,
                            support_scores=support_scores,
                            eval_scores=eval_scores,
                            selection_source="source_global_static_score",
                        ),
                        _method_row(
                            base=base,
                            method="static_embedding_mean_distance",
                            selected_expert=int(candidate_experts[embedding_idx]),
                            candidate_experts=candidate_experts,
                            support_scores=support_scores,
                            eval_scores=eval_scores,
                            selection_source="support_to_source_train_mean_l2",
                        ),
                    ]
                )
                if sensitivity_support_scores is not None and sensitivity_eval_scores is not None and sensitivity_idx >= 0:
                    sensitivity_base = {
                        **base,
                        "label_prior_used_for_routing": "source_global_laplace",
                        "spearman_support_eval_score": float(
                            spearman_corr(sensitivity_support_scores.tolist(), sensitivity_eval_scores.tolist())
                        ),
                        "support_score_by_expert_json": _json_mapping(candidate_experts, sensitivity_support_scores),
                        "eval_score_by_expert_json": _json_mapping(candidate_experts, sensitivity_eval_scores),
                    }
                    decision_rows.append(
                        _method_row(
                            base=sensitivity_base,
                            method=FAMILY_C_SENSITIVITY_METHOD,
                            selected_expert=int(candidate_experts[sensitivity_idx]),
                            candidate_experts=candidate_experts,
                            support_scores=sensitivity_support_scores,
                            eval_scores=sensitivity_eval_scores,
                            selection_source="label_marginal_support_score_source_global_laplace",
                        )
                    )

                family_a_key = (int(target_domain), int(support_seed), int(support_size))
                imported_selected = family_a_rows.get(family_a_key, -1)
                imported_available = int(imported_selected in set(candidate_experts))
                imported_rows.append(
                    {
                        "heldout_center": int(target_domain),
                        "seed": int(seed),
                        "support_size": int(support_size),
                        "support_seed": int(support_seed),
                        "method": "family_a_direct_support_nelbo_selection",
                        "selected_expert": int(imported_selected) if imported_available else -1,
                        "selection_source": str(family_a_selection_path or ""),
                        "available": int(imported_available),
                    }
                )
                decision_rows.append(
                    _method_row(
                        base=base,
                        method="family_a_direct_support_nelbo_selection",
                        selected_expert=int(imported_selected),
                        candidate_experts=candidate_experts,
                        support_scores=support_scores,
                        eval_scores=eval_scores,
                        available=imported_available,
                        selection_source=str(family_a_selection_path or ""),
                    )
                )
                protocol_rows.append(
                    {
                        **base,
                        "split_status": split.split_status,
                        "class_balanced_support_sampling_rejected": 1,
                        "target_empirical_prior_rejected": 1,
                        "true_label_routing_rejected": 1,
                    }
                )

    _write_csv(reports_dir / "label_marginal_support_nelbo_rows.csv", support_rows)
    _write_csv(reports_dir / "label_marginal_decision_table.csv", decision_rows)
    _write_csv(reports_dir / "label_marginal_class_prior_audit.csv", prior_rows)
    _write_csv(reports_dir / "label_marginal_protocol_audit.csv", protocol_rows)
    _write_csv(reports_dir / "family_c_imported_selection_baselines.csv", imported_rows)

    summary: dict[str, Any] = {
        "experiment_family": "family_c_label_marginal_support_nelbo",
        "status": "DIAGNOSTIC_ONLY",
        "primary_method": FAMILY_C_METHOD,
        "score_name": "label_marginal_nelbo_proxy",
        "score_direction": "lower_is_better",
        "primary_prior": "balanced",
        "sensitivity_priors": sensitivity_priors,
        "label_values": label_values,
        "metrics_by_method": _aggregate(decision_rows),
        "diagnostic_success_rule": (
            "Family C is diagnostic-successful only if it improves top1 oracle hit, "
            "Spearman support/eval ranking, and oracle gap over available Family C-matrix baselines."
        ),
        "family_a_comparison_note": (
            "Family A is compared only when imported selections are available and is evaluated under "
            "the Family C label-marginal eval matrix."
        ),
        "claim_boundary": {
            "allowed": "Label-marginal support scoring is a diagnostic compatibility proxy.",
            "disallowed": "Family C replaces Family A or proves class-conditional generation improves downstream utility.",
        },
        "artifacts": {
            "label_conditioned_checkpoint_provenance": "label_conditioned_checkpoint_provenance.csv",
            "label_marginal_support_nelbo_rows": "label_marginal_support_nelbo_rows.csv",
            "label_marginal_decision_table": "label_marginal_decision_table.csv",
            "label_marginal_class_prior_audit": "label_marginal_class_prior_audit.csv",
            "label_marginal_protocol_audit": "label_marginal_protocol_audit.csv",
            "family_c_imported_selection_baselines": "family_c_imported_selection_baselines.csv",
            "label_marginal_decision_summary": "label_marginal_decision_summary.json",
        },
    }
    with (reports_dir / "label_marginal_decision_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
