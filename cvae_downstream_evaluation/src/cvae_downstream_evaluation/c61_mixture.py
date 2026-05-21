"""C6.1 robust CVAE multi-source mixture downstream evaluation.

This module is post-hoc: it consumes frozen C4.1/C4.2 generators and C5.2
pre-join utility-ranker scores, fixes a mixture decision, and only then scores
target-eval utility. Multi-expert pooled mixtures are always trained in the
shared original DINO embedding space after inverse-projecting each source-local
PCA64 synthetic batch.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .downstream import balanced_accuracy, fit_locked_logistic_classifier, macro_f1
from .matrix import (
    _label,
    _load_embedding_cache,
    _read_samples_manifest,
    _records_for_split,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .schemas import (
    C42_LATENT_GMM_K1_GENERATION_MODE,
    C42_LATENT_GMM_K2_GENERATION_MODE,
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    ENSEMBLE_EXPERT_ID,
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    LATENT_GMM_PRIOR_GENERATOR_FAMILY,
    METHOD_BASELINE_ROW_TYPE,
    POSTERIOR_DECODER_MEAN_GENERATION_MODE,
    SUPPORT_NELBO_METHOD,
)


C61_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c61_cvae_mixture_downstream_v1"
C61_DEFAULT_C41_ROOT = "cvae_downstream_evaluation/artifacts/c41_heteroscedastic_decoder_full_v1"
C61_DEFAULT_C42_ROOT = "cvae_downstream_evaluation/artifacts/c42_latent_gmm_prior_v1"
C61_DEFAULT_C52_ROOT = "cvae_downstream_evaluation/artifacts/c52_utility_rank_router_v1"
SELECTOR_RIDGE_NO_EXPERT = "c52_ridge_no_expert_id_loco_utility_rank_top1"

POLICY_FIXED_HETERO_MEAN = "fixed_all_source_hetero_mean_mixture"
POLICY_FIXED_SAFE = "fixed_all_source_safe_mean_prior_mixture"
POLICY_C52_TOPK_EQUAL = "c52_ridge_no_expert_id_topk_equal_mixture"
POLICY_C52_TOPK_RANK_SOFTMAX = "c52_ridge_no_expert_id_topk_rank_softmax_mixture"
POLICY_LATE_ENSEMBLE = "late_classifier_ensemble_diagnostic_only"

SAFE_MODE_PRIORITY = ("hetero_mean", "gmm_k1", "gmm_k2", "standard_prior")
PRIMARY_POLICIES = (
    POLICY_FIXED_HETERO_MEAN,
    POLICY_FIXED_SAFE,
    POLICY_C52_TOPK_EQUAL,
    POLICY_C52_TOPK_RANK_SOFTMAX,
    POLICY_LATE_ENSEMBLE,
)

MIXTURE_GENERATOR_FAMILY = "family_c_pca64_robust_multi_source_mixture_downstream_v1"
POOLING_SPACE_DINO = "dino_original"
MIN_COMPONENT_BUDGET_PER_CLASS = 8
TOPK_COMPONENTS = 3
RANK_SOFTMAX_TAU = 2.0
RANK_SOFTMAX_MIN_WEIGHT = 0.05
RANK_SOFTMAX_MAX_WEIGHT = 0.40

FAILURE_MIXTURE_NO_GAIN = "MIXTURE_NO_GAIN"
FAILURE_DILUTION = "MIXTURE_DILUTES_STRONG_EXPERTS"
FAILURE_FRAME = "SOURCE_PCA_FRAME_MISMATCH"
FAILURE_TOPK_WEAK = "TOPK_MIXTURE_RANKER_WEAK"
FAILURE_WEIGHTING = "RANK_SOFTMAX_WEIGHTING_OVERFITS"
FAILURE_LATE_ONLY = "LATE_ENSEMBLE_ONLY_HELPS_POOLING_CONFLICT"
FAILURE_COVERAGE = "CVAE_GENERATOR_COVERAGE_INSUFFICIENT"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"
DECISION_SUCCESS = "MIXTURE_SUCCESS"
DECISION_USEFUL = "MIXTURE_USEFUL_BUT_BELOW_080"

PREJOIN_FORBIDDEN_SUBSTRINGS = (
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "oracle",
    "regret",
    "target_eval",
    "target_evaluation",
    "current_heldout_utility",
    "utility_label",
    "true_utility",
)

COMPONENT_COLUMNS = (
    "mixture_policy",
    "policy_role",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed",
    "component_key",
    "source_expert",
    "generation_mode",
    "mode_label",
    "generator_family",
    "allocated_budget_per_class",
    "desired_weight",
    "allocated_weight",
    "projection_artifact_path",
    "projection_artifact_hash",
    "generator_checkpoint_path",
    "generator_checkpoint_hash",
    "latent_prior_artifact_path",
    "latent_prior_artifact_hash",
    "diagnostic_only",
    "pooling_space",
)

DOWNSTREAM_COLUMNS = (
    "mixture_policy",
    "policy_role",
    "diagnostic_only",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed",
    "classifier_seed",
    "generator_family",
    "generation_mode",
    "budget_per_class",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "mixture_ge_080",
    "oracle_bacc_reference",
    "regret_bacc",
    "row_type",
    "n_synthetic_train",
    "n_target_eval",
    "target_eval_pool_id",
    "candidate_expert",
    "candidate_experts_hash",
    "component_keys",
    "num_components",
    "effective_num_components",
    "weight_entropy",
    "max_component_weight",
    "min_component_weight",
    "component_budget_min",
    "component_budget_max",
    "pooling_space",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "status",
    "error_message",
)

DIAGNOSTIC_COLUMNS = (
    "mixture_policy",
    "policy_role",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "generation_seed",
    "num_components",
    "effective_num_components",
    "weight_entropy",
    "max_component_weight",
    "min_component_weight",
    "component_budget_min",
    "component_budget_max",
    "synthetic_count_class_0",
    "synthetic_count_class_1",
    "synthetic_dino_norm_mean",
    "synthetic_dino_norm_std",
    "synthetic_dino_trace_cov",
    "nan_or_inf_generated",
    "pooling_space",
)

SUMMARY_COLUMNS = (
    "mixture_policy",
    "policy_role",
    "diagnostic_only",
    "n_rows",
    "mean_bacc",
    "mixture_ge_080_rate",
    "mean_oracle_bacc_reference",
    "mean_regret_bacc",
    "regret_p50",
    "regret_p75",
    "regret_p90",
    "center_variance",
    "baseline_c41_hetero_mean_bacc",
    "bacc_delta_vs_c41_hetero_mean",
    "baseline_c52_ridge_no_expert_bacc",
    "bacc_delta_vs_c52_ridge_no_expert",
    "strong_center_degrade_gt_002_count",
    "decision_label",
)

PROTOCOL_AUDIT_COLUMNS = (
    "mixture_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "heldout_source_excluded",
    "pooled_in_dino_original",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "c52_prejoin_only",
    "forbidden_prejoin_columns_present",
    "primary_modes_exclude_noise_and_gmm_k4",
    "checkpoints_retrained",
    "protocol_status",
)


@dataclass(frozen=True)
class C61RunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    support_sizes: tuple[int, ...] | None = None
    support_seeds: tuple[int, ...] | None = None
    generation_seeds: tuple[int, ...] | None = None
    classifier_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class BankCandidate:
    mode_label: str
    generator_family: str
    generation_mode: str
    model_kind: str
    latent_gmm_k: int | None = None


@dataclass(frozen=True)
class MixtureComponent:
    source_expert: str
    bank: BankCandidate
    desired_weight: float
    allocated_budget_per_class: int

    @property
    def component_key(self) -> str:
        return f"expert_{self.source_expert}::{self.bank.mode_label}"


@dataclass(frozen=True)
class GeneratedComponent:
    component: MixtureComponent
    synthetic_dino: torch.Tensor
    synthetic_labels: tuple[int, ...]
    projection_path: Path
    checkpoint_path: Path
    latent_prior_paths: tuple[Path, ...]


class _GenerationCache:
    def __init__(
        self,
        *,
        repo_root: Path,
        c41_artifacts_root: Path,
        c42_artifacts_root: Path,
        experiment_seed: int,
        train_embeddings: torch.Tensor,
        train_metadata: Sequence[Mapping[str, object]],
        device: str,
    ) -> None:
        self.repo_root = repo_root
        self.c41_root = c41_artifacts_root
        self.c42_root = c42_artifacts_root
        self.experiment_seed = int(experiment_seed)
        self.train_embeddings = train_embeddings
        self.train_metadata = train_metadata
        self.device = device
        self.projections: dict[str, object] = {}
        self.models: dict[tuple[str, str], object] = {}
        self.reference_pools: dict[str, Mapping[int, torch.Tensor]] = {}
        self.priors: dict[str, object] = {}

    def projection(self, expert: str):
        if expert not in self.projections:
            self.projections[expert] = _load_projection(self.c41_root, self.experiment_seed, expert)
        return self.projections[expert]

    def model(self, expert: str, kind: str):
        from .c41_workstation import _load_c41_model

        key = (expert, kind)
        if key not in self.models:
            self.models[key] = _load_c41_model(
                self.repo_root,
                _checkpoint_path(self.c41_root, self.experiment_seed, expert, kind),
                device=self.device,
            )
        return self.models[key]

    def pools(self, expert: str, label_values: Sequence[int]) -> Mapping[int, torch.Tensor]:
        from .c41_heteroscedastic import build_source_train_reference_pools

        if expert not in self.reference_pools:
            projection = self.projection(expert)
            projected_all = projection.transform(self.train_embeddings)
            self.reference_pools[expert] = build_source_train_reference_pools(
                train_projected_embeddings=projected_all,
                train_metadata=self.train_metadata,
                source_domain=expert,
                label_values=label_values,
            )
        return self.reference_pools[expert]

    def priors_for(self, expert: str):
        if expert not in self.priors:
            self.priors[expert] = _load_c42_priors(self.c42_root, self.experiment_seed, expert)
        return self.priors[expert]


HETERO_MEAN_BANK = BankCandidate(
    mode_label="hetero_mean",
    generator_family=HETEROSCEDASTIC_GENERATOR_FAMILY,
    generation_mode=POSTERIOR_DECODER_MEAN_GENERATION_MODE,
    model_kind="hetero_posterior",
)
STANDARD_PRIOR_BANK = BankCandidate(
    mode_label="standard_prior",
    generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
    generation_mode=C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    model_kind="standard_prior",
)
GMM_K1_BANK = BankCandidate(
    mode_label="gmm_k1",
    generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
    generation_mode=C42_LATENT_GMM_K1_GENERATION_MODE,
    model_kind="latent_gmm",
    latent_gmm_k=1,
)
GMM_K2_BANK = BankCandidate(
    mode_label="gmm_k2",
    generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
    generation_mode=C42_LATENT_GMM_K2_GENERATION_MODE,
    model_kind="latent_gmm",
    latent_gmm_k=2,
)
SAFE_BANK_BY_MODE = {
    bank.mode_label: bank
    for bank in (HETERO_MEAN_BANK, GMM_K1_BANK, GMM_K2_BANK, STANDARD_PRIOR_BANK)
}
SAFE_MODE_BY_GENERATION_MODE = {bank.generation_mode: bank for bank in SAFE_BANK_BY_MODE.values()}


def run_c61_mixture_downstream(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    c52_artifacts_root: Path,
    device: str,
    limits: C61RunLimits = C61RunLimits(),
) -> dict[str, Path]:
    prejoin_scores = load_csv_rows(c52_artifacts_root / "tables" / "c52_predicted_utility_scores_pre_join.csv")
    assert_prejoin_rows_safe(prejoin_scores)
    _assert_no_c61_forbidden_prejoin_columns(prejoin_scores)
    support_unit_rows = load_csv_rows(c41_artifacts_root / "tables" / "support_selection_units.csv")
    c52_oracle = _load_c52_oracle_reference(c52_artifacts_root)
    c41_baseline = _load_c41_baseline(c41_artifacts_root)
    c52_baseline = _load_c52_baseline(c52_artifacts_root)

    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    experiment_seed_filter = set(int(v) for v in limits.experiment_seeds) if limits.experiment_seeds else None

    component_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []

    from .c41_workstation import discover_c41_run_artifacts

    for artifact in discover_c41_run_artifacts(config=config, repo_root=repo_root):
        support = artifact.support
        if experiment_seed_filter is not None and int(support.experiment_seed) not in experiment_seed_filter:
            continue
        samples = _read_samples_manifest(support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(support.train_cache, train_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(support.test_cache, test_records, repo_root=repo_root)
        generation_cache = _GenerationCache(
            repo_root=repo_root,
            c41_artifacts_root=c41_artifacts_root,
            c42_artifacts_root=c42_artifacts_root,
            experiment_seed=int(support.experiment_seed),
            train_embeddings=train_cache.embeddings,
            train_metadata=train_cache.metadata,
            device=device,
        )

        for heldout in selected_heldout:
            heldout = str(heldout)
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            support_conditions = _support_conditions_from_rows(
                rows=support_unit_rows,
                experiment_seed=int(support.experiment_seed),
                heldout_center=heldout,
                support_sizes=limits.support_sizes,
                support_seeds=limits.support_seeds,
            )
            target_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_pool.eval_indices]
            label_values = tuple(sorted(set(target_labels).union({0, 1})))
            if label_values != (0, 1):
                raise ProtocolError(f"C6.1 expects binary labels 0/1, got {label_values}")
            target_dino = test_cache.embeddings[list(target_pool.eval_indices)].detach().cpu().float()

            for support_size, support_seed, support_eval_split_id in support_conditions:
                for policy in PRIMARY_POLICIES:
                    components = build_c61_mixture_components(
                        policy=policy,
                        candidates=candidates,
                        total_budget_per_class=int(config.primary_budget_per_class),
                        experiment_seed=int(support.experiment_seed),
                        heldout_center=heldout,
                        support_size=int(support_size),
                        support_seed=int(support_seed),
                        c52_prejoin_scores=prejoin_scores,
                    )
                    audit = _protocol_audit_row(
                        policy=policy,
                        experiment_seed=int(support.experiment_seed),
                        heldout_center=heldout,
                        support_size=int(support_size),
                        support_seed=int(support_seed),
                        support_eval_split_id=support_eval_split_id,
                        candidates=candidates,
                        components=components,
                    )
                    protocol_rows.append(audit)
                    if not components:
                        continue
                    for generation_seed in selected_generation_seeds:
                        generated = [
                            _generate_component(
                                cache=generation_cache,
                                component=component,
                                label_values=label_values,
                                generation_seed=int(generation_seed),
                            )
                            for component in components
                        ]
                        entropy = mixture_entropy_diagnostics(components)
                        component_rows.extend(
                            _component_rows(
                                policy=policy,
                                experiment_seed=int(support.experiment_seed),
                                heldout_center=heldout,
                                support_size=int(support_size),
                                support_seed=int(support_seed),
                                support_eval_split_id=support_eval_split_id,
                                generation_seed=int(generation_seed),
                                generated=generated,
                                entropy=entropy,
                            )
                        )
                        diagnostic_rows.append(
                            _diagnostic_row(
                                policy=policy,
                                experiment_seed=int(support.experiment_seed),
                                heldout_center=heldout,
                                support_size=int(support_size),
                                support_seed=int(support_seed),
                                generation_seed=int(generation_seed),
                                generated=generated,
                                entropy=entropy,
                            )
                        )
                        for classifier_seed in selected_classifier_seeds:
                            matrix_rows.append(
                                _score_mixture_row(
                                    policy=policy,
                                    experiment_seed=int(support.experiment_seed),
                                    heldout_center=heldout,
                                    support_size=int(support_size),
                                    support_seed=int(support_seed),
                                    support_eval_split_id=support_eval_split_id,
                                    generation_seed=int(generation_seed),
                                    classifier_seed=int(classifier_seed),
                                    budget_per_class=int(config.primary_budget_per_class),
                                    generated=generated,
                                    target_dino=target_dino,
                                    target_labels=target_labels,
                                    target_eval_pool_id=target_pool.target_eval_pool_id,
                                    entropy=entropy,
                                    oracle_reference=c52_oracle.get(
                                        (int(support.experiment_seed), heldout, int(support_size), int(support_seed)),
                                        math.nan,
                                    ),
                                )
                            )

    outputs = {
        "components": artifacts_root / "tables" / "c61_mixture_components_pre_join.csv",
        "matrix": artifacts_root / "tables" / "c61_mixture_downstream_matrix.csv",
        "diagnostics": artifacts_root / "tables" / "c61_mixture_diagnostics.csv",
        "protocol": artifacts_root / "tables" / "c61_protocol_audit.csv",
        "threshold": artifacts_root / "tables" / "c61_threshold_audit.csv",
        "center": artifacts_root / "tables" / "c61_center_summary.csv",
    }
    assert_c61_prejoin_rows_safe(component_rows)
    _write_csv(outputs["components"], COMPONENT_COLUMNS, component_rows)
    _write_csv(outputs["matrix"], DOWNSTREAM_COLUMNS, matrix_rows)
    _write_csv(outputs["diagnostics"], DIAGNOSTIC_COLUMNS, diagnostic_rows)
    _write_csv(outputs["protocol"], PROTOCOL_AUDIT_COLUMNS, protocol_rows)
    _write_csv(outputs["threshold"], SUMMARY_COLUMNS, build_c61_threshold_rows(matrix_rows, c41_baseline, c52_baseline))
    _write_csv(outputs["center"], _center_columns(), build_c61_center_rows(matrix_rows, c41_baseline, c52_baseline))
    return outputs


def build_c61_mixture_components(
    *,
    policy: str,
    candidates: Sequence[str],
    total_budget_per_class: int,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    c52_prejoin_scores: Sequence[Mapping[str, object]] = (),
) -> list[MixtureComponent]:
    experts = tuple(sorted(str(v) for v in candidates if str(v) != str(heldout_center)))
    if not experts:
        raise ProtocolError("C6.1 mixture cannot run without non-heldout source experts.")
    if policy == POLICY_FIXED_HETERO_MEAN:
        return _components_from_banks(experts, (HETERO_MEAN_BANK,), total_budget_per_class)
    if policy == POLICY_FIXED_SAFE or policy == POLICY_LATE_ENSEMBLE:
        banks = tuple(SAFE_BANK_BY_MODE[label] for label in select_safe_mode_prefix(total_budget_per_class, len(experts)))
        return _components_from_banks(experts, banks, total_budget_per_class)
    if policy in {POLICY_C52_TOPK_EQUAL, POLICY_C52_TOPK_RANK_SOFTMAX}:
        rows = _c52_rows_for_condition(
            c52_prejoin_scores,
            experiment_seed=experiment_seed,
            heldout_center=heldout_center,
            support_size=support_size,
            support_seed=support_seed,
        )
        if not rows:
            return []
        picked = _topk_c52_components(rows, experts, TOPK_COMPONENTS)
        if policy == POLICY_C52_TOPK_EQUAL:
            weights = {key: 1.0 / len(picked) for key in picked}
        else:
            raw = {key: math.exp(-float(rank) / RANK_SOFTMAX_TAU) for key, rank in picked.items()}
            weights = clip_and_normalize_weights(
                raw,
                min_weight=RANK_SOFTMAX_MIN_WEIGHT,
                max_weight=RANK_SOFTMAX_MAX_WEIGHT,
            )
        allocation = allocate_weighted_budget_per_class(
            total_per_class=total_budget_per_class,
            weights=weights,
        )
        _assert_component_budget_floor(allocation.values())
        return [
            MixtureComponent(
                source_expert=expert,
                bank=SAFE_MODE_BY_GENERATION_MODE[mode],
                desired_weight=float(weights[(expert, mode)]),
                allocated_budget_per_class=int(allocation[(expert, mode)]),
            )
            for expert, mode in sorted(picked)
        ]
    raise ProtocolError(f"Unknown C6.1 mixture policy: {policy}")


def select_safe_mode_prefix(total_budget_per_class: int, n_experts: int) -> tuple[str, ...]:
    if int(n_experts) <= 0:
        raise ProtocolError("n_experts must be positive.")
    selected: list[str] = []
    for label in SAFE_MODE_PRIORITY:
        candidate = [*selected, label]
        if int(total_budget_per_class) // (int(n_experts) * len(candidate)) < MIN_COMPONENT_BUDGET_PER_CLASS:
            break
        selected = candidate
    if not selected:
        raise ProtocolError(
            "C6.1 budget is too small for even all-source hetero_mean mixture: "
            f"budget={total_budget_per_class}, n_experts={n_experts}"
        )
    return tuple(selected)


def allocate_weighted_budget_per_class(
    *,
    total_per_class: int,
    weights: Mapping[object, float],
) -> dict[object, int]:
    if int(total_per_class) <= 0:
        raise ProtocolError("total_per_class must be positive.")
    if not weights:
        raise ProtocolError("Cannot allocate budget without mixture weights.")
    normalized = _normalize_weights(weights)
    floors = {key: int(math.floor(float(value) * int(total_per_class))) for key, value in normalized.items()}
    remainder = int(total_per_class) - sum(floors.values())
    fractions = sorted(
        ((float(normalized[key]) * int(total_per_class) - floors[key], _stable_component_sort_key(key), key) for key in normalized),
        key=lambda item: (-item[0], item[1]),
    )
    out = dict(floors)
    for _fraction, _sort_key, key in fractions[:remainder]:
        out[key] += 1
    return out


def clip_and_normalize_weights(
    weights: Mapping[object, float],
    *,
    min_weight: float,
    max_weight: float,
) -> dict[object, float]:
    if not weights:
        raise ProtocolError("Cannot normalize empty weights.")
    if float(min_weight) * len(weights) > 1.0 + 1.0e-12 or float(max_weight) * len(weights) < 1.0 - 1.0e-12:
        raise ProtocolError("Weight clipping bounds are infeasible for this component count.")
    current = _normalize_weights(weights)
    fixed: dict[object, float] = {}
    free = set(current)
    for _ in range(len(current) + 2):
        high = [key for key in free if current[key] > float(max_weight)]
        low = [key for key in free if current[key] < float(min_weight)]
        changed = bool(high or low)
        if high:
            key = max(high, key=lambda item: (current[item], _stable_component_sort_key(item)))
            fixed[key] = float(max_weight)
            free.remove(key)
        elif low:
            key = min(low, key=lambda item: (current[item], _stable_component_sort_key(item)))
            fixed[key] = float(min_weight)
            free.remove(key)
        remaining = 1.0 - sum(fixed.values())
        if not free:
            break
        free_mass = sum(float(weights[key]) for key in free)
        if free_mass <= 0.0:
            even = remaining / float(len(free))
            for key in free:
                current[key] = even
        else:
            for key in free:
                current[key] = remaining * float(weights[key]) / free_mass
        if not changed and all(float(min_weight) <= current[key] <= float(max_weight) for key in free):
            break
    out = {**fixed, **{key: current[key] for key in free}}
    return _normalize_weights(out)


def mixture_entropy_diagnostics(components: Sequence[MixtureComponent]) -> dict[str, float]:
    budgets = [float(component.allocated_budget_per_class) for component in components]
    total = sum(budgets)
    weights = [value / total for value in budgets] if total > 0.0 else []
    entropy = -sum(value * math.log(max(value, 1.0e-12)) for value in weights)
    return {
        "num_components": float(len(components)),
        "effective_num_components": float(math.exp(entropy)) if weights else math.nan,
        "weight_entropy": float(entropy) if weights else math.nan,
        "max_component_weight": max(weights) if weights else math.nan,
        "min_component_weight": min(weights) if weights else math.nan,
        "component_budget_min": min(budgets) if budgets else math.nan,
        "component_budget_max": max(budgets) if budgets else math.nan,
    }


def assert_c61_prejoin_rows_safe(rows: Sequence[Mapping[str, object]]) -> None:
    bad = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if any(token in str(key).lower() for token in PREJOIN_FORBIDDEN_SUBSTRINGS)
        }
    )
    if bad:
        raise ProtocolError(f"C6.1 pre-join component rows contain forbidden utility/eval columns: {bad}")


def build_c61_threshold_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    c41_baseline_by_center: Mapping[str, float],
    c52_baseline_by_center: Mapping[str, float],
) -> list[dict[str, object]]:
    c41_baseline = _mean(c41_baseline_by_center.values())
    c52_baseline = _mean(c52_baseline_by_center.values())
    out = []
    for policy in sorted({str(row["mixture_policy"]) for row in matrix_rows}):
        subset = [row for row in matrix_rows if str(row["mixture_policy"]) == policy and str(row.get("status")) == "ok"]
        values = [_float(row["bacc"]) for row in subset]
        regrets = [_float(row.get("regret_bacc")) for row in subset]
        center_means = [
            _mean(_float(row["bacc"]) for row in subset if str(row["heldout_center"]) == center)
            for center in sorted({str(row["heldout_center"]) for row in subset})
        ]
        mean_bacc = _mean(values)
        row = {
            "mixture_policy": policy,
            "policy_role": _policy_role(policy),
            "diagnostic_only": int(policy == POLICY_LATE_ENSEMBLE),
            "n_rows": len(subset),
            "mean_bacc": mean_bacc,
            "mixture_ge_080_rate": _mean(1.0 if value >= 0.80 else 0.0 for value in values),
            "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
            "mean_regret_bacc": _mean(regrets),
            "regret_p50": _quantile(regrets, 0.50),
            "regret_p75": _quantile(regrets, 0.75),
            "regret_p90": _quantile(regrets, 0.90),
            "center_variance": statistics.pvariance(center_means) if len(center_means) > 1 else 0.0,
            "baseline_c41_hetero_mean_bacc": c41_baseline,
            "bacc_delta_vs_c41_hetero_mean": mean_bacc - c41_baseline if not math.isnan(c41_baseline) else math.nan,
            "baseline_c52_ridge_no_expert_bacc": c52_baseline,
            "bacc_delta_vs_c52_ridge_no_expert": mean_bacc - c52_baseline if not math.isnan(c52_baseline) else math.nan,
            "strong_center_degrade_gt_002_count": _strong_center_degrade_count(subset, c41_baseline_by_center),
            "decision_label": "",
        }
        row["decision_label"] = _decision_label(row)
        out.append(row)
    return out


def build_c61_center_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    c41_baseline_by_center: Mapping[str, float],
    c52_baseline_by_center: Mapping[str, float],
) -> list[dict[str, object]]:
    out = []
    for policy in sorted({str(row["mixture_policy"]) for row in matrix_rows}):
        for center in sorted({str(row["heldout_center"]) for row in matrix_rows}):
            subset = [
                row
                for row in matrix_rows
                if str(row["mixture_policy"]) == policy
                and str(row["heldout_center"]) == center
                and str(row.get("status")) == "ok"
            ]
            if not subset:
                continue
            mean_bacc = _mean(_float(row["bacc"]) for row in subset)
            out.append(
                {
                    "mixture_policy": policy,
                    "heldout_center": center,
                    "n_rows": len(subset),
                    "mean_bacc": mean_bacc,
                    "mixture_ge_080_rate": _mean(1.0 if _float(row["bacc"]) >= 0.80 else 0.0 for row in subset),
                    "c41_hetero_mean_bacc": _float(c41_baseline_by_center.get(center)),
                    "delta_vs_c41_hetero_mean": mean_bacc - _float(c41_baseline_by_center.get(center)),
                    "c52_ridge_no_expert_bacc": _float(c52_baseline_by_center.get(center)),
                    "delta_vs_c52_ridge_no_expert": mean_bacc - _float(c52_baseline_by_center.get(center)),
                    "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
                    "regret_p75": _quantile([_float(row.get("regret_bacc")) for row in subset], 0.75),
                }
            )
    return out


def _components_from_banks(
    experts: Sequence[str],
    banks: Sequence[BankCandidate],
    total_budget_per_class: int,
) -> list[MixtureComponent]:
    keys = [(expert, bank.generation_mode) for expert in sorted(experts) for bank in banks]
    weights = {key: 1.0 / len(keys) for key in keys}
    allocation = allocate_weighted_budget_per_class(total_per_class=total_budget_per_class, weights=weights)
    _assert_component_budget_floor(allocation.values())
    return [
        MixtureComponent(
            source_expert=expert,
            bank=SAFE_MODE_BY_GENERATION_MODE[mode],
            desired_weight=float(weights[(expert, mode)]),
            allocated_budget_per_class=int(allocation[(expert, mode)]),
        )
        for expert, mode in keys
    ]


def _assert_component_budget_floor(values: Iterable[int]) -> None:
    clean = [int(value) for value in values]
    if clean and min(clean) < MIN_COMPONENT_BUDGET_PER_CLASS:
        raise ProtocolError(
            "C6.1 component budget fragmentation below minimum: "
            f"min={min(clean)}, required={MIN_COMPONENT_BUDGET_PER_CLASS}"
        )


def _generate_component(
    *,
    cache: _GenerationCache,
    component: MixtureComponent,
    label_values: Sequence[int],
    generation_seed: int,
) -> GeneratedComponent:
    from .c41_heteroscedastic import generate_posterior_sampled_embeddings
    from .c42_latent_gmm import generate_latent_gmm_decoder_mean, generate_standard_prior_decoder_mean

    torch = _torch_module()
    projection = cache.projection(component.source_expert)
    kind = "heteroscedastic" if component.bank.model_kind == "hetero_posterior" else "plain"
    model = cache.model(component.source_expert, kind)
    chunks = []
    labels = []
    latent_paths: list[Path] = []
    for label in label_values:
        n = int(component.allocated_budget_per_class)
        if component.bank.model_kind == "hetero_posterior":
            pools = cache.pools(component.source_expert, label_values)
            generated = generate_posterior_sampled_embeddings(
                model=model,
                reference_pool=pools[int(label)].to(next(model.parameters()).device),
                class_label=int(label),
                n_samples=n,
                seed=int(generation_seed) + int(label),
                generation_mode=component.bank.generation_mode,
            )
        elif component.bank.model_kind == "standard_prior":
            generated = generate_standard_prior_decoder_mean(
                model=model,
                class_label=int(label),
                n_samples=n,
                seed=int(generation_seed) + int(label),
            )
        elif component.bank.model_kind == "latent_gmm":
            priors = cache.priors_for(component.source_expert)
            k = int(component.bank.latent_gmm_k or 0)
            prior = priors[k][int(label)]
            latent_paths.append(_latent_prior_path(cache.c42_root, cache.experiment_seed, component.source_expert, int(label), k))
            generated = generate_latent_gmm_decoder_mean(
                model=model,
                prior=prior,
                class_label=int(label),
                n_samples=n,
                seed=int(generation_seed) + int(label),
                generation_mode=component.bank.generation_mode,
            )
        else:
            raise ProtocolError(f"Unsupported C6.1 bank model kind: {component.bank.model_kind}")
        chunks.append(generated.embeddings.detach().cpu().float())
        labels.extend(int(v) for v in generated.labels.detach().cpu().tolist())
        synthetic_pca = torch.cat(chunks, dim=0)
    synthetic_dino = projection.inverse_transform(synthetic_pca).detach().cpu().float()
    projection_path = cache.c41_root / "projections" / f"seed{cache.experiment_seed}" / f"expert_{component.source_expert}" / "pca64.pt"
    checkpoint_path = _checkpoint_path(cache.c41_root, cache.experiment_seed, component.source_expert, kind)
    return GeneratedComponent(
        component=component,
        synthetic_dino=synthetic_dino,
        synthetic_labels=tuple(labels),
        projection_path=projection_path,
        checkpoint_path=checkpoint_path,
        latent_prior_paths=tuple(latent_paths),
    )


def _score_mixture_row(
    *,
    policy: str,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    generation_seed: int,
    classifier_seed: int,
    budget_per_class: int,
    generated: Sequence[GeneratedComponent],
    target_dino: torch.Tensor,
    target_labels: Sequence[int],
    target_eval_pool_id: str,
    entropy: Mapping[str, float],
    oracle_reference: float,
) -> dict[str, object]:
    try:
        if policy == POLICY_LATE_ENSEMBLE:
            bacc, macro, auroc, auprc = _late_ensemble_score(
                generated=generated,
                target_dino=target_dino,
                target_labels=target_labels,
                classifier_seed=classifier_seed,
            )
        else:
            torch = _torch_module()
            synthetic = torch.cat([item.synthetic_dino for item in generated], dim=0)
            labels = [label for item in generated for label in item.synthetic_labels]
            prediction = fit_locked_logistic_classifier(
                _to_numpy(synthetic),
                labels,
                _to_numpy(target_dino),
                target_labels,
                classifier_seed=classifier_seed,
            )
            bacc = float(prediction.score.balanced_accuracy)
            macro = float(prediction.score.macro_f1)
            auroc = float(prediction.score.secondary_metrics.get("auroc", math.nan))
            auprc = float(prediction.score.secondary_metrics.get("auprc", math.nan))
        total_train = sum(len(item.synthetic_labels) for item in generated)
        status = "ok"
        error = ""
    except Exception as exc:
        bacc = macro = auroc = auprc = math.nan
        total_train = sum(len(item.synthetic_labels) for item in generated)
        status = "failed_c61_mixture_scoring"
        error = str(exc)
    return {
        "mixture_policy": policy,
        "policy_role": _policy_role(policy),
        "diagnostic_only": int(policy == POLICY_LATE_ENSEMBLE),
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "generation_seed": int(generation_seed),
        "classifier_seed": int(classifier_seed),
        "generator_family": MIXTURE_GENERATOR_FAMILY,
        "generation_mode": policy,
        "budget_per_class": int(budget_per_class),
        "bacc": bacc,
        "macro_f1": macro,
        "auroc": auroc,
        "auprc": auprc,
        "mixture_ge_080": int(not math.isnan(bacc) and bacc >= 0.80),
        "oracle_bacc_reference": oracle_reference,
        "regret_bacc": oracle_reference - bacc if not math.isnan(float(oracle_reference)) and not math.isnan(float(bacc)) else math.nan,
        "row_type": METHOD_BASELINE_ROW_TYPE,
        "n_synthetic_train": int(total_train),
        "n_target_eval": len(target_labels),
        "target_eval_pool_id": target_eval_pool_id,
        "candidate_expert": ENSEMBLE_EXPERT_ID,
        "candidate_experts_hash": hash_candidate_experts([item.component.component_key for item in generated]),
        "component_keys": ";".join(item.component.component_key for item in generated),
        **{key: entropy.get(key, math.nan) for key in entropy},
        "pooling_space": POOLING_SPACE_DINO,
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "status": status,
        "error_message": error,
    }


def _late_ensemble_score(
    *,
    generated: Sequence[GeneratedComponent],
    target_dino: torch.Tensor,
    target_labels: Sequence[int],
    classifier_seed: int,
) -> tuple[float, float, float, float]:
    import numpy as np  # type: ignore
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

    probabilities = []
    class_order: tuple[int, ...] | None = None
    for item in generated:
        prediction = fit_locked_logistic_classifier(
            _to_numpy(item.synthetic_dino),
            item.synthetic_labels,
            _to_numpy(target_dino),
            target_labels,
            classifier_seed=classifier_seed,
        )
        if class_order is None:
            class_order = tuple(prediction.classes)
        elif tuple(prediction.classes) != class_order:
            raise ProtocolError("C6.1 late ensemble class order mismatch.")
        probabilities.append(np.asarray(prediction.probabilities, dtype=float))
    averaged = np.mean(np.stack(probabilities, axis=0), axis=0)
    pred = [int(class_order[int(idx)]) for idx in np.argmax(averaged, axis=1)] if class_order else []
    y_true = [int(v) for v in target_labels]
    auroc = auprc = math.nan
    if class_order and len(class_order) == 2 and averaged.shape[1] == 2:
        try:
            auroc = float(roc_auc_score(y_true, averaged[:, 1]))
        except ValueError:
            auroc = math.nan
        try:
            auprc = float(average_precision_score(y_true, averaged[:, 1]))
        except ValueError:
            auprc = math.nan
    return balanced_accuracy(y_true, pred), macro_f1(y_true, pred), auroc, auprc


def _component_rows(
    *,
    policy: str,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    generation_seed: int,
    generated: Sequence[GeneratedComponent],
    entropy: Mapping[str, float],
) -> list[dict[str, object]]:
    rows = []
    total_budget = sum(item.component.allocated_budget_per_class for item in generated)
    for item in generated:
        latent_paths = item.latent_prior_paths
        rows.append(
            {
                "mixture_policy": policy,
                "policy_role": _policy_role(policy),
                "experiment_seed": int(experiment_seed),
                "heldout_center": heldout_center,
                "support_size": int(support_size),
                "support_seed": int(support_seed),
                "support_eval_split_id": support_eval_split_id,
                "generation_seed": int(generation_seed),
                "component_key": item.component.component_key,
                "source_expert": item.component.source_expert,
                "generation_mode": item.component.bank.generation_mode,
                "mode_label": item.component.bank.mode_label,
                "generator_family": item.component.bank.generator_family,
                "allocated_budget_per_class": int(item.component.allocated_budget_per_class),
                "desired_weight": float(item.component.desired_weight),
                "allocated_weight": float(item.component.allocated_budget_per_class) / max(float(total_budget), 1.0),
                "projection_artifact_path": str(item.projection_path),
                "projection_artifact_hash": _file_hash(item.projection_path),
                "generator_checkpoint_path": str(item.checkpoint_path),
                "generator_checkpoint_hash": _file_hash(item.checkpoint_path),
                "latent_prior_artifact_path": ";".join(str(path) for path in latent_paths),
                "latent_prior_artifact_hash": ";".join(_file_hash(path) for path in latent_paths),
                "diagnostic_only": int(policy == POLICY_LATE_ENSEMBLE),
                "pooling_space": POOLING_SPACE_DINO,
            }
        )
    _ = entropy
    return rows


def _diagnostic_row(
    *,
    policy: str,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    generation_seed: int,
    generated: Sequence[GeneratedComponent],
    entropy: Mapping[str, float],
) -> dict[str, object]:
    torch = _torch_module()
    synthetic = torch.cat([item.synthetic_dino for item in generated], dim=0)
    labels = [label for item in generated for label in item.synthetic_labels]
    return {
        "mixture_policy": policy,
        "policy_role": _policy_role(policy),
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "generation_seed": int(generation_seed),
        **{key: entropy.get(key, math.nan) for key in entropy},
        "synthetic_count_class_0": sum(1 for label in labels if int(label) == 0),
        "synthetic_count_class_1": sum(1 for label in labels if int(label) == 1),
        "synthetic_dino_norm_mean": float(synthetic.norm(dim=1).mean().item()),
        "synthetic_dino_norm_std": float(synthetic.norm(dim=1).std(unbiased=False).item()),
        "synthetic_dino_trace_cov": _trace_cov(synthetic),
        "nan_or_inf_generated": int(not torch.isfinite(synthetic).all().item()),
        "pooling_space": POOLING_SPACE_DINO,
    }


def _protocol_audit_row(
    *,
    policy: str,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    candidates: Sequence[str],
    components: Sequence[MixtureComponent],
) -> dict[str, object]:
    safe_modes = {item.bank.mode_label for item in components}
    pass_status = (
        str(heldout_center) not in {str(v) for v in candidates}
        and all(item.bank.mode_label in SAFE_MODE_PRIORITY for item in components)
    )
    return {
        "mixture_policy": policy,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "heldout_source_excluded": int(str(heldout_center) not in {str(v) for v in candidates}),
        "pooled_in_dino_original": 1,
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "c52_prejoin_only": int(policy not in {POLICY_C52_TOPK_EQUAL, POLICY_C52_TOPK_RANK_SOFTMAX} or bool(components)),
        "forbidden_prejoin_columns_present": 0,
        "primary_modes_exclude_noise_and_gmm_k4": int("hetero_noise" not in safe_modes and "gmm_k4" not in safe_modes),
        "checkpoints_retrained": 0,
        "protocol_status": "pass" if pass_status else "fail",
    }


def _c52_rows_for_condition(
    rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if str(row.get("selector_name")) == SELECTOR_RIDGE_NO_EXPERT
        and int(row.get("experiment_seed", -1)) == int(experiment_seed)
        and str(row.get("heldout_center")) == str(heldout_center)
        and int(row.get("support_size", -1)) == int(support_size)
        and int(row.get("support_seed", -1)) == int(support_seed)
        and str(row.get("generation_mode")) in SAFE_MODE_BY_GENERATION_MODE
    ]


def _topk_c52_components(
    rows: Sequence[Mapping[str, object]],
    experts: Sequence[str],
    top_k: int,
) -> dict[tuple[str, str], int]:
    allowed_experts = {str(v) for v in experts}
    ranked = sorted(
        [
            row
            for row in rows
            if str(row.get("candidate_expert")) in allowed_experts
            and str(row.get("generation_mode")) in SAFE_MODE_BY_GENERATION_MODE
        ],
        key=lambda row: (_float(row.get("predicted_rank_within_unit")), str(row.get("candidate_expert")), str(row.get("generation_mode"))),
    )
    out: dict[tuple[str, str], int] = {}
    for row in ranked:
        key = (str(row["candidate_expert"]), str(row["generation_mode"]))
        if key in out:
            continue
        out[key] = int(_float(row.get("predicted_rank_within_unit")))
        if len(out) >= int(top_k):
            break
    return out


def _support_conditions_from_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    experiment_seed: int,
    heldout_center: str,
    support_sizes: tuple[int, ...] | None,
    support_seeds: tuple[int, ...] | None,
) -> tuple[tuple[int, int, str], ...]:
    size_filter = set(int(v) for v in support_sizes) if support_sizes else None
    seed_filter = set(int(v) for v in support_seeds) if support_seeds else None
    out = []
    for row in rows:
        if str(row.get("method")) != SUPPORT_NELBO_METHOD:
            continue
        if int(row.get("experiment_seed", -1)) != int(experiment_seed):
            continue
        if str(row.get("heldout_center")) != str(heldout_center):
            continue
        support_size = int(row.get("support_size", 0))
        support_seed = int(row.get("support_seed", 0))
        if size_filter is not None and support_size not in size_filter:
            continue
        if seed_filter is not None and support_seed not in seed_filter:
            continue
        out.append((support_size, support_seed, str(row.get("support_eval_split_id", ""))))
    return tuple(sorted(set(out)))


def _latent_prior_path(c42_root: Path, experiment_seed: int, expert: str, label: int, k: int) -> Path:
    return c42_root / "latent_priors" / f"seed{int(experiment_seed)}" / f"expert_{expert}" / f"class_{int(label)}" / f"gmm_k{int(k)}.pt"


def _checkpoint_path(c41_root: Path, experiment_seed: int, candidate_expert: str, kind: str) -> Path:
    filename = "plain_class_conditional_pca64.pt" if kind == "plain" else "heteroscedastic_class_conditional_pca64.pt"
    return c41_root / "checkpoints" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / kind / filename


def _load_projection(c41_root: Path, experiment_seed: int, candidate_expert: str):
    torch = _torch_module()
    path = c41_root / "projections" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / "pca64.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_c42_priors(c42_root: Path, experiment_seed: int, candidate_expert: str):
    from .c42_latent_gmm import SourceClassLatentDiagGMM

    out = {}
    for k in (1, 2):
        out[k] = {}
        for label in (0, 1):
            path = _latent_prior_path(c42_root, experiment_seed, candidate_expert, label, k)
            if not path.exists():
                raise ProtocolError(f"Missing C4.2 latent prior for C6.1: {path}")
            payload = _torch_load(path)
            out[k][label] = SourceClassLatentDiagGMM.from_payload(payload)
    return out


def _load_c52_oracle_reference(c52_root: Path) -> dict[tuple[int, str, int, int], float]:
    path = c52_root / "tables" / "c52_selected_route_utility_join.csv"
    if not path.exists():
        return {}
    rows = load_csv_rows(path)
    out: dict[tuple[int, str, int, int], list[float]] = {}
    for row in rows:
        if str(row.get("selector_name")) != SELECTOR_RIDGE_NO_EXPERT:
            continue
        key = (int(row["experiment_seed"]), str(row["heldout_center"]), int(row["support_size"]), int(row["support_seed"]))
        out.setdefault(key, []).append(_float(row.get("oracle_bacc_mean")))
    return {key: _mean(values) for key, values in out.items()}


def _load_c52_baseline(c52_root: Path) -> dict[str, float]:
    path = c52_root / "tables" / "c52_selected_route_utility_join.csv"
    if not path.exists():
        return {}
    rows = [
        row
        for row in load_csv_rows(path)
        if str(row.get("selector_name")) == SELECTOR_RIDGE_NO_EXPERT
        and str(row.get("protocol_status", "pass")) == "pass"
    ]
    return {
        center: _mean(_float(row.get("selected_bacc_mean")) for row in rows if str(row.get("heldout_center")) == center)
        for center in sorted({str(row.get("heldout_center")) for row in rows})
    }


def _load_c41_baseline(c41_root: Path) -> dict[str, float]:
    path = c41_root / "tables" / "routing_to_downstream_alignment.csv"
    if not path.exists():
        return {}
    rows = [
        row
        for row in load_csv_rows(path)
        if str(row.get("method")) == SUPPORT_NELBO_METHOD
        and str(row.get("generator_family")) == HETEROSCEDASTIC_GENERATOR_FAMILY
        and str(row.get("generation_mode")) == POSTERIOR_DECODER_MEAN_GENERATION_MODE
    ]
    return {
        center: _mean(_float(row.get("selected_bacc")) for row in rows if str(row.get("heldout_center")) == center)
        for center in sorted({str(row.get("heldout_center")) for row in rows})
    }


def _decision_label(row: Mapping[str, object]) -> str:
    mean_bacc = _float(row.get("mean_bacc"))
    delta_c52 = _float(row.get("bacc_delta_vs_c52_ridge_no_expert"))
    regret_p75 = _float(row.get("regret_p75"))
    strong_degrades = int(row.get("strong_center_degrade_gt_002_count") or 0)
    if mean_bacc >= 0.80 and strong_degrades == 0:
        return DECISION_SUCCESS
    if mean_bacc >= 0.77 or delta_c52 >= 0.03 or (not math.isnan(regret_p75) and regret_p75 <= 0.05):
        return DECISION_USEFUL
    if "topk" in str(row.get("mixture_policy")) and delta_c52 <= 0.0:
        return FAILURE_TOPK_WEAK
    if "rank_softmax" in str(row.get("mixture_policy")) and delta_c52 <= 0.0:
        return FAILURE_WEIGHTING
    return FAILURE_MIXTURE_NO_GAIN


def _strong_center_degrade_count(rows: Sequence[Mapping[str, object]], baseline: Mapping[str, float]) -> int:
    count = 0
    for center, base in baseline.items():
        base_value = _float(base)
        if math.isnan(base_value) or base_value < 0.80:
            continue
        current = _mean(_float(row.get("bacc")) for row in rows if str(row.get("heldout_center")) == str(center))
        if not math.isnan(current) and current < base_value - 0.02:
            count += 1
    return count


def _policy_role(policy: str) -> str:
    if policy == POLICY_FIXED_HETERO_MEAN:
        return "primary_fixed_safe_mode"
    if policy == POLICY_FIXED_SAFE:
        return "primary_fixed_safe_mean_prior"
    if policy in {POLICY_C52_TOPK_EQUAL, POLICY_C52_TOPK_RANK_SOFTMAX}:
        return "secondary_c52_topk"
    return "diagnostic"


def _normalize_weights(weights: Mapping[object, float]) -> dict[object, float]:
    clean = {key: max(float(value), 0.0) for key, value in weights.items()}
    total = sum(clean.values())
    if total <= 0.0:
        even = 1.0 / float(len(clean))
        return {key: even for key in clean}
    return {key: value / total for key, value in clean.items()}


def _stable_component_sort_key(key: object) -> str:
    if isinstance(key, tuple):
        return "::".join(str(part) for part in key)
    return str(key)


def _trace_cov(x: torch.Tensor) -> float:
    x = x.detach().cpu().float()
    if int(x.shape[0]) < 2:
        return 0.0
    centered = x - x.mean(dim=0, keepdim=True)
    return float(centered.pow(2).sum(dim=0).sum().item() / float(x.shape[0] - 1))


def _torch_module():
    import torch  # type: ignore

    return torch


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _assert_no_c61_forbidden_prejoin_columns(rows: Sequence[Mapping[str, object]]) -> None:
    bad = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if str(key)
            not in {
                "current_heldout_utility_visible_before_selection",
                "target_support_labels_used",
                "target_eval_labels_used_for_selection",
            }
            and any(token in str(key).lower() for token in PREJOIN_FORBIDDEN_SUBSTRINGS)
        }
    )
    if bad:
        raise ProtocolError(f"C6.1 C5.2 pre-join score input has forbidden columns: {bad}")


def assert_prejoin_rows_safe(rows: Sequence[Mapping[str, object]]) -> None:
    bad = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if str(key)
            not in {
                "current_heldout_utility_visible_before_selection",
                "target_support_labels_used",
                "target_eval_labels_used_for_selection",
            }
            and any(
                token in str(key).lower()
                for token in (
                    "bacc",
                    "macro_f1",
                    "auroc",
                    "auprc",
                    "oracle",
                    "target_eval",
                    "target_evaluation",
                    "downstream",
                    "utility_label",
                    "true_utility",
                    "current_heldout_utility",
                )
            )
        }
    )
    if bad:
        raise ProtocolError(f"C6.1 pre-join C5.2 rows contain forbidden utility/eval columns: {bad}")


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _torch_load(path: Path):
    torch = _torch_module()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _center_columns() -> tuple[str, ...]:
    return (
        "mixture_policy",
        "heldout_center",
        "n_rows",
        "mean_bacc",
        "mixture_ge_080_rate",
        "c41_hetero_mean_bacc",
        "delta_vs_c41_hetero_mean",
        "c52_ridge_no_expert_bacc",
        "delta_vs_c52_ridge_no_expert",
        "mean_oracle_bacc_reference",
        "regret_p75",
    )


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return sum(clean) / float(len(clean)) if clean else math.nan


def _quantile(values: Iterable[float], q: float) -> float:
    clean = sorted(float(value) for value in values if not math.isnan(float(value)))
    if not clean:
        return math.nan
    pos = (len(clean) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] + ((clean[hi] - clean[lo]) * (pos - lo))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
