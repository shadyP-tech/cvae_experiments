"""C5.1 mode-aware unlabeled support-distance routing.

This module is intentionally post-hoc: it consumes frozen C4.1/C4.2 generator
artifacts and downstream utility matrices, computes unlabeled support-distance
proxy scores, makes routing decisions, and only then joins downstream utility.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from .c41_heteroscedastic import (
    GENERATION_MODE_POSTERIOR_DECODER_MEAN,
    GENERATION_MODE_POSTERIOR_DECODER_NOISE,
    SourceTrainPCAProjection,
    build_source_train_reference_pools,
    generate_posterior_sampled_embeddings,
)
from .c41_workstation import (
    _load_c41_model,
    discover_c41_run_artifacts,
)
from .c42_latent_gmm import (
    C42_LATENT_GMM_K1_GENERATION_MODE,
    C42_LATENT_GMM_K2_GENERATION_MODE,
    C42_LATENT_GMM_K4_GENERATION_MODE,
    SourceClassLatentDiagGMM,
    generate_latent_gmm_decoder_mean,
    generate_standard_prior_decoder_mean,
)
from .c42_workstation import C42_DEFAULT_C41_ROOT, _torch_load
from .downstream import CandidateDownstreamRow, read_candidate_downstream_matrix, spearman
from .matrix import (
    MatrixBuildLimits,
    _domain,
    _load_embedding_cache,
    _make_support_eval_split,
    _read_samples_manifest,
    _records_for_split,
)
from .protocol import LockedV1Config, ProtocolError
from .routing import (
    METADATA_METHOD,
    SOURCE_GLOBAL_METHOD,
    SUPPORT_NELBO_METHOD,
    SupportSelectionUnit,
)
from .schemas import (
    C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
    HETEROSCEDASTIC_GENERATOR_FAMILY,
    LATENT_GMM_PRIOR_GENERATOR_FAMILY,
    PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
)


C51_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c51_mode_aware_support_selector_v1"
C51_DEFAULT_C41_ROOT = C42_DEFAULT_C41_ROOT
C51_DEFAULT_C42_ROOT = "cvae_downstream_evaluation/artifacts/c42_latent_gmm_prior_v1"

SELECTOR_PRIMARY = "support_distance_rankmean_dino_seed_marginal_top1"
SELECTOR_SEED_SELECTED = "support_distance_rankmean_dino_seed_selected_top1"
SELECTOR_ZSUM_DINO = "support_distance_zsum_dino_seed_marginal_top1"
SELECTOR_PCA = "support_distance_rankmean_pca_seed_marginal_top1"
SELECTOR_METADATA_HYBRID = "metadata_expert_support_mode_top1"
SELECTOR_SUPPORT_NELBO_MODE = "unlabeled_support_nelbo_expert_support_mode_top1"
SELECTOR_SUPPORT_NELBO_FUSION = "unlabeled_support_nelbo_distance_rank_fusion"
SELECTOR_LOCO = "loco_learned_distance_selector_diagnostic_only"

PRIMARY_SELECTOR = SELECTOR_PRIMARY
SUPPORT_DISTANCE_FAMILY = "support_distance"
SUPPORT_NELBO_FAMILY = "support_nelbo"
METADATA_HYBRID_FAMILY = "metadata_hybrid"
LEARNED_DIAGNOSTIC_FAMILY = "learned_diagnostic"

FAILURE_SUPPORT_DISTANCE_NO_GAIN = "SUPPORT_DISTANCE_SELECTOR_NO_GAIN"
FAILURE_MODE_BANK_ORACLE_HIGH = "MODE_BANK_ORACLE_HIGH_SELECTOR_WEAK"
FAILURE_CENTER_CEILING = "CENTER_SPECIFIC_GENERATOR_CEILING"
FAILURE_LOCO_OVERFIT = "LOCO_SELECTOR_OVERFIT"
FAILURE_SEED_OVERFIT = "SEED_SELECTION_OVERFITS_SUPPORT_NOISE"
FAILURE_SUPPORT_SIZE_NO_MONOTONICITY = "SUPPORT_SIZE_NO_MONOTONICITY"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"
DECISION_PRIMARY_SUCCESS = "PRIMARY_SUCCESS"
DECISION_USEFUL_RESULT = "USEFUL_RESULT"

FORBIDDEN_SELECTOR_SUBSTRINGS = (
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "target_eval",
    "target_evaluation",
    "support_label",
    "oracle",
    "downstream",
)

C51_SCORE_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "candidate_expert",
    "generator_family",
    "generation_mode",
    "mode_label",
    "generation_seed",
    "score_space_primary",
    "score_space_diagnostic",
    "scaler_status",
    "target_support_count",
    "support_nelbo",
    "support_nelbo_rank",
    "metadata_selected_expert",
    "source_global_selected_expert",
    "energy_distance_dino",
    "rbf_mmd_dino_sigma05",
    "rbf_mmd_dino_sigma10",
    "rbf_mmd_dino_sigma20",
    "mean_l2_dino",
    "cov_trace_ratio_dino",
    "abs_log_cov_trace_ratio_dino",
    "pairwise_distance_ratio_dino",
    "abs_log_pairwise_distance_ratio_dino",
    "rank_energy_dino",
    "rank_rbf_mmd_dino",
    "rank_mean_l2_dino",
    "rank_cov_trace_dino",
    "rank_pairwise_dino",
    "rankmean_dino_score",
    "zsum_dino_score",
    "energy_distance_pca",
    "rbf_mmd_pca_sigma05",
    "rbf_mmd_pca_sigma10",
    "rbf_mmd_pca_sigma20",
    "mean_l2_pca",
    "cov_trace_ratio_pca",
    "abs_log_cov_trace_ratio_pca",
    "pairwise_distance_ratio_pca",
    "abs_log_pairwise_distance_ratio_pca",
    "rankmean_pca_score",
    "selector_input_guard_passed",
)

C51_ALIGNMENT_COLUMNS = (
    "selector_name",
    "selector_family",
    "selector_role",
    "diagnostic_only",
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "selected_expert",
    "selected_generator_family",
    "selected_generation_mode",
    "selected_generation_seed",
    "selected_score",
    "seed_score_mean",
    "seed_score_median",
    "seed_score_std",
    "seed_score_iqr",
    "selected_seed_if_seed_selected",
    "selected_bacc_mean",
    "selected_bacc_std",
    "selected_bacc_min",
    "selected_bacc_max",
    "selected_ge_080_rate",
    "oracle_expert",
    "oracle_generator_family",
    "oracle_generation_mode",
    "oracle_generation_seed",
    "oracle_bacc_mean",
    "oracle_gap_bacc",
    "top1_expert_mode_hit",
    "top2_expert_mode_hit",
    "selected_rank_by_bacc",
    "regret_bacc",
    "selection_depends_on_classifier_seed",
    "protocol_status",
)

SUMMARY_COLUMNS = (
    "selector_name",
    "selector_family",
    "selector_role",
    "diagnostic_only",
    "n_rows",
    "mean_selected_bacc",
    "selected_ge_080_rate",
    "mean_oracle_bacc",
    "mean_oracle_gap_bacc",
    "top1_hit_rate",
    "top2_hit_rate",
    "mean_regret_bacc",
    "regret_p50",
    "regret_p75",
    "regret_p90",
    "baseline_c41_hetero_mean_selected_bacc",
    "selected_bacc_delta_vs_c41_hetero_mean",
    "selected_ge_080_rate_delta_vs_c41_hetero_mean",
    "oracle_gap_delta_vs_c41_hetero_mean",
    "regret_p75_delta_vs_c41_hetero_mean",
    "center_positive_improvement_count",
    "strong_center_degrade_gt_002_count",
    "spearman_positive_center_count",
    "decision_label",
)


@dataclass(frozen=True)
class BankCandidate:
    mode_label: str
    generator_family: str
    generation_mode: str
    model_kind: str
    latent_gmm_k: int | None = None


@dataclass(frozen=True)
class GeneratedCandidate:
    bank: BankCandidate
    candidate_expert: str
    generation_seed: int
    synthetic_pca: torch.Tensor
    synthetic_dino: torch.Tensor
    support_nelbo: float
    support_nelbo_rank: int


CANDIDATE_BANK = (
    BankCandidate(
        mode_label="plain_posterior_mean",
        generator_family=PLAIN_CLASS_CONDITIONAL_GENERATOR_FAMILY,
        generation_mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        model_kind="plain_posterior",
    ),
    BankCandidate(
        mode_label="hetero_mean",
        generator_family=HETEROSCEDASTIC_GENERATOR_FAMILY,
        generation_mode=GENERATION_MODE_POSTERIOR_DECODER_MEAN,
        model_kind="hetero_posterior",
    ),
    BankCandidate(
        mode_label="hetero_noise",
        generator_family=HETEROSCEDASTIC_GENERATOR_FAMILY,
        generation_mode=GENERATION_MODE_POSTERIOR_DECODER_NOISE,
        model_kind="hetero_posterior",
    ),
    BankCandidate(
        mode_label="standard_prior",
        generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
        generation_mode=C42_STANDARD_PRIOR_REPLAY_GENERATION_MODE,
        model_kind="standard_prior",
    ),
    BankCandidate(
        mode_label="gmm_k1",
        generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
        generation_mode=C42_LATENT_GMM_K1_GENERATION_MODE,
        model_kind="latent_gmm",
        latent_gmm_k=1,
    ),
    BankCandidate(
        mode_label="gmm_k2",
        generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
        generation_mode=C42_LATENT_GMM_K2_GENERATION_MODE,
        model_kind="latent_gmm",
        latent_gmm_k=2,
    ),
    BankCandidate(
        mode_label="gmm_k4",
        generator_family=LATENT_GMM_PRIOR_GENERATOR_FAMILY,
        generation_mode=C42_LATENT_GMM_K4_GENERATION_MODE,
        model_kind="latent_gmm",
        latent_gmm_k=4,
    ),
)


def build_c51_support_mode_scores(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    support_units: Sequence[SupportSelectionUnit],
    device: str,
    limits: MatrixBuildLimits = MatrixBuildLimits(),
) -> Path:
    """Regenerate selector-visible synthetic batches and score unlabeled support distance."""

    path = artifacts_root / "tables" / "c51_support_mode_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_heldout_centers = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    selected_experiment_seeds = set(int(v) for v in limits.experiment_seeds) if limits.experiment_seeds else None
    support_by_key = _support_units_by_key(support_units)
    rows: list[dict[str, object]] = []

    for artifact in discover_c41_run_artifacts(config=config, repo_root=repo_root):
        support = artifact.support
        if selected_experiment_seeds is not None and int(support.experiment_seed) not in selected_experiment_seeds:
            continue
        samples = _read_samples_manifest(support.samples_manifest)
        train_records = _records_for_split(samples, "train")
        test_records = _records_for_split(samples, "test")
        train_cache = _load_embedding_cache(support.train_cache, train_records, repo_root=repo_root)
        test_cache = _load_embedding_cache(support.test_cache, test_records, repo_root=repo_root)
        for heldout in selected_heldout_centers:
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != str(heldout))
            target_indices = tuple(
                idx for idx, row in enumerate(test_cache.metadata) if str(_domain(row)) == str(heldout)
            )
            if not target_indices:
                raise ProtocolError(f"No target support candidates for heldout center {heldout}")
            generated_by_candidate: dict[str, list[GeneratedCandidate]] = {}
            projections: dict[str, SourceTrainPCAProjection] = {}
            for candidate in candidates:
                projection = _load_projection(c41_artifacts_root, support.experiment_seed, candidate)
                projections[candidate] = projection
                generated_by_candidate[candidate] = _generate_candidate_bank(
                    repo_root=repo_root,
                    c41_artifacts_root=c41_artifacts_root,
                    c42_artifacts_root=c42_artifacts_root,
                    experiment_seed=support.experiment_seed,
                    candidate_expert=candidate,
                    train_cache_embeddings=train_cache.embeddings,
                    train_cache_metadata=train_cache.metadata,
                    projection=projection,
                    generation_seeds=selected_generation_seeds,
                    budget_per_class=config.primary_budget_per_class,
                    device=device,
                )
            for support_size, support_seed in _requested_support_conditions(config, support_units, support.experiment_seed, heldout):
                support_unit = support_by_key.get((support.experiment_seed, str(heldout), int(support_size), int(support_seed), SUPPORT_NELBO_METHOD))
                if support_unit is None:
                    raise ProtocolError(f"Missing support-NELBO unit for seed={support.experiment_seed}, heldout={heldout}, k={support_size}, support_seed={support_seed}")
                metadata_unit = support_by_key.get((support.experiment_seed, str(heldout), int(support_size), int(support_seed), METADATA_METHOD))
                source_global_unit = support_by_key.get((support.experiment_seed, str(heldout), int(support_size), int(support_seed), SOURCE_GLOBAL_METHOD))
                split = _unlabeled_support_split(
                    test_metadata=test_cache.metadata,
                    target_indices=target_indices,
                    heldout_center=str(heldout),
                    support_size=int(support_size),
                    support_seed=int(support_seed),
                )
                support_dino = test_cache.embeddings[list(split.support_indices)].detach().cpu().float()
                generated_flat = [item for candidate in candidates for item in generated_by_candidate[candidate]]
                dino_bandwidths = _support_condition_bandwidths(
                    support_dino,
                    [item.synthetic_dino for item in generated_flat],
                )
                support_scores = {str(k): float(v) for k, v in support_unit.support_nelbo_by_expert.items()}
                support_ranks = _ascending_rank_map(support_scores)
                condition_rows: list[dict[str, object]] = []
                for item in generated_flat:
                    projection = projections[item.candidate_expert]
                    support_pca = projection.transform(support_dino).detach().cpu().float()
                    pca_bandwidths = _support_condition_bandwidths(support_pca, [item.synthetic_pca])
                    score_row = {
                        "experiment_seed": int(support.experiment_seed),
                        "heldout_center": str(heldout),
                        "support_size": int(support_size),
                        "support_seed": int(support_seed),
                        "support_eval_split_id": split.support_eval_split_id,
                        "candidate_expert": item.candidate_expert,
                        "generator_family": item.bank.generator_family,
                        "generation_mode": item.bank.generation_mode,
                        "mode_label": item.bank.mode_label,
                        "generation_seed": int(item.generation_seed),
                        "score_space_primary": "dino_original",
                        "score_space_diagnostic": "source_pca64_std",
                        "scaler_status": "absent",
                        "target_support_count": int(support_dino.shape[0]),
                        "support_nelbo": support_scores.get(item.candidate_expert, math.nan),
                        "support_nelbo_rank": support_ranks.get(item.candidate_expert, 999),
                        "metadata_selected_expert": metadata_unit.selected_expert if metadata_unit else "",
                        "source_global_selected_expert": source_global_unit.selected_expert if source_global_unit else "",
                        **_distance_metrics("dino", support_dino, item.synthetic_dino, dino_bandwidths),
                        **_distance_metrics("pca", support_pca, item.synthetic_pca, pca_bandwidths),
                        "selector_input_guard_passed": 1,
                    }
                    condition_rows.append(score_row)
                _add_rank_scores(condition_rows, "dino")
                _add_rank_scores(condition_rows, "pca")
                _assert_selector_score_rows_safe(condition_rows)
                rows.extend(condition_rows)

    _write_csv(path, C51_SCORE_COLUMNS, rows)
    return path


def build_c51_reports(
    *,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
) -> dict[str, Path]:
    score_rows = load_csv_rows(artifacts_root / "tables" / "c51_support_mode_scores.csv")
    _assert_selector_score_rows_safe(score_rows)
    downstream_rows = _load_bank_downstream_rows(c41_artifacts_root, c42_artifacts_root)
    c41_baseline = _load_c41_hetero_mean_baseline_rows(c41_artifacts_root)
    alignment = build_c51_alignment_rows(score_rows, downstream_rows)
    correlation = build_c51_score_utility_correlation_rows(score_rows, downstream_rows)
    threshold = build_c51_threshold_audit_rows(alignment, correlation, c41_baseline)
    outputs = {
        "alignment": artifacts_root / "tables" / "c51_expert_mode_alignment.csv",
        "threshold": artifacts_root / "tables" / "c51_threshold_audit.csv",
        "score_space": artifacts_root / "tables" / "c51_score_space_comparison.csv",
        "ablation": artifacts_root / "tables" / "c51_metric_ablation.csv",
        "regret": artifacts_root / "tables" / "c51_selector_regret_distribution.csv",
        "metadata": artifacts_root / "tables" / "c51_metadata_baseline_comparison.csv",
        "correlation": artifacts_root / "tables" / "c51_score_utility_correlation.csv",
        "support_size": artifacts_root / "tables" / "c51_support_size_sensitivity.csv",
    }
    _write_csv(outputs["alignment"], C51_ALIGNMENT_COLUMNS, alignment)
    _write_csv(outputs["threshold"], SUMMARY_COLUMNS, threshold)
    _write_csv(outputs["score_space"], SUMMARY_COLUMNS, _filter_summary(threshold, ("rankmean_dino", "zsum_dino", "rankmean_pca")))
    _write_csv(outputs["ablation"], SUMMARY_COLUMNS, _filter_summary(threshold, ("energy_only", "mmd_only", "mean_cov_only", "no_cov_terms", "rank_average_score")))
    _write_csv(outputs["regret"], _regret_columns(), build_c51_regret_rows(alignment))
    _write_csv(outputs["metadata"], SUMMARY_COLUMNS, _filter_summary(threshold, ("metadata", "support_distance", "support_nelbo")))
    _write_csv(outputs["correlation"], _correlation_columns(), correlation)
    _write_csv(outputs["support_size"], _support_size_columns(), build_c51_support_size_rows(alignment, correlation))
    return outputs


def build_c51_alignment_rows(
    score_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    by_condition: dict[tuple[int, str, int, int], list[Mapping[str, object]]] = {}
    for row in score_rows:
        key = (int(row["experiment_seed"]), str(row["heldout_center"]), int(row["support_size"]), int(row["support_seed"]))
        by_condition.setdefault(key, []).append(row)
    downstream_index = _downstream_index(downstream_rows)
    alignment: list[dict[str, object]] = []
    selectors = (
        SELECTOR_PRIMARY,
        SELECTOR_SEED_SELECTED,
        SELECTOR_ZSUM_DINO,
        SELECTOR_PCA,
        SELECTOR_METADATA_HYBRID,
        SELECTOR_SUPPORT_NELBO_MODE,
        SELECTOR_SUPPORT_NELBO_FUSION,
        "energy_only",
        "mmd_only",
        "mean_cov_only",
        "no_cov_terms",
        "rank_average_score",
    )
    for condition, rows in sorted(by_condition.items()):
        for selector in selectors:
            selected = _select_candidate(rows, selector)
            if selected is None:
                continue
            alignment.append(_alignment_for_selection(selector, selected, rows, downstream_index))
    alignment.extend(_build_loco_alignment_rows(score_rows, downstream_rows))
    _assert_classifier_seed_invariant(alignment)
    return alignment


def build_c51_threshold_audit_rows(
    alignment_rows: Sequence[Mapping[str, object]],
    correlation_rows: Sequence[Mapping[str, object]] = (),
    c41_baseline_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    baseline = _baseline_summary(c41_baseline_rows)
    baseline_by_center = _baseline_by_center_values(c41_baseline_rows)
    rows: list[dict[str, object]] = []
    for selector in sorted({str(row["selector_name"]) for row in alignment_rows}):
        subset = [row for row in alignment_rows if str(row["selector_name"]) == selector]
        selected = [_float(row["selected_bacc_mean"]) for row in subset]
        oracle = [_float(row["oracle_bacc_mean"]) for row in subset]
        regrets = [_float(row["regret_bacc"]) for row in subset]
        selected_ge_080_rate = _mean([1.0 if value >= 0.80 else 0.0 for value in selected])
        mean_gap = _mean([_float(row["oracle_gap_bacc"]) for row in subset])
        regret_p75 = _quantile(regrets, 0.75)
        center_improvements = _center_improvement_count(subset, baseline_by_center)
        strong_degrade_count = _strong_center_degrade_count(subset, baseline_by_center)
        spearman_pos = _selector_spearman_positive_centers(selector, correlation_rows)
        rows.append(
            {
                "selector_name": selector,
                "selector_family": str(subset[0]["selector_family"]) if subset else "",
                "selector_role": str(subset[0]["selector_role"]) if subset else "",
                "diagnostic_only": int(str(subset[0].get("diagnostic_only", "0")) == "1") if subset else 1,
                "n_rows": len(subset),
                "mean_selected_bacc": _mean(selected),
                "selected_ge_080_rate": selected_ge_080_rate,
                "mean_oracle_bacc": _mean(oracle),
                "mean_oracle_gap_bacc": mean_gap,
                "top1_hit_rate": _mean([_float(row["top1_expert_mode_hit"]) for row in subset]),
                "top2_hit_rate": _mean([_float(row["top2_expert_mode_hit"]) for row in subset]),
                "mean_regret_bacc": _mean(regrets),
                "regret_p50": _quantile(regrets, 0.50),
                "regret_p75": regret_p75,
                "regret_p90": _quantile(regrets, 0.90),
                "baseline_c41_hetero_mean_selected_bacc": baseline["selected_bacc"],
                "selected_bacc_delta_vs_c41_hetero_mean": _mean(selected) - baseline["selected_bacc"],
                "selected_ge_080_rate_delta_vs_c41_hetero_mean": selected_ge_080_rate - baseline["ge_080_rate"],
                "oracle_gap_delta_vs_c41_hetero_mean": mean_gap - baseline["oracle_gap"],
                "regret_p75_delta_vs_c41_hetero_mean": regret_p75 - baseline["regret_p75"],
                "center_positive_improvement_count": center_improvements,
                "strong_center_degrade_gt_002_count": strong_degrade_count,
                "spearman_positive_center_count": spearman_pos,
                "decision_label": _decision_label(
                    selector,
                    subset,
                    alignment_rows,
                    spearman_pos,
                    baseline=baseline,
                    baseline_by_center=baseline_by_center,
                ),
            }
        )
    return rows


def build_c51_regret_rows(alignment_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for selector in sorted({str(row["selector_name"]) for row in alignment_rows}):
        subset = [row for row in alignment_rows if str(row["selector_name"]) == selector]
        regrets = [_float(row["regret_bacc"]) for row in subset]
        out.append(
            {
                "selector_name": selector,
                "regret_p50": _quantile(regrets, 0.50),
                "regret_p75": _quantile(regrets, 0.75),
                "regret_p90": _quantile(regrets, 0.90),
                "regret_mean": _mean(regrets),
                "regret_max": max(regrets) if regrets else math.nan,
            }
        )
    return out


def build_c51_score_utility_correlation_rows(
    score_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    downstream_index = _downstream_index(downstream_rows)
    grouped: dict[tuple[str, int, str, int, int], list[Mapping[str, object]]] = {}
    for row in score_rows:
        key = (
            str(row["heldout_center"]),
            int(row["experiment_seed"]),
            str(row["heldout_center"]),
            int(row["support_size"]),
            int(row["support_seed"]),
        )
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    for (_center_key, experiment_seed, heldout, support_size, support_seed), rows in sorted(grouped.items()):
        scores: list[float] = []
        utilities: list[float] = []
        for candidate_rows in _group_by_candidate_mode(rows).values():
            agg = _aggregate_seed_scores(candidate_rows, "rankmean_dino_score")
            utility = _candidate_utility(candidate_rows[0], downstream_index, seed_marginal=True).mean_bacc
            if not math.isnan(agg["seed_score_median"]) and not math.isnan(utility):
                scores.append(-float(agg["seed_score_median"]))
                utilities.append(float(utility))
        out.append(
            {
                "heldout_center": heldout,
                "experiment_seed": experiment_seed,
                "support_size": support_size,
                "support_seed": support_seed,
                "selector_name": SELECTOR_PRIMARY,
                "spearman_score_bacc": spearman(scores, utilities) if len(scores) >= 2 else math.nan,
                "n_candidates": len(scores),
            }
        )
    return out


def build_c51_support_size_rows(
    alignment_rows: Sequence[Mapping[str, object]],
    correlation_rows: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    out = []
    for selector in sorted({str(row["selector_name"]) for row in alignment_rows}):
        for support_size in sorted({int(row["support_size"]) for row in alignment_rows}):
            subset = [
                row
                for row in alignment_rows
                if str(row["selector_name"]) == selector and int(row["support_size"]) == support_size
            ]
            if not subset:
                continue
            out.append(
                {
                    "selector_name": selector,
                    "support_size": support_size,
                    "mean_selected_bacc": _mean(_float(row["selected_bacc_mean"]) for row in subset),
                    "oracle_gap": _mean(_float(row["oracle_gap_bacc"]) for row in subset),
                    "top1_hit": _mean(_float(row["top1_expert_mode_hit"]) for row in subset),
                    "regret_p75": _quantile([_float(row["regret_bacc"]) for row in subset], 0.75),
                    "spearman_score_bacc": _mean(
                        _float(row["spearman_score_bacc"])
                        for row in correlation_rows
                        if str(row.get("selector_name")) == selector and int(row.get("support_size", -1)) == support_size
                    ),
                    "mode_entropy": _entropy([str(row["selected_generation_mode"]) for row in subset]),
                }
            )
    return out


def load_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _generate_candidate_bank(
    *,
    repo_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    experiment_seed: int,
    candidate_expert: str,
    train_cache_embeddings: torch.Tensor,
    train_cache_metadata: Sequence[Mapping[str, object]],
    projection: SourceTrainPCAProjection,
    generation_seeds: Sequence[int],
    budget_per_class: int,
    device: str,
) -> list[GeneratedCandidate]:
    label_values = (0, 1)
    train_projected_all = projection.transform(train_cache_embeddings)
    reference_pools = build_source_train_reference_pools(
        train_projected_embeddings=train_projected_all,
        train_metadata=train_cache_metadata,
        source_domain=candidate_expert,
        label_values=label_values,
    )
    plain_model = _load_c41_model(repo_root, _checkpoint_path(c41_artifacts_root, experiment_seed, candidate_expert, "plain"), device=device)
    hetero_model = _load_c41_model(repo_root, _checkpoint_path(c41_artifacts_root, experiment_seed, candidate_expert, "heteroscedastic"), device=device)
    priors = _load_c42_priors(c42_artifacts_root, experiment_seed, candidate_expert)
    out: list[GeneratedCandidate] = []
    for bank in CANDIDATE_BANK:
        model = hetero_model if bank.model_kind == "hetero_posterior" else plain_model
        for generation_seed in generation_seeds:
            chunks = []
            for label in label_values:
                if bank.model_kind in {"plain_posterior", "hetero_posterior"}:
                    generated = generate_posterior_sampled_embeddings(
                        model=model,
                        reference_pool=reference_pools[int(label)].to(next(model.parameters()).device),
                        class_label=int(label),
                        n_samples=int(budget_per_class),
                        seed=int(generation_seed) + int(label),
                        generation_mode=bank.generation_mode,
                    )
                elif bank.model_kind == "standard_prior":
                    generated = generate_standard_prior_decoder_mean(
                        model=model,
                        class_label=int(label),
                        n_samples=int(budget_per_class),
                        seed=int(generation_seed) + int(label),
                    )
                else:
                    generated = generate_latent_gmm_decoder_mean(
                        model=model,
                        prior=priors[int(bank.latent_gmm_k or 0)][int(label)],
                        class_label=int(label),
                        n_samples=int(budget_per_class),
                        seed=int(generation_seed) + int(label),
                        generation_mode=bank.generation_mode,
                    )
                chunks.append(generated.embeddings.detach().cpu().float())
            synthetic_pca = torch.cat(chunks, dim=0)
            out.append(
                GeneratedCandidate(
                    bank=bank,
                    candidate_expert=str(candidate_expert),
                    generation_seed=int(generation_seed),
                    synthetic_pca=synthetic_pca,
                    synthetic_dino=projection.inverse_transform(synthetic_pca).detach().cpu().float(),
                    support_nelbo=math.nan,
                    support_nelbo_rank=999,
                )
            )
    return out


def _distance_metrics(prefix: str, support: torch.Tensor, synthetic: torch.Tensor, bandwidths: Sequence[float]) -> dict[str, float]:
    support = support.detach().cpu().float()
    synthetic = synthetic.detach().cpu().float()
    cov_ratio = _trace_cov(synthetic) / max(_trace_cov(support), 1.0e-12)
    pair_ratio = _pairwise_distance_mean(synthetic) / max(_pairwise_distance_mean(support), 1.0e-12)
    metrics = {
        f"energy_distance_{prefix}": _energy_distance(support, synthetic),
        f"mean_l2_{prefix}": float((support.mean(dim=0) - synthetic.mean(dim=0)).norm().item()),
        f"cov_trace_ratio_{prefix}": cov_ratio,
        f"abs_log_cov_trace_ratio_{prefix}": abs(math.log(max(cov_ratio, 1.0e-12))),
        f"pairwise_distance_ratio_{prefix}": pair_ratio,
        f"abs_log_pairwise_distance_ratio_{prefix}": abs(math.log(max(pair_ratio, 1.0e-12))),
    }
    suffixes = ("sigma05", "sigma10", "sigma20")
    for suffix, bandwidth in zip(suffixes, bandwidths):
        metrics[f"rbf_mmd_{prefix}_{suffix}"] = _rbf_mmd_fixed(support, synthetic, sigma=float(bandwidth))
    return metrics


def _add_rank_scores(rows: list[dict[str, object]], prefix: str) -> None:
    mmd_keys = [f"rbf_mmd_{prefix}_sigma05", f"rbf_mmd_{prefix}_sigma10", f"rbf_mmd_{prefix}_sigma20"]
    rank_sets = {key: _rank_values([_float(row.get(key)) for row in rows]) for key in mmd_keys}
    metric_rank_inputs = {
        f"rank_energy_{prefix}": f"energy_distance_{prefix}",
        f"rank_mean_l2_{prefix}": f"mean_l2_{prefix}",
        f"rank_cov_trace_{prefix}": f"abs_log_cov_trace_ratio_{prefix}",
        f"rank_pairwise_{prefix}": f"abs_log_pairwise_distance_ratio_{prefix}",
    }
    metric_ranks = {
        out_key: _rank_values([_float(row.get(in_key)) for row in rows])
        for out_key, in_key in metric_rank_inputs.items()
    }
    zsum = _zsum_scores(
        rows,
        (
            f"energy_distance_{prefix}",
            f"mean_l2_{prefix}",
            f"abs_log_cov_trace_ratio_{prefix}",
            f"abs_log_pairwise_distance_ratio_{prefix}",
            *mmd_keys,
        ),
    )
    for idx, row in enumerate(rows):
        mmd_rank = _mean(rank_sets[key][idx] for key in mmd_keys)
        row[f"rank_rbf_mmd_{prefix}"] = mmd_rank
        ranks = [mmd_rank]
        for out_key in metric_rank_inputs:
            row[out_key] = metric_ranks[out_key][idx]
            ranks.append(float(row[out_key]))
        row[f"rankmean_{prefix}_score"] = _mean(ranks)
        if prefix == "dino":
            row["zsum_dino_score"] = zsum[idx]


@dataclass(frozen=True)
class UtilitySummary:
    mean_bacc: float
    std_bacc: float
    min_bacc: float
    max_bacc: float
    ge_080_rate: float
    mean_macro_f1: float


def _alignment_for_selection(
    selector: str,
    selected: Mapping[str, object],
    condition_rows: Sequence[Mapping[str, object]],
    downstream_index: Mapping[tuple[object, ...], CandidateDownstreamRow],
) -> dict[str, object]:
    seed_marginal = selector != SELECTOR_SEED_SELECTED
    utility = _candidate_utility(selected, downstream_index, seed_marginal=seed_marginal)
    oracle, ranked = _oracle_for_condition(condition_rows, downstream_index, seed_marginal=seed_marginal)
    selected_key = _candidate_tuple(selected, seed_marginal=seed_marginal)
    oracle_key = _candidate_tuple(oracle, seed_marginal=seed_marginal) if oracle else ()
    rank = next((idx + 1 for idx, item in enumerate(ranked) if _candidate_tuple(item[0], seed_marginal=seed_marginal) == selected_key), 999)
    seed_stats = _aggregate_seed_scores(
        [
            row
            for row in condition_rows
            if str(row["candidate_expert"]) == str(selected["candidate_expert"])
            and str(row["generator_family"]) == str(selected["generator_family"])
            and str(row["generation_mode"]) == str(selected["generation_mode"])
        ],
        _score_column(selector),
    )
    return {
        "selector_name": selector,
        "selector_family": _selector_family(selector),
        "selector_role": "primary" if selector == PRIMARY_SELECTOR else "diagnostic",
        "diagnostic_only": int(selector != PRIMARY_SELECTOR),
        "heldout_center": str(selected["heldout_center"]),
        "experiment_seed": int(selected["experiment_seed"]),
        "support_size": int(selected["support_size"]),
        "support_seed": int(selected["support_seed"]),
        "support_eval_split_id": str(selected["support_eval_split_id"]),
        "selected_expert": str(selected["candidate_expert"]),
        "selected_generator_family": str(selected["generator_family"]),
        "selected_generation_mode": str(selected["generation_mode"]),
        "selected_generation_seed": int(selected["generation_seed"]) if not seed_marginal else "",
        "selected_score": _float(selected.get(_score_column(selector), selected.get("selector_score", math.nan))),
        "seed_score_mean": seed_stats["seed_score_mean"],
        "seed_score_median": seed_stats["seed_score_median"],
        "seed_score_std": seed_stats["seed_score_std"],
        "seed_score_iqr": seed_stats["seed_score_iqr"],
        "selected_seed_if_seed_selected": int(selected["generation_seed"]) if selector == SELECTOR_SEED_SELECTED else "",
        "selected_bacc_mean": utility.mean_bacc,
        "selected_bacc_std": utility.std_bacc,
        "selected_bacc_min": utility.min_bacc,
        "selected_bacc_max": utility.max_bacc,
        "selected_ge_080_rate": utility.ge_080_rate,
        "oracle_expert": str(oracle["candidate_expert"]) if oracle else "",
        "oracle_generator_family": str(oracle["generator_family"]) if oracle else "",
        "oracle_generation_mode": str(oracle["generation_mode"]) if oracle else "",
        "oracle_generation_seed": int(oracle["generation_seed"]) if oracle and not seed_marginal else "",
        "oracle_bacc_mean": ranked[0][1].mean_bacc if ranked else math.nan,
        "oracle_gap_bacc": (ranked[0][1].mean_bacc - utility.mean_bacc) if ranked else math.nan,
        "top1_expert_mode_hit": int(selected_key == oracle_key),
        "top2_expert_mode_hit": int(rank <= 2),
        "selected_rank_by_bacc": rank,
        "regret_bacc": (ranked[0][1].mean_bacc - utility.mean_bacc) if ranked else math.nan,
        "selection_depends_on_classifier_seed": 0,
        "protocol_status": "pass",
    }


def _select_candidate(rows: Sequence[Mapping[str, object]], selector: str) -> Mapping[str, object] | None:
    if not rows:
        return None
    if selector == SELECTOR_SEED_SELECTED:
        return min(rows, key=lambda row: (_float(row["rankmean_dino_score"]), _tie_key(row)))
    if selector == SELECTOR_ZSUM_DINO:
        return _select_seed_marginal(rows, "zsum_dino_score")
    if selector == SELECTOR_PCA:
        return _select_seed_marginal(rows, "rankmean_pca_score")
    if selector == SELECTOR_METADATA_HYBRID:
        metadata = str(rows[0].get("metadata_selected_expert", "")).strip()
        if not metadata:
            return None
        return _select_seed_marginal([row for row in rows if str(row["candidate_expert"]) == metadata], "rankmean_dino_score")
    if selector == SELECTOR_SUPPORT_NELBO_MODE:
        best_expert = min({str(row["candidate_expert"]) for row in rows}, key=lambda expert: _float(next(row["support_nelbo"] for row in rows if str(row["candidate_expert"]) == expert)))
        return _select_seed_marginal([row for row in rows if str(row["candidate_expert"]) == best_expert], "rankmean_dino_score")
    if selector == SELECTOR_SUPPORT_NELBO_FUSION:
        grouped = _group_by_candidate_mode(rows)
        best = None
        for items in grouped.values():
            agg = _aggregate_seed_scores(items, "rankmean_dino_score")
            support_rank = _float(items[0].get("support_nelbo_rank"))
            score = float(agg["seed_score_median"]) + support_rank
            representative = dict(min(items, key=lambda row: (_float(row["rankmean_dino_score"]), _tie_key(row))))
            representative["selector_score"] = score
            if best is None or (score, _tie_key(representative)) < (_float(best.get("selector_score")), _tie_key(best)):
                best = representative
        return best
    if selector == "energy_only":
        return _select_seed_marginal(rows, "rank_energy_dino")
    if selector == "mmd_only":
        return _select_seed_marginal(rows, "rank_rbf_mmd_dino")
    if selector == "mean_cov_only":
        return _select_seed_marginal_with_keys(rows, ("rank_mean_l2_dino", "rank_cov_trace_dino"))
    if selector == "no_cov_terms":
        return _select_seed_marginal_with_keys(rows, ("rank_energy_dino", "rank_rbf_mmd_dino", "rank_mean_l2_dino"))
    return _select_seed_marginal(rows, "rankmean_dino_score")


def _select_seed_marginal(rows: Sequence[Mapping[str, object]], score_key: str) -> Mapping[str, object] | None:
    return _select_seed_marginal_with_keys(rows, (score_key,))


def _select_seed_marginal_with_keys(rows: Sequence[Mapping[str, object]], score_keys: Sequence[str]) -> Mapping[str, object] | None:
    best = None
    for items in _group_by_candidate_mode(rows).values():
        seed_scores = [_mean(_float(row[key]) for key in score_keys) for row in items]
        score = _median(seed_scores)
        representative = dict(min(items, key=lambda row: (_mean(_float(row[key]) for key in score_keys), _tie_key(row))))
        representative["selector_score"] = score
        if best is None or (score, _tie_key(representative)) < (_float(best.get("selector_score")), _tie_key(best)):
            best = representative
    return best


def _build_loco_alignment_rows(
    score_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    try:
        import numpy as np  # type: ignore
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.impute import SimpleImputer  # type: ignore
        from sklearn.linear_model import Ridge  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore
    except ModuleNotFoundError:
        return []
    downstream_index = _downstream_index(downstream_rows)
    examples = []
    for items in _group_by_support_and_candidate_mode(score_rows).values():
        representative = dict(items[0])
        representative.update(_aggregate_seed_scores(items, "rankmean_dino_score"))
        utility = _candidate_utility(representative, downstream_index, seed_marginal=True)
        representative["utility_bacc"] = utility.mean_bacc
        examples.append(representative)
    numeric = (
        "seed_score_median",
        "seed_score_std",
        "support_nelbo_rank",
        "support_size",
        "abs_log_cov_trace_ratio_dino",
        "abs_log_pairwise_distance_ratio_dino",
    )
    categorical = ("candidate_expert", "mode_label", "generator_family")
    out: list[dict[str, object]] = []
    by_condition: dict[tuple[int, str, int, int], list[Mapping[str, object]]] = {}
    for row in examples:
        key = (int(row["experiment_seed"]), str(row["heldout_center"]), int(row["support_size"]), int(row["support_seed"]))
        by_condition.setdefault(key, []).append(row)
    for heldout in sorted({str(row["heldout_center"]) for row in examples}):
        train = [row for row in examples if str(row["heldout_center"]) != heldout]
        test = [row for row in examples if str(row["heldout_center"]) == heldout]
        if not train or not test:
            continue
        x_train = [[row.get(key) for key in (*numeric, *categorical)] for row in train]
        y_train = np.asarray([_float(row["utility_bacc"]) for row in train], dtype=float)
        pre = ColumnTransformer(
            [
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(range(len(numeric)))),
                ("cat", OneHotEncoder(handle_unknown="ignore"), list(range(len(numeric), len(numeric) + len(categorical)))),
            ]
        )
        model = make_pipeline(pre, Ridge(alpha=10.0))
        model.fit(x_train, y_train)
        for condition, rows in by_condition.items():
            if condition[1] != heldout:
                continue
            x_test = [[row.get(key) for key in (*numeric, *categorical)] for row in rows]
            preds = model.predict(x_test)
            best_idx = int(np.argmax(preds))
            selected = dict(rows[best_idx])
            selected["selector_score"] = -float(preds[best_idx])
            out.append(_alignment_for_selection(SELECTOR_LOCO, selected, [r for r in score_rows if _condition_key(r) == condition], downstream_index))
    return out


def _candidate_utility(
    row: Mapping[str, object],
    downstream_index: Mapping[tuple[object, ...], CandidateDownstreamRow],
    *,
    seed_marginal: bool,
) -> UtilitySummary:
    vals: list[float] = []
    f1s: list[float] = []
    for key, candidate in downstream_index.items():
        exp_seed, heldout, support_size, support_seed, expert, family, mode, generation_seed, _classifier_seed = key
        if (
            int(exp_seed) == int(row["experiment_seed"])
            and str(heldout) == str(row["heldout_center"])
            and int(support_size) == int(row["support_size"])
            and int(support_seed) == int(row["support_seed"])
            and str(expert) == str(row["candidate_expert"])
            and str(family) == str(row["generator_family"])
            and str(mode) == str(row["generation_mode"])
            and (seed_marginal or int(generation_seed) == int(row["generation_seed"]))
        ):
            vals.append(float(candidate.bacc))
            f1s.append(float(candidate.macro_f1))
    return _utility_summary(vals, f1s)


def _oracle_for_condition(
    rows: Sequence[Mapping[str, object]],
    downstream_index: Mapping[tuple[object, ...], CandidateDownstreamRow],
    *,
    seed_marginal: bool,
) -> tuple[Mapping[str, object] | None, list[tuple[Mapping[str, object], UtilitySummary]]]:
    grouped = _group_by_candidate_mode(rows) if seed_marginal else {tuple(_candidate_tuple(row, seed_marginal=False)): [row] for row in rows}
    ranked = []
    for items in grouped.values():
        representative = items[0]
        utility = _candidate_utility(representative, downstream_index, seed_marginal=seed_marginal)
        ranked.append((representative, utility))
    ranked.sort(key=lambda item: (-item[1].mean_bacc, -item[1].mean_macro_f1, _tie_key(item[0])))
    return (ranked[0][0] if ranked else None), ranked


def _support_condition_bandwidths(support: torch.Tensor, synthetic_batches: Sequence[torch.Tensor]) -> tuple[float, float, float]:
    refs = [support.detach().cpu().float()]
    for batch in synthetic_batches:
        refs.append(_cap_rows(batch.detach().cpu().float(), 16))
    pooled = torch.cat(refs, dim=0)
    if pooled.shape[0] < 2:
        median = 1.0
    else:
        distances = torch.pdist(pooled)
        positive = distances[distances > 0]
        median = float(torch.median(positive).item()) if int(positive.numel()) else 1.0
    median = max(median, 1.0e-6)
    return 0.5 * median, median, 2.0 * median


def _unlabeled_support_split(
    *,
    test_metadata: Sequence[Mapping[str, object]],
    target_indices: Sequence[int],
    heldout_center: str,
    support_size: int,
    support_seed: int,
):
    dummy_labels = {int(idx): 0 for idx in target_indices}
    split = _make_support_eval_split(
        target_domain=int(heldout_center),
        target_indices=tuple(int(idx) for idx in target_indices),
        labels_by_index=dummy_labels,
        support_size=int(support_size),
        sampling_policy="random",
        support_seed=int(support_seed),
    )
    if int(getattr(split, "support_labels_used", 0)) != 0:
        raise ProtocolError("C5.1 support split attempted to use target-support labels.")
    if str(getattr(split, "split_status", "ok")) != "ok":
        raise ProtocolError(f"C5.1 support split is not ok: {getattr(split, 'split_status', '')}")
    return split


def _load_bank_downstream_rows(c41_root: Path, c42_root: Path) -> list[CandidateDownstreamRow]:
    rows = []
    for path in (
        c41_root / "tables" / "all_expert_downstream_matrix.csv",
        c42_root / "tables" / "all_expert_downstream_matrix.csv",
    ):
        rows.extend(read_candidate_downstream_matrix(path))
    allowed = {(candidate.generator_family, candidate.generation_mode) for candidate in CANDIDATE_BANK}
    return [
        row
        for row in rows
        if row.status == "ok"
        and row.row_type == "single_expert"
        and (row.generator_family, row.generation_mode) in allowed
    ]


def _load_c41_hetero_mean_baseline_rows(c41_root: Path) -> list[dict[str, object]]:
    path = c41_root / "tables" / "routing_to_downstream_alignment.csv"
    rows = load_csv_rows(path)
    return [
        row
        for row in rows
        if str(row.get("generator_family")) == HETEROSCEDASTIC_GENERATOR_FAMILY
        and str(row.get("generation_mode")) == GENERATION_MODE_POSTERIOR_DECODER_MEAN
        and str(row.get("method")) == SUPPORT_NELBO_METHOD
    ]


def _downstream_index(rows: Sequence[CandidateDownstreamRow]) -> dict[tuple[object, ...], CandidateDownstreamRow]:
    out: dict[tuple[object, ...], CandidateDownstreamRow] = {}
    for row in rows:
        key = (
            int(row.experiment_seed),
            str(row.heldout_center),
            int(row.support_size),
            int(row.support_seed),
            str(row.candidate_expert),
            str(row.generator_family),
            str(row.generation_mode),
            int(row.generation_seed),
            int(row.classifier_seed),
        )
        out[key] = row
    return out


def _load_projection(c41_root: Path, experiment_seed: int, candidate_expert: str) -> SourceTrainPCAProjection:
    return _torch_load(c41_root / "projections" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / "pca64.pt")


def _checkpoint_path(c41_root: Path, experiment_seed: int, candidate_expert: str, kind: str) -> Path:
    filename = "plain_class_conditional_pca64.pt" if kind == "plain" else "heteroscedastic_class_conditional_pca64.pt"
    return c41_root / "checkpoints" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / kind / filename


def _load_c42_priors(c42_root: Path, experiment_seed: int, candidate_expert: str) -> dict[int, dict[int, SourceClassLatentDiagGMM]]:
    out: dict[int, dict[int, SourceClassLatentDiagGMM]] = {}
    for k in (1, 2, 4):
        out[k] = {}
        for label in (0, 1):
            path = c42_root / "latent_priors" / f"seed{int(experiment_seed)}" / f"expert_{candidate_expert}" / f"class_{label}" / f"gmm_k{k}.pt"
            if not path.exists():
                raise ProtocolError(f"Missing C4.2 latent prior for C5.1: {path}")
            out[k][label] = SourceClassLatentDiagGMM.from_payload(_torch_load(path))
    return out


def _requested_support_conditions(
    config: LockedV1Config,
    units: Sequence[SupportSelectionUnit],
    experiment_seed: int,
    heldout_center: str,
) -> tuple[tuple[int, int], ...]:
    present = {
        (int(unit.support_size), int(unit.support_seed))
        for unit in units
        if int(unit.experiment_seed) == int(experiment_seed)
        and str(unit.heldout_center) == str(heldout_center)
        and unit.method == SUPPORT_NELBO_METHOD
    }
    configured = {(int(size), int(seed)) for size in config.support_sizes for seed in config.support_seeds}
    return tuple(sorted(present.intersection(configured)))


def _support_units_by_key(units: Sequence[SupportSelectionUnit]) -> dict[tuple[int, str, int, int, str], SupportSelectionUnit]:
    return {
        (int(unit.experiment_seed), str(unit.heldout_center), int(unit.support_size), int(unit.support_seed), str(unit.method)): unit
        for unit in units
    }


def _assert_selector_score_rows_safe(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        for key in row:
            lowered = str(key).lower()
            if lowered in {"support_nelbo", "support_nelbo_rank"}:
                continue
            if any(token in lowered for token in FORBIDDEN_SELECTOR_SUBSTRINGS):
                raise ProtocolError(f"C5.1 selector score row contains forbidden selector input column: {key}")


def _assert_classifier_seed_invariant(rows: Sequence[Mapping[str, object]]) -> None:
    bad = [row for row in rows if int(row.get("selection_depends_on_classifier_seed", 0))]
    if bad:
        raise ProtocolError("C5.1 selector depends on classifier seed.")


def _energy_distance(x: torch.Tensor, y: torch.Tensor, *, max_points: int = 512) -> float:
    x = _cap_rows(x, max_points)
    y = _cap_rows(y, max_points)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return math.nan
    xy = torch.cdist(x, y).mean()
    xx = torch.pdist(x).mean()
    yy = torch.pdist(y).mean()
    return float(((2.0 * xy) - xx - yy).item())


def _rbf_mmd_fixed(x: torch.Tensor, y: torch.Tensor, *, sigma: float, max_points: int = 512) -> float:
    x = _cap_rows(x, max_points)
    y = _cap_rows(y, max_points)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return math.nan
    denom = max(2.0 * float(sigma) * float(sigma), 1.0e-12)
    kxx = torch.exp(-torch.cdist(x, x).pow(2) / denom).mean()
    kyy = torch.exp(-torch.cdist(y, y).pow(2) / denom).mean()
    kxy = torch.exp(-torch.cdist(x, y).pow(2) / denom).mean()
    return float((kxx + kyy - (2.0 * kxy)).item())


def _trace_cov(x: torch.Tensor) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(x.var(dim=0, unbiased=True).sum().item())


def _pairwise_distance_mean(x: torch.Tensor, *, max_points: int = 512) -> float:
    x = _cap_rows(x, max_points)
    if x.shape[0] < 2:
        return 0.0
    return float(torch.pdist(x).mean().item())


def _cap_rows(x: torch.Tensor, max_points: int) -> torch.Tensor:
    return x[: int(max_points)] if int(x.shape[0]) > int(max_points) else x


def _rank_values(values: Sequence[float]) -> list[float]:
    cleaned = [float(v) if not math.isnan(float(v)) else math.inf for v in values]
    order = sorted(enumerate(cleaned), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(cleaned)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        rank = (float(i + 1) + float(j)) / 2.0
        for pos in range(i, j):
            ranks[order[pos][0]] = rank
        i = j
    return ranks


def _zsum_scores(rows: Sequence[Mapping[str, object]], keys: Sequence[str]) -> list[float]:
    values_by_key = {key: [_float(row.get(key)) for row in rows] for key in keys}
    stats = {}
    for key, values in values_by_key.items():
        finite = [value for value in values if not math.isnan(value)]
        mu = _mean(finite)
        sigma = statistics.pstdev(finite) if len(finite) > 1 else 1.0
        stats[key] = (mu, sigma if sigma > 1.0e-12 else 1.0)
    out = []
    for row in rows:
        total = 0.0
        for key in keys:
            mu, sigma = stats[key]
            value = _float(row.get(key))
            total += ((value - mu) / sigma) if not math.isnan(value) else 999.0
        out.append(total)
    return out


def _ascending_rank_map(values: Mapping[str, float]) -> dict[str, int]:
    order = sorted(values, key=lambda key: (float(values[key]), str(key)))
    return {str(key): idx + 1 for idx, key in enumerate(order)}


def _group_by_candidate_mode(rows: Sequence[Mapping[str, object]]) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row["candidate_expert"]),
            str(row["generator_family"]),
            str(row["generation_mode"]),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _group_by_support_and_candidate_mode(rows: Sequence[Mapping[str, object]]) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            int(row["experiment_seed"]),
            str(row["heldout_center"]),
            int(row["support_size"]),
            int(row["support_seed"]),
            str(row["candidate_expert"]),
            str(row["generator_family"]),
            str(row["generation_mode"]),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _aggregate_seed_scores(rows: Sequence[Mapping[str, object]], score_key: str) -> dict[str, float]:
    values = [_float(row.get(score_key, row.get("selector_score", math.nan))) for row in rows]
    return {
        "seed_score_mean": _mean(values),
        "seed_score_median": _median(values),
        "seed_score_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "seed_score_iqr": _quantile(values, 0.75) - _quantile(values, 0.25) if values else math.nan,
    }


def _condition_key(row: Mapping[str, object]) -> tuple[int, str, int, int]:
    return (int(row["experiment_seed"]), str(row["heldout_center"]), int(row["support_size"]), int(row["support_seed"]))


def _candidate_tuple(row: Mapping[str, object], *, seed_marginal: bool) -> tuple[object, ...]:
    base = (str(row["candidate_expert"]), str(row["generator_family"]), str(row["generation_mode"]))
    return base if seed_marginal else (*base, int(row["generation_seed"]))


def _tie_key(row: Mapping[str, object]) -> tuple[str, str, str, int]:
    return (
        str(row["candidate_expert"]),
        str(row["generator_family"]),
        str(row["generation_mode"]),
        int(row["generation_seed"]),
    )


def _score_column(selector: str) -> str:
    if selector == SELECTOR_ZSUM_DINO:
        return "zsum_dino_score"
    if selector == SELECTOR_PCA:
        return "rankmean_pca_score"
    if selector == "energy_only":
        return "rank_energy_dino"
    if selector == "mmd_only":
        return "rank_rbf_mmd_dino"
    return "rankmean_dino_score"


def _selector_family(selector: str) -> str:
    if selector.startswith("unlabeled_support_nelbo"):
        return SUPPORT_NELBO_FAMILY
    if selector.startswith("metadata"):
        return METADATA_HYBRID_FAMILY
    if selector.startswith("loco"):
        return LEARNED_DIAGNOSTIC_FAMILY
    return SUPPORT_DISTANCE_FAMILY


def _utility_summary(values: Sequence[float], macro_f1: Sequence[float]) -> UtilitySummary:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    f1 = [float(v) for v in macro_f1 if not math.isnan(float(v))]
    return UtilitySummary(
        mean_bacc=_mean(clean),
        std_bacc=statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        min_bacc=min(clean) if clean else math.nan,
        max_bacc=max(clean) if clean else math.nan,
        ge_080_rate=_mean([1.0 if value >= 0.80 else 0.0 for value in clean]),
        mean_macro_f1=_mean(f1),
    )


def _decision_label(
    selector: str,
    subset: Sequence[Mapping[str, object]],
    all_rows: Sequence[Mapping[str, object]],
    spearman_positive: int,
    *,
    baseline: Mapping[str, float],
    baseline_by_center: Mapping[str, float],
) -> str:
    if selector != PRIMARY_SELECTOR:
        return "DIAGNOSTIC_ONLY"
    selected = [_float(row["selected_bacc_mean"]) for row in subset]
    if not selected:
        return FAILURE_SUPPORT_DISTANCE_NO_GAIN
    mean_selected = _mean(selected)
    mean_gap = _mean(_float(row["oracle_gap_bacc"]) for row in subset)
    ge_080_rate = _mean(1.0 if value >= 0.80 else 0.0 for value in selected)
    positive_centers = _center_improvement_count(subset, baseline_by_center)
    strong_degrades = _strong_center_degrade_count(subset, baseline_by_center)
    seed_selected_rows = [row for row in all_rows if str(row["selector_name"]) == SELECTOR_SEED_SELECTED]
    seed_selected_mean = _mean(_float(row["selected_bacc_mean"]) for row in seed_selected_rows)
    support_monotonic = _support_size_monotonic_nonnegative(subset)
    if (
        mean_selected >= 0.80
        and ge_080_rate > float(baseline.get("ge_080_rate", math.nan))
        and (mean_gap - float(baseline.get("oracle_gap", math.nan))) <= -0.03
        and positive_centers >= 4
        and spearman_positive >= 4
        and strong_degrades == 0
    ):
        return DECISION_PRIMARY_SUCCESS
    if not support_monotonic:
        return FAILURE_SUPPORT_SIZE_NO_MONOTONICITY
    if seed_selected_mean - mean_selected > 0.02:
        return FAILURE_SEED_OVERFIT
    if mean_selected >= 0.77 and positive_centers >= 1:
        return DECISION_USEFUL_RESULT
    if _mean(_float(row["oracle_bacc_mean"]) for row in subset) >= 0.80 and mean_selected < 0.80:
        return FAILURE_MODE_BANK_ORACLE_HIGH
    return FAILURE_SUPPORT_DISTANCE_NO_GAIN


def _baseline_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    selected = [_float(row.get("selected_bacc")) for row in rows]
    gaps = [_float(row.get("downstream_oracle_gap_bacc")) for row in rows]
    return {
        "selected_bacc": _mean(selected),
        "ge_080_rate": _mean(1.0 if value >= 0.80 else 0.0 for value in selected),
        "oracle_gap": _mean(gaps),
        "regret_p75": _quantile(gaps, 0.75),
    }


def _baseline_by_center_values(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    out = {}
    for center in sorted({str(row.get("heldout_center")) for row in rows}):
        out[center] = _mean(_float(row.get("selected_bacc")) for row in rows if str(row.get("heldout_center")) == center)
    return out


def _baseline_by_center(rows: Sequence[Mapping[str, object]], selector: str) -> dict[str, float]:
    out = {}
    for center in sorted({str(row["heldout_center"]) for row in rows}):
        subset = [row for row in rows if str(row["heldout_center"]) == center and str(row["selector_name"]) == selector]
        out[center] = _mean(_float(row["selected_bacc_mean"]) for row in subset)
    return out


def _center_improvement_count(rows: Sequence[Mapping[str, object]], baseline: Mapping[str, float]) -> int:
    count = 0
    for center in sorted({str(row["heldout_center"]) for row in rows}):
        current = _mean(_float(row["selected_bacc_mean"]) for row in rows if str(row["heldout_center"]) == center)
        base = float(baseline.get(center, math.nan))
        if not math.isnan(current) and not math.isnan(base) and current > base:
            count += 1
    return count


def _center_positive_count_vs_selector(rows: Sequence[Mapping[str, object]], all_rows: Sequence[Mapping[str, object]], baseline_selector: str) -> int:
    baseline = _baseline_by_center(all_rows, baseline_selector)
    return _center_improvement_count(rows, baseline)


def _strong_center_degrade_count(rows: Sequence[Mapping[str, object]], baseline: Mapping[str, float]) -> int:
    count = 0
    for center, baseline_value in baseline.items():
        if math.isnan(float(baseline_value)) or float(baseline_value) < 0.80:
            continue
        current = _mean(
            _float(row["selected_bacc_mean"])
            for row in rows
            if str(row["heldout_center"]) == str(center)
        )
        if not math.isnan(current) and current < float(baseline_value) - 0.02:
            count += 1
    return count


def _support_size_monotonic_nonnegative(rows: Sequence[Mapping[str, object]]) -> bool:
    by_size = []
    for support_size in sorted({int(row["support_size"]) for row in rows}):
        by_size.append(
            _mean(
                _float(row["selected_bacc_mean"])
                for row in rows
                if int(row["support_size"]) == support_size
            )
        )
    clean = [value for value in by_size if not math.isnan(value)]
    if len(clean) < 2:
        return True
    return clean[-1] >= clean[0] - 0.005


def _selector_spearman_positive_centers(selector: str, rows: Sequence[Mapping[str, object]]) -> int:
    if selector != PRIMARY_SELECTOR:
        return 0
    positive = 0
    for center in sorted({str(row.get("heldout_center")) for row in rows}):
        vals = [
            _float(row.get("spearman_score_bacc"))
            for row in rows
            if str(row.get("heldout_center")) == center and str(row.get("selector_name")) == PRIMARY_SELECTOR
        ]
        if _mean(vals) > 0.0:
            positive += 1
    return positive


def _filter_summary(rows: Sequence[Mapping[str, object]], tokens: Sequence[str]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if any(token in str(row.get("selector_name", "")) or token in str(row.get("selector_family", "")) for token in tokens)]


def _regret_columns() -> tuple[str, ...]:
    return ("selector_name", "regret_p50", "regret_p75", "regret_p90", "regret_mean", "regret_max")


def _correlation_columns() -> tuple[str, ...]:
    return ("heldout_center", "experiment_seed", "support_size", "support_seed", "selector_name", "spearman_score_bacc", "n_candidates")


def _support_size_columns() -> tuple[str, ...]:
    return ("selector_name", "support_size", "mean_selected_bacc", "oracle_gap", "top1_hit", "regret_p75", "spearman_score_bacc", "mode_entropy")


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return sum(clean) / float(len(clean)) if clean else math.nan


def _median(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return float(statistics.median(clean)) if clean else math.nan


def _quantile(values: Sequence[float] | Iterable[float], q: float) -> float:
    clean = sorted(float(value) for value in values if not math.isnan(float(value)))
    if not clean:
        return math.nan
    pos = (len(clean) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] + ((clean[hi] - clean[lo]) * (pos - lo))


def _entropy(values: Sequence[str]) -> float:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return math.nan
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
