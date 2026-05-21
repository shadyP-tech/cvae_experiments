"""C6.2 late probability ensembles over frozen CVAE expert/mode components.

C6.2 is a dense aggregation / routing-risk reduction experiment. It reuses the
fixed C4.1/C4.2 CVAE bank, trains one locked downstream classifier per
expert/mode/generation-seed member in original DINO space, fixes an aggregation
rule before target-eval labels are consulted, and only then scores BACC.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .downstream import balanced_accuracy, macro_f1
from .matrix import (
    _label,
    _load_embedding_cache,
    _read_samples_manifest,
    _records_for_split,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .schemas import ENSEMBLE_EXPERT_ID, METHOD_BASELINE_ROW_TYPE, METADATA_METHOD, SUPPORT_NELBO_METHOD
from .c61_mixture import (
    C61_ARTIFACTS_ROOT,
    C61_DEFAULT_C41_ROOT,
    C61_DEFAULT_C42_ROOT,
    C61_DEFAULT_C52_ROOT,
    HETERO_MEAN_BANK,
    MIN_COMPONENT_BUDGET_PER_CLASS,
    POOLING_SPACE_DINO,
    RANK_SOFTMAX_TAU,
    SAFE_BANK_BY_MODE,
    SAFE_MODE_PRIORITY,
    BankCandidate,
    GeneratedComponent,
    MixtureComponent,
    _GenerationCache,
    _file_hash,
    _float,
    _generate_component,
    _load_c41_baseline,
    _load_c52_baseline,
    _load_c52_oracle_reference,
    _mean,
    _quantile,
    _support_conditions_from_rows,
    _to_numpy,
    _write_csv,
    allocate_weighted_budget_per_class,
    clip_and_normalize_weights,
    load_csv_rows,
    select_safe_mode_prefix,
)


C62_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c62_late_probability_ensemble_v1"
C62_DEFAULT_C41_ROOT = C61_DEFAULT_C41_ROOT
C62_DEFAULT_C42_ROOT = C61_DEFAULT_C42_ROOT
C62_DEFAULT_C52_ROOT = C61_DEFAULT_C52_ROOT
C62_DEFAULT_C61_ROOT = C61_ARTIFACTS_ROOT
C62_LEGACY_SUPPORT_UNITS = "cvae_downstream_evaluation/artifacts/tables/support_selection_units.csv"

POLICY_C61_REPLAY = "c61_late_ensemble_replay"
POLICY_SAFE_SINGLE = "fixed_all_source_safe_single_seed_late_ensemble"
POLICY_SAFE_MULTI = "fixed_all_source_safe_multiseed_late_ensemble"
POLICY_FIXED_TOTAL = "fixed_total_draw_safe_multiseed_late_ensemble"
POLICY_HETERO_MULTI = "fixed_all_source_hetero_mean_multiseed_late_ensemble"
POLICY_NO_STANDARD = "fixed_all_source_safe_no_standard_prior_multiseed_late_ensemble"
POLICY_METADATA_WEIGHTED = "metadata_weighted_safe_late_ensemble"
POLICY_SUPPORT_NELBO_WEIGHTED = "support_nelbo_rank_weighted_safe_late_ensemble"

PRIMARY_POLICIES = (
    POLICY_C61_REPLAY,
    POLICY_SAFE_SINGLE,
    POLICY_SAFE_MULTI,
    POLICY_FIXED_TOTAL,
    POLICY_HETERO_MULTI,
    POLICY_NO_STANDARD,
    POLICY_METADATA_WEIGHTED,
    POLICY_SUPPORT_NELBO_WEIGHTED,
)

LATE_ENSEMBLE_GENERATOR_FAMILY = "family_c_pca64_late_probability_ensemble_downstream_v1"
GLOBAL_CLASS_ORDER = (0, 1)
FIXED_TOTAL_DRAW_CONTROL = "fixed_total_draw_control"

FAILURE_NO_GAIN = "LATE_ENSEMBLE_NO_GAIN"
FAILURE_MULTI_DILUTION = "MULTISEED_AVERAGING_DILUTES_STRONG_MEMBERS"
FAILURE_STANDARD_DILUTES = "STANDARD_PRIOR_DILUTES_ENSEMBLE"
FAILURE_SUPPORT_WEIGHT = "SUPPORT_WEIGHTING_OVERFITS_PROXY"
FAILURE_METADATA = "METADATA_WEIGHTING_NO_GAIN"
FAILURE_CALIBRATION = "PROBABILITY_CALIBRATION_MISMATCH"
FAILURE_EXTRA_DRAWS = "GAIN_EXPLAINED_BY_EXTRA_SYNTHETIC_DRAWS"
FAILURE_CENTER_1_3 = "CENTER_1_3_REMAIN_CEILING"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"
DECISION_SUCCESS = "LATE_ENSEMBLE_SUCCESS"
DECISION_USEFUL = "LATE_ENSEMBLE_USEFUL"

PREJOIN_FORBIDDEN_SUBSTRINGS = (
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "oracle",
    "regret",
    "target_eval",
    "target_evaluation",
    "target_label",
    "support_label",
    "current_heldout_utility",
    "utility_label",
    "true_utility",
)

RANK_WEIGHT_MIN = 0.05
RANK_WEIGHT_MAX = 0.40
METADATA_SELECTED_EXPERT_WEIGHT = 0.50


@dataclass(frozen=True)
class C62RunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    support_sizes: tuple[int, ...] | None = None
    support_seeds: tuple[int, ...] | None = None
    generation_seeds: tuple[int, ...] | None = None
    classifier_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class EnsembleMemberSpec:
    source_expert: str
    bank: BankCandidate
    generation_seed: int
    allocated_budget_per_class: int
    weight: float
    weight_source: str
    fixed_total_draw_control: int = 0

    @property
    def member_key(self) -> str:
        return f"expert_{self.source_expert}::{self.bank.mode_label}::seed_{int(self.generation_seed)}"


@dataclass(frozen=True)
class EnsemblePlan:
    policy: str
    generation_seed_group: str
    specs: tuple[EnsembleMemberSpec, ...]
    diagnostic_only: int = 0


MEMBER_COLUMNS = (
    "ensemble_policy",
    "policy_role",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed_group",
    "member_key",
    "source_expert",
    "generation_mode",
    "mode_label",
    "generator_family",
    "generation_seed",
    "classifier_seed",
    "allocated_budget_per_class",
    "projection_artifact_path",
    "projection_artifact_hash",
    "generator_checkpoint_path",
    "generator_checkpoint_hash",
    "latent_prior_artifact_path",
    "latent_prior_artifact_hash",
    "weight",
    "weight_source",
    "fixed_total_draw_control",
    "diagnostic_only",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
)

MATRIX_COLUMNS = (
    "ensemble_policy",
    "policy_role",
    "diagnostic_only",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed_group",
    "classifier_seed",
    "generator_family",
    "generation_mode",
    "budget_per_class",
    "bacc",
    "macro_f1",
    "auroc",
    "auprc",
    "ensemble_ge_080",
    "oracle_bacc_reference",
    "regret_bacc",
    "row_type",
    "n_synthetic_train",
    "n_target_eval",
    "target_eval_pool_id",
    "candidate_expert",
    "candidate_experts_hash",
    "member_keys",
    "num_members",
    "effective_num_members",
    "weight_entropy",
    "max_member_weight",
    "min_member_weight",
    "member_budget_min",
    "member_budget_max",
    "probability_aggregation",
    "prediction_rule",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "status",
    "error_message",
)

MEMBER_DIAGNOSTIC_COLUMNS = (
    "ensemble_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "generation_seed_group",
    "classifier_seed",
    "member_key",
    "source_expert",
    "mode_label",
    "generation_seed",
    "member_mean_confidence",
    "member_entropy",
    "member_logit_norm",
    "weight",
    "weight_source",
    "target_eval_labels_used_for_member_fit",
)

PROBABILITY_DIAGNOSTIC_COLUMNS = (
    "ensemble_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "generation_seed_group",
    "classifier_seed",
    "member_mean_confidence",
    "member_entropy",
    "member_logit_norm",
    "pairwise_prediction_disagreement",
    "probability_average_bacc",
    "logit_average_bacc",
    "hard_vote_bacc",
    "probability_calibration_mismatch",
    "diagnostic_only_logit_average",
    "diagnostic_only_hard_vote",
)

PROTOCOL_COLUMNS = (
    "ensemble_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "heldout_source_excluded",
    "pooled_in_dino_original",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "target_eval_threshold_search",
    "checkpoints_retrained",
    "prejoin_forbidden_columns_present",
    "primary_modes_exclude_noise_and_gmm_k4",
    "weights_prejoin_only",
    "class_probability_alignment_checked",
    "protocol_status",
)

SUMMARY_COLUMNS = (
    "ensemble_policy",
    "policy_role",
    "diagnostic_only",
    "n_rows",
    "mean_bacc",
    "ensemble_ge_080_rate",
    "mean_oracle_bacc_reference",
    "mean_regret_bacc",
    "regret_p50",
    "regret_p75",
    "regret_p90",
    "center_variance",
    "c61_late_ensemble_mean_bacc",
    "mean_delta_vs_c61_late_ensemble",
    "paired_positive_center_seed_cells",
    "paired_center_seed_cells",
    "c61_replay_max_abs_bacc_delta",
    "c61_replay_matches_within_tolerance",
    "c52_ridge_no_expert_mean_bacc",
    "mean_delta_vs_c52_ridge_no_expert",
    "uniform_dense_minus_metadata_weighted",
    "metadata_weighted_minus_uniform_dense",
    "center_1_delta_vs_c61_late",
    "center_3_delta_vs_c61_late",
    "strong_center_degrade_gt_002_count",
    "decision_label",
)

CENTER_COLUMNS = (
    "ensemble_policy",
    "heldout_center",
    "n_rows",
    "mean_bacc",
    "ensemble_ge_080_rate",
    "c61_late_ensemble_bacc",
    "delta_vs_c61_late_ensemble",
    "c52_ridge_no_expert_bacc",
    "delta_vs_c52_ridge_no_expert",
    "mean_oracle_bacc_reference",
    "regret_p75",
)


def run_c62_late_ensemble(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    c52_artifacts_root: Path,
    c61_artifacts_root: Path,
    device: str,
    limits: C62RunLimits = C62RunLimits(),
) -> dict[str, Path]:
    support_unit_rows = load_csv_rows(c41_artifacts_root / "tables" / "support_selection_units.csv")
    legacy_support_path = repo_root / C62_LEGACY_SUPPORT_UNITS
    combined_support_rows = list(support_unit_rows)
    if legacy_support_path.exists():
        combined_support_rows.extend(load_csv_rows(legacy_support_path))
    assert_c62_prejoin_rows_safe(_selector_visible_support_rows(combined_support_rows))
    c61_late_rows = _load_c61_late_rows(c61_artifacts_root)
    c61_late_by_center = _center_baseline(c61_late_rows)
    c61_late_by_condition = _c61_late_by_condition(c61_late_rows)
    c52_baseline = _load_c52_baseline(c52_artifacts_root)
    c52_oracle = _load_c52_oracle_reference(c52_artifacts_root)

    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    experiment_seed_filter = set(int(v) for v in limits.experiment_seeds) if limits.experiment_seeds else None

    member_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    member_diagnostics: list[dict[str, object]] = []
    probability_diagnostics: list[dict[str, object]] = []

    from .c41_workstation import discover_c41_run_artifacts

    for artifact in discover_c41_run_artifacts(config=config, repo_root=repo_root):
        support = artifact.support
        experiment_seed = int(support.experiment_seed)
        if experiment_seed_filter is not None and experiment_seed not in experiment_seed_filter:
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
            experiment_seed=experiment_seed,
            train_embeddings=train_cache.embeddings,
            train_metadata=train_cache.metadata,
            device=device,
        )

        for heldout in selected_heldout:
            heldout = str(heldout)
            candidates = tuple(str(c) for c in config.candidate_domains if str(c) != heldout)
            support_conditions = _support_conditions_from_rows(
                rows=support_unit_rows,
                experiment_seed=experiment_seed,
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
            if tuple(sorted(set(target_labels).union({0, 1}))) != GLOBAL_CLASS_ORDER:
                raise ProtocolError(f"C6.2 expects binary labels {GLOBAL_CLASS_ORDER}, got {sorted(set(target_labels))}")
            target_dino = test_cache.embeddings[list(target_pool.eval_indices)].detach().cpu().float()
            for support_size, support_seed, support_eval_split_id in support_conditions:
                support_rows_for_condition = [
                    row
                    for row in combined_support_rows
                    if int(row.get("experiment_seed", -1)) == experiment_seed
                    and str(row.get("heldout_center")) == heldout
                    and int(row.get("support_size", -1)) == int(support_size)
                    and int(row.get("support_seed", -1)) == int(support_seed)
                ]
                plans = build_c62_ensemble_plans(
                    policies=PRIMARY_POLICIES,
                    candidates=candidates,
                    total_budget_per_class=int(config.primary_budget_per_class),
                    generation_seeds=selected_generation_seeds,
                    support_rows=support_rows_for_condition,
                )
                for plan in plans:
                    protocol_rows.append(
                        _protocol_row(
                            plan=plan,
                            experiment_seed=experiment_seed,
                            heldout_center=heldout,
                            support_size=int(support_size),
                            support_seed=int(support_seed),
                            support_eval_split_id=support_eval_split_id,
                            candidates=candidates,
                        )
                    )
                    generated = [
                        _generate_member(
                            cache=generation_cache,
                            spec=spec,
                            label_values=GLOBAL_CLASS_ORDER,
                        )
                        for spec in plan.specs
                    ]
                    for classifier_seed in selected_classifier_seeds:
                        member_rows.extend(
                            _member_rows(
                                plan=plan,
                                experiment_seed=experiment_seed,
                                heldout_center=heldout,
                                support_size=int(support_size),
                                support_seed=int(support_seed),
                                support_eval_split_id=support_eval_split_id,
                                classifier_seed=int(classifier_seed),
                                generated=generated,
                            )
                        )
                        score = _score_late_ensemble_row(
                            plan=plan,
                            experiment_seed=experiment_seed,
                            heldout_center=heldout,
                            support_size=int(support_size),
                            support_seed=int(support_seed),
                            support_eval_split_id=support_eval_split_id,
                            classifier_seed=int(classifier_seed),
                            budget_per_class=int(config.primary_budget_per_class),
                            generated=generated,
                            target_dino=target_dino,
                            target_labels=target_labels,
                            target_eval_pool_id=target_pool.target_eval_pool_id,
                            oracle_reference=c52_oracle.get(
                                (experiment_seed, heldout, int(support_size), int(support_seed)),
                                math.nan,
                            ),
                        )
                        matrix_rows.append(score.matrix_row)
                        member_diagnostics.extend(score.member_diagnostics)
                        probability_diagnostics.append(score.probability_diagnostics)

    outputs = {
        "members": artifacts_root / "tables" / "c62_ensemble_members_pre_join.csv",
        "matrix": artifacts_root / "tables" / "c62_late_ensemble_downstream_matrix.csv",
        "threshold": artifacts_root / "tables" / "c62_threshold_audit.csv",
        "center": artifacts_root / "tables" / "c62_center_summary.csv",
        "protocol": artifacts_root / "tables" / "c62_protocol_audit.csv",
        "member_diagnostics": artifacts_root / "tables" / "c62_member_diagnostics.csv",
        "probability_diagnostics": artifacts_root / "tables" / "c62_probability_diagnostics.csv",
    }
    assert_c62_prejoin_rows_safe(member_rows)
    _write_csv(outputs["members"], MEMBER_COLUMNS, member_rows)
    _write_csv(outputs["matrix"], MATRIX_COLUMNS, matrix_rows)
    _write_csv(
        outputs["threshold"],
        SUMMARY_COLUMNS,
        build_c62_threshold_rows(
            matrix_rows,
            c61_late_rows=c61_late_rows,
            c61_late_by_center=c61_late_by_center,
            c61_late_by_condition=c61_late_by_condition,
            c52_baseline_by_center=c52_baseline,
        ),
    )
    _write_csv(outputs["center"], CENTER_COLUMNS, build_c62_center_rows(matrix_rows, c61_late_by_center, c52_baseline))
    _write_csv(outputs["protocol"], PROTOCOL_COLUMNS, protocol_rows)
    _write_csv(outputs["member_diagnostics"], MEMBER_DIAGNOSTIC_COLUMNS, member_diagnostics)
    _write_csv(outputs["probability_diagnostics"], PROBABILITY_DIAGNOSTIC_COLUMNS, probability_diagnostics)
    return outputs


def build_c62_ensemble_plans(
    *,
    policies: Sequence[str],
    candidates: Sequence[str],
    total_budget_per_class: int,
    generation_seeds: Sequence[int],
    support_rows: Sequence[Mapping[str, object]] = (),
) -> list[EnsemblePlan]:
    experts = tuple(sorted(str(v) for v in candidates))
    if not experts:
        raise ProtocolError("C6.2 cannot build an ensemble without non-heldout source experts.")
    seeds = tuple(int(seed) for seed in generation_seeds)
    if not seeds:
        raise ProtocolError("C6.2 cannot build an ensemble without generation seeds.")
    out: list[EnsemblePlan] = []
    for policy in policies:
        if policy in {POLICY_C61_REPLAY, POLICY_SAFE_SINGLE}:
            for seed in seeds:
                out.append(
                    _build_plan_for_seed_group(
                        policy=policy,
                        experts=experts,
                        total_budget_per_class=total_budget_per_class,
                        seeds=(seed,),
                        support_rows=support_rows,
                    )
                )
        else:
            out.append(
                _build_plan_for_seed_group(
                    policy=policy,
                    experts=experts,
                    total_budget_per_class=total_budget_per_class,
                    seeds=seeds,
                    support_rows=support_rows,
                )
            )
    return out


def assert_c62_prejoin_rows_safe(rows: Sequence[Mapping[str, object]]) -> None:
    allowed = {
        "target_support_labels_used",
        "target_eval_labels_used_for_selection",
        "target_eval_labels_used_for_member_fit",
    }
    bad = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if str(key) not in allowed and any(token in str(key).lower() for token in PREJOIN_FORBIDDEN_SUBSTRINGS)
        }
    )
    if bad:
        raise ProtocolError(f"C6.2 pre-join rows contain forbidden utility/eval columns: {bad}")


@dataclass(frozen=True)
class _GeneratedMember:
    spec: EnsembleMemberSpec
    generated: GeneratedComponent


@dataclass(frozen=True)
class _ProbabilityScore:
    matrix_row: dict[str, object]
    member_diagnostics: tuple[dict[str, object], ...]
    probability_diagnostics: dict[str, object]


def _build_plan_for_seed_group(
    *,
    policy: str,
    experts: Sequence[str],
    total_budget_per_class: int,
    seeds: Sequence[int],
    support_rows: Sequence[Mapping[str, object]],
) -> EnsemblePlan:
    banks = _policy_banks(policy, total_budget_per_class, len(experts))
    fixed_total = int(policy == POLICY_FIXED_TOTAL)
    while True:
        probability_key_weights = _policy_key_weights(policy, experts, banks, support_rows)
        budget_key_weights = _uniform_key_weights(experts, banks)
        component_allocation = allocate_weighted_budget_per_class(
            total_per_class=total_budget_per_class,
            weights=budget_key_weights,
        )
        specs: list[EnsembleMemberSpec] = []
        for expert in sorted(experts):
            for bank in banks:
                key = (str(expert), bank.mode_label)
                if key not in component_allocation:
                    continue
                component_budget = int(component_allocation[key])
                seed_budgets = (
                    _split_budget_across_seeds(component_budget, tuple(int(seed) for seed in seeds))
                    if fixed_total
                    else {int(seed): component_budget for seed in seeds}
                )
                for seed, seed_budget in sorted(seed_budgets.items()):
                    if int(seed_budget) <= 0:
                        continue
                    specs.append(
                        EnsembleMemberSpec(
                            source_expert=str(expert),
                            bank=bank,
                            generation_seed=int(seed),
                            allocated_budget_per_class=int(seed_budget),
                            weight=float(probability_key_weights[key]) / float(len(seeds)),
                            weight_source=_weight_source(policy, support_rows),
                            fixed_total_draw_control=fixed_total,
                        )
                    )
        if policy == POLICY_FIXED_TOTAL or min(spec.allocated_budget_per_class for spec in specs) >= MIN_COMPONENT_BUDGET_PER_CLASS:
            break
        if policy in {POLICY_METADATA_WEIGHTED, POLICY_SUPPORT_NELBO_WEIGHTED} and len(banks) > 1:
            banks = banks[:-1]
            continue
        _assert_member_budget_floor(spec.allocated_budget_per_class for spec in specs)
    normalized = _normalize_member_weights(specs)
    group = "seed_" + str(seeds[0]) if len(seeds) == 1 else "all:" + "|".join(str(seed) for seed in seeds)
    return EnsemblePlan(
        policy=policy,
        generation_seed_group=group,
        specs=tuple(normalized),
        diagnostic_only=int(policy == POLICY_C61_REPLAY),
    )


def _policy_banks(policy: str, total_budget_per_class: int, n_experts: int) -> tuple[BankCandidate, ...]:
    if policy == POLICY_HETERO_MULTI:
        return (HETERO_MEAN_BANK,)
    if policy == POLICY_NO_STANDARD:
        labels = tuple(label for label in select_safe_mode_prefix(total_budget_per_class, n_experts) if label != "standard_prior")
        return tuple(SAFE_BANK_BY_MODE[label] for label in labels)
    if policy in {
        POLICY_C61_REPLAY,
        POLICY_SAFE_SINGLE,
        POLICY_SAFE_MULTI,
        POLICY_FIXED_TOTAL,
        POLICY_METADATA_WEIGHTED,
        POLICY_SUPPORT_NELBO_WEIGHTED,
    }:
        return tuple(SAFE_BANK_BY_MODE[label] for label in select_safe_mode_prefix(total_budget_per_class, n_experts))
    raise ProtocolError(f"Unknown C6.2 ensemble policy: {policy}")


def _policy_key_weights(
    policy: str,
    experts: Sequence[str],
    banks: Sequence[BankCandidate],
    support_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], float]:
    expert_weights = _expert_weights(policy, experts, support_rows)
    mode_weight = 1.0 / float(len(banks))
    raw = {
        (str(expert), bank.mode_label): float(expert_weights[str(expert)]) * mode_weight
        for expert in sorted(experts)
        for bank in banks
    }
    return _normalize(raw)


def _uniform_key_weights(experts: Sequence[str], banks: Sequence[BankCandidate]) -> dict[tuple[str, str], float]:
    keys = [(str(expert), bank.mode_label) for expert in sorted(experts) for bank in banks]
    value = 1.0 / float(len(keys))
    return {key: value for key in keys}


def _expert_weights(
    policy: str,
    experts: Sequence[str],
    support_rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    experts = tuple(sorted(str(v) for v in experts))
    if policy == POLICY_METADATA_WEIGHTED:
        selected = _selected_expert_for_method(support_rows, METADATA_METHOD)
        if selected in experts and len(experts) > 1:
            remaining = (1.0 - METADATA_SELECTED_EXPERT_WEIGHT) / float(len(experts) - 1)
            return {expert: (METADATA_SELECTED_EXPERT_WEIGHT if expert == selected else remaining) for expert in experts}
    if policy == POLICY_SUPPORT_NELBO_WEIGHTED:
        scores = _support_nelbo_scores(support_rows)
        available = {expert: float(scores[expert]) for expert in experts if expert in scores}
        if available:
            ranked = _ascending_rank_map(available)
            raw = {expert: math.exp(-float(ranked.get(expert, len(experts))) / RANK_SOFTMAX_TAU) for expert in experts}
            return clip_and_normalize_weights(raw, min_weight=RANK_WEIGHT_MIN, max_weight=RANK_WEIGHT_MAX)
    even = 1.0 / float(len(experts))
    return {expert: even for expert in experts}


def _weight_source(policy: str, support_rows: Sequence[Mapping[str, object]]) -> str:
    if policy == POLICY_METADATA_WEIGHTED:
        selected = _selected_expert_for_method(support_rows, METADATA_METHOD)
        return f"metadata_selected_expert:{selected}" if selected else "metadata_missing_uniform"
    if policy == POLICY_SUPPORT_NELBO_WEIGHTED:
        return "support_nelbo_rank"
    return "uniform"


def _generate_member(
    *,
    cache: _GenerationCache,
    spec: EnsembleMemberSpec,
    label_values: Sequence[int],
) -> _GeneratedMember:
    component = MixtureComponent(
        source_expert=spec.source_expert,
        bank=spec.bank,
        desired_weight=spec.weight,
        allocated_budget_per_class=int(spec.allocated_budget_per_class),
    )
    return _GeneratedMember(
        spec=spec,
        generated=_generate_component(
            cache=cache,
            component=component,
            label_values=label_values,
            generation_seed=int(spec.generation_seed),
        ),
    )


def _score_late_ensemble_row(
    *,
    plan: EnsemblePlan,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    classifier_seed: int,
    budget_per_class: int,
    generated: Sequence[_GeneratedMember],
    target_dino: object,
    target_labels: Sequence[int],
    target_eval_pool_id: str,
    oracle_reference: float,
) -> _ProbabilityScore:
    try:
        probs = []
        log_probs = []
        hard_preds = []
        member_rows = []
        weights = []
        for member in generated:
            prediction = _fit_member_probabilities(
                synthetic_embeddings=_to_numpy(member.generated.synthetic_dino),
                synthetic_labels=member.generated.synthetic_labels,
                target_embeddings=_to_numpy(target_dino),
                classifier_seed=classifier_seed,
            )
            aligned = align_probabilities_to_class_order(
                prediction["probabilities"],
                prediction["classes"],
                GLOBAL_CLASS_ORDER,
            )
            probs.append(aligned)
            log_probs.append(_log_probabilities(aligned))
            hard_preds.append(fixed_predictions_from_probabilities(aligned, GLOBAL_CLASS_ORDER))
            weights.append(float(member.spec.weight))
            member_diag = _member_probability_diagnostics(aligned)
            member_rows.append(
                {
                    "ensemble_policy": plan.policy,
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": heldout_center,
                    "support_size": int(support_size),
                    "support_seed": int(support_seed),
                    "generation_seed_group": plan.generation_seed_group,
                    "classifier_seed": int(classifier_seed),
                    "member_key": member.spec.member_key,
                    "source_expert": member.spec.source_expert,
                    "mode_label": member.spec.bank.mode_label,
                    "generation_seed": int(member.spec.generation_seed),
                    **member_diag,
                    "weight": float(member.spec.weight),
                    "weight_source": member.spec.weight_source,
                    "target_eval_labels_used_for_member_fit": 0,
                }
            )
        weights_arr = _np_array(weights)
        weights_arr = weights_arr / max(float(weights_arr.sum()), 1.0e-12)
        stacked = _np_stack(probs)
        mean_prob = (stacked * weights_arr[:, None, None]).sum(axis=0)
        mean_log_prob = (_np_stack(log_probs) * weights_arr[:, None, None]).sum(axis=0)
        hard_vote_pred = _hard_vote_predictions(hard_preds, GLOBAL_CLASS_ORDER)
        prob_pred = fixed_predictions_from_probabilities(mean_prob, GLOBAL_CLASS_ORDER)
        logit_pred = _predictions_from_log_probs(mean_log_prob, GLOBAL_CLASS_ORDER)
        prob_metrics = _score_predictions_and_probabilities(target_labels, prob_pred, mean_prob, GLOBAL_CLASS_ORDER)
        logit_bacc = balanced_accuracy(target_labels, logit_pred)
        hard_bacc = balanced_accuracy(target_labels, hard_vote_pred)
        total_train = sum(len(member.generated.synthetic_labels) for member in generated)
        entropy = ensemble_weight_diagnostics([member.spec for member in generated])
        status = "ok"
        error = ""
    except Exception as exc:
        prob_metrics = {"bacc": math.nan, "macro_f1": math.nan, "auroc": math.nan, "auprc": math.nan}
        logit_bacc = hard_bacc = math.nan
        total_train = sum(len(member.generated.synthetic_labels) for member in generated)
        entropy = ensemble_weight_diagnostics([member.spec for member in generated])
        member_rows = []
        probs = []
        hard_preds = []
        status = "failed_c62_late_ensemble_scoring"
        error = str(exc)
    disagreement = _pairwise_prediction_disagreement(hard_preds)
    mean_member_diag = _mean_member_diagnostics(member_rows)
    bacc = float(prob_metrics["bacc"])
    matrix_row = {
        "ensemble_policy": plan.policy,
        "policy_role": _policy_role(plan.policy),
        "diagnostic_only": int(plan.diagnostic_only),
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "generation_seed_group": plan.generation_seed_group,
        "classifier_seed": int(classifier_seed),
        "generator_family": LATE_ENSEMBLE_GENERATOR_FAMILY,
        "generation_mode": plan.policy,
        "budget_per_class": int(budget_per_class),
        "bacc": bacc,
        "macro_f1": prob_metrics["macro_f1"],
        "auroc": prob_metrics["auroc"],
        "auprc": prob_metrics["auprc"],
        "ensemble_ge_080": int(not math.isnan(bacc) and bacc >= 0.80),
        "oracle_bacc_reference": oracle_reference,
        "regret_bacc": oracle_reference - bacc if not math.isnan(_float(oracle_reference)) and not math.isnan(bacc) else math.nan,
        "row_type": METHOD_BASELINE_ROW_TYPE,
        "n_synthetic_train": int(total_train),
        "n_target_eval": len(target_labels),
        "target_eval_pool_id": target_eval_pool_id,
        "candidate_expert": ENSEMBLE_EXPERT_ID,
        "candidate_experts_hash": hash_candidate_experts(member.spec.member_key for member in generated),
        "member_keys": ";".join(member.spec.member_key for member in generated),
        **entropy,
        "probability_aggregation": "weighted_probability_average",
        "prediction_rule": "binary_p_positive_ge_0_5" if len(GLOBAL_CLASS_ORDER) == 2 else "multiclass_argmax",
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "status": status,
        "error_message": error,
    }
    probability_row = {
        "ensemble_policy": plan.policy,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "generation_seed_group": plan.generation_seed_group,
        "classifier_seed": int(classifier_seed),
        **mean_member_diag,
        "pairwise_prediction_disagreement": disagreement,
        "probability_average_bacc": bacc,
        "logit_average_bacc": logit_bacc,
        "hard_vote_bacc": hard_bacc,
        "probability_calibration_mismatch": int(
            not math.isnan(bacc)
            and (not math.isnan(logit_bacc) and logit_bacc > bacc + 0.01 or not math.isnan(hard_bacc) and hard_bacc > bacc + 0.01)
        ),
        "diagnostic_only_logit_average": 1,
        "diagnostic_only_hard_vote": 1,
    }
    return _ProbabilityScore(matrix_row=matrix_row, member_diagnostics=tuple(member_rows), probability_diagnostics=probability_row)


def align_probabilities_to_class_order(probabilities: object, classes: Sequence[int], global_class_order: Sequence[int]) -> object:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    source_classes = tuple(int(v) for v in classes)
    target_classes = tuple(int(v) for v in global_class_order)
    if probs.ndim != 2:
        raise ProtocolError("Member probabilities must be a 2D array.")
    if set(source_classes) != set(target_classes):
        raise ProtocolError(f"C6.2 member class mismatch: member={source_classes}, global={target_classes}")
    aligned = np.zeros((probs.shape[0], len(target_classes)), dtype=float)
    for out_idx, cls in enumerate(target_classes):
        aligned[:, out_idx] = probs[:, source_classes.index(cls)]
    return aligned


def fixed_predictions_from_probabilities(probabilities: object, global_class_order: Sequence[int]) -> list[int]:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    classes = tuple(int(v) for v in global_class_order)
    if len(classes) == 2:
        positive_idx = classes.index(1) if 1 in classes else 1
        negative = classes[1 - positive_idx]
        positive = classes[positive_idx]
        return [positive if float(value) >= 0.5 else negative for value in probs[:, positive_idx]]
    return [classes[int(idx)] for idx in np.argmax(probs, axis=1)]


def ensemble_weight_diagnostics(specs: Sequence[EnsembleMemberSpec]) -> dict[str, float]:
    weights = [max(float(spec.weight), 0.0) for spec in specs]
    total = sum(weights)
    normalized = [value / total for value in weights] if total > 0.0 else []
    entropy = -sum(value * math.log(max(value, 1.0e-12)) for value in normalized)
    budgets = [float(spec.allocated_budget_per_class) for spec in specs]
    return {
        "num_members": float(len(specs)),
        "effective_num_members": float(math.exp(entropy)) if normalized else math.nan,
        "weight_entropy": float(entropy) if normalized else math.nan,
        "max_member_weight": max(normalized) if normalized else math.nan,
        "min_member_weight": min(normalized) if normalized else math.nan,
        "member_budget_min": min(budgets) if budgets else math.nan,
        "member_budget_max": max(budgets) if budgets else math.nan,
    }


def build_c62_threshold_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    *,
    c61_late_rows: Sequence[Mapping[str, object]],
    c61_late_by_center: Mapping[str, float],
    c61_late_by_condition: Mapping[tuple[int, str, int, int, int], float],
    c52_baseline_by_center: Mapping[str, float],
) -> list[dict[str, object]]:
    c61_mean = _mean(_float(row.get("bacc")) for row in c61_late_rows)
    c52_mean = _mean(c52_baseline_by_center.values())
    policy_means = {
        policy: _mean(_float(row.get("bacc")) for row in matrix_rows if str(row.get("ensemble_policy")) == policy and str(row.get("status")) == "ok")
        for policy in sorted({str(row.get("ensemble_policy")) for row in matrix_rows})
    }
    uniform_mean = policy_means.get(POLICY_SAFE_MULTI, math.nan)
    metadata_mean = policy_means.get(POLICY_METADATA_WEIGHTED, math.nan)
    out = []
    for policy in sorted(policy_means):
        subset = [row for row in matrix_rows if str(row.get("ensemble_policy")) == policy and str(row.get("status")) == "ok"]
        values = [_float(row.get("bacc")) for row in subset]
        regrets = [_float(row.get("regret_bacc")) for row in subset]
        center_means = [
            _mean(_float(row.get("bacc")) for row in subset if str(row.get("heldout_center")) == center)
            for center in sorted({str(row.get("heldout_center")) for row in subset})
        ]
        mean_bacc = _mean(values)
        replay_delta = _c61_replay_max_abs_delta(subset, c61_late_rows) if policy == POLICY_C61_REPLAY else math.nan
        row = {
            "ensemble_policy": policy,
            "policy_role": _policy_role(policy),
            "diagnostic_only": int(policy == POLICY_C61_REPLAY),
            "n_rows": len(subset),
            "mean_bacc": mean_bacc,
            "ensemble_ge_080_rate": _mean(1.0 if value >= 0.80 else 0.0 for value in values),
            "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
            "mean_regret_bacc": _mean(regrets),
            "regret_p50": _quantile(regrets, 0.50),
            "regret_p75": _quantile(regrets, 0.75),
            "regret_p90": _quantile(regrets, 0.90),
            "center_variance": statistics.pvariance(center_means) if len(center_means) > 1 else 0.0,
            "c61_late_ensemble_mean_bacc": c61_mean,
            "mean_delta_vs_c61_late_ensemble": mean_bacc - c61_mean if not math.isnan(c61_mean) else math.nan,
            "paired_positive_center_seed_cells": _paired_positive_cells(subset, c61_late_by_condition)[0],
            "paired_center_seed_cells": _paired_positive_cells(subset, c61_late_by_condition)[1],
            "c61_replay_max_abs_bacc_delta": replay_delta,
            "c61_replay_matches_within_tolerance": int(policy != POLICY_C61_REPLAY or (not math.isnan(replay_delta) and replay_delta <= 1.0e-9)),
            "c52_ridge_no_expert_mean_bacc": c52_mean,
            "mean_delta_vs_c52_ridge_no_expert": mean_bacc - c52_mean if not math.isnan(c52_mean) else math.nan,
            "uniform_dense_minus_metadata_weighted": uniform_mean - metadata_mean if not math.isnan(uniform_mean) and not math.isnan(metadata_mean) else math.nan,
            "metadata_weighted_minus_uniform_dense": metadata_mean - uniform_mean if not math.isnan(uniform_mean) and not math.isnan(metadata_mean) else math.nan,
            "center_1_delta_vs_c61_late": _center_delta(subset, "1", c61_late_by_center),
            "center_3_delta_vs_c61_late": _center_delta(subset, "3", c61_late_by_center),
            "strong_center_degrade_gt_002_count": _strong_center_degrade_count(subset, c61_late_by_center),
            "decision_label": "",
        }
        row["decision_label"] = _decision_label(row)
        out.append(row)
    return out


def build_c62_center_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    c61_late_by_center: Mapping[str, float],
    c52_baseline_by_center: Mapping[str, float],
) -> list[dict[str, object]]:
    out = []
    policies = sorted({str(row.get("ensemble_policy")) for row in matrix_rows})
    centers = sorted({str(row.get("heldout_center")) for row in matrix_rows})
    for policy in policies:
        for center in centers:
            subset = [
                row
                for row in matrix_rows
                if str(row.get("ensemble_policy")) == policy
                and str(row.get("heldout_center")) == center
                and str(row.get("status")) == "ok"
            ]
            if not subset:
                continue
            mean_bacc = _mean(_float(row.get("bacc")) for row in subset)
            out.append(
                {
                    "ensemble_policy": policy,
                    "heldout_center": center,
                    "n_rows": len(subset),
                    "mean_bacc": mean_bacc,
                    "ensemble_ge_080_rate": _mean(1.0 if _float(row.get("bacc")) >= 0.80 else 0.0 for row in subset),
                    "c61_late_ensemble_bacc": _float(c61_late_by_center.get(center)),
                    "delta_vs_c61_late_ensemble": mean_bacc - _float(c61_late_by_center.get(center)),
                    "c52_ridge_no_expert_bacc": _float(c52_baseline_by_center.get(center)),
                    "delta_vs_c52_ridge_no_expert": mean_bacc - _float(c52_baseline_by_center.get(center)),
                    "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
                    "regret_p75": _quantile([_float(row.get("regret_bacc")) for row in subset], 0.75),
                }
            )
    return out


def _fit_member_probabilities(
    *,
    synthetic_embeddings: object,
    synthetic_labels: Sequence[int],
    target_embeddings: object,
    classifier_seed: int,
) -> dict[str, object]:
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("C6.2 downstream member fitting requires numpy and scikit-learn.") from exc

    x_syn = np.asarray(synthetic_embeddings, dtype=float)
    y_syn = np.asarray(synthetic_labels, dtype=int)
    x_eval = np.asarray(target_embeddings, dtype=float)
    if x_syn.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if x_syn.shape[1] != x_eval.shape[1]:
        raise ValueError("Synthetic and target embeddings must share the original DINO frame.")
    if sorted(set(int(v) for v in y_syn.tolist())) != [0, 1]:
        raise ValueError("Locked v1 classifier expects exactly binary synthetic labels.")
    scaler = StandardScaler()
    x_syn_scaled = scaler.fit_transform(x_syn)
    x_eval_scaled = scaler.transform(x_eval)
    clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, class_weight=None, random_state=int(classifier_seed))
    clf.fit(x_syn_scaled, y_syn)
    return {
        "probabilities": clf.predict_proba(x_eval_scaled),
        "classes": tuple(int(v) for v in clf.classes_.tolist()),
    }


def _score_predictions_and_probabilities(
    target_labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: object,
    class_order: Sequence[int],
) -> dict[str, float]:
    import numpy as np  # type: ignore
    from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

    y_true = [int(v) for v in target_labels]
    pred = [int(v) for v in predictions]
    probs = np.asarray(probabilities, dtype=float)
    auroc = auprc = math.nan
    classes = tuple(int(v) for v in class_order)
    if len(classes) == 2 and probs.shape[1] == 2 and 1 in classes:
        p1 = probs[:, classes.index(1)]
        try:
            auroc = float(roc_auc_score(y_true, p1))
        except ValueError:
            auroc = math.nan
        try:
            auprc = float(average_precision_score(y_true, p1))
        except ValueError:
            auprc = math.nan
    return {
        "bacc": balanced_accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred),
        "auroc": auroc,
        "auprc": auprc,
    }


def _member_rows(
    *,
    plan: EnsemblePlan,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    classifier_seed: int,
    generated: Sequence[_GeneratedMember],
) -> list[dict[str, object]]:
    rows = []
    for member in generated:
        item = member.generated
        latent_paths = item.latent_prior_paths
        rows.append(
            {
                "ensemble_policy": plan.policy,
                "policy_role": _policy_role(plan.policy),
                "experiment_seed": int(experiment_seed),
                "heldout_center": heldout_center,
                "support_size": int(support_size),
                "support_seed": int(support_seed),
                "support_eval_split_id": support_eval_split_id,
                "generation_seed_group": plan.generation_seed_group,
                "member_key": member.spec.member_key,
                "source_expert": member.spec.source_expert,
                "generation_mode": member.spec.bank.generation_mode,
                "mode_label": member.spec.bank.mode_label,
                "generator_family": member.spec.bank.generator_family,
                "generation_seed": int(member.spec.generation_seed),
                "classifier_seed": int(classifier_seed),
                "allocated_budget_per_class": int(member.spec.allocated_budget_per_class),
                "projection_artifact_path": str(item.projection_path),
                "projection_artifact_hash": _file_hash(item.projection_path),
                "generator_checkpoint_path": str(item.checkpoint_path),
                "generator_checkpoint_hash": _file_hash(item.checkpoint_path),
                "latent_prior_artifact_path": ";".join(str(path) for path in latent_paths),
                "latent_prior_artifact_hash": ";".join(_file_hash(path) for path in latent_paths),
                "weight": float(member.spec.weight),
                "weight_source": member.spec.weight_source,
                "fixed_total_draw_control": int(member.spec.fixed_total_draw_control),
                "diagnostic_only": int(plan.diagnostic_only),
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
            }
        )
    return rows


def _protocol_row(
    *,
    plan: EnsemblePlan,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    candidates: Sequence[str],
) -> dict[str, object]:
    modes = {spec.bank.mode_label for spec in plan.specs}
    pass_status = str(heldout_center) not in {str(candidate) for candidate in candidates}
    return {
        "ensemble_policy": plan.policy,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "heldout_source_excluded": int(str(heldout_center) not in {str(candidate) for candidate in candidates}),
        "pooled_in_dino_original": 0,
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "target_eval_threshold_search": 0,
        "checkpoints_retrained": 0,
        "prejoin_forbidden_columns_present": 0,
        "primary_modes_exclude_noise_and_gmm_k4": int("hetero_noise" not in modes and "gmm_k4" not in modes),
        "weights_prejoin_only": 1,
        "class_probability_alignment_checked": 1,
        "protocol_status": "pass" if pass_status else "fail",
    }


def _split_budget_across_seeds(total: int, seeds: Sequence[int]) -> dict[int, int]:
    seeds = tuple(int(seed) for seed in seeds)
    base = int(total) // len(seeds)
    remainder = int(total) - (base * len(seeds))
    return {seed: base + (1 if idx < remainder else 0) for idx, seed in enumerate(sorted(seeds))}


def _normalize_member_weights(specs: Sequence[EnsembleMemberSpec]) -> tuple[EnsembleMemberSpec, ...]:
    total = sum(max(float(spec.weight), 0.0) for spec in specs)
    if total <= 0.0:
        value = 1.0 / float(len(specs))
        return tuple(_replace_spec_weight(spec, value) for spec in specs)
    return tuple(_replace_spec_weight(spec, max(float(spec.weight), 0.0) / total) for spec in specs)


def _replace_spec_weight(spec: EnsembleMemberSpec, weight: float) -> EnsembleMemberSpec:
    return EnsembleMemberSpec(
        source_expert=spec.source_expert,
        bank=spec.bank,
        generation_seed=spec.generation_seed,
        allocated_budget_per_class=spec.allocated_budget_per_class,
        weight=float(weight),
        weight_source=spec.weight_source,
        fixed_total_draw_control=spec.fixed_total_draw_control,
    )


def _assert_member_budget_floor(values: Iterable[int]) -> None:
    clean = [int(value) for value in values]
    if clean and min(clean) < MIN_COMPONENT_BUDGET_PER_CLASS:
        raise ProtocolError(
            "C6.2 member budget fragmentation below minimum: "
            f"min={min(clean)}, required={MIN_COMPONENT_BUDGET_PER_CLASS}"
        )


def _selected_expert_for_method(rows: Sequence[Mapping[str, object]], method: str) -> str:
    picked = sorted(
        [str(row.get("selected_expert")) for row in rows if str(row.get("method")) == method and str(row.get("selected_expert", "")).strip()]
    )
    return picked[0] if picked else ""


def _support_nelbo_scores(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    for row in rows:
        if str(row.get("method")) != SUPPORT_NELBO_METHOD:
            continue
        raw = row.get("support_nelbo_by_expert_json", "{}")
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            return {}
        return {str(key): float(value) for key, value in payload.items()}
    return {}


def _ascending_rank_map(scores: Mapping[str, float]) -> dict[str, int]:
    return {
        str(key): int(idx + 1)
        for idx, (key, _value) in enumerate(sorted(scores.items(), key=lambda item: (float(item[1]), str(item[0]))))
    }


def _selector_visible_support_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    safe_keys = {
        "heldout_center",
        "experiment_seed",
        "support_size",
        "support_seed",
        "method",
        "selected_expert",
        "candidate_experts",
        "support_nelbo_by_expert_json",
        "target_expert_excluded",
        "support_eval_split_id",
    }
    return [{key: row.get(key, "") for key in safe_keys if key in row} for row in rows]


def _normalize(weights: Mapping[object, float]) -> dict[object, float]:
    clean = {key: max(float(value), 0.0) for key, value in weights.items()}
    total = sum(clean.values())
    if total <= 0.0:
        even = 1.0 / float(len(clean))
        return {key: even for key in clean}
    return {key: value / total for key, value in clean.items()}


def _np_array(values: object):
    import numpy as np  # type: ignore

    return np.asarray(values, dtype=float)


def _np_stack(values: Sequence[object]):
    import numpy as np  # type: ignore

    return np.stack([np.asarray(value, dtype=float) for value in values], axis=0)


def _log_probabilities(probabilities: object):
    import numpy as np  # type: ignore

    return np.log(np.clip(np.asarray(probabilities, dtype=float), 1.0e-8, 1.0))


def _predictions_from_log_probs(log_probs: object, global_class_order: Sequence[int]) -> list[int]:
    import numpy as np  # type: ignore

    arr = np.asarray(log_probs, dtype=float)
    classes = tuple(int(value) for value in global_class_order)
    return [classes[int(idx)] for idx in np.argmax(arr, axis=1)]


def _hard_vote_predictions(predictions: Sequence[Sequence[int]], global_class_order: Sequence[int]) -> list[int]:
    classes = tuple(int(value) for value in global_class_order)
    if not predictions:
        return []
    n = len(predictions[0])
    out = []
    for idx in range(n):
        counts = {cls: 0 for cls in classes}
        for pred in predictions:
            counts[int(pred[idx])] = counts.get(int(pred[idx]), 0) + 1
        out.append(max(classes, key=lambda cls: (counts.get(cls, 0), -classes.index(cls))))
    return out


def _member_probability_diagnostics(probabilities: object) -> dict[str, float]:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1.0e-8, 1.0)), axis=1)
    logp = np.log(np.clip(probs, 1.0e-8, 1.0))
    return {
        "member_mean_confidence": float(np.mean(np.max(probs, axis=1))),
        "member_entropy": float(np.mean(entropy)),
        "member_logit_norm": float(np.mean(np.linalg.norm(logp, axis=1))),
    }


def _mean_member_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        "member_mean_confidence": _mean(_float(row.get("member_mean_confidence")) for row in rows),
        "member_entropy": _mean(_float(row.get("member_entropy")) for row in rows),
        "member_logit_norm": _mean(_float(row.get("member_logit_norm")) for row in rows),
    }


def _pairwise_prediction_disagreement(predictions: Sequence[Sequence[int]]) -> float:
    if len(predictions) < 2:
        return 0.0
    rates = []
    for idx, left in enumerate(predictions):
        for right in predictions[idx + 1 :]:
            if not left:
                continue
            rates.append(sum(1 for a, b in zip(left, right) if int(a) != int(b)) / float(len(left)))
    return _mean(rates)


def _load_c61_late_rows(c61_root: Path) -> list[dict[str, object]]:
    path = c61_root / "tables" / "c61_mixture_downstream_matrix.csv"
    if not path.exists():
        return []
    return [
        row
        for row in load_csv_rows(path)
        if str(row.get("mixture_policy")) == "late_classifier_ensemble_diagnostic_only"
        and str(row.get("status")) == "ok"
    ]


def _center_baseline(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        center: _mean(_float(row.get("bacc")) for row in rows if str(row.get("heldout_center")) == center)
        for center in sorted({str(row.get("heldout_center")) for row in rows})
    }


def _c61_late_by_condition(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, str, int, int, int], float]:
    out: dict[tuple[int, str, int, int, int], list[float]] = {}
    for row in rows:
        key = (
            int(row.get("experiment_seed", -1)),
            str(row.get("heldout_center")),
            int(row.get("support_size", -1)),
            int(row.get("support_seed", -1)),
            int(row.get("classifier_seed", -1)),
        )
        out.setdefault(key, []).append(_float(row.get("bacc")))
    return {key: _mean(values) for key, values in out.items()}


def _c61_replay_max_abs_delta(
    rows: Sequence[Mapping[str, object]],
    c61_late_rows: Sequence[Mapping[str, object]],
) -> float:
    c61 = {}
    for row in c61_late_rows:
        key = (
            int(row.get("experiment_seed", -1)),
            str(row.get("heldout_center")),
            int(row.get("support_size", -1)),
            int(row.get("support_seed", -1)),
            f"seed_{int(row.get('generation_seed', -1))}",
            int(row.get("classifier_seed", -1)),
        )
        c61[key] = _float(row.get("bacc"))
    deltas = []
    for row in rows:
        key = (
            int(row.get("experiment_seed", -1)),
            str(row.get("heldout_center")),
            int(row.get("support_size", -1)),
            int(row.get("support_seed", -1)),
            str(row.get("generation_seed_group")),
            int(row.get("classifier_seed", -1)),
        )
        if key in c61:
            deltas.append(abs(_float(row.get("bacc")) - c61[key]))
    return max(deltas) if deltas else math.nan


def _paired_positive_cells(
    rows: Sequence[Mapping[str, object]],
    c61_by_condition: Mapping[tuple[int, str, int, int, int], float],
) -> tuple[int, int]:
    grouped: dict[tuple[int, str], list[float]] = {}
    baseline: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        condition = (
            int(row.get("experiment_seed", -1)),
            str(row.get("heldout_center")),
            int(row.get("support_size", -1)),
            int(row.get("support_seed", -1)),
            int(row.get("classifier_seed", -1)),
        )
        if condition not in c61_by_condition:
            continue
        cell = (condition[0], condition[1])
        grouped.setdefault(cell, []).append(_float(row.get("bacc")))
        baseline.setdefault(cell, []).append(_float(c61_by_condition[condition]))
    deltas = [_mean(grouped[cell]) - _mean(baseline[cell]) for cell in grouped]
    return (sum(1 for delta in deltas if delta > 0.0), len(deltas))


def _center_delta(rows: Sequence[Mapping[str, object]], center: str, baseline: Mapping[str, float]) -> float:
    current = _mean(_float(row.get("bacc")) for row in rows if str(row.get("heldout_center")) == str(center))
    return current - _float(baseline.get(str(center)))


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


def _decision_label(row: Mapping[str, object]) -> str:
    mean_bacc = _float(row.get("mean_bacc"))
    delta = _float(row.get("mean_delta_vs_c61_late_ensemble"))
    strong_degrades = int(row.get("strong_center_degrade_gt_002_count") or 0)
    if mean_bacc >= 0.80 and strong_degrades == 0:
        return DECISION_SUCCESS
    if delta >= 0.01 or _float(row.get("ensemble_ge_080_rate")) >= 0.50:
        return DECISION_USEFUL
    policy = str(row.get("ensemble_policy"))
    if policy == POLICY_SAFE_MULTI and delta < 0.0:
        return FAILURE_MULTI_DILUTION
    if policy == POLICY_NO_STANDARD and delta > 0.0:
        return FAILURE_STANDARD_DILUTES
    if policy == POLICY_METADATA_WEIGHTED:
        return FAILURE_METADATA
    if policy == POLICY_SUPPORT_NELBO_WEIGHTED:
        return FAILURE_SUPPORT_WEIGHT
    if policy == POLICY_FIXED_TOTAL and delta < 0.0:
        return FAILURE_EXTRA_DRAWS
    if _float(row.get("center_1_delta_vs_c61_late")) < 0.0 and _float(row.get("center_3_delta_vs_c61_late")) < 0.0:
        return FAILURE_CENTER_1_3
    return FAILURE_NO_GAIN


def _policy_role(policy: str) -> str:
    if policy == POLICY_SAFE_MULTI:
        return "primary_uniform_dense_late_ensemble"
    if policy in {POLICY_SAFE_SINGLE, POLICY_FIXED_TOTAL}:
        return "required_budget_control"
    if policy in {POLICY_METADATA_WEIGHTED, POLICY_SUPPORT_NELBO_WEIGHTED}:
        return "secondary_weighted_dense_late_ensemble"
    if policy == POLICY_C61_REPLAY:
        return "replay_bridge"
    return "ablation"
