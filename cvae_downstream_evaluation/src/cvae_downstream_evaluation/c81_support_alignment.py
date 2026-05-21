"""C8.1 unlabeled target-support alignment over C6.3 geometric ensembles."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .c61_mixture import (
    _GenerationCache,
    _file_hash,
    _float,
    _load_c52_oracle_reference,
    _mean,
    _quantile,
    _support_conditions_from_rows,
    _to_numpy,
    _write_csv,
    load_csv_rows,
)
from .c62_late_ensemble import (
    C62_LEGACY_SUPPORT_UNITS,
    GLOBAL_CLASS_ORDER,
    _GeneratedMember,
    _fit_member_probabilities,
    _generate_member,
    _hard_vote_predictions,
    _member_probability_diagnostics,
    _np_array,
    _np_stack,
    _selector_visible_support_rows,
    align_probabilities_to_class_order,
    assert_c62_prejoin_rows_safe,
    ensemble_weight_diagnostics,
    fixed_predictions_from_probabilities,
)
from .c63_geometric_ensemble import (
    C63_ARTIFACTS_ROOT,
    C63_DEFAULT_C41_ROOT,
    C63_DEFAULT_C42_ROOT,
    C63_DEFAULT_C52_ROOT,
    C63_DEFAULT_C62_ROOT,
    C63RunLimits,
    GEOMETRIC_GENERATOR_FAMILY,
    GEOMETRIC_SOFTMAX_TEMPERATURE,
    LOG_PROBABILITY_EPSILON,
    POLICY_GEOM_SAFE_MULTI,
    _center_baseline,
    _empty_geometric_diagnostics,
    _geometric_diagnostics,
    _member_counts,
    _policy_role,
    _predictions_from_scores,
    build_c63_ensemble_plans,
    geometric_pool_probabilities,
    normalize_weights,
)
from .downstream import balanced_accuracy
from .matrix import (
    _domain,
    _label,
    _load_embedding_cache,
    _make_support_eval_split,
    _read_samples_manifest,
    _records_for_split,
    _sample_id,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .schemas import ENSEMBLE_EXPERT_ID, METHOD_BASELINE_ROW_TYPE
from .splits import assert_disjoint_ids


C81_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c81_unlabeled_support_alignment_geometric_ensemble_v1"
C81_DEFAULT_C41_ROOT = C63_DEFAULT_C41_ROOT
C81_DEFAULT_C42_ROOT = C63_DEFAULT_C42_ROOT
C81_DEFAULT_C52_ROOT = C63_DEFAULT_C52_ROOT
C81_DEFAULT_C62_ROOT = C63_DEFAULT_C62_ROOT
C81_DEFAULT_C63_ROOT = C63_ARTIFACTS_ROOT

POLICY_SUPPORT_ALIGNED_GEOM = "fixed_all_source_safe_multiseed_support_aligned_geometric_late_ensemble"
ALIGN_IDENTITY = "identity_no_alignment"
ALIGN_MEAN_ONLY = "mean_only"
ALIGN_DIAG_ALPHA025 = "diag_coral_alpha025"
ALIGN_DIAG_ALPHA05 = "mean_diagvar_shrinkage_coral_alpha05"
ALIGN_DIAG_ALPHA075 = "diag_coral_alpha075"
ALIGN_FULL_CORAL = "full_coral_shrinkage_diagnostic"

DEFAULT_ALIGNMENT_POLICIES = (
    ALIGN_IDENTITY,
    ALIGN_MEAN_ONLY,
    ALIGN_DIAG_ALPHA025,
    ALIGN_DIAG_ALPHA05,
    ALIGN_DIAG_ALPHA075,
)
PRIMARY_ALIGNMENT_POLICY = ALIGN_DIAG_ALPHA05

ALIGNMENT_GENERATOR_FAMILY = "family_c_pca64_unlabeled_support_aligned_geometric_ensemble_downstream_v1"
VARIANCE_FLOOR = 1.0e-4
SCALE_MIN = 0.5
SCALE_MAX = 2.0
MIN_SUPPORT_FOR_DIAGVAR = 8
SCALE_CLIP_FRAC_THRESHOLD = 0.50
BOOTSTRAP_REPLICATES = 20
BOOTSTRAP_SCALE_STD_THRESHOLD = 0.15

DECISION_TARGET_ALIGNMENT_GAIN = "C81_TARGET_ALIGNMENT_GAIN"
DECISION_085_CANDIDATE = "C81_085_ALIGNMENT_CANDIDATE"
FAILURE_REPLAY = "C63_REPLAY_MISMATCH"
FAILURE_NO_GAIN = "SUPPORT_ALIGNMENT_NO_GAIN"
FAILURE_OVERFITS = "SUPPORT_ALIGNMENT_OVERFITS_SMALL_SUPPORT"
FAILURE_CLASS_GEOMETRY = "SUPPORT_ALIGNMENT_DISTORTS_CLASS_GEOMETRY"
FAILURE_WEAK_CENTERS = "WEAK_CENTERS_REMAIN_CEILING"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"

REPLAY_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class C81RunLimits(C63RunLimits):
    pass


@dataclass(frozen=True)
class SupportEvalPools:
    support_indices: tuple[int, ...]
    eval_indices: tuple[int, ...]
    support_sample_ids: tuple[str, ...]
    eval_sample_ids: tuple[str, ...]
    support_eval_split_id: str
    target_eval_pool_id: str


@dataclass(frozen=True)
class AlignmentTransform:
    policy_requested: str
    policy_applied: str
    alpha: float
    mu_s: object
    mu_t: object
    mu_a: object
    scale: object
    linear_matrix: object | None
    fallback_trigger: str
    fallback_policy: str
    fallback_reason: str
    scale_clip_frac: float
    nan_or_inf_alignment: int


@dataclass(frozen=True)
class _C81Score:
    matrix_row: dict[str, object]
    member_rows: tuple[dict[str, object], ...]
    alignment_rows: tuple[dict[str, object], ...]
    class_geometry_rows: tuple[dict[str, object], ...]
    probability_row: dict[str, object]
    disagreement_row: dict[str, object]
    protocol_row: dict[str, object]


MATRIX_COLUMNS = (
    "ensemble_policy",
    "source_c63_policy",
    "alignment_policy",
    "alignment_policy_applied",
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
    "c63_primary_bacc",
    "delta_vs_c63_primary",
    "row_type",
    "n_synthetic_train",
    "n_target_support",
    "n_target_eval",
    "target_eval_pool_id",
    "candidate_expert",
    "candidate_experts_hash",
    "member_keys",
    "c63_member_bank_hash",
    "c81_member_bank_hash",
    "identity_replay_probability_hash",
    "identity_replay_member_count_match",
    "identity_replay_class_order_match",
    "num_members",
    "num_experts",
    "num_modes",
    "num_generation_seeds",
    "num_classifier_seeds",
    "effective_num_members",
    "effective_num_members_after_weighting",
    "weight_entropy",
    "max_member_weight",
    "min_member_weight",
    "member_budget_min",
    "member_budget_max",
    "aggregation_rule",
    "log_probability_epsilon",
    "geometric_softmax_temperature",
    "temperature_tuned",
    "prediction_rule",
    "alignment_space",
    "alignment_uses_target_support_x",
    "alignment_applied_to_synthetic_train_only",
    "target_support_labels_used",
    "target_eval_features_used_for_alignment",
    "target_eval_features_transformed",
    "target_eval_labels_used_for_alignment",
    "target_eval_labels_used_for_selection",
    "support_eval_disjoint",
    "status",
    "error_message",
)

MEMBER_COLUMNS = (
    "ensemble_policy",
    "alignment_policy",
    "alignment_policy_applied",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed_group",
    "classifier_seed",
    "member_key",
    "source_expert",
    "generation_mode",
    "mode_label",
    "generator_family",
    "generation_seed",
    "allocated_budget_per_class",
    "projection_artifact_path",
    "projection_artifact_hash",
    "generator_checkpoint_path",
    "generator_checkpoint_hash",
    "latent_prior_artifact_path",
    "latent_prior_artifact_hash",
    "weight",
    "weight_source",
    "target_support_labels_used",
    "target_eval_labels_used_for_member_fit",
)

ALIGNMENT_COLUMNS = (
    "ensemble_policy",
    "alignment_policy",
    "alignment_policy_applied",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed_group",
    "classifier_seed",
    "member_key",
    "source_expert",
    "mode_label",
    "generation_seed",
    "target_support_count",
    "synthetic_mean_l2_to_support_before",
    "synthetic_mean_l2_to_support_after",
    "synthetic_cov_trace_ratio_to_support_before",
    "synthetic_cov_trace_ratio_to_support_after",
    "synthetic_pairwise_distance_ratio_to_support_before",
    "synthetic_pairwise_distance_ratio_to_support_after",
    "support_effective_rank",
    "scale_clip_frac",
    "nan_or_inf_alignment",
    "support_bootstrap_mean_shift_std",
    "support_bootstrap_scale_std",
    "support_bootstrap_scale_std_mean",
    "support_bootstrap_unstable_flag",
    "fallback_trigger",
    "fallback_policy",
    "fallback_reason",
)

CLASS_GEOMETRY_COLUMNS = (
    "ensemble_policy",
    "alignment_policy",
    "alignment_policy_applied",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "generation_seed_group",
    "classifier_seed",
    "member_key",
    "source_expert",
    "mode_label",
    "generation_seed",
    "synthetic_class_centroid_distance_mean_before",
    "synthetic_class_centroid_distance_mean_after",
    "synthetic_within_class_trace_before",
    "synthetic_within_class_trace_after",
    "synthetic_fisher_ratio_before",
    "synthetic_fisher_ratio_after",
    "class_centroid_collapse_flag",
)

PROBABILITY_COLUMNS = (
    "ensemble_policy",
    "alignment_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "generation_seed_group",
    "classifier_seed",
    "member_entropy_before",
    "member_entropy_after",
    "ensemble_entropy_before",
    "ensemble_entropy_after",
    "member_disagreement_before",
    "member_disagreement_after",
    "probability_average_bacc_after",
    "geometric_bacc_after",
    "hard_vote_bacc_after",
)

DISAGREEMENT_COLUMNS = (
    "ensemble_policy",
    "alignment_policy",
    "heldout_center",
    "experiment_seed",
    "support_seed",
    "support_size",
    "classifier_seed",
    "mean_member_entropy_before",
    "mean_member_entropy_after",
    "ensemble_entropy_before",
    "ensemble_entropy_after",
    "member_disagreement_before",
    "member_disagreement_after",
    "bacc",
    "delta_vs_c63_primary",
)

PROTOCOL_COLUMNS = (
    "ensemble_policy",
    "alignment_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "heldout_source_excluded",
    "alignment_space",
    "alignment_uses_target_support_x",
    "alignment_applied_to_synthetic_train_only",
    "target_support_labels_used",
    "target_eval_features_used_for_alignment",
    "target_eval_features_transformed",
    "target_eval_labels_used_for_alignment",
    "target_eval_labels_used_for_selection",
    "support_eval_disjoint",
    "target_eval_threshold_search",
    "checkpoints_retrained",
    "member_bank_hash_match",
    "identity_replay_member_count_match",
    "identity_replay_class_order_match",
    "protocol_status",
)

SUMMARY_COLUMNS = (
    "alignment_policy",
    "n_rows",
    "mean_bacc",
    "ensemble_ge_080_rate",
    "mean_oracle_bacc_reference",
    "mean_regret_bacc",
    "regret_p50",
    "regret_p75",
    "regret_p90",
    "c63_primary_mean_bacc",
    "mean_delta_vs_c63_primary",
    "paired_positive_seed_cells_vs_c63",
    "paired_seed_cells_vs_c63",
    "identity_replay_max_abs_bacc_delta",
    "identity_replay_matches_within_tolerance",
    "member_bank_hash_match_rate",
    "center_1_delta_vs_c63",
    "center_3_delta_vs_c63",
    "weak_center_gain_ge_003_count",
    "strong_center_degrade_gt_002_count",
    "seed_positive_gain_rate",
    "class_centroid_collapse_rate",
    "support_bootstrap_unstable_rate",
    "decision_label",
)

CENTER_COLUMNS = (
    "alignment_policy",
    "heldout_center",
    "n_rows",
    "mean_bacc",
    "ensemble_ge_080_rate",
    "c63_primary_bacc",
    "delta_vs_c63_primary",
    "mean_oracle_bacc_reference",
    "regret_p75",
)


def run_c81_support_alignment(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    c52_artifacts_root: Path,
    c62_artifacts_root: Path,
    c63_artifacts_root: Path,
    device: str,
    limits: C81RunLimits = C81RunLimits(),
    enable_full_coral_diagnostic: bool = False,
) -> dict[str, Path]:
    support_unit_rows = load_csv_rows(c41_artifacts_root / "tables" / "support_selection_units.csv")
    combined_support_rows = list(support_unit_rows)
    legacy_support_path = repo_root / C62_LEGACY_SUPPORT_UNITS
    if legacy_support_path.exists():
        combined_support_rows.extend(load_csv_rows(legacy_support_path))
    assert_c62_prejoin_rows_safe(_selector_visible_support_rows(combined_support_rows))

    c63_primary_rows = _load_c63_primary_rows(c63_artifacts_root)
    c63_by_condition = _bacc_by_condition(c63_primary_rows)
    c63_by_center = _center_baseline(c63_primary_rows)
    c52_oracle = _load_c52_oracle_reference(c52_artifacts_root)

    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    experiment_seed_filter = set(int(v) for v in limits.experiment_seeds) if limits.experiment_seeds else None
    alignment_policies = list(DEFAULT_ALIGNMENT_POLICIES)
    if enable_full_coral_diagnostic:
        alignment_policies.append(ALIGN_FULL_CORAL)

    member_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    class_geometry_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    disagreement_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []

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
            target_union_pool = build_target_eval_pool(
                test_metadata=test_cache.metadata,
                heldout_center=heldout,
                support_sizes=config.support_sizes,
                support_seeds=config.support_seeds,
            )
            target_labels = [_label(test_cache.metadata[idx]) for idx in target_union_pool.eval_indices]
            if tuple(sorted(set(target_labels).union({0, 1}))) != GLOBAL_CLASS_ORDER:
                raise ProtocolError(f"C8.1 expects binary labels {GLOBAL_CLASS_ORDER}, got {sorted(set(target_labels))}")
            target_dino = test_cache.embeddings[list(target_union_pool.eval_indices)].detach().cpu().float()

            for support_size, support_seed, support_eval_split_id in support_conditions:
                pools = build_unlabeled_support_eval_pools(
                    test_metadata=test_cache.metadata,
                    heldout_center=heldout,
                    support_size=int(support_size),
                    support_seed=int(support_seed),
                    support_eval_split_id=support_eval_split_id,
                    union_eval_indices=target_union_pool.eval_indices,
                    union_target_eval_pool_id=target_union_pool.target_eval_pool_id,
                )
                support_dino = test_cache.embeddings[list(pools.support_indices)].detach().cpu().float()
                support_rows_for_condition = [
                    row
                    for row in combined_support_rows
                    if int(row.get("experiment_seed", -1)) == experiment_seed
                    and str(row.get("heldout_center")) == heldout
                    and int(row.get("support_size", -1)) == int(support_size)
                    and int(row.get("support_seed", -1)) == int(support_seed)
                ]
                plans = build_c63_ensemble_plans(
                    policies=(POLICY_GEOM_SAFE_MULTI,),
                    candidates=candidates,
                    total_budget_per_class=int(config.primary_budget_per_class),
                    generation_seeds=selected_generation_seeds,
                    support_rows=support_rows_for_condition,
                )
                if len(plans) != 1:
                    raise ProtocolError("C8.1 expects exactly one C6.3 safe multiseed plan.")
                plan = plans[0]
                generated = [
                    _generate_member(cache=generation_cache, spec=spec, label_values=GLOBAL_CLASS_ORDER)
                    for spec in plan.specs
                ]
                for classifier_seed in selected_classifier_seeds:
                    c63_key = (experiment_seed, heldout, int(support_size), int(support_seed), int(classifier_seed))
                    c63_primary_bacc = c63_by_condition.get(c63_key, math.nan)
                    c63_member_hash = _c63_member_bank_hash(
                        c63_primary_rows,
                        experiment_seed=experiment_seed,
                        heldout_center=heldout,
                        support_size=int(support_size),
                        support_seed=int(support_seed),
                        classifier_seed=int(classifier_seed),
                    )
                    for alignment_policy in alignment_policies:
                        score = _score_c81_row(
                            plan=plan,
                            alignment_policy=alignment_policy,
                            experiment_seed=experiment_seed,
                            heldout_center=heldout,
                            support_size=int(support_size),
                            support_seed=int(support_seed),
                            support_eval_split_id=support_eval_split_id,
                            classifier_seed=int(classifier_seed),
                            budget_per_class=int(config.primary_budget_per_class),
                            generated=generated,
                            target_support_dino=support_dino,
                            target_dino=target_dino,
                            target_labels=target_labels,
                            target_eval_pool_id=pools.target_eval_pool_id,
                            oracle_reference=c52_oracle.get(
                                (experiment_seed, heldout, int(support_size), int(support_seed)),
                                math.nan,
                            ),
                            c63_primary_bacc=c63_primary_bacc,
                            c63_member_bank_hash=c63_member_hash,
                            support_eval_disjoint=int(
                                set(pools.support_sample_ids).isdisjoint(set(pools.eval_sample_ids))
                            ),
                        )
                        matrix_rows.append(score.matrix_row)
                        member_rows.extend(score.member_rows)
                        alignment_rows.extend(score.alignment_rows)
                        class_geometry_rows.extend(score.class_geometry_rows)
                        probability_rows.append(score.probability_row)
                        disagreement_rows.append(score.disagreement_row)
                        protocol_rows.append(score.protocol_row)

    outputs = {
        "matrix": artifacts_root / "tables" / "c81_support_alignment_matrix.csv",
        "members": artifacts_root / "tables" / "c81_ensemble_members_pre_join.csv",
        "alignment": artifacts_root / "tables" / "c81_alignment_diagnostics.csv",
        "variant": artifacts_root / "tables" / "c81_alignment_variant_comparison.csv",
        "geometry": artifacts_root / "tables" / "c81_class_geometry_diagnostics.csv",
        "probability": artifacts_root / "tables" / "c81_probability_diagnostics.csv",
        "disagreement": artifacts_root / "tables" / "c81_member_disagreement_diagnostics.csv",
        "center": artifacts_root / "tables" / "c81_center_summary.csv",
        "threshold": artifacts_root / "tables" / "c81_threshold_audit.csv",
        "protocol": artifacts_root / "tables" / "c81_protocol_audit.csv",
    }
    assert_c62_prejoin_rows_safe(member_rows)
    _write_csv(outputs["matrix"], MATRIX_COLUMNS, matrix_rows)
    _write_csv(outputs["members"], MEMBER_COLUMNS, member_rows)
    _write_csv(outputs["alignment"], ALIGNMENT_COLUMNS, alignment_rows)
    _write_csv(outputs["variant"], SUMMARY_COLUMNS, build_c81_threshold_rows(matrix_rows, c63_primary_rows, alignment_rows, class_geometry_rows))
    _write_csv(outputs["geometry"], CLASS_GEOMETRY_COLUMNS, class_geometry_rows)
    _write_csv(outputs["probability"], PROBABILITY_COLUMNS, probability_rows)
    _write_csv(outputs["disagreement"], DISAGREEMENT_COLUMNS, disagreement_rows)
    _write_csv(outputs["center"], CENTER_COLUMNS, build_c81_center_rows(matrix_rows, c63_by_center))
    _write_csv(outputs["threshold"], SUMMARY_COLUMNS, build_c81_threshold_rows(matrix_rows, c63_primary_rows, alignment_rows, class_geometry_rows))
    _write_csv(outputs["protocol"], PROTOCOL_COLUMNS, protocol_rows)
    return outputs


def build_unlabeled_support_eval_pools(
    *,
    test_metadata: Sequence[Mapping[str, object]],
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    union_eval_indices: Sequence[int],
    union_target_eval_pool_id: str,
) -> SupportEvalPools:
    target_indices = tuple(
        idx for idx, row in enumerate(test_metadata) if str(_domain(row)) == str(heldout_center)
    )
    dummy_labels = {idx: 0 for idx in target_indices}
    split = _make_support_eval_split(
        target_domain=int(heldout_center),
        target_indices=target_indices,
        labels_by_index=dummy_labels,
        support_size=int(support_size),
        sampling_policy="random",
        support_seed=int(support_seed),
    )
    if int(getattr(split, "support_labels_used", 0)) != 0:
        raise ProtocolError("C8.1 support split attempted to use target-support labels.")
    if str(getattr(split, "split_status", "ok")) != "ok":
        raise ProtocolError(f"C8.1 support split is not ok: {getattr(split, 'split_status', '')}")
    support_indices = tuple(int(idx) for idx in split.support_indices)
    eval_indices = tuple(int(idx) for idx in union_eval_indices)
    support_ids = tuple(str(_sample_id(test_metadata[idx])) for idx in support_indices)
    eval_ids = tuple(str(_sample_id(test_metadata[idx])) for idx in eval_indices)
    assert_disjoint_ids(support_ids, eval_ids)
    split_id = str(getattr(split, "support_eval_split_id", support_eval_split_id) or support_eval_split_id)
    return SupportEvalPools(
        support_indices=support_indices,
        eval_indices=eval_indices,
        support_sample_ids=tuple(sorted(support_ids)),
        eval_sample_ids=tuple(sorted(eval_ids)),
        support_eval_split_id=split_id,
        target_eval_pool_id=union_target_eval_pool_id,
    )


def fit_alignment_transform(
    synthetic_x: object,
    support_x: object,
    *,
    policy: str,
    alpha: float,
    bootstrap_seed: int = 0,
) -> tuple[AlignmentTransform, dict[str, float]]:
    import numpy as np  # type: ignore

    x_s = np.asarray(synthetic_x, dtype=float)
    x_t = np.asarray(support_x, dtype=float)
    if x_s.ndim != 2 or x_t.ndim != 2:
        raise ProtocolError("C8.1 alignment expects 2D synthetic/support features.")
    if x_s.shape[1] != x_t.shape[1]:
        raise ProtocolError("C8.1 synthetic/support DINO dimensions do not match.")
    requested = str(policy)
    policy_applied = requested
    fallback_trigger = ""
    fallback_policy = ""
    fallback_reason = ""
    if requested == ALIGN_IDENTITY:
        alpha = 0.0
    elif requested == ALIGN_MEAN_ONLY:
        alpha = 1.0
    elif requested == ALIGN_DIAG_ALPHA025:
        alpha = 0.25
    elif requested == ALIGN_DIAG_ALPHA05:
        alpha = 0.50
    elif requested == ALIGN_DIAG_ALPHA075:
        alpha = 0.75
    elif requested == ALIGN_FULL_CORAL:
        alpha = 1.0
        if x_t.shape[0] < 2 * x_t.shape[1]:
            policy_applied = ALIGN_MEAN_ONLY
            alpha = 1.0
            fallback_trigger = "full_coral_support_lt_2d"
            fallback_policy = ALIGN_MEAN_ONLY
            fallback_reason = "full CORAL disabled for unstable high-dimensional support covariance"
    else:
        raise ProtocolError(f"Unknown C8.1 alignment policy: {requested}")

    if policy_applied not in {ALIGN_IDENTITY, ALIGN_MEAN_ONLY, ALIGN_FULL_CORAL} and x_t.shape[0] < MIN_SUPPORT_FOR_DIAGVAR:
        policy_applied = ALIGN_MEAN_ONLY
        alpha = 1.0
        fallback_trigger = "n_support_lt_min_support_for_diagvar"
        fallback_policy = ALIGN_MEAN_ONLY
        fallback_reason = f"n_support={x_t.shape[0]} < {MIN_SUPPORT_FOR_DIAGVAR}"

    mu_s = x_s.mean(axis=0)
    mu_t = x_t.mean(axis=0)
    var_s = x_s.var(axis=0)
    var_t = x_t.var(axis=0)
    if policy_applied == ALIGN_IDENTITY:
        mu_a = mu_s.copy()
        scale = np.ones_like(mu_s)
        linear_matrix = None
    elif policy_applied == ALIGN_MEAN_ONLY:
        mu_a = mu_t.copy()
        scale = np.ones_like(mu_s)
        linear_matrix = None
    elif policy_applied == ALIGN_FULL_CORAL:
        mu_a = mu_t.copy()
        scale = np.ones_like(mu_s)
        linear_matrix = _full_coral_matrix(x_s, x_t, alpha=float(alpha))
    else:
        mu_a = (1.0 - float(alpha)) * mu_s + float(alpha) * mu_t
        var_a = (1.0 - float(alpha)) * var_s + float(alpha) * var_t
        raw_scale = np.sqrt(np.maximum(var_a, VARIANCE_FLOOR) / np.maximum(var_s, VARIANCE_FLOOR))
        scale = np.clip(raw_scale, SCALE_MIN, SCALE_MAX)
        linear_matrix = None
    scale_clip_frac = float(np.mean((scale <= SCALE_MIN + 1.0e-12) | (scale >= SCALE_MAX - 1.0e-12)))
    matrix_finite = True if linear_matrix is None else bool(np.isfinite(linear_matrix).all())
    nan_or_inf = int(not np.isfinite(mu_a).all() or not np.isfinite(scale).all() or not matrix_finite)
    if (scale_clip_frac > SCALE_CLIP_FRAC_THRESHOLD or nan_or_inf) and policy_applied not in {ALIGN_IDENTITY, ALIGN_MEAN_ONLY}:
        policy_applied = ALIGN_MEAN_ONLY
        alpha = 1.0
        mu_a = mu_t.copy()
        scale = np.ones_like(mu_s)
        linear_matrix = None
        fallback_trigger = "scale_clip_or_nonfinite"
        fallback_policy = ALIGN_MEAN_ONLY
        fallback_reason = f"scale_clip_frac={scale_clip_frac:.6f}, nan_or_inf={nan_or_inf}"
        scale_clip_frac = 0.0
        nan_or_inf = 0
    transform = AlignmentTransform(
        policy_requested=requested,
        policy_applied=policy_applied,
        alpha=float(alpha),
        mu_s=mu_s,
        mu_t=mu_t,
        mu_a=mu_a,
        scale=scale,
        linear_matrix=linear_matrix,
        fallback_trigger=fallback_trigger,
        fallback_policy=fallback_policy,
        fallback_reason=fallback_reason,
        scale_clip_frac=scale_clip_frac,
        nan_or_inf_alignment=nan_or_inf,
    )
    bootstrap = _support_bootstrap_diagnostics(x_s, x_t, policy=policy_applied, alpha=float(alpha), seed=int(bootstrap_seed))
    return transform, bootstrap


def apply_alignment_transform(x: object, transform: AlignmentTransform) -> object:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    centered = arr - np.asarray(transform.mu_s, dtype=float)
    if transform.linear_matrix is not None:
        return np.asarray(transform.mu_a, dtype=float) + centered @ np.asarray(transform.linear_matrix, dtype=float)
    return np.asarray(transform.mu_a, dtype=float) + np.asarray(transform.scale, dtype=float) * centered


def hash_member_bank(specs: Sequence[object]) -> str:
    payload = json.dumps(
        {
            "member_keys": sorted(str(spec.member_key) for spec in specs),
            "class_order": GLOBAL_CLASS_ORDER,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _score_c81_row(
    *,
    plan: object,
    alignment_policy: str,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    classifier_seed: int,
    budget_per_class: int,
    generated: Sequence[_GeneratedMember],
    target_support_dino: object,
    target_dino: object,
    target_labels: Sequence[int],
    target_eval_pool_id: str,
    oracle_reference: float,
    c63_primary_bacc: float,
    c63_member_bank_hash: str,
    support_eval_disjoint: int,
) -> _C81Score:
    try:
        import numpy as np  # type: ignore

        probs_after = []
        probs_before = []
        hard_after = []
        hard_before = []
        weights = []
        member_rows = []
        alignment_rows = []
        class_geometry_rows = []
        c81_member_hash = hash_member_bank(plan.specs)
        support_arr = np.asarray(_to_numpy(target_support_dino), dtype=float)
        target_arr = np.asarray(_to_numpy(target_dino), dtype=float)
        for member in generated:
            x_before = np.asarray(_to_numpy(member.generated.synthetic_dino), dtype=float)
            y_syn = tuple(int(v) for v in member.generated.synthetic_labels)
            transform, bootstrap = fit_alignment_transform(
                x_before,
                support_arr,
                policy=alignment_policy,
                alpha=0.5,
                bootstrap_seed=_stable_int_seed(experiment_seed, heldout_center, support_size, support_seed, classifier_seed, member.spec.member_key, alignment_policy),
            )
            x_after = apply_alignment_transform(x_before, transform)
            before_prediction = _fit_member_probabilities(
                synthetic_embeddings=x_before,
                synthetic_labels=y_syn,
                target_embeddings=target_arr,
                classifier_seed=classifier_seed,
            )
            after_prediction = _fit_member_probabilities(
                synthetic_embeddings=x_after,
                synthetic_labels=y_syn,
                target_embeddings=target_arr,
                classifier_seed=classifier_seed,
            )
            before_aligned = align_probabilities_to_class_order(before_prediction["probabilities"], before_prediction["classes"], GLOBAL_CLASS_ORDER)
            after_aligned = align_probabilities_to_class_order(after_prediction["probabilities"], after_prediction["classes"], GLOBAL_CLASS_ORDER)
            probs_before.append(before_aligned)
            probs_after.append(after_aligned)
            hard_before.append(fixed_predictions_from_probabilities(before_aligned, GLOBAL_CLASS_ORDER))
            hard_after.append(fixed_predictions_from_probabilities(after_aligned, GLOBAL_CLASS_ORDER))
            weights.append(float(member.spec.weight))
            member_rows.append(_member_row(plan, member, transform, experiment_seed, heldout_center, support_size, support_seed, support_eval_split_id, classifier_seed))
            alignment_rows.append(
                {
                    "ensemble_policy": POLICY_SUPPORT_ALIGNED_GEOM,
                    "alignment_policy": alignment_policy,
                    "alignment_policy_applied": transform.policy_applied,
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": heldout_center,
                    "support_size": int(support_size),
                    "support_seed": int(support_seed),
                    "support_eval_split_id": support_eval_split_id,
                    "generation_seed_group": plan.generation_seed_group,
                    "classifier_seed": int(classifier_seed),
                    "member_key": member.spec.member_key,
                    "source_expert": member.spec.source_expert,
                    "mode_label": member.spec.bank.mode_label,
                    "generation_seed": int(member.spec.generation_seed),
                    "target_support_count": int(support_arr.shape[0]),
                    **_alignment_distribution_diagnostics(x_before, x_after, support_arr),
                    "support_effective_rank": _effective_rank(support_arr),
                    "scale_clip_frac": transform.scale_clip_frac,
                    "nan_or_inf_alignment": transform.nan_or_inf_alignment,
                    **bootstrap,
                    "fallback_trigger": transform.fallback_trigger,
                    "fallback_policy": transform.fallback_policy,
                    "fallback_reason": transform.fallback_reason,
                }
            )
            class_geometry_rows.append(
                {
                    "ensemble_policy": POLICY_SUPPORT_ALIGNED_GEOM,
                    "alignment_policy": alignment_policy,
                    "alignment_policy_applied": transform.policy_applied,
                    "experiment_seed": int(experiment_seed),
                    "heldout_center": heldout_center,
                    "support_size": int(support_size),
                    "support_seed": int(support_seed),
                    "support_eval_split_id": support_eval_split_id,
                    "generation_seed_group": plan.generation_seed_group,
                    "classifier_seed": int(classifier_seed),
                    "member_key": member.spec.member_key,
                    "source_expert": member.spec.source_expert,
                    "mode_label": member.spec.bank.mode_label,
                    "generation_seed": int(member.spec.generation_seed),
                    **_class_geometry_diagnostics(x_before, x_after, y_syn),
                }
            )
        weights_arr = _np_array(normalize_weights(weights))
        stacked_after = _np_stack(probs_after)
        stacked_before = _np_stack(probs_before)
        mean_after = (stacked_after * weights_arr[:, None, None]).sum(axis=0)
        _, geom_after = geometric_pool_probabilities(stacked_after, weights_arr)
        _, geom_before = geometric_pool_probabilities(stacked_before, weights_arr)
        hard_vote_after = _hard_vote_predictions(hard_after, GLOBAL_CLASS_ORDER)
        probability_pred_after = fixed_predictions_from_probabilities(mean_after, GLOBAL_CLASS_ORDER)
        geometric_scores_after, geometric_prob_after = geometric_pool_probabilities(stacked_after, weights_arr)
        geometric_pred_after = _predictions_from_scores(geometric_scores_after, GLOBAL_CLASS_ORDER)
        primary_prob = geom_after
        primary_pred = geometric_pred_after
        primary_metrics = _score_predictions_and_probabilities(target_labels, primary_pred, primary_prob, GLOBAL_CLASS_ORDER)
        probability_metrics = _score_predictions_and_probabilities(target_labels, probability_pred_after, mean_after, GLOBAL_CLASS_ORDER)
        geometric_metrics = _score_predictions_and_probabilities(target_labels, geometric_pred_after, geometric_prob_after, GLOBAL_CLASS_ORDER)
        hard_bacc = balanced_accuracy(target_labels, hard_vote_after)
        before_pred = fixed_predictions_from_probabilities(geom_before, GLOBAL_CLASS_ORDER)
        before_bacc = balanced_accuracy(target_labels, before_pred)
        entropy = ensemble_weight_diagnostics([member.spec for member in generated])
        member_counts = _member_counts([member.spec for member in generated])
        after_diag = _geometric_diagnostics(
            stacked=stacked_after,
            mean_prob=mean_after,
            geometric_prob=geometric_prob_after,
            geometric_pred=geometric_pred_after,
            hard_preds=hard_after,
            target_labels=target_labels,
            member_entropy_values=[_member_probability_diagnostics(prob)["member_entropy"] for prob in probs_after],
        )
        before_diag = _geometric_diagnostics(
            stacked=stacked_before,
            mean_prob=(stacked_before * weights_arr[:, None, None]).sum(axis=0),
            geometric_prob=geom_before,
            geometric_pred=before_pred,
            hard_preds=hard_before,
            target_labels=target_labels,
            member_entropy_values=[_member_probability_diagnostics(prob)["member_entropy"] for prob in probs_before],
        )
        total_train = sum(len(member.generated.synthetic_labels) for member in generated)
        status = "ok"
        error = ""
        prob_hash = _probability_hash(geometric_prob_after)
    except Exception as exc:
        primary_metrics = probability_metrics = geometric_metrics = {"bacc": math.nan, "macro_f1": math.nan, "auroc": math.nan, "auprc": math.nan}
        hard_bacc = math.nan
        entropy = ensemble_weight_diagnostics([member.spec for member in generated])
        member_counts = _member_counts([member.spec for member in generated])
        after_diag = before_diag = _empty_geometric_diagnostics()
        total_train = sum(len(member.generated.synthetic_labels) for member in generated)
        status = "failed_c81_support_alignment_scoring"
        error = str(exc)
        member_rows = []
        alignment_rows = []
        class_geometry_rows = []
        c81_member_hash = hash_member_bank(plan.specs)
        prob_hash = ""
        before_bacc = math.nan
    bacc = float(primary_metrics["bacc"])
    member_hash_match = int(bool(c63_member_bank_hash) and c63_member_bank_hash == c81_member_hash)
    matrix_row = {
        "ensemble_policy": POLICY_SUPPORT_ALIGNED_GEOM,
        "source_c63_policy": POLICY_GEOM_SAFE_MULTI,
        "alignment_policy": alignment_policy,
        "alignment_policy_applied": _common_applied_policy(alignment_rows),
        "policy_role": "primary" if alignment_policy == PRIMARY_ALIGNMENT_POLICY else "diagnostic",
        "diagnostic_only": int(alignment_policy != PRIMARY_ALIGNMENT_POLICY),
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "generation_seed_group": plan.generation_seed_group,
        "classifier_seed": int(classifier_seed),
        "generator_family": ALIGNMENT_GENERATOR_FAMILY,
        "generation_mode": alignment_policy,
        "budget_per_class": int(budget_per_class),
        "bacc": bacc,
        "macro_f1": primary_metrics["macro_f1"],
        "auroc": primary_metrics["auroc"],
        "auprc": primary_metrics["auprc"],
        "ensemble_ge_080": int(not math.isnan(bacc) and bacc >= 0.80),
        "oracle_bacc_reference": oracle_reference,
        "regret_bacc": oracle_reference - bacc if not math.isnan(_float(oracle_reference)) and not math.isnan(bacc) else math.nan,
        "c63_primary_bacc": c63_primary_bacc,
        "delta_vs_c63_primary": bacc - _float(c63_primary_bacc) if not math.isnan(_float(c63_primary_bacc)) and not math.isnan(bacc) else math.nan,
        "row_type": METHOD_BASELINE_ROW_TYPE,
        "n_synthetic_train": int(total_train),
        "n_target_support": int(len(_to_numpy(target_support_dino))),
        "n_target_eval": len(target_labels),
        "target_eval_pool_id": target_eval_pool_id,
        "candidate_expert": ENSEMBLE_EXPERT_ID,
        "candidate_experts_hash": hash_candidate_experts(member.spec.member_key for member in generated),
        "member_keys": ";".join(member.spec.member_key for member in generated),
        "c63_member_bank_hash": c63_member_bank_hash,
        "c81_member_bank_hash": c81_member_hash,
        "identity_replay_probability_hash": prob_hash if alignment_policy == ALIGN_IDENTITY else "",
        "identity_replay_member_count_match": int(len(generated) > 0),
        "identity_replay_class_order_match": 1,
        **member_counts,
        **entropy,
        "effective_num_members_after_weighting": entropy.get("effective_num_members", math.nan),
        "aggregation_rule": "weighted_log_probability_geometric_pooling",
        "log_probability_epsilon": LOG_PROBABILITY_EPSILON,
        "geometric_softmax_temperature": GEOMETRIC_SOFTMAX_TEMPERATURE,
        "temperature_tuned": 0,
        "prediction_rule": "argmax_weighted_log_probability",
        "alignment_space": "dino_original",
        "alignment_uses_target_support_x": 1,
        "alignment_applied_to_synthetic_train_only": 1,
        "target_support_labels_used": 0,
        "target_eval_features_used_for_alignment": 0,
        "target_eval_features_transformed": 0,
        "target_eval_labels_used_for_alignment": 0,
        "target_eval_labels_used_for_selection": 0,
        "support_eval_disjoint": int(support_eval_disjoint),
        "status": status,
        "error_message": error,
    }
    probability_row = {
        "ensemble_policy": POLICY_SUPPORT_ALIGNED_GEOM,
        "alignment_policy": alignment_policy,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "generation_seed_group": plan.generation_seed_group,
        "classifier_seed": int(classifier_seed),
        "member_entropy_before": before_diag["mean_member_entropy"],
        "member_entropy_after": after_diag["mean_member_entropy"],
        "ensemble_entropy_before": before_diag["geometric_entropy"],
        "ensemble_entropy_after": after_diag["geometric_entropy"],
        "member_disagreement_before": before_diag["member_disagreement_entropy"],
        "member_disagreement_after": after_diag["member_disagreement_entropy"],
        "probability_average_bacc_after": probability_metrics["bacc"],
        "geometric_bacc_after": geometric_metrics["bacc"],
        "hard_vote_bacc_after": hard_bacc,
    }
    disagreement_row = {
        "ensemble_policy": POLICY_SUPPORT_ALIGNED_GEOM,
        "alignment_policy": alignment_policy,
        "heldout_center": heldout_center,
        "experiment_seed": int(experiment_seed),
        "support_seed": int(support_seed),
        "support_size": int(support_size),
        "classifier_seed": int(classifier_seed),
        "mean_member_entropy_before": before_diag["mean_member_entropy"],
        "mean_member_entropy_after": after_diag["mean_member_entropy"],
        "ensemble_entropy_before": before_diag["geometric_entropy"],
        "ensemble_entropy_after": after_diag["geometric_entropy"],
        "member_disagreement_before": before_diag["member_disagreement_entropy"],
        "member_disagreement_after": after_diag["member_disagreement_entropy"],
        "bacc": bacc,
        "delta_vs_c63_primary": bacc - _float(c63_primary_bacc) if not math.isnan(_float(c63_primary_bacc)) and not math.isnan(bacc) else math.nan,
    }
    protocol_pass = (
        int(support_eval_disjoint) == 1
        and member_hash_match == 1
        and matrix_row["target_support_labels_used"] == 0
        and matrix_row["target_eval_features_used_for_alignment"] == 0
        and matrix_row["target_eval_features_transformed"] == 0
        and matrix_row["target_eval_labels_used_for_selection"] == 0
    )
    protocol_row = {
        "ensemble_policy": POLICY_SUPPORT_ALIGNED_GEOM,
        "alignment_policy": alignment_policy,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "heldout_source_excluded": int(str(heldout_center) not in {str(member.spec.source_expert) for member in generated}),
        "alignment_space": "dino_original",
        "alignment_uses_target_support_x": 1,
        "alignment_applied_to_synthetic_train_only": 1,
        "target_support_labels_used": 0,
        "target_eval_features_used_for_alignment": 0,
        "target_eval_features_transformed": 0,
        "target_eval_labels_used_for_alignment": 0,
        "target_eval_labels_used_for_selection": 0,
        "support_eval_disjoint": int(support_eval_disjoint),
        "target_eval_threshold_search": 0,
        "checkpoints_retrained": 0,
        "member_bank_hash_match": member_hash_match,
        "identity_replay_member_count_match": matrix_row["identity_replay_member_count_match"],
        "identity_replay_class_order_match": matrix_row["identity_replay_class_order_match"],
        "protocol_status": "pass" if protocol_pass else "fail",
    }
    _ = before_bacc
    return _C81Score(
        matrix_row=matrix_row,
        member_rows=tuple(member_rows),
        alignment_rows=tuple(alignment_rows),
        class_geometry_rows=tuple(class_geometry_rows),
        probability_row=probability_row,
        disagreement_row=disagreement_row,
        protocol_row=protocol_row,
    )


def build_c81_threshold_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    c63_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    class_geometry_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    c63_mean = _mean(_float(row.get("bacc")) for row in c63_rows)
    out = []
    for policy in sorted({str(row.get("alignment_policy")) for row in matrix_rows}):
        subset = [row for row in matrix_rows if str(row.get("alignment_policy")) == policy and str(row.get("status")) == "ok"]
        values = [_float(row.get("bacc")) for row in subset]
        deltas = [_float(row.get("delta_vs_c63_primary")) for row in subset]
        regrets = [_float(row.get("regret_bacc")) for row in subset]
        replay_delta = _identity_replay_max_abs_delta(subset) if policy == ALIGN_IDENTITY else math.nan
        row = {
            "alignment_policy": policy,
            "n_rows": len(subset),
            "mean_bacc": _mean(values),
            "ensemble_ge_080_rate": _mean(1.0 if value >= 0.80 else 0.0 for value in values),
            "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
            "mean_regret_bacc": _mean(regrets),
            "regret_p50": _quantile(regrets, 0.50),
            "regret_p75": _quantile(regrets, 0.75),
            "regret_p90": _quantile(regrets, 0.90),
            "c63_primary_mean_bacc": c63_mean,
            "mean_delta_vs_c63_primary": _mean(deltas),
            "paired_positive_seed_cells_vs_c63": sum(1 for value in deltas if not math.isnan(value) and value > 0.0),
            "paired_seed_cells_vs_c63": sum(1 for value in deltas if not math.isnan(value)),
            "identity_replay_max_abs_bacc_delta": replay_delta,
            "identity_replay_matches_within_tolerance": int(policy != ALIGN_IDENTITY or (not math.isnan(replay_delta) and replay_delta <= REPLAY_TOLERANCE)),
            "member_bank_hash_match_rate": _mean(1.0 if str(row.get("c63_member_bank_hash")) == str(row.get("c81_member_bank_hash")) and row.get("c63_member_bank_hash") else 0.0 for row in subset),
            "center_1_delta_vs_c63": _center_delta(subset, "1"),
            "center_3_delta_vs_c63": _center_delta(subset, "3"),
            "weak_center_gain_ge_003_count": _weak_center_gain_count(subset),
            "strong_center_degrade_gt_002_count": _strong_center_degrade_count(subset),
            "seed_positive_gain_rate": _seed_positive_gain_rate(subset),
            "class_centroid_collapse_rate": _class_collapse_rate(class_geometry_rows, policy),
            "support_bootstrap_unstable_rate": _support_unstable_rate(alignment_rows, policy),
            "decision_label": "",
        }
        row["decision_label"] = _decision_label(row)
        out.append(row)
    return out


def build_c81_center_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    c63_by_center: Mapping[str, float],
) -> list[dict[str, object]]:
    out = []
    for policy in sorted({str(row.get("alignment_policy")) for row in matrix_rows}):
        for center in sorted({str(row.get("heldout_center")) for row in matrix_rows}):
            subset = [
                row
                for row in matrix_rows
                if str(row.get("alignment_policy")) == policy
                and str(row.get("heldout_center")) == center
                and str(row.get("status")) == "ok"
            ]
            if not subset:
                continue
            mean_bacc = _mean(_float(row.get("bacc")) for row in subset)
            c63_value = _float(c63_by_center.get(center))
            out.append(
                {
                    "alignment_policy": policy,
                    "heldout_center": center,
                    "n_rows": len(subset),
                    "mean_bacc": mean_bacc,
                    "ensemble_ge_080_rate": _mean(1.0 if _float(row.get("bacc")) >= 0.80 else 0.0 for row in subset),
                    "c63_primary_bacc": c63_value,
                    "delta_vs_c63_primary": mean_bacc - c63_value if not math.isnan(c63_value) else math.nan,
                    "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
                    "regret_p75": _quantile([_float(row.get("regret_bacc")) for row in subset], 0.75),
                }
            )
    return out


def _member_row(
    plan: object,
    member: _GeneratedMember,
    transform: AlignmentTransform,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    support_eval_split_id: str,
    classifier_seed: int,
) -> dict[str, object]:
    item = member.generated
    return {
        "ensemble_policy": POLICY_SUPPORT_ALIGNED_GEOM,
        "alignment_policy": transform.policy_requested,
        "alignment_policy_applied": transform.policy_applied,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "generation_seed_group": plan.generation_seed_group,
        "classifier_seed": int(classifier_seed),
        "member_key": member.spec.member_key,
        "source_expert": member.spec.source_expert,
        "generation_mode": member.spec.bank.generation_mode,
        "mode_label": member.spec.bank.mode_label,
        "generator_family": member.spec.bank.generator_family,
        "generation_seed": int(member.spec.generation_seed),
        "allocated_budget_per_class": int(member.spec.allocated_budget_per_class),
        "projection_artifact_path": str(item.projection_path),
        "projection_artifact_hash": _file_hash(item.projection_path),
        "generator_checkpoint_path": str(item.checkpoint_path),
        "generator_checkpoint_hash": _file_hash(item.checkpoint_path),
        "latent_prior_artifact_path": ";".join(str(path) for path in item.latent_prior_paths),
        "latent_prior_artifact_hash": ";".join(_file_hash(path) for path in item.latent_prior_paths),
        "weight": float(member.spec.weight),
        "weight_source": member.spec.weight_source,
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_member_fit": 0,
    }


def _alignment_distribution_diagnostics(x_before: object, x_after: object, support_x: object) -> dict[str, float]:
    import numpy as np  # type: ignore

    before = np.asarray(x_before, dtype=float)
    after = np.asarray(x_after, dtype=float)
    support = np.asarray(support_x, dtype=float)
    support_trace = max(_trace_cov_np(support), 1.0e-12)
    support_pairwise = max(_mean_pairwise_distance(support), 1.0e-12)
    return {
        "synthetic_mean_l2_to_support_before": float(np.linalg.norm(before.mean(axis=0) - support.mean(axis=0))),
        "synthetic_mean_l2_to_support_after": float(np.linalg.norm(after.mean(axis=0) - support.mean(axis=0))),
        "synthetic_cov_trace_ratio_to_support_before": float(_trace_cov_np(before) / support_trace),
        "synthetic_cov_trace_ratio_to_support_after": float(_trace_cov_np(after) / support_trace),
        "synthetic_pairwise_distance_ratio_to_support_before": float(_mean_pairwise_distance(before) / support_pairwise),
        "synthetic_pairwise_distance_ratio_to_support_after": float(_mean_pairwise_distance(after) / support_pairwise),
    }


def _class_geometry_diagnostics(x_before: object, x_after: object, labels: Sequence[int]) -> dict[str, float | int]:
    before = _class_geometry_stats(x_before, labels)
    after = _class_geometry_stats(x_after, labels)
    return {
        "synthetic_class_centroid_distance_mean_before": before["centroid_distance"],
        "synthetic_class_centroid_distance_mean_after": after["centroid_distance"],
        "synthetic_within_class_trace_before": before["within_trace"],
        "synthetic_within_class_trace_after": after["within_trace"],
        "synthetic_fisher_ratio_before": before["fisher_ratio"],
        "synthetic_fisher_ratio_after": after["fisher_ratio"],
        "class_centroid_collapse_flag": int(
            (
                not math.isnan(before["centroid_distance"])
                and after["centroid_distance"] < 0.75 * before["centroid_distance"]
            )
            or (
                not math.isnan(before["fisher_ratio"])
                and after["fisher_ratio"] < 0.75 * before["fisher_ratio"]
            )
        ),
    }


def _class_geometry_stats(x: object, labels: Sequence[int]) -> dict[str, float]:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    y = np.asarray(labels, dtype=int)
    classes = sorted(set(int(v) for v in y.tolist()))
    if len(classes) < 2:
        return {"centroid_distance": math.nan, "within_trace": math.nan, "fisher_ratio": math.nan}
    centroids = []
    traces = []
    for cls in classes:
        subset = arr[y == cls]
        if subset.size == 0:
            continue
        centroids.append(subset.mean(axis=0))
        traces.append(_trace_cov_np(subset))
    if len(centroids) < 2:
        return {"centroid_distance": math.nan, "within_trace": math.nan, "fisher_ratio": math.nan}
    centroid_distance = float(np.linalg.norm(centroids[1] - centroids[0]))
    within_trace = float(np.mean(traces)) if traces else math.nan
    return {
        "centroid_distance": centroid_distance,
        "within_trace": within_trace,
        "fisher_ratio": float((centroid_distance ** 2) / max(within_trace, 1.0e-12)) if not math.isnan(within_trace) else math.nan,
    }


def _support_bootstrap_diagnostics(x_s: object, x_t: object, *, policy: str, alpha: float, seed: int) -> dict[str, float | int]:
    import numpy as np  # type: ignore

    source = np.asarray(x_s, dtype=float)
    support = np.asarray(x_t, dtype=float)
    if support.shape[0] <= 1 or policy == ALIGN_IDENTITY:
        return {
            "support_bootstrap_mean_shift_std": 0.0,
            "support_bootstrap_scale_std": 0.0,
            "support_bootstrap_scale_std_mean": 0.0,
            "support_bootstrap_unstable_flag": 0,
        }
    rng = np.random.default_rng(int(seed))
    full_mu_t = support.mean(axis=0)
    mu_s = source.mean(axis=0)
    var_s = source.var(axis=0)
    mean_shifts = []
    scales = []
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, support.shape[0], size=support.shape[0])
        boot = support[idx]
        mu_t = boot.mean(axis=0)
        var_t = boot.var(axis=0)
        mean_shifts.append(float(np.linalg.norm(mu_t - full_mu_t)))
        if policy == ALIGN_MEAN_ONLY:
            scales.append(np.ones(source.shape[1]))
        else:
            var_a = (1.0 - float(alpha)) * var_s + float(alpha) * var_t
            raw_scale = np.sqrt(np.maximum(var_a, VARIANCE_FLOOR) / np.maximum(var_s, VARIANCE_FLOOR))
            scales.append(np.clip(raw_scale, SCALE_MIN, SCALE_MAX))
    scale_arr = np.asarray(scales, dtype=float)
    scale_std_by_dim = scale_arr.std(axis=0)
    mean_shift_std = float(statistics.pstdev(mean_shifts)) if len(mean_shifts) > 1 else 0.0
    scale_std_mean = float(np.mean(scale_std_by_dim))
    synthetic_support_mean_l2_before = float(np.linalg.norm(mu_s - full_mu_t))
    unstable = int(scale_std_mean > BOOTSTRAP_SCALE_STD_THRESHOLD or mean_shift_std > 0.25 * synthetic_support_mean_l2_before)
    return {
        "support_bootstrap_mean_shift_std": mean_shift_std,
        "support_bootstrap_scale_std": float(np.linalg.norm(scale_std_by_dim)),
        "support_bootstrap_scale_std_mean": scale_std_mean,
        "support_bootstrap_unstable_flag": unstable,
    }


def _trace_cov_np(x: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return 0.0
    centered = arr - arr.mean(axis=0, keepdims=True)
    return float(np.sum(centered * centered) / max(arr.shape[0] - 1, 1))


def _mean_pairwise_distance(x: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return 0.0
    limit = min(arr.shape[0], 256)
    arr = arr[:limit]
    diffs = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diffs * diffs, axis=2))
    upper = dist[np.triu_indices(dist.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _effective_rank(x: object) -> float:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return 0.0
    centered = arr - arr.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular * singular
    total = float(power.sum())
    if total <= 0:
        return 0.0
    probs = power / total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1.0e-12, 1.0))))
    return float(math.exp(entropy))


def _full_coral_matrix(source: object, support: object, *, alpha: float) -> object:
    import numpy as np  # type: ignore

    x_s = np.asarray(source, dtype=float)
    x_t = np.asarray(support, dtype=float)
    cov_s = _cov_np(x_s) + np.eye(x_s.shape[1]) * VARIANCE_FLOOR
    cov_t = _cov_np(x_t) + np.eye(x_t.shape[1]) * VARIANCE_FLOOR
    cov_a = (1.0 - float(alpha)) * cov_s + float(alpha) * cov_t
    return _matrix_invsqrt(cov_s) @ _matrix_sqrt(cov_a)


def _cov_np(x: object) -> object:
    import numpy as np  # type: ignore

    arr = np.asarray(x, dtype=float)
    if arr.shape[0] <= 1:
        return np.eye(arr.shape[1]) * VARIANCE_FLOOR
    centered = arr - arr.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / float(max(arr.shape[0] - 1, 1))


def _matrix_sqrt(matrix: object) -> object:
    import numpy as np  # type: ignore

    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    values = np.maximum(values, VARIANCE_FLOOR)
    return (vectors * np.sqrt(values)) @ vectors.T


def _matrix_invsqrt(matrix: object) -> object:
    import numpy as np  # type: ignore

    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    values = np.maximum(values, VARIANCE_FLOOR)
    return (vectors * (1.0 / np.sqrt(values))) @ vectors.T


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
    out = {
        "bacc": balanced_accuracy(y_true, pred),
        "macro_f1": _macro_f1(y_true, pred),
        "auroc": math.nan,
        "auprc": math.nan,
    }
    if len(class_order) == 2 and probs.shape[1] == 2:
        pos_idx = tuple(int(v) for v in class_order).index(1)
        try:
            out["auroc"] = float(roc_auc_score(y_true, probs[:, pos_idx]))
        except ValueError:
            out["auroc"] = math.nan
        try:
            out["auprc"] = float(average_precision_score(y_true, probs[:, pos_idx]))
        except ValueError:
            out["auprc"] = math.nan
    return out


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    from .downstream import macro_f1

    return macro_f1(y_true, y_pred)


def _load_c63_primary_rows(c63_root: Path) -> list[dict[str, object]]:
    path = c63_root / "tables" / "c63_geometric_late_ensemble_downstream_matrix.csv"
    if not path.exists():
        return []
    return [
        row
        for row in load_csv_rows(path)
        if str(row.get("ensemble_policy")) == POLICY_GEOM_SAFE_MULTI
        and str(row.get("status")) == "ok"
    ]


def _bacc_by_condition(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, str, int, int, int], float]:
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


def _c63_member_bank_hash(
    rows: Sequence[Mapping[str, object]],
    *,
    experiment_seed: int,
    heldout_center: str,
    support_size: int,
    support_seed: int,
    classifier_seed: int,
) -> str:
    matches = [
        row
        for row in rows
        if int(row.get("experiment_seed", -1)) == int(experiment_seed)
        and str(row.get("heldout_center")) == str(heldout_center)
        and int(row.get("support_size", -1)) == int(support_size)
        and int(row.get("support_seed", -1)) == int(support_seed)
        and int(row.get("classifier_seed", -1)) == int(classifier_seed)
    ]
    if not matches:
        return ""
    member_keys = str(matches[0].get("member_keys", ""))
    payload = json.dumps(
        {"member_keys": sorted(part for part in member_keys.split(";") if part), "class_order": GLOBAL_CLASS_ORDER},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _probability_hash(probabilities: object) -> str:
    import numpy as np  # type: ignore

    arr = np.asarray(probabilities, dtype=float)
    rounded = np.round(arr, decimals=12)
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def _common_applied_policy(rows: Sequence[Mapping[str, object]]) -> str:
    values = sorted({str(row.get("alignment_policy_applied")) for row in rows if str(row.get("alignment_policy_applied", ""))})
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "mixed:" + "|".join(values)


def _identity_replay_max_abs_delta(rows: Sequence[Mapping[str, object]]) -> float:
    deltas = [abs(_float(row.get("delta_vs_c63_primary"))) for row in rows if not math.isnan(_float(row.get("delta_vs_c63_primary")))]
    return max(deltas) if deltas else math.nan


def _center_delta(rows: Sequence[Mapping[str, object]], center: str) -> float:
    subset = [row for row in rows if str(row.get("heldout_center")) == str(center)]
    return _mean(_float(row.get("delta_vs_c63_primary")) for row in subset)


def _weak_center_gain_count(rows: Sequence[Mapping[str, object]]) -> int:
    return sum(1 for center in ("1", "3") if _center_delta(rows, center) >= 0.03)


def _strong_center_degrade_count(rows: Sequence[Mapping[str, object]]) -> int:
    count = 0
    for center in sorted({str(row.get("heldout_center")) for row in rows}):
        c63_center = _mean(_float(row.get("c63_primary_bacc")) for row in rows if str(row.get("heldout_center")) == center)
        if c63_center >= 0.80 and _center_delta(rows, center) < -0.02:
            count += 1
    return count


def _seed_positive_gain_rate(rows: Sequence[Mapping[str, object]]) -> float:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("experiment_seed", -1)), []).append(_float(row.get("delta_vs_c63_primary")))
    values = [_mean(items) for items in grouped.values()]
    return _mean(1.0 if value > 0.0 else 0.0 for value in values)


def _class_collapse_rate(rows: Sequence[Mapping[str, object]], policy: str) -> float:
    subset = [row for row in rows if str(row.get("alignment_policy")) == policy]
    return _mean(_float(row.get("class_centroid_collapse_flag")) for row in subset)


def _support_unstable_rate(rows: Sequence[Mapping[str, object]], policy: str) -> float:
    subset = [row for row in rows if str(row.get("alignment_policy")) == policy]
    return _mean(_float(row.get("support_bootstrap_unstable_flag")) for row in subset)


def _decision_label(row: Mapping[str, object]) -> str:
    mean_bacc = _float(row.get("mean_bacc"))
    mean_delta = _float(row.get("mean_delta_vs_c63_primary"))
    if int(_float(row.get("identity_replay_matches_within_tolerance"))) == 0 and str(row.get("alignment_policy")) == ALIGN_IDENTITY:
        return FAILURE_REPLAY
    if _float(row.get("member_bank_hash_match_rate")) < 1.0:
        return FAILURE_REPLAY
    if _float(row.get("class_centroid_collapse_rate")) > 0.0:
        return FAILURE_CLASS_GEOMETRY
    if _float(row.get("support_bootstrap_unstable_rate")) > 0.25:
        return FAILURE_OVERFITS
    if (mean_delta >= 0.02 or mean_bacc >= 0.85) and _float(row.get("strong_center_degrade_gt_002_count")) == 0:
        return DECISION_085_CANDIDATE if mean_bacc >= 0.85 else DECISION_TARGET_ALIGNMENT_GAIN
    if mean_delta >= 0.005 and _float(row.get("weak_center_gain_ge_003_count")) >= 1:
        return DECISION_TARGET_ALIGNMENT_GAIN
    if _float(row.get("weak_center_gain_ge_003_count")) == 0:
        return FAILURE_WEAK_CENTERS
    return FAILURE_NO_GAIN


def _stable_int_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
