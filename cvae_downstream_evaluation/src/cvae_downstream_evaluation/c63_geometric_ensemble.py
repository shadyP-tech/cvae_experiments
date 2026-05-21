"""C6.3 predeclared geometric late ensembles.

C6.3 is a narrow aggregation-rule ablation over C6.2. It reuses the same frozen
CVAE expert/mode member construction and downstream member classifiers, then
changes only the late aggregation operator from arithmetic probability averaging
to weighted log-probability / geometric pooling.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .downstream import balanced_accuracy
from .matrix import (
    _label,
    _load_embedding_cache,
    _read_samples_manifest,
    _records_for_split,
    build_target_eval_pool,
    hash_candidate_experts,
)
from .protocol import LockedV1Config, ProtocolError
from .schemas import ENSEMBLE_EXPERT_ID, METHOD_BASELINE_ROW_TYPE
from .c61_mixture import (
    C61_DEFAULT_C41_ROOT,
    C61_DEFAULT_C42_ROOT,
    C61_DEFAULT_C52_ROOT,
    _GenerationCache,
    _file_hash,
    _float,
    _generate_component,
    _load_c52_oracle_reference,
    _mean,
    _quantile,
    _support_conditions_from_rows,
    _to_numpy,
    _write_csv,
    load_csv_rows,
)
from .c62_late_ensemble import (
    C62_ARTIFACTS_ROOT,
    C62_LEGACY_SUPPORT_UNITS,
    C62RunLimits,
    EnsembleMemberSpec,
    EnsemblePlan,
    GLOBAL_CLASS_ORDER,
    POLICY_FIXED_TOTAL as C62_POLICY_FIXED_TOTAL,
    POLICY_HETERO_MULTI as C62_POLICY_HETERO_MULTI,
    POLICY_METADATA_WEIGHTED as C62_POLICY_METADATA_WEIGHTED,
    POLICY_NO_STANDARD as C62_POLICY_NO_STANDARD,
    POLICY_SAFE_MULTI as C62_POLICY_SAFE_MULTI,
    POLICY_SAFE_SINGLE as C62_POLICY_SAFE_SINGLE,
    POLICY_SUPPORT_NELBO_WEIGHTED as C62_POLICY_SUPPORT_NELBO_WEIGHTED,
    _GeneratedMember,
    _fit_member_probabilities,
    _generate_member,
    _hard_vote_predictions,
    _member_probability_diagnostics,
    _np_array,
    _np_stack,
    _score_predictions_and_probabilities,
    _selector_visible_support_rows,
    align_probabilities_to_class_order,
    assert_c62_prejoin_rows_safe,
    build_c62_ensemble_plans,
    ensemble_weight_diagnostics,
    fixed_predictions_from_probabilities,
)


C63_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c63_geometric_late_ensemble_v1"
C63_DEFAULT_C41_ROOT = C61_DEFAULT_C41_ROOT
C63_DEFAULT_C42_ROOT = C61_DEFAULT_C42_ROOT
C63_DEFAULT_C52_ROOT = C61_DEFAULT_C52_ROOT
C63_DEFAULT_C62_ROOT = C62_ARTIFACTS_ROOT

POLICY_C62_REPLAY = "c62_probability_average_replay"
POLICY_GEOM_SAFE_SINGLE = "fixed_all_source_safe_single_seed_geometric_late_ensemble"
POLICY_GEOM_SAFE_MULTI = "fixed_all_source_safe_multiseed_geometric_late_ensemble"
POLICY_GEOM_FIXED_TOTAL = "fixed_total_draw_safe_multiseed_geometric_late_ensemble"
POLICY_GEOM_HETERO = "fixed_all_source_hetero_mean_multiseed_geometric_late_ensemble"
POLICY_GEOM_NO_STANDARD = "fixed_all_source_safe_no_standard_prior_multiseed_geometric_late_ensemble"
POLICY_GEOM_METADATA = "metadata_weighted_safe_geometric_late_ensemble"
POLICY_GEOM_SUPPORT_NELBO = "support_nelbo_rank_weighted_safe_geometric_late_ensemble"

PRIMARY_POLICIES = (
    POLICY_C62_REPLAY,
    POLICY_GEOM_SAFE_SINGLE,
    POLICY_GEOM_SAFE_MULTI,
    POLICY_GEOM_FIXED_TOTAL,
    POLICY_GEOM_HETERO,
    POLICY_GEOM_NO_STANDARD,
    POLICY_GEOM_METADATA,
    POLICY_GEOM_SUPPORT_NELBO,
)

SOURCE_POLICY_BY_C63 = {
    POLICY_C62_REPLAY: C62_POLICY_SAFE_MULTI,
    POLICY_GEOM_SAFE_SINGLE: C62_POLICY_SAFE_SINGLE,
    POLICY_GEOM_SAFE_MULTI: C62_POLICY_SAFE_MULTI,
    POLICY_GEOM_FIXED_TOTAL: C62_POLICY_FIXED_TOTAL,
    POLICY_GEOM_HETERO: C62_POLICY_HETERO_MULTI,
    POLICY_GEOM_NO_STANDARD: C62_POLICY_NO_STANDARD,
    POLICY_GEOM_METADATA: C62_POLICY_METADATA_WEIGHTED,
    POLICY_GEOM_SUPPORT_NELBO: C62_POLICY_SUPPORT_NELBO_WEIGHTED,
}

GEOMETRIC_GENERATOR_FAMILY = "family_c_pca64_geometric_late_ensemble_downstream_v1"
LOG_PROBABILITY_EPSILON = 1.0e-8
GEOMETRIC_SOFTMAX_TEMPERATURE = 1.0
REPLAY_TOLERANCE = 1.0e-9

DECISION_SUCCESS = "GEOMETRIC_LATE_ENSEMBLE_SUCCESS"
DECISION_USEFUL = "GEOMETRIC_LATE_ENSEMBLE_USEFUL"
FAILURE_NO_GAIN = "GEOMETRIC_LATE_ENSEMBLE_NO_GAIN"
FAILURE_OVERPENALIZES = "GEOMETRIC_OVERPENALIZES_DISAGREEMENT"
FAILURE_WEAK_CENTERS = "WEAK_CENTERS_REMAIN_CEILING"
FAILURE_UNSTABLE = "GAIN_NOT_PAIRED_STABLE"
FAILURE_REPLAY = "C62_PROBABILITY_REPLAY_MISMATCH"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_LABEL_ACCESS"


@dataclass(frozen=True)
class C63RunLimits(C62RunLimits):
    pass


@dataclass(frozen=True)
class _C63Score:
    matrix_row: dict[str, object]
    probability_diagnostics: dict[str, object]
    disagreement_diagnostics: dict[str, object]


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
    "geometric_prob_diagnostic_only",
    "prediction_rule",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "status",
    "error_message",
)

PROTOCOL_COLUMNS = (
    "ensemble_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "heldout_source_excluded",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "target_eval_threshold_search",
    "checkpoints_retrained",
    "prejoin_forbidden_columns_present",
    "primary_modes_exclude_noise_and_gmm_k4",
    "weights_prejoin_only",
    "weights_normalized_per_cell",
    "class_probability_alignment_checked",
    "temperature_tuned",
    "protocol_status",
)

PROBABILITY_COLUMNS = (
    "ensemble_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "generation_seed_group",
    "classifier_seed",
    "aggregation_rule",
    "probability_average_bacc",
    "geometric_bacc",
    "hard_vote_bacc",
    "probability_average_entropy",
    "geometric_entropy",
    "clip_rate_by_class",
    "clip_rate_class_0",
    "clip_rate_class_1",
    "clip_rate_by_member",
    "fraction_predictions_with_any_clipped_member",
    "mean_min_probability_predicted_class",
    "mean_min_probability_true_class",
    "true_class_probability_diagnostics_post_eval_only",
    "geometric_prob_diagnostic_only",
)

DISAGREEMENT_COLUMNS = (
    "heldout_center",
    "experiment_seed",
    "support_seed",
    "support_size",
    "classifier_seed",
    "aggregation_rule",
    "mean_member_entropy",
    "member_disagreement_entropy",
    "mean_pairwise_js_divergence",
    "mean_vote_margin",
    "fraction_any_member_clipped",
    "fraction_majority_disagreement",
    "bacc",
    "delta_vs_c62",
)

COMPARISON_COLUMNS = (
    "ensemble_policy",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "classifier_seed",
    "generation_seed_group",
    "probability_average_bacc",
    "geometric_bacc",
    "hard_vote_bacc",
    "delta_geometric_vs_probability",
    "c62_probability_average_bacc",
    "delta_vs_c62_probability_average",
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
    "c62_probability_average_mean_bacc",
    "mean_delta_vs_c62_probability_average",
    "paired_positive_center_seed_cells_vs_c62",
    "paired_center_seed_cells_vs_c62",
    "c62_replay_max_abs_bacc_delta",
    "c62_replay_matches_within_tolerance",
    "center_1_delta_vs_c62",
    "center_3_delta_vs_c62",
    "strong_center_degrade_gt_002_count",
    "decision_label",
)

CENTER_COLUMNS = (
    "ensemble_policy",
    "heldout_center",
    "n_rows",
    "mean_bacc",
    "ensemble_ge_080_rate",
    "c62_probability_average_bacc",
    "delta_vs_c62_probability_average",
    "mean_oracle_bacc_reference",
    "regret_p75",
)


def run_c63_geometric_ensemble(
    *,
    config: LockedV1Config,
    repo_root: Path,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    c52_artifacts_root: Path,
    c62_artifacts_root: Path,
    device: str,
    limits: C63RunLimits = C63RunLimits(),
) -> dict[str, Path]:
    support_unit_rows = load_csv_rows(c41_artifacts_root / "tables" / "support_selection_units.csv")
    combined_support_rows = list(support_unit_rows)
    legacy_support_path = repo_root / C62_LEGACY_SUPPORT_UNITS
    if legacy_support_path.exists():
        combined_support_rows.extend(load_csv_rows(legacy_support_path))
    assert_c62_prejoin_rows_safe(_selector_visible_support_rows(combined_support_rows))
    c62_primary_rows = _load_c62_primary_rows(c62_artifacts_root)
    c62_by_condition = _c62_by_condition(c62_primary_rows)
    c62_by_center = _center_baseline(c62_primary_rows)
    c52_oracle = _load_c52_oracle_reference(c52_artifacts_root)

    selected_generation_seeds = limits.generation_seeds or tuple(config.generation_seeds)
    selected_classifier_seeds = limits.classifier_seeds or tuple(config.classifier_seeds)
    selected_heldout = limits.heldout_centers or tuple(str(v) for v in config.candidate_domains)
    experiment_seed_filter = set(int(v) for v in limits.experiment_seeds) if limits.experiment_seeds else None

    member_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    protocol_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    disagreement_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

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
                raise ProtocolError(f"C6.3 expects binary labels {GLOBAL_CLASS_ORDER}, got {sorted(set(target_labels))}")
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
                plans = build_c63_ensemble_plans(
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
                        _generate_member(cache=generation_cache, spec=spec, label_values=GLOBAL_CLASS_ORDER)
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
                        baseline_key = (
                            experiment_seed,
                            heldout,
                            int(support_size),
                            int(support_seed),
                            int(classifier_seed),
                        )
                        c62_baseline = c62_by_condition.get(baseline_key, math.nan)
                        score = _score_c63_row(
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
                            c62_probability_average_bacc=c62_baseline,
                        )
                        matrix_rows.append(score.matrix_row)
                        probability_rows.append(score.probability_diagnostics)
                        disagreement_rows.append(score.disagreement_diagnostics)
                        comparison_rows.append(_comparison_row(score, c62_baseline))

    outputs = {
        "members": artifacts_root / "tables" / "c63_ensemble_members_pre_join.csv",
        "matrix": artifacts_root / "tables" / "c63_geometric_late_ensemble_downstream_matrix.csv",
        "threshold": artifacts_root / "tables" / "c63_threshold_audit.csv",
        "center": artifacts_root / "tables" / "c63_center_summary.csv",
        "protocol": artifacts_root / "tables" / "c63_protocol_audit.csv",
        "probability": artifacts_root / "tables" / "c63_probability_diagnostics.csv",
        "comparison": artifacts_root / "tables" / "c63_aggregation_rule_comparison.csv",
        "disagreement": artifacts_root / "tables" / "c63_member_disagreement_diagnostics.csv",
    }
    assert_c62_prejoin_rows_safe(member_rows)
    _write_csv(outputs["members"], MEMBER_COLUMNS, member_rows)
    _write_csv(outputs["matrix"], MATRIX_COLUMNS, matrix_rows)
    _write_csv(outputs["threshold"], SUMMARY_COLUMNS, build_c63_threshold_rows(matrix_rows, c62_primary_rows, c62_by_center, c62_by_condition))
    _write_csv(outputs["center"], CENTER_COLUMNS, build_c63_center_rows(matrix_rows, c62_by_center))
    _write_csv(outputs["protocol"], PROTOCOL_COLUMNS, protocol_rows)
    _write_csv(outputs["probability"], PROBABILITY_COLUMNS, probability_rows)
    _write_csv(outputs["comparison"], COMPARISON_COLUMNS, comparison_rows)
    _write_csv(outputs["disagreement"], DISAGREEMENT_COLUMNS, disagreement_rows)
    return outputs


def build_c63_ensemble_plans(
    *,
    policies: Sequence[str],
    candidates: Sequence[str],
    total_budget_per_class: int,
    generation_seeds: Sequence[int],
    support_rows: Sequence[Mapping[str, object]] = (),
) -> list[EnsemblePlan]:
    out: list[EnsemblePlan] = []
    for policy in policies:
        source = SOURCE_POLICY_BY_C63.get(policy)
        if source is None:
            raise ProtocolError(f"Unknown C6.3 ensemble policy: {policy}")
        source_plans = build_c62_ensemble_plans(
            policies=(source,),
            candidates=candidates,
            total_budget_per_class=total_budget_per_class,
            generation_seeds=generation_seeds,
            support_rows=support_rows,
        )
        for source_plan in source_plans:
            out.append(
                EnsemblePlan(
                    policy=policy,
                    generation_seed_group=source_plan.generation_seed_group,
                    specs=_normalize_specs(source_plan.specs),
                    diagnostic_only=int(policy == POLICY_C62_REPLAY),
                )
            )
    return out


def geometric_pool_probabilities(
    probabilities: object,
    weights: Sequence[float],
    *,
    epsilon: float = LOG_PROBABILITY_EPSILON,
    temperature: float = GEOMETRIC_SOFTMAX_TEMPERATURE,
) -> tuple[object, object]:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 3:
        raise ProtocolError("C6.3 geometric pooling expects member x sample x class probabilities.")
    weights_arr = np.asarray(normalize_weights(weights), dtype=float)
    if len(weights_arr) != probs.shape[0]:
        raise ProtocolError("C6.3 weight count does not match member probability count.")
    if float(temperature) != 1.0:
        raise ProtocolError("C6.3 temperature tuning is forbidden; temperature must be 1.0.")
    scores = (np.log(np.clip(probs, float(epsilon), 1.0)) * weights_arr[:, None, None]).sum(axis=0)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    geometric_prob = np.exp(shifted)
    geometric_prob = geometric_prob / np.clip(geometric_prob.sum(axis=1, keepdims=True), 1.0e-12, None)
    return scores, geometric_prob


def normalize_weights(weights: Sequence[float]) -> tuple[float, ...]:
    clean = tuple(max(float(value), 0.0) for value in weights)
    total = sum(clean)
    if total <= 0.0:
        return tuple(1.0 / float(len(clean)) for _ in clean)
    return tuple(value / total for value in clean)


def _score_c63_row(
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
    c62_probability_average_bacc: float,
) -> _C63Score:
    try:
        probs = []
        hard_preds = []
        member_entropy_values = []
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
            hard_preds.append(fixed_predictions_from_probabilities(aligned, GLOBAL_CLASS_ORDER))
            weights.append(float(member.spec.weight))
            member_entropy_values.append(_member_probability_diagnostics(aligned)["member_entropy"])
        stacked = _np_stack(probs)
        weights_arr = _np_array(normalize_weights(weights))
        mean_prob = (stacked * weights_arr[:, None, None]).sum(axis=0)
        geometric_scores, geometric_prob = geometric_pool_probabilities(stacked, weights_arr)
        hard_vote_pred = _hard_vote_predictions(hard_preds, GLOBAL_CLASS_ORDER)
        probability_pred = fixed_predictions_from_probabilities(mean_prob, GLOBAL_CLASS_ORDER)
        geometric_pred = _predictions_from_scores(geometric_scores, GLOBAL_CLASS_ORDER)
        primary_prob = mean_prob if plan.policy == POLICY_C62_REPLAY else geometric_prob
        primary_pred = probability_pred if plan.policy == POLICY_C62_REPLAY else geometric_pred
        primary_rule = "weighted_probability_average" if plan.policy == POLICY_C62_REPLAY else "weighted_log_probability_geometric_pooling"
        primary_metrics = _score_predictions_and_probabilities(target_labels, primary_pred, primary_prob, GLOBAL_CLASS_ORDER)
        probability_metrics = _score_predictions_and_probabilities(target_labels, probability_pred, mean_prob, GLOBAL_CLASS_ORDER)
        geometric_metrics = _score_predictions_and_probabilities(target_labels, geometric_pred, geometric_prob, GLOBAL_CLASS_ORDER)
        hard_bacc = balanced_accuracy(target_labels, hard_vote_pred)
        entropy = ensemble_weight_diagnostics([member.spec for member in generated])
        member_counts = _member_counts([member.spec for member in generated])
        diag = _geometric_diagnostics(
            stacked=stacked,
            mean_prob=mean_prob,
            geometric_prob=geometric_prob,
            geometric_pred=geometric_pred,
            hard_preds=hard_preds,
            target_labels=target_labels,
            member_entropy_values=member_entropy_values,
        )
        total_train = sum(len(member.generated.synthetic_labels) for member in generated)
        status = "ok"
        error = ""
    except Exception as exc:
        primary_metrics = probability_metrics = geometric_metrics = {
            "bacc": math.nan,
            "macro_f1": math.nan,
            "auroc": math.nan,
            "auprc": math.nan,
        }
        hard_bacc = math.nan
        entropy = ensemble_weight_diagnostics([member.spec for member in generated])
        member_counts = _member_counts([member.spec for member in generated])
        diag = _empty_geometric_diagnostics()
        total_train = sum(len(member.generated.synthetic_labels) for member in generated)
        primary_rule = "weighted_probability_average" if plan.policy == POLICY_C62_REPLAY else "weighted_log_probability_geometric_pooling"
        status = "failed_c63_geometric_scoring"
        error = str(exc)
    bacc = float(primary_metrics["bacc"])
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
        "generator_family": GEOMETRIC_GENERATOR_FAMILY,
        "generation_mode": plan.policy,
        "budget_per_class": int(budget_per_class),
        "bacc": bacc,
        "macro_f1": primary_metrics["macro_f1"],
        "auroc": primary_metrics["auroc"],
        "auprc": primary_metrics["auprc"],
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
        **member_counts,
        **entropy,
        "effective_num_members_after_weighting": entropy.get("effective_num_members", math.nan),
        "aggregation_rule": primary_rule,
        "log_probability_epsilon": LOG_PROBABILITY_EPSILON,
        "geometric_softmax_temperature": GEOMETRIC_SOFTMAX_TEMPERATURE,
        "temperature_tuned": 0,
        "geometric_prob_diagnostic_only": int(plan.policy != POLICY_C62_REPLAY),
        "prediction_rule": "argmax_weighted_log_probability" if plan.policy != POLICY_C62_REPLAY else "binary_p_positive_ge_0_5",
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
        "aggregation_rule": primary_rule,
        "probability_average_bacc": probability_metrics["bacc"],
        "geometric_bacc": geometric_metrics["bacc"],
        "hard_vote_bacc": hard_bacc,
        **diag,
        "geometric_prob_diagnostic_only": int(plan.policy != POLICY_C62_REPLAY),
    }
    disagreement_row = {
        "heldout_center": heldout_center,
        "experiment_seed": int(experiment_seed),
        "support_seed": int(support_seed),
        "support_size": int(support_size),
        "classifier_seed": int(classifier_seed),
        "aggregation_rule": primary_rule,
        "mean_member_entropy": diag["mean_member_entropy"],
        "member_disagreement_entropy": diag["member_disagreement_entropy"],
        "mean_pairwise_js_divergence": diag["mean_pairwise_js_divergence"],
        "mean_vote_margin": diag["mean_vote_margin"],
        "fraction_any_member_clipped": diag["fraction_predictions_with_any_clipped_member"],
        "fraction_majority_disagreement": diag["fraction_majority_disagreement"],
        "bacc": bacc,
        "delta_vs_c62": bacc - _float(c62_probability_average_bacc),
    }
    return _C63Score(
        matrix_row=matrix_row,
        probability_diagnostics=probability_row,
        disagreement_diagnostics=disagreement_row,
    )


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


def build_c63_threshold_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    c62_primary_rows: Sequence[Mapping[str, object]],
    c62_by_center: Mapping[str, float],
    c62_by_condition: Mapping[tuple[int, str, int, int, int], float],
) -> list[dict[str, object]]:
    c62_mean = _mean(_float(row.get("bacc")) for row in c62_primary_rows)
    out = []
    for policy in sorted({str(row.get("ensemble_policy")) for row in matrix_rows}):
        subset = [row for row in matrix_rows if str(row.get("ensemble_policy")) == policy and str(row.get("status")) == "ok"]
        values = [_float(row.get("bacc")) for row in subset]
        regrets = [_float(row.get("regret_bacc")) for row in subset]
        center_means = [
            _mean(_float(row.get("bacc")) for row in subset if str(row.get("heldout_center")) == center)
            for center in sorted({str(row.get("heldout_center")) for row in subset})
        ]
        replay_delta = _c62_replay_max_abs_delta(subset, c62_primary_rows) if policy == POLICY_C62_REPLAY else math.nan
        paired = _paired_positive_cells(subset, c62_by_condition)
        mean_bacc = _mean(values)
        row = {
            "ensemble_policy": policy,
            "policy_role": _policy_role(policy),
            "diagnostic_only": int(policy == POLICY_C62_REPLAY),
            "n_rows": len(subset),
            "mean_bacc": mean_bacc,
            "ensemble_ge_080_rate": _mean(1.0 if value >= 0.80 else 0.0 for value in values),
            "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
            "mean_regret_bacc": _mean(regrets),
            "regret_p50": _quantile(regrets, 0.50),
            "regret_p75": _quantile(regrets, 0.75),
            "regret_p90": _quantile(regrets, 0.90),
            "center_variance": statistics.pvariance(center_means) if len(center_means) > 1 else 0.0,
            "c62_probability_average_mean_bacc": c62_mean,
            "mean_delta_vs_c62_probability_average": mean_bacc - c62_mean if not math.isnan(c62_mean) else math.nan,
            "paired_positive_center_seed_cells_vs_c62": paired[0],
            "paired_center_seed_cells_vs_c62": paired[1],
            "c62_replay_max_abs_bacc_delta": replay_delta,
            "c62_replay_matches_within_tolerance": int(policy != POLICY_C62_REPLAY or (not math.isnan(replay_delta) and replay_delta <= REPLAY_TOLERANCE)),
            "center_1_delta_vs_c62": _center_delta(subset, "1", c62_by_center),
            "center_3_delta_vs_c62": _center_delta(subset, "3", c62_by_center),
            "strong_center_degrade_gt_002_count": _strong_center_degrade_count(subset, c62_by_center),
            "decision_label": "",
        }
        row["decision_label"] = _decision_label(row)
        out.append(row)
    return out


def build_c63_center_rows(
    matrix_rows: Sequence[Mapping[str, object]],
    c62_by_center: Mapping[str, float],
) -> list[dict[str, object]]:
    out = []
    for policy in sorted({str(row.get("ensemble_policy")) for row in matrix_rows}):
        for center in sorted({str(row.get("heldout_center")) for row in matrix_rows}):
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
                    "c62_probability_average_bacc": _float(c62_by_center.get(center)),
                    "delta_vs_c62_probability_average": mean_bacc - _float(c62_by_center.get(center)),
                    "mean_oracle_bacc_reference": _mean(_float(row.get("oracle_bacc_reference")) for row in subset),
                    "regret_p75": _quantile([_float(row.get("regret_bacc")) for row in subset], 0.75),
                }
            )
    return out


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
    weights = normalize_weights([spec.weight for spec in plan.specs])
    pass_status = (
        str(heldout_center) not in {str(candidate) for candidate in candidates}
        and abs(sum(weights) - 1.0) <= 1.0e-12
    )
    return {
        "ensemble_policy": plan.policy,
        "experiment_seed": int(experiment_seed),
        "heldout_center": heldout_center,
        "support_size": int(support_size),
        "support_seed": int(support_seed),
        "support_eval_split_id": support_eval_split_id,
        "heldout_source_excluded": int(str(heldout_center) not in {str(candidate) for candidate in candidates}),
        "target_support_labels_used": 0,
        "target_eval_labels_used_for_selection": 0,
        "target_eval_threshold_search": 0,
        "checkpoints_retrained": 0,
        "prejoin_forbidden_columns_present": 0,
        "primary_modes_exclude_noise_and_gmm_k4": int("hetero_noise" not in modes and "gmm_k4" not in modes),
        "weights_prejoin_only": 1,
        "weights_normalized_per_cell": int(abs(sum(weights) - 1.0) <= 1.0e-12),
        "class_probability_alignment_checked": 1,
        "temperature_tuned": 0,
        "protocol_status": "pass" if pass_status else "fail",
    }


def _geometric_diagnostics(
    *,
    stacked: object,
    mean_prob: object,
    geometric_prob: object,
    geometric_pred: Sequence[int],
    hard_preds: Sequence[Sequence[int]],
    target_labels: Sequence[int],
    member_entropy_values: Sequence[float],
) -> dict[str, object]:
    import numpy as np  # type: ignore

    probs = np.asarray(stacked, dtype=float)
    mean_prob = np.asarray(mean_prob, dtype=float)
    geometric_prob = np.asarray(geometric_prob, dtype=float)
    clipped = probs <= LOG_PROBABILITY_EPSILON
    class_clip_rates = clipped.mean(axis=(0, 1))
    pred_idx = np.asarray([GLOBAL_CLASS_ORDER.index(int(value)) for value in geometric_pred], dtype=int)
    true_idx = np.asarray([GLOBAL_CLASS_ORDER.index(int(value)) for value in target_labels], dtype=int)
    sample_idx = np.arange(probs.shape[1])
    pred_member_probs = probs[:, sample_idx, pred_idx]
    true_member_probs = probs[:, sample_idx, true_idx]
    vote_stats = _vote_statistics(hard_preds)
    return {
        "clip_rate_by_class": "|".join(f"{GLOBAL_CLASS_ORDER[idx]}:{float(value)}" for idx, value in enumerate(class_clip_rates)),
        "clip_rate_class_0": float(class_clip_rates[0]) if len(class_clip_rates) > 0 else math.nan,
        "clip_rate_class_1": float(class_clip_rates[1]) if len(class_clip_rates) > 1 else math.nan,
        "clip_rate_by_member": float(clipped.mean(axis=(1, 2)).mean()) if probs.size else math.nan,
        "fraction_predictions_with_any_clipped_member": float((pred_member_probs <= LOG_PROBABILITY_EPSILON).any(axis=0).mean()),
        "mean_min_probability_predicted_class": float(pred_member_probs.min(axis=0).mean()),
        "mean_min_probability_true_class": float(true_member_probs.min(axis=0).mean()),
        "true_class_probability_diagnostics_post_eval_only": 1,
        "probability_average_entropy": _entropy(mean_prob),
        "geometric_entropy": _entropy(geometric_prob),
        "mean_member_entropy": _mean(member_entropy_values),
        "member_disagreement_entropy": vote_stats["member_disagreement_entropy"],
        "mean_pairwise_js_divergence": _mean_pairwise_js_divergence(probs),
        "mean_vote_margin": vote_stats["mean_vote_margin"],
        "fraction_majority_disagreement": vote_stats["fraction_majority_disagreement"],
    }


def _empty_geometric_diagnostics() -> dict[str, object]:
    return {
        "clip_rate_by_class": "",
        "clip_rate_class_0": math.nan,
        "clip_rate_class_1": math.nan,
        "clip_rate_by_member": math.nan,
        "fraction_predictions_with_any_clipped_member": math.nan,
        "mean_min_probability_predicted_class": math.nan,
        "mean_min_probability_true_class": math.nan,
        "true_class_probability_diagnostics_post_eval_only": 1,
        "probability_average_entropy": math.nan,
        "geometric_entropy": math.nan,
        "mean_member_entropy": math.nan,
        "member_disagreement_entropy": math.nan,
        "mean_pairwise_js_divergence": math.nan,
        "mean_vote_margin": math.nan,
        "fraction_majority_disagreement": math.nan,
    }


def _comparison_row(score: _C63Score, c62_probability_average_bacc: float) -> dict[str, object]:
    row = score.matrix_row
    diag = score.probability_diagnostics
    return {
        "ensemble_policy": row.get("ensemble_policy"),
        "experiment_seed": row.get("experiment_seed"),
        "heldout_center": row.get("heldout_center"),
        "support_size": row.get("support_size"),
        "support_seed": row.get("support_seed"),
        "classifier_seed": row.get("classifier_seed"),
        "generation_seed_group": row.get("generation_seed_group"),
        "probability_average_bacc": diag.get("probability_average_bacc"),
        "geometric_bacc": diag.get("geometric_bacc"),
        "hard_vote_bacc": diag.get("hard_vote_bacc"),
        "delta_geometric_vs_probability": _float(diag.get("geometric_bacc")) - _float(diag.get("probability_average_bacc")),
        "c62_probability_average_bacc": c62_probability_average_bacc,
        "delta_vs_c62_probability_average": _float(row.get("bacc")) - _float(c62_probability_average_bacc),
    }


def _normalize_specs(specs: Sequence[EnsembleMemberSpec]) -> tuple[EnsembleMemberSpec, ...]:
    weights = normalize_weights([spec.weight for spec in specs])
    return tuple(_replace_spec_weight(spec, float(weights[idx])) for idx, spec in enumerate(specs))


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


def _member_counts(specs: Sequence[EnsembleMemberSpec]) -> dict[str, int]:
    return {
        "num_experts": len({spec.source_expert for spec in specs}),
        "num_modes": len({spec.bank.mode_label for spec in specs}),
        "num_generation_seeds": len({int(spec.generation_seed) for spec in specs}),
        "num_classifier_seeds": 1,
    }


def _predictions_from_scores(scores: object, global_class_order: Sequence[int]) -> list[int]:
    import numpy as np  # type: ignore

    arr = np.asarray(scores, dtype=float)
    classes = tuple(int(value) for value in global_class_order)
    return [classes[int(idx)] for idx in np.argmax(arr, axis=1)]


def _entropy(probabilities: object) -> float:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    return float(np.mean(-np.sum(probs * np.log(np.clip(probs, LOG_PROBABILITY_EPSILON, 1.0)), axis=1)))


def _vote_statistics(hard_preds: Sequence[Sequence[int]]) -> dict[str, float]:
    import numpy as np  # type: ignore

    if not hard_preds:
        return {"member_disagreement_entropy": math.nan, "mean_vote_margin": math.nan, "fraction_majority_disagreement": math.nan}
    preds = np.asarray(hard_preds, dtype=int)
    entropies = []
    margins = []
    majority_disagreements = []
    for col in preds.T:
        counts = np.asarray([np.sum(col == cls) for cls in GLOBAL_CLASS_ORDER], dtype=float)
        shares = counts / max(float(counts.sum()), 1.0)
        entropies.append(float(-np.sum(shares * np.log(np.clip(shares, LOG_PROBABILITY_EPSILON, 1.0)))))
        ordered = np.sort(shares)[::-1]
        margins.append(float(ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)))
        majority_disagreements.append(float(1.0 - ordered[0]))
    return {
        "member_disagreement_entropy": float(np.mean(entropies)),
        "mean_vote_margin": float(np.mean(margins)),
        "fraction_majority_disagreement": float(np.mean(majority_disagreements)),
    }


def _mean_pairwise_js_divergence(probabilities: object) -> float:
    import numpy as np  # type: ignore

    probs = np.asarray(probabilities, dtype=float)
    if probs.shape[0] < 2:
        return 0.0
    values = []
    for left_idx in range(probs.shape[0]):
        for right_idx in range(left_idx + 1, probs.shape[0]):
            left = np.clip(probs[left_idx], LOG_PROBABILITY_EPSILON, 1.0)
            right = np.clip(probs[right_idx], LOG_PROBABILITY_EPSILON, 1.0)
            middle = 0.5 * (left + right)
            kl_left = np.sum(left * (np.log(left) - np.log(middle)), axis=1)
            kl_right = np.sum(right * (np.log(right) - np.log(middle)), axis=1)
            values.append(float(np.mean(0.5 * (kl_left + kl_right))))
    return _mean(values)


def _load_c62_primary_rows(c62_root: Path) -> list[dict[str, object]]:
    path = c62_root / "tables" / "c62_late_ensemble_downstream_matrix.csv"
    if not path.exists():
        return []
    return [
        row
        for row in load_csv_rows(path)
        if str(row.get("ensemble_policy")) == C62_POLICY_SAFE_MULTI
        and str(row.get("status")) == "ok"
    ]


def _center_baseline(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        center: _mean(_float(row.get("bacc")) for row in rows if str(row.get("heldout_center")) == center)
        for center in sorted({str(row.get("heldout_center")) for row in rows})
    }


def _c62_by_condition(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, str, int, int, int], float]:
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


def _c62_replay_max_abs_delta(
    rows: Sequence[Mapping[str, object]],
    c62_rows: Sequence[Mapping[str, object]],
) -> float:
    baseline = {}
    for row in c62_rows:
        key = (
            int(row.get("experiment_seed", -1)),
            str(row.get("heldout_center")),
            int(row.get("support_size", -1)),
            int(row.get("support_seed", -1)),
            str(row.get("generation_seed_group")),
            int(row.get("classifier_seed", -1)),
        )
        baseline[key] = _float(row.get("bacc"))
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
        if key in baseline:
            deltas.append(abs(_float(row.get("bacc")) - baseline[key]))
    return max(deltas) if deltas else math.nan


def _paired_positive_cells(
    rows: Sequence[Mapping[str, object]],
    baseline_by_condition: Mapping[tuple[int, str, int, int, int], float],
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
        if condition not in baseline_by_condition:
            continue
        cell = (condition[0], condition[1])
        grouped.setdefault(cell, []).append(_float(row.get("bacc")))
        baseline.setdefault(cell, []).append(_float(baseline_by_condition[condition]))
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
    if str(row.get("ensemble_policy")) == POLICY_C62_REPLAY and int(row.get("c62_replay_matches_within_tolerance") or 0) != 1:
        return FAILURE_REPLAY
    mean_bacc = _float(row.get("mean_bacc"))
    delta = _float(row.get("mean_delta_vs_c62_probability_average"))
    paired_positive = int(row.get("paired_positive_center_seed_cells_vs_c62") or 0)
    strong_degrades = int(row.get("strong_center_degrade_gt_002_count") or 0)
    center_1_delta = _float(row.get("center_1_delta_vs_c62"))
    center_3_delta = _float(row.get("center_3_delta_vs_c62"))
    ge80 = _float(row.get("ensemble_ge_080_rate"))
    if mean_bacc >= 0.80 and delta >= 0.01 and paired_positive >= 10 and strong_degrades == 0:
        return DECISION_SUCCESS
    if delta > 0.0 and paired_positive < 8:
        return FAILURE_UNSTABLE
    if (delta >= 0.01 or (center_1_delta >= 0.02 and center_3_delta >= 0.02 and delta >= 0.0) or ge80 >= 0.70) and paired_positive >= 8:
        return DECISION_USEFUL
    if center_1_delta < 0.0 and center_3_delta < 0.0:
        return FAILURE_WEAK_CENTERS
    if delta < -0.01:
        return FAILURE_OVERPENALIZES
    return FAILURE_NO_GAIN


def _policy_role(policy: str) -> str:
    if policy == POLICY_GEOM_SAFE_MULTI:
        return "primary_geometric_dense_late_ensemble"
    if policy in {POLICY_GEOM_SAFE_SINGLE, POLICY_GEOM_FIXED_TOTAL}:
        return "required_budget_control"
    if policy in {POLICY_GEOM_METADATA, POLICY_GEOM_SUPPORT_NELBO}:
        return "secondary_proxy_weighted_geometric"
    if policy == POLICY_C62_REPLAY:
        return "replay_bridge"
    return "ablation"
