"""C5.2 source-LOCO utility-ranking router.

This module consumes frozen C4.1/C4.2/C5.1 artifacts. It first writes
selector-visible predicted utility scores without current-heldout utility, then
joins downstream utility only after route selection.
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

from .c41_heteroscedastic import GENERATION_MODE_POSTERIOR_DECODER_MEAN
from .c42_workstation import C42_DEFAULT_C41_ROOT
from .c51_mode_aware import (
    C51_ARTIFACTS_ROOT,
    C51_DEFAULT_C42_ROOT,
    _load_bank_downstream_rows,
    load_csv_rows,
)
from .downstream import CandidateDownstreamRow, spearman
from .protocol import ProtocolError
from .schemas import HETEROSCEDASTIC_GENERATOR_FAMILY, SUPPORT_NELBO_METHOD


C52_ARTIFACTS_ROOT = "cvae_downstream_evaluation/artifacts/c52_utility_rank_router_v1"
C52_DEFAULT_C41_ROOT = C42_DEFAULT_C41_ROOT
C52_DEFAULT_C42_ROOT = C51_DEFAULT_C42_ROOT
C52_DEFAULT_C51_ROOT = C51_ARTIFACTS_ROOT

SELECTOR_RIDGE = "c52_ridge_loco_utility_rank_top1"
SELECTOR_HGBR = "c52_hgbr_loco_utility_rank_top1"
SELECTOR_RIDGE_NO_EXPERT = "c52_ridge_no_expert_id_loco_utility_rank_top1"
SELECTOR_HGBR_NO_EXPERT = "c52_hgbr_no_expert_id_loco_utility_rank_top1"

PRIMARY_SELECTOR = SELECTOR_RIDGE
NOISE_GENERATION_MODE = "posterior_sample_decoder_noise"
FEATURE_SET_WITH_EXPERT = "with_expert_id"
FEATURE_SET_NO_EXPERT = "no_expert_id"

DECISION_PRIMARY_SUCCESS = "PRIMARY_SUCCESS"
DECISION_USEFUL_RESULT = "USEFUL_RESULT"
FAILURE_NO_TRANSFER = "UTILITY_RANKER_NO_TRANSFER"
FAILURE_MODE_BANK_HIGH = "MODE_BANK_ORACLE_HIGH_ROUTER_WEAK"
FAILURE_RIDGE_BEATS_HGBR = "RIDGE_BEATS_HGBR_NONLINEAR_OVERFIT"
FAILURE_EXPERT_ID_PRIOR = "EXPERT_ID_PRIOR_NO_QUERY_SIGNAL"
FAILURE_PROTOCOL = "PROTOCOL_FAILURE_TARGET_UTILITY_LEAKAGE"

PREJOIN_FORBIDDEN_SUBSTRINGS = (
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

ALLOWED_ZERO_AUDIT_COLUMNS = {
    "current_heldout_utility_visible_before_selection",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
}

EXAMPLE_COLUMNS = (
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "candidate_expert",
    "generator_family",
    "generation_mode",
    "mode_label",
    "primary_candidate_eligible",
    "diagnostic_only",
    "generation_seed_count",
    "support_nelbo_mean",
    "support_nelbo_rank_within_unit",
    "support_nelbo_z_within_unit",
    "support_nelbo_delta_to_best_within_unit",
    "metadata_match",
    "source_global_match",
    "rankmean_dino_median",
    "rankmean_dino_mean",
    "rankmean_dino_std",
    "rankmean_dino_iqr",
    "zsum_dino_median",
    "rankmean_pca_median",
    "rank_energy_dino_median",
    "rank_rbf_mmd_dino_median",
    "rank_mean_l2_dino_median",
    "rank_cov_trace_dino_median",
    "rank_pairwise_dino_median",
    "utility_label_bacc",
    "utility_label_bacc_std",
    "utility_label_bacc_min",
    "utility_label_bacc_max",
    "utility_label_ge_080_rate",
    "utility_label_macro_f1",
    "utility_label_generation_seed_count",
    "utility_label_classifier_seed_count",
)

PREJOIN_SCORE_COLUMNS = (
    "selector_name",
    "selector_family",
    "selector_role",
    "diagnostic_only",
    "model_type",
    "feature_set",
    "current_heldout_center",
    "training_heldout_centers",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "candidate_expert",
    "generator_family",
    "generation_mode",
    "mode_label",
    "predicted_utility_score",
    "predicted_rank_within_unit",
    "support_nelbo_rank_within_unit",
    "candidate_scores_available_before_selection",
    "feature_hash",
    "current_heldout_utility_visible_before_selection",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
)

SELECTED_PREJOIN_COLUMNS = (
    "selector_name",
    "selector_family",
    "selector_role",
    "diagnostic_only",
    "model_type",
    "feature_set",
    "current_heldout_center",
    "training_heldout_centers",
    "experiment_seed",
    "heldout_center",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "selected_expert",
    "selected_generator_family",
    "selected_generation_mode",
    "selected_mode_label",
    "selected_predicted_utility_score",
    "predicted_margin_to_top2",
    "candidate_scores_available_before_selection",
    "selected_feature_hash",
    "selected_route_key",
    "current_heldout_utility_visible_before_selection",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
)

UTILITY_JOIN_COLUMNS = (
    "selector_name",
    "selector_family",
    "selector_role",
    "diagnostic_only",
    "model_type",
    "feature_set",
    "heldout_center",
    "experiment_seed",
    "support_size",
    "support_seed",
    "support_eval_split_id",
    "selected_expert",
    "selected_generator_family",
    "selected_generation_mode",
    "selected_mode_label",
    "selected_predicted_utility_score",
    "selected_bacc_mean",
    "selected_bacc_std",
    "selected_bacc_min",
    "selected_bacc_max",
    "selected_ge_080_rate",
    "selected_macro_f1_mean",
    "oracle_expert",
    "oracle_generator_family",
    "oracle_generation_mode",
    "oracle_bacc_mean",
    "oracle_gap_bacc",
    "selected_rank_by_bacc",
    "top1_oracle_hit",
    "top2_oracle_hit",
    "regret_bacc",
    "spearman_predicted_vs_bacc",
    "candidate_scores_available_before_selection",
    "selected_route_key",
    "downstream_matrix_join_hash",
    "current_heldout_utility_visible_before_selection",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
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
    "mean_spearman",
    "median_spearman",
    "positive_spearman_rate",
    "regret_p50",
    "regret_p75",
    "regret_p90",
    "baseline_c41_hetero_mean_selected_bacc",
    "selected_bacc_delta_vs_c41_hetero_mean",
    "c41_to_oracle_gap_closed_fraction",
    "center_positive_improvement_count",
    "strong_center_degrade_gt_002_count",
    "decision_label",
)

PROTOCOL_AUDIT_COLUMNS = (
    "selector_name",
    "current_heldout_center",
    "training_heldout_centers",
    "n_train_examples",
    "n_score_examples",
    "current_heldout_center_in_training",
    "pre_join_forbidden_columns_present",
    "current_heldout_utility_visible_before_selection",
    "target_support_labels_used",
    "target_eval_labels_used_for_selection",
    "primary_candidate_bank_excludes_noise",
    "protocol_status",
)


@dataclass(frozen=True)
class C52RunLimits:
    experiment_seeds: tuple[int, ...] | None = None
    heldout_centers: tuple[str, ...] | None = None
    support_sizes: tuple[int, ...] | None = None
    support_seeds: tuple[int, ...] | None = None


@dataclass(frozen=True)
class UtilitySummary:
    mean_bacc: float
    std_bacc: float
    min_bacc: float
    max_bacc: float
    ge_080_rate: float
    mean_macro_f1: float
    generation_seed_count: int
    classifier_seed_count: int


@dataclass(frozen=True)
class SelectorSpec:
    name: str
    model_type: str
    feature_set: str
    role: str
    diagnostic_only: int


SELECTOR_SPECS = (
    SelectorSpec(SELECTOR_RIDGE, "ridge", FEATURE_SET_WITH_EXPERT, "primary", 0),
    SelectorSpec(SELECTOR_HGBR, "hgbr", FEATURE_SET_WITH_EXPERT, "comparator", 0),
    SelectorSpec(SELECTOR_RIDGE_NO_EXPERT, "ridge", FEATURE_SET_NO_EXPERT, "ablation", 1),
    SelectorSpec(SELECTOR_HGBR_NO_EXPERT, "hgbr", FEATURE_SET_NO_EXPERT, "ablation", 1),
)

NUMERIC_FEATURES = (
    "support_size",
    "support_nelbo_rank_within_unit",
    "support_nelbo_z_within_unit",
    "support_nelbo_delta_to_best_within_unit",
    "metadata_match",
    "source_global_match",
    "generation_seed_count",
    "rankmean_dino_median",
    "rankmean_dino_mean",
    "rankmean_dino_std",
    "rankmean_dino_iqr",
    "zsum_dino_median",
    "rankmean_pca_median",
    "rank_energy_dino_median",
    "rank_rbf_mmd_dino_median",
    "rank_mean_l2_dino_median",
    "rank_cov_trace_dino_median",
    "rank_pairwise_dino_median",
)

CATEGORICAL_WITH_EXPERT = (
    "candidate_expert",
    "generator_family",
    "generation_mode",
    "mode_label",
)
CATEGORICAL_NO_EXPERT = (
    "generator_family",
    "generation_mode",
    "mode_label",
)


def run_c52_utility_ranker(
    *,
    artifacts_root: Path,
    c41_artifacts_root: Path,
    c42_artifacts_root: Path,
    c51_artifacts_root: Path,
    limits: C52RunLimits = C52RunLimits(),
) -> dict[str, Path]:
    score_rows = load_csv_rows(c51_artifacts_root / "tables" / "c51_support_mode_scores.csv")
    downstream_rows = _load_bank_downstream_rows(c41_artifacts_root, c42_artifacts_root)
    downstream_hash = _file_hash(c41_artifacts_root / "tables" / "all_expert_downstream_matrix.csv") + "." + _file_hash(c42_artifacts_root / "tables" / "all_expert_downstream_matrix.csv")
    score_rows = _apply_score_limits(score_rows, limits)
    examples = build_c52_router_examples(score_rows, downstream_rows)
    prejoin_scores, protocol_audit = build_c52_prejoin_predictions(
        examples,
        current_heldout_centers=limits.heldout_centers,
    )
    assert_prejoin_rows_safe(prejoin_scores)
    selected = build_c52_selected_routes_prejoin(prejoin_scores)
    assert_prejoin_rows_safe(selected)
    utility_join = build_c52_utility_join(selected, prejoin_scores, examples, downstream_hash=downstream_hash)
    baseline_rows = _load_c41_hetero_mean_baseline_rows(c41_artifacts_root)
    threshold = build_c52_threshold_audit_rows(utility_join, baseline_rows)
    center = build_c52_center_summary_rows(utility_join, baseline_rows)
    outputs = {
        "examples": artifacts_root / "tables" / "c52_router_training_examples.csv",
        "prejoin_scores": artifacts_root / "tables" / "c52_predicted_utility_scores_pre_join.csv",
        "selected_prejoin": artifacts_root / "tables" / "c52_selected_routes_pre_join.csv",
        "utility_join": artifacts_root / "tables" / "c52_selected_route_utility_join.csv",
        "threshold": artifacts_root / "tables" / "c52_threshold_audit.csv",
        "center": artifacts_root / "tables" / "c52_center_summary.csv",
        "protocol": artifacts_root / "tables" / "c52_protocol_audit.csv",
    }
    _write_csv(outputs["examples"], EXAMPLE_COLUMNS, examples)
    _write_csv(outputs["prejoin_scores"], PREJOIN_SCORE_COLUMNS, prejoin_scores)
    _write_csv(outputs["selected_prejoin"], SELECTED_PREJOIN_COLUMNS, selected)
    _write_csv(outputs["utility_join"], UTILITY_JOIN_COLUMNS, utility_join)
    _write_csv(outputs["threshold"], SUMMARY_COLUMNS, threshold)
    _write_csv(outputs["center"], _center_columns(), center)
    _write_csv(outputs["protocol"], PROTOCOL_AUDIT_COLUMNS, protocol_audit)
    return outputs


def build_c52_router_examples(
    score_rows: Sequence[Mapping[str, object]],
    downstream_rows: Sequence[CandidateDownstreamRow],
) -> list[dict[str, object]]:
    utility_index = _utility_by_candidate_mode(downstream_rows)
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in score_rows:
        key = _candidate_mode_key(row)
        grouped.setdefault(key, []).append(row)
    examples: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        first = rows[0]
        utility = utility_index.get(key, _utility_summary([], []))
        generation_mode = str(first["generation_mode"])
        eligible = int(generation_mode != NOISE_GENERATION_MODE)
        example = {
            "experiment_seed": int(first["experiment_seed"]),
            "heldout_center": str(first["heldout_center"]),
            "support_size": int(first["support_size"]),
            "support_seed": int(first["support_seed"]),
            "support_eval_split_id": str(first.get("support_eval_split_id", "")),
            "candidate_expert": str(first["candidate_expert"]),
            "generator_family": str(first["generator_family"]),
            "generation_mode": generation_mode,
            "mode_label": str(first.get("mode_label", generation_mode)),
            "primary_candidate_eligible": eligible,
            "diagnostic_only": int(not eligible),
            "generation_seed_count": len({int(row["generation_seed"]) for row in rows}),
            "support_nelbo_mean": _mean(_float(row.get("support_nelbo")) for row in rows),
            "support_nelbo_rank_within_unit": _mean(_float(row.get("support_nelbo_rank")) for row in rows),
            "support_nelbo_z_within_unit": math.nan,
            "support_nelbo_delta_to_best_within_unit": math.nan,
            "metadata_match": int(str(first.get("metadata_selected_expert", "")) == str(first["candidate_expert"])),
            "source_global_match": int(str(first.get("source_global_selected_expert", "")) == str(first["candidate_expert"])),
            "rankmean_dino_median": _median(_float(row.get("rankmean_dino_score")) for row in rows),
            "rankmean_dino_mean": _mean(_float(row.get("rankmean_dino_score")) for row in rows),
            "rankmean_dino_std": _std(_float(row.get("rankmean_dino_score")) for row in rows),
            "rankmean_dino_iqr": _iqr(_float(row.get("rankmean_dino_score")) for row in rows),
            "zsum_dino_median": _median(_float(row.get("zsum_dino_score")) for row in rows),
            "rankmean_pca_median": _median(_float(row.get("rankmean_pca_score")) for row in rows),
            "rank_energy_dino_median": _median(_float(row.get("rank_energy_dino")) for row in rows),
            "rank_rbf_mmd_dino_median": _median(_float(row.get("rank_rbf_mmd_dino")) for row in rows),
            "rank_mean_l2_dino_median": _median(_float(row.get("rank_mean_l2_dino")) for row in rows),
            "rank_cov_trace_dino_median": _median(_float(row.get("rank_cov_trace_dino")) for row in rows),
            "rank_pairwise_dino_median": _median(_float(row.get("rank_pairwise_dino")) for row in rows),
            "utility_label_bacc": utility.mean_bacc,
            "utility_label_bacc_std": utility.std_bacc,
            "utility_label_bacc_min": utility.min_bacc,
            "utility_label_bacc_max": utility.max_bacc,
            "utility_label_ge_080_rate": utility.ge_080_rate,
            "utility_label_macro_f1": utility.mean_macro_f1,
            "utility_label_generation_seed_count": utility.generation_seed_count,
            "utility_label_classifier_seed_count": utility.classifier_seed_count,
        }
        examples.append(example)
    _add_support_nelbo_unit_features(examples)
    return examples


def build_c52_prejoin_predictions(
    examples: Sequence[Mapping[str, object]],
    *,
    current_heldout_centers: Sequence[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _require_sklearn()
    import numpy as np  # type: ignore

    rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    heldout_centers = (
        tuple(str(center) for center in current_heldout_centers)
        if current_heldout_centers
        else tuple(sorted({str(row["heldout_center"]) for row in examples}))
    )
    for current in heldout_centers:
        for spec in SELECTOR_SPECS:
            train = [
                row
                for row in examples
                if str(row["heldout_center"]) != current
                and int(row.get("primary_candidate_eligible", 0)) == 1
                and not math.isnan(_float(row.get("utility_label_bacc")))
            ]
            score = [
                row
                for row in examples
                if str(row["heldout_center"]) == current
                and int(row.get("primary_candidate_eligible", 0)) == 1
            ]
            train_centers = sorted({str(row["heldout_center"]) for row in train})
            audit.append(
                {
                    "selector_name": spec.name,
                    "current_heldout_center": current,
                    "training_heldout_centers": ";".join(train_centers),
                    "n_train_examples": len(train),
                    "n_score_examples": len(score),
                    "current_heldout_center_in_training": int(current in train_centers),
                    "pre_join_forbidden_columns_present": 0,
                    "current_heldout_utility_visible_before_selection": 0,
                    "target_support_labels_used": 0,
                    "target_eval_labels_used_for_selection": 0,
                    "primary_candidate_bank_excludes_noise": int(all(str(row["generation_mode"]) != NOISE_GENERATION_MODE for row in score)),
                    "protocol_status": "pass",
                }
            )
            if not train or not score:
                continue
            model = _fit_selector_model(train, spec)
            predictions = model.predict(_feature_matrix(score, spec))
            for example, pred in zip(score, list(np.asarray(predictions, dtype=float))):
                rows.append(
                    {
                        "selector_name": spec.name,
                        "selector_family": "utility_ranker",
                        "selector_role": spec.role,
                        "diagnostic_only": spec.diagnostic_only,
                        "model_type": spec.model_type,
                        "feature_set": spec.feature_set,
                        "current_heldout_center": current,
                        "training_heldout_centers": ";".join(train_centers),
                        "experiment_seed": int(example["experiment_seed"]),
                        "heldout_center": str(example["heldout_center"]),
                        "support_size": int(example["support_size"]),
                        "support_seed": int(example["support_seed"]),
                        "support_eval_split_id": str(example["support_eval_split_id"]),
                        "candidate_expert": str(example["candidate_expert"]),
                        "generator_family": str(example["generator_family"]),
                        "generation_mode": str(example["generation_mode"]),
                        "mode_label": str(example["mode_label"]),
                        "predicted_utility_score": float(pred),
                        "predicted_rank_within_unit": math.nan,
                        "support_nelbo_rank_within_unit": _float(example.get("support_nelbo_rank_within_unit")),
                        "candidate_scores_available_before_selection": 0,
                        "feature_hash": _feature_hash(example, spec),
                        "current_heldout_utility_visible_before_selection": 0,
                        "target_support_labels_used": 0,
                        "target_eval_labels_used_for_selection": 0,
                    }
                )
    _rank_predictions_within_units(rows)
    assert_prejoin_rows_safe(rows)
    return rows, audit


def build_c52_selected_routes_prejoin(prejoin_scores: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    selected = []
    for key, rows in sorted(_group_by_selector_condition(prejoin_scores).items()):
        ranked = sorted(rows, key=lambda row: (-_float(row["predicted_utility_score"]), _tie_key(row)))
        if not ranked:
            continue
        top = ranked[0]
        top2 = ranked[1] if len(ranked) > 1 else None
        selected.append(
            {
                "selector_name": str(top["selector_name"]),
                "selector_family": str(top["selector_family"]),
                "selector_role": str(top["selector_role"]),
                "diagnostic_only": int(top["diagnostic_only"]),
                "model_type": str(top["model_type"]),
                "feature_set": str(top["feature_set"]),
                "current_heldout_center": str(top["current_heldout_center"]),
                "training_heldout_centers": str(top["training_heldout_centers"]),
                "experiment_seed": int(top["experiment_seed"]),
                "heldout_center": str(top["heldout_center"]),
                "support_size": int(top["support_size"]),
                "support_seed": int(top["support_seed"]),
                "support_eval_split_id": str(top["support_eval_split_id"]),
                "selected_expert": str(top["candidate_expert"]),
                "selected_generator_family": str(top["generator_family"]),
                "selected_generation_mode": str(top["generation_mode"]),
                "selected_mode_label": str(top["mode_label"]),
                "selected_predicted_utility_score": _float(top["predicted_utility_score"]),
                "predicted_margin_to_top2": _float(top["predicted_utility_score"]) - _float(top2["predicted_utility_score"]) if top2 else math.nan,
                "candidate_scores_available_before_selection": len(ranked),
                "selected_feature_hash": str(top["feature_hash"]),
                "selected_route_key": _selected_route_key(top),
                "current_heldout_utility_visible_before_selection": 0,
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
            }
        )
    assert_prejoin_rows_safe(selected)
    return selected


def build_c52_utility_join(
    selected_rows: Sequence[Mapping[str, object]],
    prejoin_scores: Sequence[Mapping[str, object]],
    examples: Sequence[Mapping[str, object]],
    *,
    downstream_hash: str,
) -> list[dict[str, object]]:
    utility_by_key = {_candidate_mode_key(row): row for row in examples}
    scores_by_condition = _group_by_selector_condition(prejoin_scores)
    out = []
    for selected in selected_rows:
        selected_key = (
            int(selected["experiment_seed"]),
            str(selected["heldout_center"]),
            int(selected["support_size"]),
            int(selected["support_seed"]),
            str(selected["selected_expert"]),
            str(selected["selected_generator_family"]),
            str(selected["selected_generation_mode"]),
        )
        utility = utility_by_key.get(selected_key)
        condition_key = (
            str(selected["selector_name"]),
            int(selected["experiment_seed"]),
            str(selected["heldout_center"]),
            int(selected["support_size"]),
            int(selected["support_seed"]),
        )
        candidates = scores_by_condition.get(condition_key, [])
        ranked = _rank_true_utility(candidates, utility_by_key)
        oracle = ranked[0] if ranked else None
        selected_rank = next(
            (
                idx + 1
                for idx, item in enumerate(ranked)
                if _candidate_key_from_score(item[0]) == selected_key
            ),
            999,
        )
        spearman_value = _spearman_for_condition(candidates, utility_by_key)
        selected_bacc = _float(utility.get("utility_label_bacc") if utility else math.nan)
        oracle_bacc = _float(oracle[1].get("utility_label_bacc") if oracle else math.nan)
        out.append(
            {
                "selector_name": str(selected["selector_name"]),
                "selector_family": str(selected["selector_family"]),
                "selector_role": str(selected["selector_role"]),
                "diagnostic_only": int(selected["diagnostic_only"]),
                "model_type": str(selected["model_type"]),
                "feature_set": str(selected["feature_set"]),
                "heldout_center": str(selected["heldout_center"]),
                "experiment_seed": int(selected["experiment_seed"]),
                "support_size": int(selected["support_size"]),
                "support_seed": int(selected["support_seed"]),
                "support_eval_split_id": str(selected["support_eval_split_id"]),
                "selected_expert": str(selected["selected_expert"]),
                "selected_generator_family": str(selected["selected_generator_family"]),
                "selected_generation_mode": str(selected["selected_generation_mode"]),
                "selected_mode_label": str(selected["selected_mode_label"]),
                "selected_predicted_utility_score": _float(selected["selected_predicted_utility_score"]),
                "selected_bacc_mean": selected_bacc,
                "selected_bacc_std": _float(utility.get("utility_label_bacc_std") if utility else 0.0),
                "selected_bacc_min": _float(utility.get("utility_label_bacc_min") if utility else selected_bacc),
                "selected_bacc_max": _float(utility.get("utility_label_bacc_max") if utility else selected_bacc),
                "selected_ge_080_rate": _float(utility.get("utility_label_ge_080_rate") if utility else math.nan),
                "selected_macro_f1_mean": _float(utility.get("utility_label_macro_f1") if utility else math.nan),
                "oracle_expert": str(oracle[0]["candidate_expert"]) if oracle else "",
                "oracle_generator_family": str(oracle[0]["generator_family"]) if oracle else "",
                "oracle_generation_mode": str(oracle[0]["generation_mode"]) if oracle else "",
                "oracle_bacc_mean": oracle_bacc,
                "oracle_gap_bacc": oracle_bacc - selected_bacc if not math.isnan(oracle_bacc) and not math.isnan(selected_bacc) else math.nan,
                "selected_rank_by_bacc": selected_rank,
                "top1_oracle_hit": int(selected_rank == 1),
                "top2_oracle_hit": int(selected_rank <= 2),
                "regret_bacc": oracle_bacc - selected_bacc if not math.isnan(oracle_bacc) and not math.isnan(selected_bacc) else math.nan,
                "spearman_predicted_vs_bacc": spearman_value,
                "candidate_scores_available_before_selection": int(selected["candidate_scores_available_before_selection"]),
                "selected_route_key": str(selected["selected_route_key"]),
                "downstream_matrix_join_hash": downstream_hash,
                "current_heldout_utility_visible_before_selection": 0,
                "target_support_labels_used": 0,
                "target_eval_labels_used_for_selection": 0,
                "protocol_status": "pass",
            }
        )
    return out


def build_c52_threshold_audit_rows(
    utility_rows: Sequence[Mapping[str, object]],
    c41_baseline_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = _baseline_summary(c41_baseline_rows)
    baseline_by_center = _baseline_by_center(c41_baseline_rows)
    summaries: dict[str, dict[str, float]] = {}
    out = []
    for selector in sorted({str(row["selector_name"]) for row in utility_rows}):
        subset = [row for row in utility_rows if str(row["selector_name"]) == selector]
        selected = [_float(row["selected_bacc_mean"]) for row in subset]
        oracle = [_float(row["oracle_bacc_mean"]) for row in subset]
        gaps = [_float(row["oracle_gap_bacc"]) for row in subset]
        regrets = [_float(row["regret_bacc"]) for row in subset]
        spearman_values = [_float(row["spearman_predicted_vs_bacc"]) for row in subset]
        mean_selected = _mean(selected)
        mean_oracle = _mean(oracle)
        c41_gap = float(baseline["oracle_bacc"] - baseline["selected_bacc"])
        closed = (mean_selected - baseline["selected_bacc"]) / c41_gap if c41_gap > 1.0e-12 else math.nan
        summary = {
            "mean_selected": mean_selected,
            "mean_spearman": _mean(spearman_values),
            "closed": closed,
        }
        summaries[selector] = summary
        out.append(
            {
                "selector_name": selector,
                "selector_family": str(subset[0]["selector_family"]) if subset else "",
                "selector_role": str(subset[0]["selector_role"]) if subset else "",
                "diagnostic_only": int(subset[0]["diagnostic_only"]) if subset else 1,
                "n_rows": len(subset),
                "mean_selected_bacc": mean_selected,
                "selected_ge_080_rate": _mean(1.0 if value >= 0.80 else 0.0 for value in selected),
                "mean_oracle_bacc": mean_oracle,
                "mean_oracle_gap_bacc": _mean(gaps),
                "top1_hit_rate": _mean(_float(row["top1_oracle_hit"]) for row in subset),
                "top2_hit_rate": _mean(_float(row["top2_oracle_hit"]) for row in subset),
                "mean_spearman": _mean(spearman_values),
                "median_spearman": _median(spearman_values),
                "positive_spearman_rate": _mean(1.0 if value > 0.0 else 0.0 for value in spearman_values),
                "regret_p50": _quantile(regrets, 0.50),
                "regret_p75": _quantile(regrets, 0.75),
                "regret_p90": _quantile(regrets, 0.90),
                "baseline_c41_hetero_mean_selected_bacc": baseline["selected_bacc"],
                "selected_bacc_delta_vs_c41_hetero_mean": mean_selected - baseline["selected_bacc"],
                "c41_to_oracle_gap_closed_fraction": closed,
                "center_positive_improvement_count": _center_positive_count(subset, baseline_by_center),
                "strong_center_degrade_gt_002_count": _strong_center_degrade_count(subset, baseline_by_center),
                "decision_label": "",
            }
        )
    no_expert = {
        SELECTOR_RIDGE: SELECTOR_RIDGE_NO_EXPERT,
        SELECTOR_HGBR: SELECTOR_HGBR_NO_EXPERT,
    }
    ridge_mean = summaries.get(SELECTOR_RIDGE, {}).get("mean_selected", math.nan)
    for row in out:
        selector = str(row["selector_name"])
        row["decision_label"] = _decision_label(
            selector=selector,
            row=row,
            baseline=baseline,
            no_expert_summary=summaries.get(no_expert.get(selector, ""), {}),
            ridge_mean=ridge_mean,
        )
    return out


def build_c52_center_summary_rows(
    utility_rows: Sequence[Mapping[str, object]],
    c41_baseline_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    baseline = _baseline_by_center(c41_baseline_rows)
    out = []
    for selector in sorted({str(row["selector_name"]) for row in utility_rows}):
        for center in sorted({str(row["heldout_center"]) for row in utility_rows}):
            subset = [row for row in utility_rows if str(row["selector_name"]) == selector and str(row["heldout_center"]) == center]
            if not subset:
                continue
            selected = [_float(row["selected_bacc_mean"]) for row in subset]
            base = baseline.get(center, math.nan)
            out.append(
                {
                    "selector_name": selector,
                    "heldout_center": center,
                    "n_rows": len(subset),
                    "mean_selected_bacc": _mean(selected),
                    "c41_hetero_mean_selected_bacc": base,
                    "delta_vs_c41_hetero_mean": _mean(selected) - base if not math.isnan(base) else math.nan,
                    "mean_oracle_bacc": _mean(_float(row["oracle_bacc_mean"]) for row in subset),
                    "mean_oracle_gap_bacc": _mean(_float(row["oracle_gap_bacc"]) for row in subset),
                    "top1_hit_rate": _mean(_float(row["top1_oracle_hit"]) for row in subset),
                    "mean_spearman": _mean(_float(row["spearman_predicted_vs_bacc"]) for row in subset),
                    "regret_p75": _quantile([_float(row["regret_bacc"]) for row in subset], 0.75),
                }
            )
    return out


def assert_prejoin_rows_safe(rows: Sequence[Mapping[str, object]]) -> None:
    bad = sorted(
        {
            str(key)
            for row in rows
            for key in row
            if str(key) not in ALLOWED_ZERO_AUDIT_COLUMNS
            if any(token in str(key).lower() for token in PREJOIN_FORBIDDEN_SUBSTRINGS)
        }
    )
    if bad:
        raise ProtocolError(f"C5.2 pre-join selector rows contain forbidden utility/eval columns: {bad}")


def _fit_selector_model(rows: Sequence[Mapping[str, object]], spec: SelectorSpec):
    from sklearn.compose import ColumnTransformer  # type: ignore
    from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.linear_model import Ridge  # type: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

    categorical = CATEGORICAL_WITH_EXPERT if spec.feature_set == FEATURE_SET_WITH_EXPERT else CATEGORICAL_NO_EXPERT
    encoder = _dense_one_hot_encoder(OneHotEncoder)
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(range(len(NUMERIC_FEATURES)))),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), encoder), list(range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(categorical)))),
        ]
    )
    estimator = (
        Ridge(alpha=10.0)
        if spec.model_type == "ridge"
        else HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=7,
            l2_regularization=1.0,
            random_state=17,
        )
    )
    model = make_pipeline(pre, estimator)
    model.fit(_feature_matrix(rows, spec), [_float(row["utility_label_bacc"]) for row in rows])
    return model


def _feature_matrix(rows: Sequence[Mapping[str, object]], spec: SelectorSpec) -> list[list[object]]:
    categorical = CATEGORICAL_WITH_EXPERT if spec.feature_set == FEATURE_SET_WITH_EXPERT else CATEGORICAL_NO_EXPERT
    return [[row.get(key) for key in (*NUMERIC_FEATURES, *categorical)] for row in rows]


def _dense_one_hot_encoder(cls):
    try:
        return cls(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return cls(handle_unknown="ignore", sparse=False)


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ProtocolError("C5.2 requires scikit-learn for Ridge/HGBR utility ranking.") from exc


def _utility_by_candidate_mode(rows: Sequence[CandidateDownstreamRow]) -> dict[tuple[object, ...], UtilitySummary]:
    grouped: dict[tuple[object, ...], list[CandidateDownstreamRow]] = {}
    for row in rows:
        if row.status != "ok" or row.row_type != "single_expert":
            continue
        key = (
            int(row.experiment_seed),
            str(row.heldout_center),
            int(row.support_size),
            int(row.support_seed),
            str(row.candidate_expert),
            str(row.generator_family),
            str(row.generation_mode),
        )
        grouped.setdefault(key, []).append(row)
    return {
        key: _utility_summary([row.bacc for row in items], [row.macro_f1 for row in items], items)
        for key, items in grouped.items()
    }


def _utility_summary(
    bacc: Sequence[float],
    macro_f1: Sequence[float],
    rows: Sequence[CandidateDownstreamRow] = (),
) -> UtilitySummary:
    clean = [float(value) for value in bacc if not math.isnan(float(value))]
    return UtilitySummary(
        mean_bacc=_mean(clean),
        std_bacc=statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        min_bacc=min(clean) if clean else math.nan,
        max_bacc=max(clean) if clean else math.nan,
        ge_080_rate=_mean(1.0 if value >= 0.80 else 0.0 for value in clean),
        mean_macro_f1=_mean(float(value) for value in macro_f1 if not math.isnan(float(value))),
        generation_seed_count=len({int(row.generation_seed) for row in rows}),
        classifier_seed_count=len({int(row.classifier_seed) for row in rows}),
    )


def _add_support_nelbo_unit_features(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (int(row["experiment_seed"]), str(row["heldout_center"]), int(row["support_size"]), int(row["support_seed"]))
        grouped.setdefault(key, []).append(row)
    for items in grouped.values():
        values = [_float(row.get("support_nelbo_mean")) for row in items]
        finite = [value for value in values if not math.isnan(value)]
        mu = _mean(finite)
        sigma = statistics.pstdev(finite) if len(finite) > 1 else 1.0
        sigma = sigma if sigma > 1.0e-12 else 1.0
        best = min(finite) if finite else math.nan
        for row in items:
            value = _float(row.get("support_nelbo_mean"))
            row["support_nelbo_z_within_unit"] = (value - mu) / sigma if not math.isnan(value) else math.nan
            row["support_nelbo_delta_to_best_within_unit"] = value - best if not math.isnan(value) and not math.isnan(best) else math.nan


def _rank_predictions_within_units(rows: list[dict[str, object]]) -> None:
    for _key, items in _group_by_selector_condition(rows).items():
        ranked = sorted(items, key=lambda row: (-_float(row["predicted_utility_score"]), _tie_key(row)))
        for rank, row in enumerate(ranked, start=1):
            row["predicted_rank_within_unit"] = rank
            row["candidate_scores_available_before_selection"] = len(ranked)


def _rank_true_utility(
    score_rows: Sequence[Mapping[str, object]],
    utility_by_key: Mapping[tuple[object, ...], Mapping[str, object]],
) -> list[tuple[Mapping[str, object], Mapping[str, object]]]:
    ranked = []
    for row in score_rows:
        utility = utility_by_key.get(_candidate_key_from_score(row))
        if utility is None:
            continue
        ranked.append((row, utility))
    ranked.sort(key=lambda item: (-_float(item[1]["utility_label_bacc"]), _tie_key(item[0])))
    return ranked


def _spearman_for_condition(
    score_rows: Sequence[Mapping[str, object]],
    utility_by_key: Mapping[tuple[object, ...], Mapping[str, object]],
) -> float:
    preds = []
    utils = []
    for row in score_rows:
        utility = utility_by_key.get(_candidate_key_from_score(row))
        if utility is None:
            continue
        preds.append(_float(row["predicted_utility_score"]))
        utils.append(_float(utility["utility_label_bacc"]))
    return spearman(preds, utils) if len(preds) >= 2 else math.nan


def _load_c41_hetero_mean_baseline_rows(c41_root: Path) -> list[dict[str, object]]:
    rows = load_csv_rows(c41_root / "tables" / "routing_to_downstream_alignment.csv")
    return [
        row
        for row in rows
        if str(row.get("generator_family")) == HETEROSCEDASTIC_GENERATOR_FAMILY
        and str(row.get("generation_mode")) == GENERATION_MODE_POSTERIOR_DECODER_MEAN
        and str(row.get("method")) == SUPPORT_NELBO_METHOD
    ]


def _baseline_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        "selected_bacc": _mean(_float(row.get("selected_bacc")) for row in rows),
        "oracle_bacc": _mean(_float(row.get("oracle_bacc")) for row in rows),
    }


def _baseline_by_center(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    out = {}
    for center in sorted({str(row.get("heldout_center")) for row in rows}):
        out[center] = _mean(_float(row.get("selected_bacc")) for row in rows if str(row.get("heldout_center")) == center)
    return out


def _decision_label(
    *,
    selector: str,
    row: Mapping[str, object],
    baseline: Mapping[str, float],
    no_expert_summary: Mapping[str, float],
    ridge_mean: float,
) -> str:
    if int(row.get("diagnostic_only", 0)):
        return "DIAGNOSTIC_ONLY"
    mean_selected = _float(row["mean_selected_bacc"])
    closed = _float(row["c41_to_oracle_gap_closed_fraction"])
    positive_centers = int(row["center_positive_improvement_count"])
    mean_spearman = _float(row["mean_spearman"])
    if selector == SELECTOR_HGBR and not math.isnan(ridge_mean) and mean_selected <= ridge_mean - 0.02:
        return FAILURE_RIDGE_BEATS_HGBR
    no_expert_selected = _float(no_expert_summary.get("mean_selected"))
    no_expert_spearman = _float(no_expert_summary.get("mean_spearman"))
    if not math.isnan(no_expert_selected) and mean_selected >= no_expert_selected + 0.02 and no_expert_spearman <= 0.0:
        return FAILURE_EXPERT_ID_PRIOR
    if mean_selected >= 0.80 or (closed >= 0.50 and positive_centers >= 4 and mean_spearman > 0.0):
        return DECISION_PRIMARY_SUCCESS
    if closed >= 0.50 or mean_selected >= 0.77:
        return DECISION_USEFUL_RESULT
    if _float(row["mean_oracle_bacc"]) >= 0.80:
        return FAILURE_MODE_BANK_HIGH
    if mean_selected <= _float(baseline.get("selected_bacc")):
        return FAILURE_NO_TRANSFER
    return FAILURE_NO_TRANSFER


def _center_positive_count(rows: Sequence[Mapping[str, object]], baseline: Mapping[str, float]) -> int:
    count = 0
    for center in sorted({str(row["heldout_center"]) for row in rows}):
        current = _mean(_float(row["selected_bacc_mean"]) for row in rows if str(row["heldout_center"]) == center)
        base = _float(baseline.get(center))
        if not math.isnan(current) and not math.isnan(base) and current > base:
            count += 1
    return count


def _strong_center_degrade_count(rows: Sequence[Mapping[str, object]], baseline: Mapping[str, float]) -> int:
    count = 0
    for center, base in baseline.items():
        base_value = _float(base)
        if math.isnan(base_value) or base_value < 0.80:
            continue
        current = _mean(_float(row["selected_bacc_mean"]) for row in rows if str(row["heldout_center"]) == str(center))
        if not math.isnan(current) and current < base_value - 0.02:
            count += 1
    return count


def _apply_score_limits(rows: Sequence[Mapping[str, object]], limits: C52RunLimits) -> list[Mapping[str, object]]:
    out = []
    experiment_seeds = set(int(v) for v in limits.experiment_seeds) if limits.experiment_seeds else None
    support_sizes = set(int(v) for v in getattr(limits, "support_sizes", ()) or ()) or None
    support_seeds = set(int(v) for v in getattr(limits, "support_seeds", ()) or ()) or None
    for row in rows:
        if experiment_seeds is not None and int(row["experiment_seed"]) not in experiment_seeds:
            continue
        if support_sizes is not None and int(row["support_size"]) not in support_sizes:
            continue
        if support_seeds is not None and int(row["support_seed"]) not in support_seeds:
            continue
        out.append(row)
    return out


def _candidate_mode_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["experiment_seed"]),
        str(row["heldout_center"]),
        int(row["support_size"]),
        int(row["support_seed"]),
        str(row["candidate_expert"]),
        str(row["generator_family"]),
        str(row["generation_mode"]),
    )


def _candidate_key_from_score(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["experiment_seed"]),
        str(row["heldout_center"]),
        int(row["support_size"]),
        int(row["support_seed"]),
        str(row["candidate_expert"]),
        str(row["generator_family"]),
        str(row["generation_mode"]),
    )


def _group_by_selector_condition(rows: Sequence[Mapping[str, object]]) -> dict[tuple[object, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            str(row["selector_name"]),
            int(row["experiment_seed"]),
            str(row["heldout_center"]),
            int(row["support_size"]),
            int(row["support_seed"]),
        )
        grouped.setdefault(key, []).append(row)
    return grouped


def _tie_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (str(row["candidate_expert"]), str(row["generator_family"]), str(row["generation_mode"]))


def _feature_hash(row: Mapping[str, object], spec: SelectorSpec) -> str:
    categorical = CATEGORICAL_WITH_EXPERT if spec.feature_set == FEATURE_SET_WITH_EXPERT else CATEGORICAL_NO_EXPERT
    payload = {key: row.get(key) for key in (*NUMERIC_FEATURES, *categorical)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _selected_route_key(row: Mapping[str, object]) -> str:
    payload = [
        str(row["selector_name"]),
        str(row["experiment_seed"]),
        str(row["heldout_center"]),
        str(row["support_size"]),
        str(row["support_seed"]),
        str(row["candidate_expert"]),
        str(row["generator_family"]),
        str(row["generation_mode"]),
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _center_columns() -> tuple[str, ...]:
    return (
        "selector_name",
        "heldout_center",
        "n_rows",
        "mean_selected_bacc",
        "c41_hetero_mean_selected_bacc",
        "delta_vs_c41_hetero_mean",
        "mean_oracle_bacc",
        "mean_oracle_gap_bacc",
        "top1_hit_rate",
        "mean_spearman",
        "regret_p75",
    )


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return sum(clean) / float(len(clean)) if clean else math.nan


def _median(values: Iterable[float]) -> float:
    clean = sorted(float(value) for value in values if not math.isnan(float(value)))
    return float(statistics.median(clean)) if clean else math.nan


def _std(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.pstdev(clean) if len(clean) > 1 else 0.0


def _iqr(values: Iterable[float]) -> float:
    clean = sorted(float(value) for value in values if not math.isnan(float(value)))
    if not clean:
        return math.nan
    return _quantile(clean, 0.75) - _quantile(clean, 0.25)


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
