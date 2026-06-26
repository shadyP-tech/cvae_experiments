from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import decentralized_adaptive_gmm_prior as d1a
import decentralized_reliability_weighted_gmm_prior as d12
from artifact_table_specs import TableOutput, write_table_outputs
from decision_markdown import write_decision_markdown_text
from decentralized_support_nelbo_reliability_gmm_prior import (
    DecentralizedSupportNelboReliabilityConfig,
    PRIMARY_SUPPORT_RELIABILITY_METHOD,
    PROTOCOL_WORDING,
    ROW_SHUFFLED_LABEL_CONTROL,
    ROW_SHUFFLED_SUMMARY_CONTROL,
    ROW_SHUFFLED_SUPPORT_CONTROL,
)
from preservation_repair import _float, _format_float
from protocol import build_leakage_report
from reporting import write_protocol_finalization


def _write_artifacts(
    root: Path,
    cfg: DecentralizedSupportNelboReliabilityConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    summary_manifest_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    support_score_rows: Sequence[Mapping[str, object]],
    support_weight_rows: Sequence[Mapping[str, object]],
    combined_weight_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    alignment_rows: Sequence[Mapping[str, object]],
    centerwise_rows: Sequence[Mapping[str, object]],
    late_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    coverage_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    matrix_columns = _matrix_columns()
    write_table_outputs(
        root,
        (
            TableOutput("tables/decentralized_support_nelbo_reliability_downstream_matrix.csv", matrix_rows, columns=matrix_columns),
            TableOutput("tables/decentralized_support_nelbo_reliability_gap_summary.csv", gap_rows, columns=matrix_columns),
            TableOutput("tables/decentralized_support_nelbo_reliability_summary.csv", [dict(decision)]),
            TableOutput("tables/support_eval_split_manifest.csv", split_rows),
            TableOutput("tables/support_nelbo_scores.csv", support_score_rows),
            TableOutput("tables/support_nelbo_weight_manifest.csv", support_weight_rows),
            TableOutput("tables/combined_weight_manifest.csv", combined_weight_rows),
            TableOutput("tables/support_nelbo_alignment_matrix.csv", alignment_rows),
            TableOutput("tables/source_reliability_manifest.csv", reliability_rows),
            TableOutput("tables/centerwise_delta_summary.csv", centerwise_rows),
            TableOutput("tables/late_aggregation_matrix.csv", late_rows, columns=matrix_columns),
            TableOutput("tables/real_feature_reference_matrix.csv", real_feature_rows, columns=matrix_columns),
            TableOutput("tables/generated_component_coverage_audit.csv", coverage_rows),
            TableOutput("tables/weak_source_audit.csv", weak_rows),
            TableOutput("tables/nearest_neighbor_memorization_audit.csv", nn_rows),
            TableOutput("tables/negative_control_summary.csv", [_negative_control_summary(decision)]),
            TableOutput("tables/exported_source_summary_manifest.csv", summary_manifest_rows, columns=d1a._summary_manifest_columns()),
            TableOutput("tables/source_summary_diagnostics.csv", diagnostic_rows, columns=d1a._diagnostic_columns()),
            TableOutput("manifests/decentralized_support_nelbo_reliability_prior_model_manifest.csv", model_manifest_rows),
        ),
    )
    leakage = build_leakage_report(
        target_support_labels_for_selection=False,
        target_eval_labels_for_scoring_only=True,
        target_expert_excluded=target_expert_excluded,
        oracle_rows_diagnostic_only=True,
        extra_violations=protocol_violations,
    )
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest={
            "schema_version": "cvae_rebuild_decentralized_support_nelbo_reliability_gmm_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "target_conditioned_support_nelbo_x_reliability_decentralized_composition",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "support_eval_disjoint": True,
            "target_expert_excluded": target_expert_excluded,
            "exported_source_summaries_are_target_agnostic": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "adaptive_k_selection_uses_source_local_fit_statistics_only": True,
            "source_reliability_uses_source_local_eval_only": True,
            "support_nelbo_uses_unlabeled_target_support_only": True,
            "decision_baselines_recomputed_on_support_excluded_eval_subset": True,
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "target-conditioned support-NELBO compatibility-weighted composition; no metadata-routing claim, "
                "no formal privacy claim, no centralized source-union deployability claim, and no exact utility-prediction claim"
            ),
        },
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)


def _matrix_columns() -> tuple[str, ...]:
    return d12._matrix_columns() + (
        "support_size",
        "support_seed",
        "support_eval_split_id",
        "support_labels_used",
        "n_target_eval_after_support",
        "support_eval_min_class_count",
        "support_eval_ineligible_reason",
        "support_weight_json",
        "support_score_json",
        "raw_support_nelbo_json",
        "calibrated_support_nelbo_json",
        "combined_weight_json",
        "combined_budget_per_class_json",
        "support_nelbo_tau",
        "mean_l1_distance_from_reliability_only",
        "delta_vs_d1_2_reliability_support_eval_reference",
        "delta_vs_equal_support_eval_reference",
    )


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": PRIMARY_SUPPORT_RELIABILITY_METHOD,
        "control_methods": f"{ROW_SHUFFLED_SUPPORT_CONTROL}|{ROW_SHUFFLED_SUMMARY_CONTROL}|{ROW_SHUFFLED_LABEL_CONTROL}",
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "shuffled_support_control_center_equal_mean_bacc": decision.get("shuffled_support_control_center_equal_mean_bacc", math.nan),
        "shuffled_support_control_gap": decision.get("shuffled_support_control_gap", math.nan),
        "shuffled_summary_control_center_equal_mean_bacc": decision.get("shuffled_summary_control_center_equal_mean_bacc", math.nan),
        "shuffled_label_control_center_equal_mean_bacc": decision.get("shuffled_label_control_center_equal_mean_bacc", math.nan),
        "control_center_equal_mean_bacc": decision.get("negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "control_competitive": _float(decision.get("negative_control_gap")) < 0.03,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# D1.3: Support-NELBO x Reliability Target-Conditioned Decentralized Composition",
            "",
            "## Summary",
            "",
            f"- Primary method: `{PRIMARY_SUPPORT_RELIABILITY_METHOD}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'D1_3_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs D1.2 reliability support-eval reference: {_format_float(decision.get('delta_vs_d1_2_reliability_support_eval_reference'))}",
            f"- Delta vs equal support-eval reference: {_format_float(decision.get('delta_vs_equal_support_eval_reference'))}",
            f"- Centers beating D1.2: {decision.get('centers_beating_d1_2_reliability', '')}",
            f"- Seeds beating D1.2: {decision.get('seeds_beating_d1_2_reliability', '')}",
            f"- Support-NELBO vs downstream Spearman: {_format_float(decision.get('spearman_support_nelbo_vs_downstream_utility'))}",
            f"- Top-1 downstream oracle hit: {_format_float(decision.get('top1_downstream_oracle_hit'))}",
            f"- Top-2 downstream oracle containment: {_format_float(decision.get('top2_downstream_oracle_containment'))}",
            f"- Downstream oracle gap: {_format_float(decision.get('downstream_oracle_gap'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Strongest negative control: `{decision.get('strongest_negative_control_method', '')}` at {_format_float(decision.get('strongest_negative_control_center_equal_mean_bacc'))}",
            f"- Shuffled-support control gap: {_format_float(decision.get('shuffled_support_control_gap'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This is target-conditioned support-NELBO compatibility-weighted composition.",
            "It is not metadata routing and it is not a formal privacy result.",
            "All decision baselines are recomputed on the same support-excluded target eval subset.",
            "",
            "## Supported Claim If PASS",
            "",
            "Target-conditioned support-NELBO improves utility ranking and weighting of decentralized generative experts beyond source-local reliability alone.",
            "",
        ]
    )
    write_decision_markdown_text(root, text)


def _resolved_config(cfg: DecentralizedSupportNelboReliabilityConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "repair_artifact_root": str(cfg.repair_artifact_root),
        "feature_cache_root": str(cfg.feature_cache_root),
        "backbone": cfg.backbone,
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "replicate_seeds": list(cfg.replicate_seeds),
        "support_seeds": list(cfg.support_seeds),
        "support_size": cfg.support_size,
        "support_size_diagnostics": list(cfg.support_size_diagnostics),
        "align_support_and_generation_seed": cfg.align_support_and_generation_seed,
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "min_per_source_per_class": cfg.min_per_source_per_class,
        "primary_variant": cfg.primary_variant,
        "primary_method": cfg.primary_method,
        "candidate_components_per_source_class": list(cfg.candidate_components_per_source_class),
        "min_samples_per_component": cfg.min_samples_per_component,
        "source_weighting": cfg.source_weighting,
        "gmm_covariance_type": cfg.gmm_covariance_type,
        "gmm_reg_covar": cfg.gmm_reg_covar,
        "gmm_n_init": cfg.gmm_n_init,
        "gmm_max_iter": cfg.gmm_max_iter,
        "min_component_weight": cfg.min_component_weight,
        "variance_floor": cfg.variance_floor,
        "primary_pooling": cfg.primary_pooling,
        "reliability_floor_score": cfg.reliability_floor_score,
        "support_nelbo_tau": cfg.support_nelbo_tau,
        "tau_diagnostics": list(cfg.tau_diagnostics),
        "support_alpha": cfg.support_alpha,
        "reliability_alpha": cfg.reliability_alpha,
        "classifier": {
            "type": cfg.classifier_type,
            "solver": cfg.classifier_solver,
            "C": cfg.classifier_c,
            "max_iter": cfg.classifier_max_iter,
            "class_weight": cfg.classifier_class_weight,
            "classifier_seed": cfg.classifier_seed,
        },
    }
