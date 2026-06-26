from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import decentralized_component_union_prior as cu
from artifact_table_specs import TableOutput, write_table_outputs
from decision_markdown import write_decision_markdown
from labeled_support_random_vs_dense_policy_calibration import (
    LabeledSupportPolicyCalibrationConfig,
    POLICY_DENSE,
    POLICY_RANDOM_BAG,
    POLICY_SHRINK050,
    PROTOCOL_WORDING,
    ROW_RANDOM_SINGLE_MASS,
)
from preservation_repair import _format_float
from reporting import write_protocol_finalization


def _write_artifacts(
    root: Path,
    cfg: LabeledSupportPolicyCalibrationConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    policy_score_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    switch_event_rows: Sequence[Mapping[str, object]],
    probability_manifest_rows: Sequence[Mapping[str, object]],
    random_bag_manifest_rows: Sequence[Mapping[str, object]],
    utility_alignment_rows: Sequence[Mapping[str, object]],
    quantization_rows: Sequence[Mapping[str, object]],
    common_eval_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    target_oracle_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    tail_rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    leakage: Mapping[str, object],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_table_outputs(
        root,
        (
            TableOutput("tables/labeled_support_policy_downstream_matrix.csv", matrix_rows),
            TableOutput("tables/labeled_support_policy_summary.csv", [dict(summary)]),
            TableOutput("tables/labeled_support_tail_metric_summary.csv", tail_rows),
            TableOutput("tables/labeled_support_split_manifest.csv", split_rows),
            TableOutput("tables/labeled_support_policy_score_matrix.csv", policy_score_rows),
            TableOutput("tables/labeled_support_policy_selection_manifest.csv", selection_rows),
            TableOutput("tables/policy_switch_event_table.csv", switch_event_rows),
            TableOutput("tables/candidate_policy_probability_manifest.csv", probability_manifest_rows),
            TableOutput("tables/random_bag_manifest.csv", random_bag_manifest_rows),
            TableOutput("tables/support_to_target_utility_alignment.csv", utility_alignment_rows),
            TableOutput("tables/support_size_quantization_audit.csv", quantization_rows),
            TableOutput("tables/support_size_common_eval_audit.csv", common_eval_rows),
            TableOutput("tables/negative_control_summary.csv", negative_rows),
            TableOutput("tables/oracle_policy_gap_summary.csv", oracle_rows),
            TableOutput("tables/labeled_support_target_oracle_audit.csv", target_oracle_rows),
            TableOutput("tables/eligibility_audit.csv", eligibility_rows),
            TableOutput("tables/runtime_memory_audit.csv", runtime_rows),
            TableOutput("tables/component_manifest.csv", component_manifest_rows),
            TableOutput("tables/component_coverage_audit.csv", component_coverage_rows),
            TableOutput("tables/paired_generation_audit.csv", paired_generation_rows),
        ),
    )
    write_protocol_finalization(
        root,
        leakage_report=leakage,
        protocol_manifest=_protocol_manifest(cfg, target_expert_excluded, protocol_violations),
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, summary, leakage_status=str(leakage.get("status", "")))


def _labeled_support_leakage_report(
    *,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
    support_eval_disjoint: bool,
) -> dict[str, object]:
    violations = list(str(v) for v in protocol_violations)
    if not target_expert_excluded:
        violations.append("target_expert_not_excluded")
    if not support_eval_disjoint:
        violations.append("support_eval_overlap")
    return {
        "schema_version": "cvae_rebuild_labeled_support_tier2_leakage_report_v1",
        "status": "PASS" if not violations else "FAIL",
        "violations": violations,
        "protocol_tier": "tier2_labeled_target_support_calibration",
        "target_support_labels_for_policy_selection": True,
        "target_eval_labels_for_scoring_only": True,
        "support_eval_disjoint": bool(support_eval_disjoint),
        "target_expert_excluded": bool(target_expert_excluded),
        "support_labels_do_not_train_classifiers": True,
        "support_labels_do_not_modify_generation": True,
        "support_labels_do_not_tune_hyperparameters": True,
        "oracle_rows_diagnostic_only": True,
    }


def _protocol_manifest(cfg: LabeledSupportPolicyCalibrationConfig, target_expert_excluded: bool, protocol_violations: Sequence[str]) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_labeled_support_random_vs_dense_policy_calibration_protocol_v1",
        "experiment_name": cfg.name,
        "protocol_tier": "tier2_labeled_target_support_calibration",
        "experiment_type": "few_shot_target_local_utility_calibration",
        "primary_method": cfg.primary_method,
        "primary_variant": cfg.primary_variant,
        "target_support_labels_for_policy_selection": True,
        "target_eval_labels_for_scoring_only": True,
        "support_eval_disjoint": True,
        "class_balanced_support": True,
        "support_labels_do_not_train_classifiers": True,
        "support_labels_do_not_modify_generation": True,
        "support_labels_do_not_tune_hyperparameters": True,
        "support_labels_do_not_choose_support_size": True,
        "support_labels_do_not_change_candidate_policies": True,
        "target_expert_excluded": target_expert_excluded,
        "adoption_eligible_policies": [POLICY_RANDOM_BAG, POLICY_DENSE],
        "diagnostic_policies": [POLICY_SHRINK050, ROW_RANDOM_SINGLE_MASS, cu.ROW_SOURCE_UNION_K16_REFERENCE, cu.ROW_REAL_FEATURE_DENSE_REFERENCE],
        "primary_labeled_support_size": cfg.primary_labeled_support_size,
        "diagnostic_labeled_support_sizes": list(cfg.diagnostic_labeled_support_sizes),
        "primary_switch_quantum": cfg.primary_switch_quantum,
        "support_quantum_by_size": {str(key): value for key, value in cfg.support_quantum_by_size.items()},
        "oracle_rows_diagnostic_only": True,
        "support8_support32_common_eval_diagnostic_only": True,
        "skip_nearest_neighbor_audit": cfg.skip_nearest_neighbor_audit,
        "protocol_violations": list(protocol_violations),
        "protocol_wording": PROTOCOL_WORDING,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    lines = [
        "# Labeled Support16 Random-vs-Dense Policy Calibration v1",
        "",
        "## Primary Verdict",
        "",
        f"- Primary method: `{decision.get('primary_method', '')}`",
        f"- Primary verdict: `{decision.get('primary_verdict', '')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Leakage status: `{leakage_status}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Bottom20 mean BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Worst seed-center BACC: {_format_float(decision.get('worst_seed_center_bacc'))}",
        f"- Delta vs random mass-bag: {_format_float(decision.get('delta_vs_random_mass_bag'))}",
        f"- Bottom20 delta vs random mass-bag: {_format_float(decision.get('bottom20_delta_vs_random_mass_bag'))}",
        f"- Within-cell random-vs-dense policy AUC: {_format_float(decision.get('within_cell_pairwise_policy_auc'))}",
        f"- Aggregate Spearman support-vs-target BACC: {_format_float(decision.get('aggregate_spearman_support_bacc_vs_target_bacc'))}",
        f"- Dense switch precision vs target oracle: {_format_float(decision.get('dense_switch_precision_against_target_oracle'))}",
        f"- Dense switch recall vs target oracle: {_format_float(decision.get('dense_switch_recall_against_target_oracle'))}",
        f"- Oracle gap reduction vs random: {_format_float(decision.get('oracle_gap_reduction_vs_random'))}",
        f"- selected_random_mass_bag_rate: {_format_float(decision.get('selected_random_mass_bag_rate'))}",
        f"- selected_dense_reliability_rate: {_format_float(decision.get('selected_dense_reliability_rate'))}",
        "",
        "## Protocol Boundary",
        "",
        PROTOCOL_WORDING,
        "",
        "Support8/support32 and common-eval rows are diagnostic-only and cannot rescue a failed support16 primary.",
        "Oracle rows are diagnostic-only and cannot affect policy selection, thresholds, candidate policies, or adoption.",
        "",
        "## Supported Claim If Successful",
        "",
        "A small class-balanced labeled target-support set can detect when a high-mean random mass-bag composition is unsafe and switch to a dense reliability policy, improving weak-regime robustness without retraining source experts or using target-evaluation labels.",
        "",
    ]
    write_decision_markdown(root, lines)


def _resolved_config(cfg: LabeledSupportPolicyCalibrationConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "support_seeds": list(cfg.support_seeds),
        "primary_labeled_support_size": cfg.primary_labeled_support_size,
        "diagnostic_labeled_support_sizes": list(cfg.diagnostic_labeled_support_sizes),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "primary_switch_quantum": cfg.primary_switch_quantum,
    }
