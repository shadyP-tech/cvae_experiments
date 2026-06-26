from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from artifact_table_specs import TableOutput, write_table_outputs
from decision_markdown import write_decision_markdown
from preservation_repair import _format_float
from reporting import write_protocol_finalization
from target_support_regime_risk_gated_component_union import (
    COMPACT_FEATURES,
    PROTOCOL_WORDING,
    ROW_THRESHOLD_SENSITIVITY_PREFIX,
    TargetSupportRiskGateConfig,
)


def _write_artifacts(
    root: Path,
    cfg: TargetSupportRiskGateConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    split_rows: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, object]],
    source_inner_rows: Sequence[Mapping[str, object]],
    lopo_rows: Sequence[Mapping[str, object]],
    feature_ablation_rows: Sequence[Mapping[str, object]],
    model_rows: Sequence[Mapping[str, object]],
    selection_rows: Sequence[Mapping[str, object]],
    probability_manifest_rows: Sequence[Mapping[str, object]],
    random_bag_manifest_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
    oracle_rows: Sequence[Mapping[str, object]],
    target_oracle_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
    tail_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_table_outputs(
        root,
        (
            TableOutput("tables/risk_gated_downstream_matrix.csv", matrix_rows),
            TableOutput("tables/risk_gated_summary.csv", [dict(decision)]),
            TableOutput("tables/risk_gated_tail_metric_summary.csv", tail_rows),
            TableOutput("tables/support_eval_split_manifest.csv", split_rows),
            TableOutput("tables/support_regime_feature_matrix.csv", feature_rows),
            TableOutput("tables/source_inner_gate_training_matrix.csv", source_inner_rows),
            TableOutput("tables/source_inner_lopo_gate_audit.csv", lopo_rows),
            TableOutput("tables/risk_gate_feature_ablation_summary.csv", feature_ablation_rows),
            TableOutput("tables/risk_gate_threshold_sensitivity_audit.csv", _threshold_sensitivity_rows(matrix_rows)),
            TableOutput("tables/risk_gate_selection_manifest.csv", selection_rows),
            TableOutput("tables/candidate_policy_probability_manifest.csv", probability_manifest_rows),
            TableOutput("tables/random_bag_manifest.csv", random_bag_manifest_rows),
            TableOutput("tables/negative_control_summary.csv", negative_rows),
            TableOutput("tables/oracle_policy_gap_summary.csv", oracle_rows),
            TableOutput("tables/risk_gate_target_oracle_audit.csv", target_oracle_rows),
            TableOutput("tables/eligibility_audit.csv", eligibility_rows),
            TableOutput("tables/runtime_memory_audit.csv", runtime_rows),
            TableOutput("tables/component_manifest.csv", component_manifest_rows),
            TableOutput("tables/component_coverage_audit.csv", component_coverage_rows),
            TableOutput("tables/paired_generation_audit.csv", paired_generation_rows),
            TableOutput("manifests/risk_gate_model_manifest.csv", model_rows),
        ),
    )
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest=_protocol_manifest(cfg, target_expert_excluded, protocol_violations),
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision, leakage_status=leakage.status)


def _threshold_sensitivity_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "threshold_low": row.get("threshold_low", ""),
            "threshold_high": row.get("threshold_high", ""),
            "experiment_seed": row.get("experiment_seed", ""),
            "heldout_center": row.get("heldout_center", ""),
            "support_seed": row.get("support_seed", ""),
            "selected_policy": row.get("selected_policy", ""),
            "bacc": row.get("bacc", math.nan),
            "macro_f1": row.get("macro_f1", math.nan),
            "diagnostic_only": True,
        }
        for row in rows
        if str(row.get("prior_method", "")).startswith(ROW_THRESHOLD_SENSITIVITY_PREFIX)
    ]


def _protocol_manifest(cfg: TargetSupportRiskGateConfig, target_expert_excluded: bool, protocol_violations: Sequence[str]) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_target_support_regime_risk_gate_protocol_v1",
        "experiment_name": cfg.name,
        "experiment_type": "target_support_regime_risk_policy_selection",
        "primary_method": cfg.primary_method,
        "primary_variant": cfg.primary_variant,
        "support_size": cfg.support_size,
        "support_size_diagnostics": list(cfg.support_size_diagnostics),
        "nested_support_max_size": cfg.nested_support_max_size,
        "support_labels_used": False,
        "target_eval_labels_for_scoring_only": True,
        "target_expert_excluded": target_expert_excluded,
        "gate_training_pooling": "across_source_inner_support_seeds",
        "real_target_support_seed_used_for_gate_training": False,
        "n_source_inner_training_episodes_expected": 12,
        "center_id_used_as_feature": False,
        "compact_feature_set": list(COMPACT_FEATURES),
        "risk_low_threshold": cfg.risk_low_threshold,
        "risk_high_threshold": cfg.risk_high_threshold,
        "threshold_sensitivity_diagnostic_only": True,
        "oracle_rows_diagnostic_only": True,
        "skip_nearest_neighbor_audit": cfg.skip_nearest_neighbor_audit,
        "protocol_violations": list(protocol_violations),
        "protocol_wording": PROTOCOL_WORDING,
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    lines = [
        "# Target-Support Regime-Risk Gated Component Policy v1",
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
        f"- LOPO gate verdict: `{decision.get('lopo_gate_verdict', '')}`",
        f"- LOPO gate AUC: {_format_float(decision.get('lopo_gate_auc'))}",
        f"- LOPO control AUC: {_format_float(decision.get('lopo_control_auc'))}",
        f"- Target-oracle risk recall: {_format_float(decision.get('target_oracle_risk_recall'))}",
        f"- Centers represented: {_format_float(decision.get('n_centers_represented'))}",
        f"- Control tail clear: `{decision.get('control_tail_clear', '')}`",
        f"- selected_random_mass_bag_rate: {_format_float(decision.get('selected_random_mass_bag_rate'))}",
        f"- selected_shrink050_rate: {_format_float(decision.get('selected_shrink050_rate'))}",
        f"- selected_dense_reliability_rate: {_format_float(decision.get('selected_dense_reliability_rate'))}",
        f"- fallback_rate: {_format_float(decision.get('fallback_rate'))}",
        f"- untrained_gate_rate: {_format_float(decision.get('untrained_gate_rate'))}",
        "",
        "## Protocol Boundary",
        "",
        PROTOCOL_WORDING,
        "",
        "The primary verdict above is printed before threshold-sensitivity diagnostics. Threshold-sensitivity rows are audit-only and cannot rescue a failed primary.",
        "A failed source-inner LOPO audit blocks thesis-facing adoption even if target tail metrics improve.",
        "",
        "## Supported Claim If Successful",
        "",
        "Unlabeled target-support statistics can identify high-risk target regimes and select a safer fixed composition policy, improving weak-regime robustness without target-evaluation labels.",
        "",
    ]
    write_decision_markdown(root, lines)


def _resolved_config(cfg: TargetSupportRiskGateConfig) -> dict[str, object]:
    return {
        "name": cfg.name,
        "artifact_root": str(cfg.artifact_root),
        "experiment_seeds": list(cfg.experiment_seeds),
        "heldout_centers": list(cfg.heldout_centers),
        "support_seeds": list(cfg.support_seeds),
        "support_size": cfg.support_size,
        "support_size_diagnostics": list(cfg.support_size_diagnostics),
        "synthetic_per_class_total": cfg.synthetic_per_class_total,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "risk_thresholds": [cfg.risk_low_threshold, cfg.risk_high_threshold],
        "compact_features": list(COMPACT_FEATURES),
    }
