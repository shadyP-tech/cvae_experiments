from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from domain_regime import normalize_domain_regime
from preservation_repair import _float, _format_float
from reporting import write_csv_rows, write_json, write_protocol_finalization
from splits import candidate_experts
from component_union_tailrisk_anchored_mass_bagged import (
    CENTER3_FAILURE_AUDIT_CELLS,
    CENTER3_FAILURE_PRIMARY_CELL,
    FIXED_BETA050_POSITIVE_UNION_PRIMARY_POOLING,
    HARM_GATED_POSITIVE_UNION_PRIMARY_POOLING,
    HARM_GATED_PRIMARY_SELECTABLE_RULES,
    POSITIVE_UNION_PRIMARY_POOLING,
    POSITIVE_UNION_RULE_BETA050,
    PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD,
    PRIMARY_HARM_GATED_POSITIVE_UNION_METHOD,
    PRIMARY_MULTIPANEL_TAILRISK_METHOD,
    PRIMARY_POSITIVE_UNION_METHOD,
    PRIMARY_TAILRISK_METHOD,
    _center3_failure_audit_role,
    _is_center3_failure_audit_cell,
    _midogpp_contract_info,
    _negative_control_summary,
    _oracle_gap_rows,
    _panel_ece_source_inner_rows,
    _panel_summary_rows,
    _random_mass_bag_summary,
    _resolved_config,
    _resolved_fixed_beta050_config,
    _resolved_harm_gated_positive_union_config,
    _resolved_multipanel_config,
    _resolved_positive_union_config,
    _safe_int,
    _tail_metric_summary_rows,
)


def _write_multipanel_artifacts(
    root: Path,
    cfg: MultipanelTailRiskConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    seed_diagnostic_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    panel_disagreement_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    confidence_rows: Sequence[Mapping[str, object]],
    failure_rows: Sequence[Mapping[str, object]],
    center3_failure_cell_rows: Sequence[Mapping[str, object]],
    center3_failure_sample_rows: Sequence[Mapping[str, object]],
    center3_failure_pooling_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "multipanel_tailrisk_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_seed_diagnostic_matrix.csv", seed_diagnostic_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "multipanel_tailrisk_failure_decomposition.csv", failure_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_panel_disagreement.csv", panel_disagreement_rows)
    write_csv_rows(root / "tables" / "panel_ece_source_inner.csv", _panel_ece_source_inner_rows(calibration_rows))
    write_csv_rows(root / "tables" / "panel_confidence_summary.csv", confidence_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_probability_invariants.csv", invariant_rows)
    write_csv_rows(root / "tables" / "multipanel_tailrisk_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "manifests" / "multipanel_tailrisk_model_manifest.csv", model_manifest_rows)
    _write_center3_failure_audit_artifacts(
        root,
        cell_rows=center3_failure_cell_rows,
        sample_rows=center3_failure_sample_rows,
        pooling_rows=center3_failure_pooling_rows,
        source_weight_rows=source_weight_rows,
        component_coverage_rows=component_coverage_rows,
    )
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest=_multipanel_protocol_manifest_payload(
            cfg,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        ),
        resolved_config=_resolved_multipanel_config(cfg),
    )
    _write_multipanel_decision_summary(root, decision)


def _multipanel_protocol_manifest_payload(
    cfg: MultipanelTailRiskConfig,
    *,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_tailrisk_multipanel_component_union_protocol_v1",
        "experiment_name": cfg.name,
        "primary_method": cfg.primary_method,
        "experiment_type": "source_only_tailrisk_multipanel_mass_bag_stabilization",
        "target_expert_excluded": bool(target_expert_excluded),
        "target_support_used": False,
        "target_support_labels_for_selection": False,
        "target_eval_labels_for_scoring_only": True,
        "selection_used_target_labels": False,
        "target_calibration_metrics_audit_only": True,
        "center3_failure_audit_diagnostic_only": True,
        "center3_failure_audit_target_labels_post_prediction_only": True,
        "center3_failure_audit_cells": [f"{seed}xcenter{center}" for seed, center in CENTER3_FAILURE_AUDIT_CELLS],
        "source_inner_calibration_primary": True,
        "target_conditioned_point_compatibility_estimate": False,
        "fixed_all_source_inclusion": True,
        "panel_seeds_are_evaluation_replicates": False,
        "decision_cell": "experiment_seed_x_heldout_center",
        "primary_pooling_rule": "blend_per_seed_then_equal_probability_pool",
        "blend_alpha_locked": cfg.blend_alpha,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
        "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
        "prior_tailrisk_comparator": (
            ""
            if cfg.prior_tailrisk_artifact_root is None
            else str(cfg.prior_tailrisk_artifact_root / "tables" / "tailrisk_downstream_matrix.csv")
        ),
        "claim_boundary": (
            "stabilized source-only dense stochastic generative composition; "
            "not compatibility routing, target adaptation, or target-label-driven method choice"
        ),
        "protocol_violations": list(protocol_violations),
    }


def _write_positive_union_artifacts(
    root: Path,
    cfg: SourceInnerPositiveUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    source_inner_selection_rows: Sequence[Mapping[str, object]],
    candidate_rule_rows: Sequence[Mapping[str, object]],
    class_conditional_rows: Sequence[Mapping[str, object]],
    effective_threshold_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    harm_rows: Sequence[Mapping[str, object]],
    per_source_harm_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    source_pool_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "positive_union_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "positive_union_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "positive_union_source_inner_selection.csv", source_inner_selection_rows)
    write_csv_rows(root / "tables" / "positive_union_candidate_rule_matrix.csv", candidate_rule_rows)
    write_csv_rows(root / "tables" / "positive_union_class_conditional_audit.csv", class_conditional_rows)
    write_csv_rows(root / "tables" / "positive_union_effective_threshold_audit.csv", effective_threshold_rows)
    write_csv_rows(root / "tables" / "positive_union_paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "positive_union_harm_audit.csv", harm_rows)
    write_csv_rows(root / "tables" / "positive_union_source_inner_per_source_harm_audit.csv", per_source_harm_rows)
    write_csv_rows(root / "tables" / "positive_union_probability_invariants.csv", invariant_rows)
    write_csv_rows(root / "tables" / "positive_union_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "manifests" / "positive_union_model_manifest.csv", model_manifest_rows)
    write_csv_rows(root / "manifests" / "positive_union_source_pool_manifest.csv", source_pool_rows)
    write_json(
        root / "manifests" / "positive_union_rule_selection_manifest.json",
        {
            "schema_version": "cvae_rebuild_positive_union_rule_selection_manifest_v1",
            "domain_regime": normalize_domain_regime(cfg.domain_regime),
            "selection_rows": _positive_union_rule_selection_manifest_rows(cfg, source_inner_selection_rows, source_pool_rows),
        },
    )
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest=_positive_union_protocol_manifest_payload(
            cfg,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        ),
        resolved_config=_resolved_positive_union_config(cfg),
    )
    _write_positive_union_decision_summary(root, decision)


def _positive_union_protocol_manifest_payload(
    cfg: SourceInnerPositiveUnionConfig,
    *,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> dict[str, object]:
    contract_info = _midogpp_contract_info(cfg)
    eligible_domain_ids = list(contract_info.eligible_domain_ids) if contract_info is not None else list(cfg.heldout_centers)
    return {
        "schema_version": "cvae_rebuild_source_inner_positive_union_protocol_v1",
        "experiment_name": cfg.name,
        "primary_method": cfg.primary_method,
        "experiment_type": "source_only_class_conditional_positive_union_tailrisk_repair",
        "domain_regime": normalize_domain_regime(cfg.domain_regime),
        "eligible_domain_ids": eligible_domain_ids,
        "expected_source_count": int(len(eligible_domain_ids) - 1),
        "domain_4_excluded": "4" not in set(eligible_domain_ids),
        "all_eligible_heldouts_complete": True,
        "dataset_contract_fingerprints": contract_info.fingerprints if contract_info is not None else {},
        "target_expert_excluded": bool(target_expert_excluded),
        "target_support_used": False,
        "target_support_labels_for_selection": False,
        "target_eval_labels_for_scoring_only": True,
        "selection_used_target_labels": False,
        "target_calibration_metrics_audit_only": True,
        "target_eval_candidate_rule_metrics_audit_only": True,
        "target_conditioned_point_compatibility_estimate": False,
        "compatibility_router": False,
        "fixed_all_source_inclusion": True,
        "panel_seeds_are_evaluation_replicates": False,
        "decision_cell": "experiment_seed_x_heldout_center",
        "source_inner_selection_primary": True,
        "source_inner_validation_scope": "pooled_non_target_source_validation_rows",
        "source_inner_per_source_harm_audit": True,
        "positive_label": cfg.positive_label,
        "prediction_threshold": cfg.prediction_threshold,
        "minimum_source_inner_positive_count": cfg.min_source_inner_positive_count,
        "positive_union_eps": cfg.positive_union_eps,
        "candidate_pooling_rules": list(cfg.candidate_pooling_rules),
        "primary_pooling_rule": POSITIVE_UNION_PRIMARY_POOLING,
        "blend_alpha_locked": cfg.blend_alpha,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
        "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
        "prior_tailrisk_comparator": (
            ""
            if cfg.prior_tailrisk_artifact_root is None
            else str(cfg.prior_tailrisk_artifact_root / "tables" / "tailrisk_downstream_matrix.csv")
        ),
        "claim_boundary": (
            "source-inner selected class-conditional aggregation repair after fixed dense source-only "
            "CVAE seed-blend aggregation; not compatibility routing, target adaptation, or target-label tuning"
        ),
        "protocol_violations": list(protocol_violations),
    }


def _positive_union_rule_selection_manifest_rows(
    cfg: SourceInnerPositiveUnionConfig,
    selection_rows: Sequence[Mapping[str, object]],
    source_pool_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    pool_by_cell = {
        (str(row.get("experiment_seed")), str(row.get("heldout_domain_id"))): row
        for row in source_pool_rows
    }
    out = []
    for row in selection_rows:
        seed = str(row.get("experiment_seed", ""))
        heldout = str(row.get("heldout_center", ""))
        fallback_sources = list(candidate_experts(cfg.heldout_centers, heldout))
        pool = pool_by_cell.get((seed, heldout), {})
        out.append(
            {
                "experiment_seed": seed,
                "heldout_domain": heldout,
                "candidate_sources": pool.get("source_domain_ids", json.dumps(fallback_sources)),
                "expected_source_count": pool.get("expected_source_count", len(cfg.heldout_centers) - 1),
                "actual_source_count": pool.get("actual_source_count", len(fallback_sources)),
                "selected_rule": row.get("selected_rule", ""),
                "selected_beta": row.get("selected_beta", ""),
                "source_inner_positive_count": row.get("source_inner_positive_count", ""),
                "source_inner_negative_count": row.get("source_inner_negative_count", ""),
                "selection_signal": "source_inner_only",
                "target_labels_used_for_selection": False,
                "target_support_used": False,
            }
        )
    return out


def _write_fixed_beta050_positive_union_artifacts(
    root: Path,
    cfg: FixedBeta050PositiveUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    candidate_rule_rows: Sequence[Mapping[str, object]],
    class_conditional_rows: Sequence[Mapping[str, object]],
    effective_threshold_rows: Sequence[Mapping[str, object]],
    rare_positive_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    harm_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    retrospective_reference_rows: Sequence[Mapping[str, object]],
    source_inner_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "fixed_beta050_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "fixed_beta050_candidate_rule_matrix.csv", candidate_rule_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_class_conditional_audit.csv", class_conditional_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_effective_threshold_audit.csv", effective_threshold_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_rare_positive_opportunity_audit.csv", rare_positive_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_harm_audit.csv", harm_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_probability_invariants.csv", invariant_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_retrospective_reference.csv", retrospective_reference_rows)
    write_csv_rows(root / "tables" / "fixed_beta050_source_inner_diagnostics.csv", source_inner_rows)
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest=_fixed_beta050_protocol_manifest_payload(
            cfg,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        ),
        resolved_config=_resolved_fixed_beta050_config(cfg),
    )
    _write_fixed_beta050_decision_summary(root, decision)


def _fixed_beta050_protocol_manifest_payload(
    cfg: FixedBeta050PositiveUnionConfig,
    *,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_fixed_beta050_positive_union_confirmation_protocol_v1",
        "experiment_name": cfg.name,
        "primary_method": cfg.primary_method,
        "experiment_type": "source_only_fixed_beta050_positive_union_confirmation",
        "target_expert_excluded": bool(target_expert_excluded),
        "target_support_used": False,
        "target_support_labels_for_selection": False,
        "target_eval_labels_for_scoring_only": True,
        "selection_used_target_labels": False,
        "target_eval_candidate_rule_metrics_audit_only": True,
        "target_conditioned_point_compatibility_estimate": False,
        "compatibility_router": False,
        "fixed_all_source_inclusion": True,
        "panel_seeds_are_evaluation_replicates": False,
        "decision_cell": "experiment_seed_x_heldout_center",
        "beta_rule": "fixed_global_beta050",
        "fixed_pooling_rule": cfg.fixed_pooling_rule,
        "fixed_beta": cfg.fixed_beta,
        "beta_origin": "hypothesis_generated_from_prior_positive_union_diagnostic",
        "development_experiment_seeds": list(cfg.development_experiment_seeds),
        "primary_confirmation_experiment_seeds": list(cfg.confirmation_experiment_seeds),
        "no_posthoc_beta_selection": True,
        "old_cells_retrospective_reference_only": True,
        "source_inner_selection_primary": False,
        "source_inner_diagnostics_only": True,
        "positive_label": cfg.positive_label,
        "prediction_threshold": cfg.prediction_threshold,
        "positive_union_eps": cfg.positive_union_eps,
        "candidate_pooling_rules": list(cfg.candidate_pooling_rules),
        "primary_pooling_rule": FIXED_BETA050_POSITIVE_UNION_PRIMARY_POOLING,
        "rare_positive_definition": {
            "class1_count_lte": cfg.rare_positive_count_threshold,
            "positive_prevalence_lte": cfg.rare_positive_prevalence_threshold,
        },
        "blend_alpha_locked": cfg.blend_alpha,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
        "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
        "prior_tailrisk_comparator": "retrospective/contextual_only_for_development_seeds",
        "claim_boundary": (
            "fixed global beta050 positive-evidence pooling after dense source-only CVAE seed-blend "
            "aggregation; not source-inner selected, not compatibility routing, not target adaptation, "
            "and not target-threshold tuning"
        ),
        "protocol_violations": list(protocol_violations),
    }


def _write_harm_gated_positive_union_artifacts(
    root: Path,
    cfg: SourceInnerHarmGatedPositiveUnionConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    source_inner_selection_rows: Sequence[Mapping[str, object]],
    candidate_rule_rows: Sequence[Mapping[str, object]],
    class_conditional_rows: Sequence[Mapping[str, object]],
    effective_threshold_rows: Sequence[Mapping[str, object]],
    rare_positive_rows: Sequence[Mapping[str, object]],
    paired_delta_rows: Sequence[Mapping[str, object]],
    harm_rows: Sequence[Mapping[str, object]],
    source_inner_harm_gate_rows: Sequence[Mapping[str, object]],
    proxy_validity_rows: Sequence[Mapping[str, object]],
    selected_rule_distribution_rows: Sequence[Mapping[str, object]],
    replacement_seed_rows: Sequence[Mapping[str, object]],
    invariant_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    retrospective_reference_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "harm_gated_positive_union_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "harm_gated_positive_union_source_inner_selection.csv", source_inner_selection_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_candidate_rule_matrix.csv", candidate_rule_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_class_conditional_audit.csv", class_conditional_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_effective_threshold_audit.csv", effective_threshold_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_rare_positive_opportunity_audit.csv", rare_positive_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_paired_deltas.csv", paired_delta_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_harm_audit.csv", harm_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_source_inner_harm_gate_audit.csv", source_inner_harm_gate_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_proxy_validity_audit.csv", proxy_validity_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_selected_rule_distribution.csv", selected_rule_distribution_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_replacement_seed_audit.csv", replacement_seed_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_probability_invariants.csv", invariant_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "harm_gated_positive_union_retrospective_development_reference.csv", retrospective_reference_rows)
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest=_harm_gated_protocol_manifest_payload(
            cfg,
            replacement_seed_rows=replacement_seed_rows,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        ),
        resolved_config=_resolved_harm_gated_positive_union_config(cfg),
    )
    _write_harm_gated_positive_union_decision_summary(root, decision)


def _harm_gated_protocol_manifest_payload(
    cfg: SourceInnerHarmGatedPositiveUnionConfig,
    *,
    replacement_seed_rows: Sequence[Mapping[str, object]],
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> dict[str, object]:
    completed = []
    if replacement_seed_rows:
        try:
            completed = json.loads(str(replacement_seed_rows[-1].get("completed_primary_experiment_seeds", "[]")))
        except json.JSONDecodeError:
            completed = []
    return {
        "schema_version": "cvae_rebuild_source_inner_harm_gated_positive_union_protocol_v1",
        "experiment_name": cfg.name,
        "primary_method": cfg.primary_method,
        "experiment_type": "source_only_harm_gated_positive_union_confirmation",
        "target_expert_excluded": bool(target_expert_excluded),
        "development_experiment_seeds": list(cfg.development_experiment_seeds),
        "primary_requested_experiment_seeds": list(cfg.primary_requested_experiment_seeds),
        "reserve_experiment_seeds": list(cfg.reserve_experiment_seeds),
        "reserve_seed_policy": cfg.reserve_seed_policy,
        "cell_level_reserve_stitching_allowed": False,
        "primary_confirmation_experiment_seeds": completed if completed else "resolved_after_reserve_replacement",
        "selection_used_target_labels": False,
        "target_support_used": False,
        "target_eval_labels_for_scoring_only": True,
        "target_support_labels_for_selection": False,
        "target_conditioned_point_compatibility_estimate": False,
        "compatibility_router": False,
        "target_threshold_tuning": False,
        "target_label_calibration": False,
        "fixed_all_source_inclusion": True,
        "panel_seeds_are_evaluation_replicates": False,
        "decision_cell": "experiment_seed_x_heldout_center",
        "positive_class_label": 1,
        "rare_positive_class_label": 1,
        "class_order": [0, 1],
        "probability_column_positive": 1,
        "positive_label": cfg.positive_label,
        "prediction_threshold": cfg.prediction_threshold,
        "candidate_pooling_rules": list(cfg.candidate_pooling_rules),
        "primary_selectable_rules": list(cfg.primary_selectable_rules),
        "beta100_primary_selectable": False,
        "selector_thresholds_frozen_before_primary": True,
        "selector_threshold_source": cfg.selector_threshold_source,
        "selector_thresholds_may_be_changed_after_primary": False,
        "minimum_source_inner_positive_count": cfg.min_source_inner_positive_count,
        "beta050_min_source_inner_positive_count": cfg.beta050_min_source_inner_positive_count,
        "harm_gate_bacc_noninferiority_margin": cfg.harm_gate_bacc_noninferiority_margin,
        "beta025_class0_recall_margin": cfg.beta025_class0_recall_margin,
        "beta025_predicted_positive_rate_delta": cfg.beta025_predicted_positive_rate_delta,
        "beta050_class0_recall_margin": cfg.beta050_class0_recall_margin,
        "beta050_precision_margin": cfg.beta050_precision_margin,
        "beta050_predicted_positive_rate_delta": cfg.beta050_predicted_positive_rate_delta,
        "primary_pooling_rule": HARM_GATED_POSITIVE_UNION_PRIMARY_POOLING,
        "blend_alpha_locked": cfg.blend_alpha,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
        "nearest_neighbor_memorization_audit_skipped": bool(cfg.skip_nearest_neighbor_audit),
        "nearest_neighbor_memorization_audit_skip_reason": "virchow2_memory_safety" if cfg.skip_nearest_neighbor_audit else "",
        "panel_seed_groups": {panel: list(seeds) for panel, seeds in cfg.panel_seed_groups},
        "rare_positive_definition": {
            "class1_count_lte": cfg.rare_positive_count_threshold,
            "positive_prevalence_lte": cfg.rare_positive_prevalence_threshold,
        },
        "claim_boundary": (
            "source-inner harm-gated positive-evidence pooling after dense source-only CVAE seed-blend "
            "aggregation; not compatibility routing, not target adaptation, not target-threshold tuning, "
            "and not target-support calibration"
        ),
        "protocol_violations": list(protocol_violations),
    }


def _write_center3_failure_audit_artifacts(
    root: Path,
    *,
    cell_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
    pooling_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
) -> None:
    audit_root = root / "center3_failure_audit"
    filtered_source_weights = _center3_failure_filtered_existing_rows(source_weight_rows)
    filtered_component_coverage = _center3_failure_filtered_existing_rows(component_coverage_rows)
    write_csv_rows(audit_root / "center3_failure_cell_summary.csv", cell_rows)
    write_csv_rows(audit_root / "center3_failure_sample_audit.csv", sample_rows)
    write_csv_rows(audit_root / "center3_failure_pooling_path.csv", pooling_rows)
    write_csv_rows(audit_root / "center3_failure_source_weight_comparison.csv", filtered_source_weights)
    write_csv_rows(audit_root / "center3_failure_component_coverage_comparison.csv", filtered_component_coverage)
    _write_center3_failure_conclusion(
        audit_root / "center3_failure_conclusion.md",
        cell_rows=cell_rows,
        sample_rows=sample_rows,
        source_weight_rows=filtered_source_weights,
        component_coverage_rows=filtered_component_coverage,
    )


def _center3_failure_filtered_existing_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    for row in rows:
        experiment_seed = _safe_int(row.get("experiment_seed"), default=-1)
        heldout_center = str(row.get("heldout_center", ""))
        if not _is_center3_failure_audit_cell(experiment_seed, heldout_center):
            continue
        out.append(
            {
                "audit_only": True,
                "target_eval_labels_used_for_audit_only": True,
                "selection_used_target_labels": False,
                "audit_cell_role": _center3_failure_audit_role(experiment_seed, heldout_center),
                **dict(row),
            }
        )
    return out


def _write_center3_failure_conclusion(
    path: Path,
    *,
    cell_rows: Sequence[Mapping[str, object]],
    sample_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
) -> None:
    primary_cell = [
        row
        for row in cell_rows
        if int(_safe_int(row.get("experiment_seed"), default=-1)) == CENTER3_FAILURE_PRIMARY_CELL[0]
        and str(row.get("heldout_center")) == CENTER3_FAILURE_PRIMARY_CELL[1]
    ]
    final = next((row for row in primary_cell if row.get("audit_method") == "final_v2"), {})
    best_seed_delta = _float(final.get("delta_best_individual_seed_blend_minus_final_v2", math.nan))
    class0_recall = _float(final.get("class0_recall", math.nan))
    class1_recall = _float(final.get("class1_recall", math.nan))
    class0_pred = _safe_int(final.get("class0_predicted_count"), default=0)
    class1_pred = _safe_int(final.get("class1_predicted_count"), default=0)
    n_eval = _safe_int(final.get("n_target_eval"), default=0)
    seed101_suppressed = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("seed_101_correct_final_wrong") is True or str(row.get("seed_101_correct_final_wrong")) == "True")
    )
    seed127_suppressed = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("seed_127_correct_final_wrong") is True or str(row.get("seed_127_correct_final_wrong")) == "True")
    )
    final_correct_seed101_wrong = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("final_correct_seed_101_wrong") is True or str(row.get("final_correct_seed_101_wrong")) == "True")
    )
    final_correct_seed127_wrong = sum(
        1
        for row in sample_rows
        if row.get("audit_cell_role") == "primary_center3_failure"
        and (row.get("final_correct_seed_127_wrong") is True or str(row.get("final_correct_seed_127_wrong")) == "True")
    )
    flags: list[str] = []
    if not final:
        flags.append("insufficient_row_level_evidence")
    if n_eval and (class0_pred == 0 or class1_pred == 0 or class0_pred == n_eval or class1_pred == n_eval):
        flags.append("class_collapse")
    elif math.isfinite(class0_recall) and math.isfinite(class1_recall) and min(class0_recall, class1_recall) <= 0.05:
        flags.append("near_class_collapse")
    if math.isfinite(best_seed_delta) and best_seed_delta >= 0.10:
        flags.append("probability_pooling_suppresses_best_seed")
    if seed101_suppressed > final_correct_seed101_wrong or seed127_suppressed > final_correct_seed127_wrong:
        if "probability_pooling_suppresses_best_seed" not in flags:
            flags.append("probability_pooling_suppresses_best_seed")
    mean_incorrect_conf = _float(final.get("mean_confidence_incorrect", math.nan))
    if math.isfinite(mean_incorrect_conf) and mean_incorrect_conf >= 0.65:
        flags.append("confident_wrong_predictions")
    if not flags:
        flags.append("no_single_dominant_failure_mode_from_compact_audit")

    lines = [
        "# Center3 Failure Audit",
        "",
        "## Scope",
        "",
        "Diagnostic-only audit of predefined cells. Target labels are used only after fixed prediction bundles exist, for scoring and failure analysis.",
        "",
        "## Primary Cell",
        "",
        f"- Cell: `{CENTER3_FAILURE_PRIMARY_CELL[0]} x center{CENTER3_FAILURE_PRIMARY_CELL[1]}`",
        f"- Final v2 BACC: {_format_float(final.get('bacc', math.nan)) if final else 'nan'}",
        f"- Final class0 recall: {_format_float(class0_recall)}",
        f"- Final class1 recall: {_format_float(class1_recall)}",
        f"- Predicted class counts: class0={class0_pred}, class1={class1_pred}, n={n_eval}",
        f"- Best individual seed-blend delta over final: {_format_float(best_seed_delta)}",
        f"- Seed101 correct while final wrong: {seed101_suppressed}",
        f"- Seed127 correct while final wrong: {seed127_suppressed}",
        f"- Final correct while seed101 wrong: {final_correct_seed101_wrong}",
        f"- Final correct while seed127 wrong: {final_correct_seed127_wrong}",
        "",
        "## Assigned Failure Mode",
        "",
        f"- `{ '|'.join(flags) }`",
        "",
        "## Artifact Evidence",
        "",
        f"- Cell/pooling rows: {len(cell_rows)}",
        f"- Sample audit rows: {len(sample_rows)}",
        f"- Source-weight comparison rows: {len(source_weight_rows)}",
        f"- Component-coverage comparison rows: {len(component_coverage_rows)}",
        "",
        "## Protocol Boundary",
        "",
        "This audit must not be used to select seeds, calibrate on target labels, change pooling policy, or claim target-compatible expert discovery. Any follow-up method must be predeclared separately.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")




def _write_multipanel_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Multi-Panel Tail-Risk Mass-Bag Stabilization v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_MULTIPANEL_TAILRISK_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'MULTIPANEL_TAILRISK_STABILIZATION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Intersection mean BACC: {_format_float(decision.get('intersection_center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Frozen bottom-20 BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs prior tailrisk: {_format_float(decision.get('delta_vs_prior_tailrisk_intersection'))}",
        f"- Delta vs canonical random mass-bag: {_format_float(decision.get('delta_vs_canonical_random_mass_bag_intersection'))}",
        f"- Frozen bottom20 median delta: {_format_float(decision.get('frozen_bottom20_median_delta_vs_prior_tailrisk'))}",
        f"- Worst per-center regression vs prior tailrisk: {_format_float(decision.get('worst_per_center_regression_vs_prior_tailrisk'))}",
        f"- Tail-risk transfer flag: `{decision.get('tailrisk_transfer_flag')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a source-only stochastic-composition stabilization experiment. It is not a compatibility router and does not claim random mass-bag discovers target-compatible experts.",
        "",
        "The primary method blends each seed-specific shrink050 anchor with its seed-specific random mass-bag, then probability-pools the nine predeclared seed blends before computing metrics.",
        "",
        "Target evaluation labels are scoring/audit only and never choose seeds, alpha, source set, calibration, classifier, or pass/fail policy.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_positive_union_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Source-Inner Class-Conditional Positive Union v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_POSITIVE_UNION_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'SOURCE_INNER_POSITIVE_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Selected rule counts: `{decision.get('selected_rule_counts_json', '{}')}`",
        f"- Insufficient source-inner positive-count cells: {decision.get('insufficient_source_inner_positive_count_cells', 0)}",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Intersection mean BACC: {_format_float(decision.get('intersection_center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Frozen bottom-20 BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs prior tailrisk: {_format_float(decision.get('delta_vs_prior_tailrisk_intersection'))}",
        f"- Delta vs v2 arithmetic multipanel: {_format_float(decision.get('delta_vs_v2_arithmetic_intersection'))}",
        f"- Frozen bottom20 median delta: {_format_float(decision.get('frozen_bottom20_median_delta_vs_prior_tailrisk'))}",
        f"- Worst per-center regression vs prior tailrisk: {_format_float(decision.get('worst_per_center_regression_vs_prior_tailrisk'))}",
        f"- Worst seed-center regression vs prior tailrisk: {_format_float(decision.get('worst_seed_center_regression_vs_prior_tailrisk'))}",
        f"- Tail-risk transfer flag: `{decision.get('tailrisk_transfer_flag')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a source-inner selected class-conditional aggregation repair over fixed CVAE seed-blend probabilities. It is not compatibility routing, target adaptation, target-threshold tuning, or target-compatible expert discovery.",
        "",
        "Target labels are used only after the source-inner rule is fixed, for scoring and audit rows.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_fixed_beta050_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Fixed Beta050 Positive-Union Confirmation v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_FIXED_BETA050_POSITIVE_UNION_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'FIXED_BETA050_POSITIVE_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Fixed rule: `{decision.get('fixed_rule', POSITIVE_UNION_RULE_BETA050)}`",
        f"- Fixed beta: {_format_float(decision.get('fixed_beta', 0.5))}",
        f"- Development seeds: `{decision.get('development_experiment_seeds_json', '[]')}`",
        f"- Primary confirmation seeds: `{decision.get('primary_confirmation_experiment_seeds_json', '[]')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Intersection mean BACC: {_format_float(decision.get('intersection_center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Frozen arithmetic bottom-20 BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs v2 arithmetic multipanel: {_format_float(decision.get('delta_vs_v2_arithmetic_intersection'))}",
        f"- Frozen bottom20 median delta vs arithmetic: {_format_float(decision.get('frozen_bottom20_median_delta_vs_v2_arithmetic'))}",
        f"- Assessable rare-positive cells: {decision.get('n_assessable_rare_positive_cells', 0)}",
        f"- Rare-positive recall mean delta vs arithmetic: {_format_float(decision.get('rare_positive_recall_mean_delta_vs_arithmetic'))}",
        f"- Worst per-center regression vs arithmetic: {_format_float(decision.get('worst_per_center_regression_vs_v2_arithmetic'))}",
        f"- Worst seed-center regression vs arithmetic: {_format_float(decision.get('worst_seed_center_regression_vs_v2_arithmetic'))}",
        f"- Tail-risk transfer flag: `{decision.get('tailrisk_transfer_flag')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a fixed global beta050 confirmation. The beta was hypothesis-generated from prior diagnostic seeds `[42,43,44]` and is predeclared before evaluating fresh seeds.",
        "",
        "This is not source-inner selected, not compatibility routing, not target adaptation, and not target-threshold tuning. Target labels are scoring/audit only after fixed predictions exist.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_harm_gated_positive_union_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Source-Inner Harm-Gated Positive-Union v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_HARM_GATED_POSITIVE_UNION_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'HARM_GATED_POSITIVE_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Completed primary seeds: `{decision.get('completed_primary_experiment_seeds_json', '[]')}`",
        f"- Valid primary cells: {decision.get('n_valid_primary_cells', 0)}",
        f"- Selected rule counts: `{decision.get('selected_rule_counts_json', '{}')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Intersection mean BACC: {_format_float(decision.get('intersection_center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Frozen arithmetic bottom-20 BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Delta vs v2 arithmetic multipanel: {_format_float(decision.get('delta_vs_v2_arithmetic_intersection'))}",
        f"- Delta vs fixed beta050 diagnostic: {_format_float(decision.get('delta_vs_fixed_beta050_intersection'))}",
        f"- Frozen bottom20 median delta vs arithmetic: {_format_float(decision.get('frozen_bottom20_median_delta_vs_v2_arithmetic'))}",
        f"- Assessable rare-positive cells: {decision.get('n_assessable_rare_positive_cells', 0)}",
        f"- Rare-positive recall mean delta vs arithmetic: {_format_float(decision.get('rare_positive_recall_mean_delta_vs_arithmetic'))}",
        f"- Worst per-center regression vs arithmetic: {_format_float(decision.get('worst_per_center_regression_vs_v2_arithmetic'))}",
        f"- Worst seed-center regression vs arithmetic: {_format_float(decision.get('worst_seed_center_regression_vs_v2_arithmetic'))}",
        f"- Tail-risk transfer flag: `{decision.get('tailrisk_transfer_flag')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a source-only harm-gated positive-evidence pooling confirmation. The thresholds are frozen from retrospective development evidence before evaluating primary seeds.",
        "",
        "This is not compatibility routing, not target adaptation, not target-support calibration, and not target-threshold tuning. Target labels are scoring/audit only after the source-inner rule is fixed.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")




def _write_artifacts(
    root: Path,
    cfg: TailRiskAnchoredConfig,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    eligibility_rows: Sequence[Mapping[str, object]],
    blend_manifest_rows: Sequence[Mapping[str, object]],
    complementarity_rows: Sequence[Mapping[str, object]],
    calibration_rows: Sequence[Mapping[str, object]],
    shuffled_null_rows: Sequence[Mapping[str, object]],
    shuffled_null_summary: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage: object,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> None:
    write_csv_rows(root / "tables" / "tailrisk_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "tailrisk_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "tailrisk_panel_summary.csv", _panel_summary_rows(matrix_rows))
    write_csv_rows(root / "tables" / "tailrisk_tail_metric_summary.csv", _tail_metric_summary_rows(matrix_rows))
    write_csv_rows(root / "tables" / "tailrisk_probability_blend_manifest.csv", blend_manifest_rows)
    write_csv_rows(root / "tables" / "tailrisk_complementarity_audit.csv", complementarity_rows)
    write_csv_rows(root / "tables" / "tailrisk_calibration_audit.csv", calibration_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "oracle_gap_summary.csv", _oracle_gap_rows(matrix_rows))
    write_csv_rows(root / "tables" / "random_mass_bag_control_summary.csv", _random_mass_bag_summary(matrix_rows))
    write_csv_rows(root / "tables" / "shuffled_reliability_null_summary.csv", shuffled_null_summary)
    write_csv_rows(root / "tables" / "anchor_reproducibility_audit.csv", anchor_rows)
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "shuffled_reliability_null_matrix.csv", shuffled_null_rows)
    write_csv_rows(root / "manifests" / "tailrisk_component_union_model_manifest.csv", model_manifest_rows)
    write_protocol_finalization(
        root,
        leakage_report=leakage.to_json_dict(),
        protocol_manifest=_tailrisk_anchored_protocol_manifest_payload(
            cfg,
            protocol_violations=protocol_violations,
            target_expert_excluded=target_expert_excluded,
        ),
        resolved_config=_resolved_config(cfg),
    )
    _write_decision_summary(root, decision)


def _tailrisk_anchored_protocol_manifest_payload(
    cfg: TailRiskAnchoredConfig,
    *,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
) -> dict[str, object]:
    return {
        "schema_version": "cvae_rebuild_tailrisk_anchored_component_union_protocol_v1",
        "experiment_name": cfg.name,
        "primary_method": cfg.primary_method,
        "experiment_type": "source_only_tailrisk_anchored_mass_uncertainty_component_union",
        "target_expert_excluded": bool(target_expert_excluded),
        "target_support_used": False,
        "target_support_labels_for_selection": False,
        "target_eval_labels_for_scoring_only": True,
        "target_calibration_metrics_audit_only": True,
        "target_conditioned_point_compatibility_estimate": False,
        "fixed_all_source_inclusion": True,
        "blend_alpha_locked": cfg.blend_alpha,
        "random_mass_bag_size": cfg.random_mass_bag_size,
        "random_mass_bag_distribution": "dirichlet_uniform_alpha4",
        "source_ablation_diagnostic_only": True,
        "oracle_rows_diagnostic_only": True,
        "claim_boundary": (
            "source-only robustness aggregation under component/source-mass uncertainty; "
            "not learned routing, sparse expert selection, target adaptation, formal privacy, "
            "or causal reliability validation"
        ),
        "protocol_violations": list(protocol_violations),
    }


def _write_decision_summary(root: Path, decision: Mapping[str, object]) -> None:
    lines = [
        "# Tail-Risk Anchored Mass-Uncertainty Component-Union v1",
        "",
        "## Summary",
        "",
        f"- Primary method: `{decision.get('primary_method', PRIMARY_TAILRISK_METHOD)}`",
        f"- Primary verdict: `{decision.get('primary_verdict', 'TAILRISK_ANCHORED_COMPONENT_UNION_FAIL')}`",
        f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
        f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
        f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
        f"- Center 3 BACC: {_format_float(decision.get('center3_bacc'))}",
        f"- Bottom-20 cell mean BACC: {_format_float(decision.get('bottom20_cell_mean_bacc'))}",
        f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
        f"- Shrink050 BACC: {_format_float(decision.get('shrink050_center_equal_mean_bacc'))}",
        f"- Random mass-bag BACC: {_format_float(decision.get('random_mass_bag_center_equal_mean_bacc'))}",
        f"- Center3 delta vs random mass-bag: {_format_float(decision.get('center3_delta_vs_random_mass_bag'))}",
        f"- Bottom20 delta vs random mass-bag: {_format_float(decision.get('bottom20_delta_vs_random_mass_bag'))}",
        f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
        f"- Complementarity nontrivial: `{decision.get('complementarity_nontrivial_on_center3_or_bottom20')}`",
        f"- Fresh panel preserves tail direction: `{decision.get('fresh_panel_preserves_tail_direction')}`",
        f"- Leakage status: `{decision.get('leakage_status', '')}`",
        "",
        "## Protocol Boundary",
        "",
        "This is a locked source-only robustness aggregation audit. It uses no target support, no target-conditioned point compatibility estimate, and no sparse expert selection.",
        "",
        "The primary method averages fixed prediction probabilities from a reliability-shrink050 component-union anchor and an 11-member Dirichlet-uniform random mass-bag ensemble with alpha 0.50/0.50.",
        "",
        "Target evaluation labels and target calibration metrics are audit/scoring only and never choose alpha, weights, source set, classifier, or decision logic.",
        "",
        "Safe claim if successful: in Virchow2 CVAE-generated feature aggregation, fixed source-only probability blending of a conservative reliability-weighted component union with a random mass-bag ensemble can reduce weak-center tail risk when the two compositions make complementary errors.",
        "",
    ]
    (root / "reports" / "decision_summary.md").write_text("\n".join(lines), encoding="utf-8")


