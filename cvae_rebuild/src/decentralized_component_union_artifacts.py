from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from decentralized_component_union_prior import (
    MATCHED_SHUFFLED_RELIABILITY_PREFIX,
    MATCHED_SHUFFLED_RELIABILITY_SHRINK050_PREFIX,
    PROTOCOL_WORDING,
    ROW_RANDOM_MASS_BAG_CONTROL,
    ROW_RANDOM_SOURCE_MASS_CONTROL,
    ROW_SHUFFLED_LABEL_CONTROL,
    ROW_SHUFFLED_RELIABILITY_CONTROL,
    ROW_SHUFFLED_SUMMARY_CONTROL,
)
from preservation_repair import _format_float
from protocol import build_leakage_report
from reporting import write_csv_rows, write_protocol_finalization


def _write_artifacts(
    root: Path,
    cfg: object,
    *,
    matrix_rows: Sequence[Mapping[str, object]],
    gap_rows: Sequence[Mapping[str, object]],
    component_manifest_rows: Sequence[Mapping[str, object]],
    source_summary_rows: Sequence[Mapping[str, object]],
    prototype_manifest_rows: Sequence[Mapping[str, object]],
    component_coverage_rows: Sequence[Mapping[str, object]],
    source_weight_rows: Sequence[Mapping[str, object]],
    reliability_rows: Sequence[Mapping[str, object]],
    source_ablation_rows: Sequence[Mapping[str, object]],
    paired_generation_rows: Sequence[Mapping[str, object]],
    weak_rows: Sequence[Mapping[str, object]],
    nn_rows: Sequence[Mapping[str, object]],
    real_feature_rows: Sequence[Mapping[str, object]],
    late_rows: Sequence[Mapping[str, object]],
    model_manifest_rows: Sequence[Mapping[str, object]],
    anchor_rows: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
    leakage_status: str,
    protocol_violations: Sequence[str],
    target_expert_excluded: bool,
    matched_shuffled_reliability_lambda: Callable[[object], float],
    matched_shuffled_reliability_null_rows: Callable[[Sequence[Mapping[str, object]]], list[Mapping[str, object]]],
    panel_summary_rows: Callable[[Sequence[Mapping[str, object]], object], list[dict[str, object]]],
    shuffled_reliability_cell_delta_rows: Callable[[Sequence[Mapping[str, object]], object], list[dict[str, object]]],
    shuffled_reliability_center_summary_rows: Callable[[Sequence[Mapping[str, object]], object], list[dict[str, object]]],
    random_mass_bag_control_summary_rows: Callable[[Sequence[Mapping[str, object]]], list[dict[str, object]]],
    oracle_gap_summary_rows: Callable[[Sequence[Mapping[str, object]], object], list[dict[str, object]]],
    eligibility_rows: Callable[[Sequence[Mapping[str, object]], object], list[dict[str, object]]],
    resolved_config: Callable[[object], dict[str, object]],
) -> None:
    matched_null_rows = matched_shuffled_reliability_null_rows(matrix_rows)
    panel_rows = panel_summary_rows(matrix_rows, cfg)
    null_cell_delta_rows = shuffled_reliability_cell_delta_rows(matrix_rows, cfg)
    null_center_rows = shuffled_reliability_center_summary_rows(matrix_rows, cfg)
    write_csv_rows(root / "tables" / "component_union_downstream_matrix.csv", matrix_rows)
    write_csv_rows(root / "tables" / "component_union_gap_summary.csv", gap_rows)
    write_csv_rows(root / "tables" / "component_union_summary.csv", [dict(decision)])
    write_csv_rows(root / "tables" / "component_union_panel_summary.csv", panel_rows)
    write_csv_rows(
        root / "tables" / "shuffled_reliability_null_matrix.csv",
        matched_null_rows,
        columns=None if matched_null_rows else (
            "experiment_seed",
            "heldout_center",
            "panel",
            "replicate_seed",
            "prior_method",
            "control_permutation_id",
            "bacc",
            "macro_f1",
            "status",
        ),
    )
    write_csv_rows(root / "tables" / "shuffled_reliability_null_summary.csv", [_null_summary_output(decision)])
    write_csv_rows(
        root / "tables" / "shuffled_reliability_cell_delta_summary.csv",
        null_cell_delta_rows,
        columns=(
            "experiment_seed",
            "heldout_center",
            "replicate_seed",
            "panel",
            "null_perm_id",
            "primary_bacc",
            "null_bacc",
            "delta_primary_minus_null",
        ),
    )
    write_csv_rows(
        root / "tables" / "shuffled_reliability_center_summary.csv",
        null_center_rows,
        columns=(
            "heldout_center",
            "panel",
            "primary_center_bacc",
            "null_mean_center_bacc",
            "null_p95_center_bacc",
            "primary_minus_null_mean",
            "primary_above_null_p95",
        ),
    )
    write_csv_rows(root / "tables" / "component_manifest.csv", component_manifest_rows)
    write_csv_rows(root / "tables" / "source_summary_diagnostics.csv", source_summary_rows)
    write_csv_rows(root / "tables" / "prototype_manifest.csv", prototype_manifest_rows)
    write_csv_rows(root / "tables" / "component_coverage_audit.csv", component_coverage_rows)
    write_csv_rows(root / "tables" / "source_weight_manifest.csv", source_weight_rows)
    write_csv_rows(root / "tables" / "source_reliability_manifest.csv", reliability_rows)
    write_csv_rows(root / "tables" / "source_ablation_audit.csv", source_ablation_rows)
    write_csv_rows(root / "tables" / "paired_generation_audit.csv", paired_generation_rows)
    write_csv_rows(root / "tables" / "weak_source_audit.csv", weak_rows)
    write_csv_rows(root / "tables" / "nearest_neighbor_memorization_audit.csv", nn_rows)
    write_csv_rows(root / "tables" / "real_feature_reference_matrix.csv", real_feature_rows)
    write_csv_rows(root / "tables" / "late_aggregation_reference_matrix.csv", late_rows)
    write_csv_rows(root / "tables" / "negative_control_summary.csv", [_negative_control_summary(decision)])
    write_csv_rows(root / "tables" / "random_mass_bag_control_summary.csv", random_mass_bag_control_summary_rows(matrix_rows))
    write_csv_rows(root / "tables" / "anchor_reproducibility_audit.csv", anchor_rows)
    write_csv_rows(root / "tables" / "oracle_gap_summary.csv", oracle_gap_summary_rows(matrix_rows, cfg))
    write_csv_rows(root / "tables" / "eligibility_audit.csv", eligibility_rows(matrix_rows, cfg))
    write_csv_rows(root / "manifests" / "decentralized_component_union_prior_model_manifest.csv", model_manifest_rows)
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
            "schema_version": "cvae_rebuild_decentralized_component_union_prior_protocol_manifest_v1",
            "experiment_name": cfg.name,
            "experiment_type": "decentralized_component_level_generative_expert_composition",
            "primary_variant": cfg.primary_variant,
            "primary_method": cfg.primary_method,
            "primary_shrink_lambda": cfg.primary_shrink_lambda,
            "canonical_replicate_seeds": list(cfg.replicate_seeds),
            "fresh_replicate_seeds": list(cfg.fresh_replicate_seeds),
            "random_mass_bag_control_size": cfg.random_mass_bag_control_size,
            "target_support_labels_for_selection": False,
            "target_eval_labels_for_scoring_only": True,
            "target_expert_excluded": target_expert_excluded,
            "fixed_all_source_inclusion": True,
            "tests_target_conditioned_routing": False,
            "tests_composition_granularity": True,
            "exported_source_summaries_are_target_agnostic": True,
            "raw_source_embedding_pooling_for_prior_fit": False,
            "pooled_classifier_frame": "raw_embedding_frame_after_source_inverse_pca",
            "source_union_references_diagnostic_only": True,
            "source_ablation_diagnostic_only": True,
            "matched_shuffled_reliability_null_permutations": cfg.matched_shuffled_reliability_null_permutations,
            "matched_shuffled_reliability_null_lambda": matched_shuffled_reliability_lambda(cfg) if cfg.matched_shuffled_reliability_null_permutations else "",
            "oracle_rows_diagnostic_only": True,
            "protocol_wording": PROTOCOL_WORDING,
            "claim_boundary": (
                "component-level generative expert composition using source-only reliability-weighted dense mass allocation where configured; "
                "no target-specific compatibility routing claim, "
                "no support-NELBO downstream claim, and no formal privacy claim"
            ),
            "cache_policy": {
                "component_summaries": "source/seed/class/config",
                "generated_pools": "source_weight_hash+latent_seed+component_summary_hash",
                "classifier_predictions": "generated_pool_hash+classifier_config+eval_fold",
            },
        },
        resolved_config=resolved_config(cfg),
    )
    _write_decision_summary(root, decision, leakage_status=leakage_status)


def _negative_control_summary(decision: Mapping[str, object]) -> dict[str, object]:
    return {
        "primary_method": decision.get("primary_method", ""),
        "control_methods": "|".join(
            [
                ROW_SHUFFLED_SUMMARY_CONTROL,
                ROW_SHUFFLED_LABEL_CONTROL,
                ROW_SHUFFLED_RELIABILITY_CONTROL,
                ROW_RANDOM_SOURCE_MASS_CONTROL,
                ROW_RANDOM_MASS_BAG_CONTROL,
                f"{MATCHED_SHUFFLED_RELIABILITY_PREFIX}*",
                f"{MATCHED_SHUFFLED_RELIABILITY_SHRINK050_PREFIX}*",
            ]
        ),
        "primary_center_equal_mean_bacc": decision.get("center_equal_mean_bacc", math.nan),
        "strongest_negative_control_method": decision.get("strongest_negative_control_method", ""),
        "strongest_negative_control_center_equal_mean_bacc": decision.get("strongest_negative_control_center_equal_mean_bacc", math.nan),
        "negative_control_gap": decision.get("negative_control_gap", math.nan),
        "control_competitive": "NEGATIVE_CONTROL_COMPETITIVE" in str(decision.get("diagnostic_flags", "")),
        "matched_null_empirical_p_value": decision.get("empirical_p_value", math.nan),
        "primary_minus_null_mean": decision.get("primary_minus_null_mean", math.nan),
        "primary_minus_null_p95": decision.get("primary_minus_null_p95", math.nan),
        "random_mass_bag_control_center_equal_mean_bacc": decision.get("random_mass_bag_control_center_equal_mean_bacc", math.nan),
        "delta_vs_random_mass_bag_control": decision.get("delta_vs_random_mass_bag_control", math.nan),
    }


def _null_summary_output(decision: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "n_null_permutations",
        "effective_unique_null_patterns",
        "primary_center_equal_mean_bacc",
        "null_mean_center_equal_bacc",
        "null_p90_center_equal_bacc",
        "null_p95_center_equal_bacc",
        "null_max_center_equal_bacc",
        "empirical_p_value",
        "primary_minus_null_mean",
        "primary_minus_null_p95",
        "paired_cell_mean_delta_vs_null_mean",
        "paired_cell_win_fraction_vs_null",
    )
    return {field: decision.get(field, math.nan) for field in fields}


def _write_decision_summary(root: Path, decision: Mapping[str, object], *, leakage_status: str) -> None:
    text = "\n".join(
        [
            "# Decentralized Component-Level Generative Expert Composition",
            "",
            "## Summary",
            "",
            f"- Primary method: `{decision.get('primary_method', '')}`",
            f"- Primary verdict: `{decision.get('primary_verdict', 'COMPONENT_UNION_FAIL')}`",
            f"- Diagnostic flags: `{decision.get('diagnostic_flags', '')}`",
            f"- Center-equal mean BACC: {_format_float(decision.get('center_equal_mean_bacc'))}",
            f"- Seed-cell mean BACC: {_format_float(decision.get('seed_cell_mean_bacc'))}",
            f"- Center-equal macro-F1: {_format_float(decision.get('center_equal_macro_f1'))}",
            f"- Min center BACC: {_format_float(decision.get('min_center_bacc'))}",
            f"- Seed std BACC: {_format_float(decision.get('seed_std_bacc'))}",
            f"- Delta vs D1.2 reliability all4: {_format_float(decision.get('delta_vs_d1_2_reliability_all4'))}",
            f"- Delta vs full reliability-weighted dense all4: {_format_float(decision.get('delta_vs_full_reliability_weighted_dense_all4'))}",
            f"- Delta vs equal all4: {_format_float(decision.get('delta_vs_equal_all4'))}",
            f"- Delta vs component shrink025: {_format_float(decision.get('delta_vs_component_shrink025'))}",
            f"- Delta vs random mass bag control: {_format_float(decision.get('delta_vs_random_mass_bag_control'))}",
            f"- Retention vs source-union K16: {_format_float(decision.get('retention_vs_source_union_k16'))}",
            f"- Retention vs center-balanced K16: {_format_float(decision.get('retention_vs_center_balanced_k16'))}",
            f"- Oracle gap vs source-union K16: {_format_float(decision.get('oracle_gap_vs_source_union_k16'))}",
            f"- Oracle gap vs real-feature dense: {_format_float(decision.get('oracle_gap_vs_real_feature_dense'))}",
            f"- Delta vs real-feature dense reference: {_format_float(decision.get('delta_vs_real_source_embedding_dense_reference'))}",
            f"- Negative-control gap: {_format_float(decision.get('negative_control_gap'))}",
            f"- Matched shuffled-null permutations: {decision.get('n_null_permutations', 0)}",
            f"- Matched shuffled-null empirical p-value: {_format_float(decision.get('empirical_p_value'))}",
            f"- Effective unique shuffled-null patterns: {decision.get('effective_unique_null_patterns', 0)}",
            f"- Primary minus matched null mean: {_format_float(decision.get('primary_minus_null_mean'))}",
            f"- Primary minus matched null p95: {_format_float(decision.get('primary_minus_null_p95'))}",
            f"- Paired-cell win fraction vs matched null: {_format_float(decision.get('paired_cell_win_fraction_vs_null'))}",
            f"- Canonical panel BACC: {_format_float(decision.get('canonical_center_equal_mean_bacc'))}",
            f"- Fresh panel BACC: {_format_float(decision.get('fresh_center_equal_mean_bacc'))}",
            f"- Fresh panel preserves canonical direction: `{decision.get('fresh_panel_preserves_canonical_direction', '')}`",
            f"- Anchor reproducibility pass: `{decision.get('anchor_reproducibility_pass', '')}`",
            f"- Max source-ablation drop: {_format_float(decision.get('max_source_ablation_drop_bacc'))}",
            f"- Max source-ablation gain: {_format_float(decision.get('max_source_ablation_gain_bacc'))}",
            f"- Source-ablation max abs delta: {_format_float(decision.get('source_ablation_max_abs_delta'))}",
            f"- Source-ablation shrink025 v2 reference max abs delta: {_format_float(decision.get('source_ablation_reference_shrink025_v2_max_abs_delta'))}",
            f"- Source-ablation hybrid v1 reference max abs delta: {_format_float(decision.get('source_ablation_reference_hybrid_v1_max_abs_delta'))}",
            f"- Source-ablation mass-bagged v1 reference max abs delta: {_format_float(decision.get('source_ablation_reference_mass_bagged_v1_max_abs_delta'))}",
            f"- Source-ablation reduction vs shrink025 v2: {_format_float(decision.get('source_ablation_reduction_vs_shrink025_v2'))}",
            f"- Source-ablation reduction vs hybrid v1: {_format_float(decision.get('source_ablation_reduction_vs_hybrid_v1'))}",
            f"- Source-ablation reduction vs mass-bagged v1: {_format_float(decision.get('source_ablation_reduction_vs_mass_bagged_v1'))}",
            f"- Leakage status: `{leakage_status}`",
            "",
            "## Protocol Boundary",
            "",
            PROTOCOL_WORDING,
            "",
            "This experiment does not test target-conditioned routing.",
            "It tests whether decentralized generative composition should operate at component/prototype granularity rather than whole-source granularity.",
            "The routing decision is fixed: use all non-heldout source experts.",
            "For shrink025/shrink050 confirmation audits, the deployed decision is source/component mass allocation, not sparse source selection.",
            "Target evaluation labels are used only for final scoring.",
            "",
            "## Supported Claim If Successful",
            "",
            "Source-only reliability contains weak compatibility information when used as a regularized dense prior over component-union experts, if the primary clears matched shuffled-reliability and random-mass controls.",
            "",
        ]
    )
    path = root / "reports" / "decision_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
