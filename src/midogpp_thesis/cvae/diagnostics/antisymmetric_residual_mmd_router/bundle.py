"""Closed-world artifact contract for the cross-fitted diagnostic."""

from __future__ import annotations


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/support_partition_lock.json",
    "manifests/crossfit_surface_lock.json",
    "manifests/source_products_lock.json",
    "manifests/router_plan_lock.json",
    "manifests/global_case_prediction_seal.json",
    "manifests/content_index.json",
    "arrays/source_prefix_blocks.npy",
    "arrays/router_states.npz",
    "arrays/target_predictions.npz",
    "tables/support_partitions.csv",
    "tables/crossfit_folds.csv",
    "tables/source_block_index.csv",
    "tables/compatibility_case_energy.csv",
    "tables/compatibility_scores.csv",
    "tables/case_router_plans.csv",
    "tables/case_target_assignments.csv",
    "tables/target_prediction_index.csv",
    "tables/target_metrics.csv",
    "tables/paired_deltas.csv",
    "reports/phase_01_source_products_complete.json",
    "reports/phase_02_router_plans_complete.json",
    "reports/phase_03_predictions_sealed.json",
    "reports/phase_04_scoring_complete.json",
    "reports/label_access_report.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)

CONTENT_INDEX_MEMBERS = tuple(
    member
    for member in REQUIRED_FILES
    if member
    not in {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)


__all__ = ("CONTENT_INDEX_MEMBERS", "REQUIRED_FILES")
