"""Closed-world canonical member contract for the Stage-70 result bundle."""

REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/policy_binding.json",
    "manifests/evaluation_plan.json",
    "manifests/prediction_seal.json",
    "manifests/content_index.json",
    "checkpoints/source/source_cache.json",
    "checkpoints/predictions/prediction_cache.json",
    "reports/workstation_preflight.json",
    "reports/leakage_report.json",
    "reports/label_access.json",
    "reports/primary_result.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/seed_cell_metrics.csv",
    "tables/ensemble_metrics.csv",
    "tables/center_contrasts.csv",
    "tables/contrast_inference.csv",
    "tables/oracle_diagnostics.csv",
    "tables/prediction_index.csv",
)

__all__ = ("REQUIRED_FILES",)
