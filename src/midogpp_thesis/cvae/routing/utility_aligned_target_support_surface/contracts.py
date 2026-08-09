"""Frozen identities for the label-free target-support feature producer."""

from ..utility_aligned_identities import (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    METADATA_PROFILE_ARTIFACT_ID as METADATA_ARTIFACT_ID,
    TARGET_SUPPORT_CACHE_ARTIFACT_ID as CACHE_ARTIFACT_ID,
    TARGET_SUPPORT_PARENT_RESERVATION_ARTIFACT_ID as RESERVATION_ARTIFACT_ID,
    TARGET_SUPPORT_PRODUCER_EXPERIMENT_ID as EXPERIMENT_ID,
    TARGET_SUPPORT_SURFACE_ARTIFACT_ID as OUTPUT_ARTIFACT_ID,
)

INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    RESERVATION_ARTIFACT_ID,
    CACHE_ARTIFACT_ID,
    METADATA_ARTIFACT_ID,
)

REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/target_reservation.json",
    "manifests/target_support_cache_lock.json",
    "manifests/source_generation_lock.json",
    "manifests/case_bootstrap_plans.json",
    "manifests/target_support_surface_lock.json",
    "manifests/target_support_action_shifts_lock.json",
    "manifests/content_index.json",
    "tables/target_candidate_features.csv",
    "tables/target_candidate_feature_bootstraps.csv",
    "tables/target_support_action_shifts.csv",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
)

__all__ = (
    "CACHE_ARTIFACT_ID", "EXPERIMENT_ID", "INPUT_ARTIFACT_IDS",
    "OUTPUT_ARTIFACT_ID", "REQUIRED_FILES", "RESERVATION_ARTIFACT_ID",
)
