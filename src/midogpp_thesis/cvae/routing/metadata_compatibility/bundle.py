"""Closed-world file contract for the metadata compatibility artifact."""

from __future__ import annotations


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/metadata_profile_lock.json",
    "manifests/compatibility_lock.json",
    "manifests/content_index.json",
    "reports/compatibility_decision.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/metadata_profiles.csv",
    "tables/compatibility_scores.csv",
)

CONTENT_INDEX_MEMBERS = tuple(
    relative
    for relative in REQUIRED_FILES
    if relative
    not in {
        "manifests/content_index.json",
        "reports/run_state.json",
        "reports/validation_report.json",
    }
)


__all__ = ("CONTENT_INDEX_MEMBERS", "REQUIRED_FILES")
