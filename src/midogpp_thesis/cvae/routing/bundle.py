"""Closed-world file contract for the Stage-60 equal-union artifact."""

from __future__ import annotations


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/protocol_manifest.json",
    "manifests/policy_lock.json",
    "manifests/equal_union_policy_plan.json",
    "manifests/content_index.json",
    "reports/policy_decision.json",
    "reports/leakage_report.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/policy_assignments.csv",
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
