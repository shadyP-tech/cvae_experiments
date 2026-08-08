"""Closed-world file contract for the fresh residual-topup Stage-60 lock."""

from __future__ import annotations


REQUIRED_FILES = (
    "config.resolved.yaml",
    "provenance/input_artifacts.json",
    "manifests/fresh_surface_attestation.json",
    "manifests/protocol_manifest.json",
    "manifests/policy_lock.json",
    "manifests/action_library.json",
    "manifests/content_index.json",
    "reports/protocol_report.json",
    "reports/leakage_report.json",
    "reports/policy_decision.json",
    "reports/run_state.json",
    "reports/validation_report.json",
    "tables/proxy_ballots.csv",
    "tables/proxy_ranks.csv",
    "tables/policy_actions.csv",
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
