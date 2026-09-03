"""Cohesive execution services for the HARP v13 phase coordinator."""

from .admission import (
    assert_pristine_output,
    authorization_provenance,
    dedicated_scratch,
    exact_output_root,
    validate_parent_ledger,
    validate_preflight,
    validate_pristine_or_label_free_recovery,
)
from .bindings import (
    bind_admission_artifact,
    bind_development_artifact,
    bind_model_artifact,
    bind_target_action_artifact,
    reconstruct_frozen_routes_for_evaluation,
    validate_in_memory_route_bindings,
)
from .completion import (
    commit_completion_state,
    validate_content_index,
    write_content_index,
)
from .menus import validate_complete_physical_menus
from .reports import (
    TerminalReportBundle,
    build_leakage_report,
    prelabel_route_summary,
    write_terminal_reports,
)


__all__ = (
    "TerminalReportBundle",
    "assert_pristine_output",
    "authorization_provenance",
    "bind_admission_artifact",
    "bind_development_artifact",
    "bind_model_artifact",
    "bind_target_action_artifact",
    "build_leakage_report",
    "commit_completion_state",
    "dedicated_scratch",
    "exact_output_root",
    "prelabel_route_summary",
    "reconstruct_frozen_routes_for_evaluation",
    "validate_complete_physical_menus",
    "validate_content_index",
    "validate_in_memory_route_bindings",
    "validate_parent_ledger",
    "validate_preflight",
    "validate_pristine_or_label_free_recovery",
    "write_content_index",
    "write_terminal_reports",
)
