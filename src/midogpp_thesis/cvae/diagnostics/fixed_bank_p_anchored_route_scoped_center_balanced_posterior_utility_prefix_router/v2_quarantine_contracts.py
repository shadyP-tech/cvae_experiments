"""Frozen phase, inventory, and receipt contracts for CBPUPR v2 quarantine."""

from __future__ import annotations

import re

from .bundle import ALLOWED_DIRECTORIES
from .constants import EXPECTED_OUTER_PLAN_COUNT, EXPECTED_PSEUDO_ROUTE_COUNT
from .terminal_access_journal import TERMINAL_ACCESS_JOURNAL_ORDER


V2_TERMINAL_PHASE = "TERMINAL_LABELS_METRICS_AND_CONTROLS"
V2_FINAL_PHASE = "CONTENT_AND_TWO_FRESH_PROCESS_VALIDATION"
V2_TERMINAL_PERSISTENCE_ORDER = (
    *TERMINAL_ACCESS_JOURNAL_ORDER,
    "reports/label_capability_report.json",
    "tables/terminal_method_metrics.json",
    "tables/terminal_center_contrasts.json",
    "tables/terminal_case_oracles.json",
    "manifests/terminal_evaluation_seal.json",
    "reports/diagnostic_summary.json",
    "reports/leakage_report.json",
    "reports/publication_decision.json",
    "reports/runtime_summary.json",
)
V2_FINAL_PERSISTENCE_ORDER = (
    "manifests/content_index.json",
    "reports/fresh_process_attestation.json",
    "reports/validation_report.json",
)

V2_TERMINAL_FAILURE_ARTIFACT_DIRECTORIES = ALLOWED_DIRECTORIES
V2_TERMINAL_FAILURE_SCRATCH_DIRECTORIES = frozenset(
    {
        "prediction_cache",
        "prediction_cache/checkpoints",
        "source_generation",
        "source_generation/arrays",
        "source_generation/checkpoints",
        "source_generation/manifests",
    }
)
V2_TERMINAL_FAILURE_SCRATCH_FILES = frozenset(
    {
        "source_generation/arrays/frozen_source_streams.npy",
        "source_generation/manifests/frozen_source_stream_index.json",
        "source_generation/manifests/frozen_source_stream_lock.json",
    }
)
SCRATCH_TO_ARTIFACT_SOURCE_MEMBERS = {
    "source_generation/arrays/frozen_source_streams.npy": (
        "arrays/frozen_source_streams.npy"
    ),
    "source_generation/manifests/frozen_source_stream_index.json": (
        "manifests/frozen_source_stream_index.json"
    ),
    "source_generation/manifests/frozen_source_stream_lock.json": (
        "manifests/frozen_source_stream_lock.json"
    ),
}

QUARANTINE_SUFFIX = re.compile(
    r"\.quarantine-v2-terminal-failure-([0-9]{8}T[0-9]{6}Z)"
)
RUN_STATE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "phase",
        "error",
        "error_class",
        "updated_at_utc",
        "cross_run_recovery_allowed",
        "terminal_recovery_allowed",
    }
)
PROTOCOL_MANIFEST_BASE_KEYS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "output_artifact_id",
        "config_contract_hash",
        "protocol_contract_hash",
        "stage",
        "claim_scope",
        "claim_role",
        "input_artifact_hashes",
        "cache_binding_hash",
        "pre_gpu_firewall",
        "exact_six_original_inputs",
        "previous_stage90_output_or_checkpoint_used",
        "test_split_previously_consumed",
        "fresh_evidence",
        "publication_status",
        "protocol_manifest_hash",
    }
)
CAPABILITY_KEYS = frozenset(
    {
        "schema_version",
        "plan_seal_hash",
        "event_count",
        "events",
        "target_candidate_seal_complete",
        "pre_evaluation_seal_complete",
        "pseudo_evaluation_route_count",
        "calibration_seal_complete",
        "decision_count",
        "aggregate_seal_complete",
        "terminal_opened",
        "raw_labels_persisted",
        "audit_hash",
    }
)
EXPECTED_PRETERMINAL_CAPABILITY_EVENT_COUNT = (
    9 * 8 + 9 * 8 * 7 + EXPECTED_OUTER_PLAN_COUNT + EXPECTED_PSEUDO_ROUTE_COUNT
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

AUDIT_SCHEMA = "fixed_bank_cbpupr_v2_terminal_failure_quarantine_audit_v1"
RECEIPT_SCHEMA = "fixed_bank_cbpupr_v2_terminal_failure_quarantine_receipt_v1"
ELIGIBLE_NEXT_ACTION = "PRESERVE_ONLY_NO_REUSE_NO_RERUN"


__all__ = tuple(name for name in globals() if name.isupper())
