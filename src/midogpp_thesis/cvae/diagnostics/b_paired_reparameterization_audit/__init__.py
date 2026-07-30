"""Public, import-light contracts for the paired reparameterization audit."""

from __future__ import annotations

from .config import (
    AUDIT_CANDIDATES,
    AUDIT_CENTERS,
    CONTRACT_ARTIFACT_ID,
    CONTROLLED_CANDIDATES,
    FEATURE_CACHE_ARTIFACT_ID,
    FIXED_ANTITHETIC_CANDIDATE,
    FIXED_ONE_EPSILON_CANDIDATE,
    INITIALIZATION_SEEDS,
    LEGACY_CANDIDATE,
    SNAPSHOT_ARTIFACT_ID,
    SNAPSHOT_INPUT_URI,
    AuditConfig,
    ClaimFirewall,
    DecisionThresholds,
    FrozenBRecipe,
    HistoricalLineage,
    LegacyExpectation,
    SnapshotBuildConfig,
    audit_config_from_mapping,
    load_audit_config,
    load_snapshot_build_config,
    snapshot_build_config_from_mapping,
)
from .protocol import (
    AuditKeyRecord,
    assert_candidate_use,
    build_key_record,
    comparison_pairs,
    compute_key_hash,
    compute_pair_id,
    key_inventory_hash,
    key_record_from_mapping,
    validate_key_inventory,
    validate_key_record,
)
from .snapshot import (
    HASH_PROMOTED,
    PENDING_HASH_PROMOTION,
    ArrayBinding,
    AuditSnapshot,
    CenterPreparedData,
    ContentEntry,
    PreparedPartition,
    build_snapshot,
    load_snapshot,
    snapshot_from_mapping,
    snapshot_manifest_hash,
    validate_snapshot,
)
from .trace import (
    EpsilonTraceLedger,
    EpsilonTraceSpec,
    LoadedEpsilonTrace,
    load_epsilon_trace,
    trace_content_hash,
)


def __getattr__(name: str) -> object:
    """Keep CLI discovery light while exposing execution entrypoints."""

    if name == "build_snapshot_from_config":
        from .snapshot_builder import build_snapshot_from_config

        return build_snapshot_from_config
    if name == "run_b_paired_reparameterization_audit":
        from .runner import run_b_paired_reparameterization_audit

        return run_b_paired_reparameterization_audit
    if name in {"assert_valid_audit_bundle", "validate_audit_bundle"}:
        from .validation import assert_valid_audit_bundle, validate_audit_bundle

        return {
            "assert_valid_audit_bundle": assert_valid_audit_bundle,
            "validate_audit_bundle": validate_audit_bundle,
        }[name]
    raise AttributeError(name)

__all__ = (
    "AUDIT_CANDIDATES",
    "AUDIT_CENTERS",
    "CONTRACT_ARTIFACT_ID",
    "CONTROLLED_CANDIDATES",
    "FEATURE_CACHE_ARTIFACT_ID",
    "FIXED_ANTITHETIC_CANDIDATE",
    "FIXED_ONE_EPSILON_CANDIDATE",
    "HASH_PROMOTED",
    "INITIALIZATION_SEEDS",
    "LEGACY_CANDIDATE",
    "PENDING_HASH_PROMOTION",
    "SNAPSHOT_ARTIFACT_ID",
    "SNAPSHOT_INPUT_URI",
    "ArrayBinding",
    "AuditConfig",
    "AuditKeyRecord",
    "AuditSnapshot",
    "ClaimFirewall",
    "CenterPreparedData",
    "ContentEntry",
    "DecisionThresholds",
    "EpsilonTraceLedger",
    "EpsilonTraceSpec",
    "FrozenBRecipe",
    "HistoricalLineage",
    "LegacyExpectation",
    "LoadedEpsilonTrace",
    "PreparedPartition",
    "SnapshotBuildConfig",
    "assert_candidate_use",
    "audit_config_from_mapping",
    "build_key_record",
    "build_snapshot",
    "build_snapshot_from_config",
    "comparison_pairs",
    "compute_key_hash",
    "compute_pair_id",
    "key_inventory_hash",
    "key_record_from_mapping",
    "load_audit_config",
    "load_epsilon_trace",
    "load_snapshot",
    "load_snapshot_build_config",
    "run_b_paired_reparameterization_audit",
    "snapshot_from_mapping",
    "snapshot_manifest_hash",
    "snapshot_build_config_from_mapping",
    "trace_content_hash",
    "validate_key_inventory",
    "validate_key_record",
    "validate_snapshot",
    "validate_audit_bundle",
    "assert_valid_audit_bundle",
)
