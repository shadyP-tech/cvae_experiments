"""Pure construction and reading of metadata profile and compatibility locks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from .config import MetadataCompatibilityConfig
from .contracts import (
    CLAIM_SCOPE,
    DOMAIN_AXIS,
    DOMAIN_MAPPING_MEMBER,
    DOMAIN_MAPPING_SHA256,
    ELIGIBLE_CENTERS,
    EXCLUDED_CENTERS,
    EXPECTED_COMPATIBILITY_LOCK_HASH,
    EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH,
    EXPECTED_CONFIG_CONTRACT_HASH,
    EXPECTED_METADATA_PROFILE_LOCK_HASH,
    EXPECTED_METADATA_PROFILE_TABLE_HASH,
    EXPECTED_PROFILE_COUNT,
    EXPECTED_SCORE_COUNT,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_ID,
    MetadataCompatibilityLock,
    MetadataProfile,
    CompatibilityScore,
    ORDERED_AXES,
    SCORING_FAMILY,
    SCORING_NAMESPACE,
    SOURCES_PER_TARGET,
)
from .profiles import metadata_profile_rows
from .scoring import compatibility_score_table_hash, metadata_profile_table_hash


def build_metadata_profile_lock(
    config: MetadataCompatibilityConfig,
    profiles: Mapping[str, MetadataProfile],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_metadata_profile_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "input_artifact_id": INPUT_ARTIFACT_ID,
        "input_member": DOMAIN_MAPPING_MEMBER,
        "domain_mapping_sha256": config.expected_domain_mapping_sha256,
        "domain_axis": DOMAIN_AXIS,
        "ordered_axes": list(ORDERED_AXES),
        "eligible_centers": list(ELIGIBLE_CENTERS),
        "excluded_centers": list(EXCLUDED_CENTERS),
        "profile_count": EXPECTED_PROFILE_COUNT,
        "profiles": list(metadata_profile_rows(profiles)),
        "metadata_profile_table_hash": metadata_profile_table_hash(profiles),
        "parsed_input_fields": ["domain_axis", "domain_name_to_id"],
        "all_other_input_fields_ignored": True,
        "profile_values_only": True,
        "center_4_profile_emitted": False,
    }
    payload["metadata_profile_lock_hash"] = stable_hash(payload)
    if payload["metadata_profile_lock_hash"] != EXPECTED_METADATA_PROFILE_LOCK_HASH:
        raise ProtocolError("Metadata profile-lock semantic identity drifted.")
    return payload


def build_compatibility_lock(
    config: MetadataCompatibilityConfig,
    metadata_profile_lock: Mapping[str, object],
    scores: tuple[CompatibilityScore, ...],
) -> MetadataCompatibilityLock:
    payload: dict[str, object] = {
        "schema_version": (
            "midogpp_uniform_b_v2_metadata_exact_match_compatibility_lock_v1"
        ),
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "input_artifact_id": INPUT_ARTIFACT_ID,
        "domain_mapping_sha256": config.expected_domain_mapping_sha256,
        "metadata_profile_lock_hash": metadata_profile_lock.get(
            "metadata_profile_lock_hash"
        ),
        "metadata_profile_table_hash": metadata_profile_lock.get(
            "metadata_profile_table_hash"
        ),
        "compatibility_score_table_hash": compatibility_score_table_hash(scores),
        "scoring_family": SCORING_FAMILY,
        "scoring_namespace": SCORING_NAMESPACE,
        "ordered_axes": list(ORDERED_AXES),
        "component_weights": {axis: 1 for axis in ORDERED_AXES},
        "scorer_inputs": "metadata_profile_values_only",
        "center_or_domain_ids_passed_to_scorer": False,
        "directionality": "all_ordered_target_source_pairs",
        "target_expert_excluded": True,
        "eligible_target_count": len(ELIGIBLE_CENTERS),
        "sources_per_target": SOURCES_PER_TARGET,
        "ordered_score_count": EXPECTED_SCORE_COUNT,
        "minimum_score": 0,
        "maximum_score": len(ORDERED_AXES),
        "metadata_score_is_proxy_only": True,
        "ranking_performed": False,
        "selection_performed": False,
        "weighting_performed": False,
        "nelbo_computed": False,
        "true_utility_computed": False,
    }
    payload["compatibility_lock_hash"] = stable_hash(payload)
    lock = MetadataCompatibilityLock(payload)
    if lock.compatibility_lock_hash != EXPECTED_COMPATIBILITY_LOCK_HASH:
        raise ProtocolError("Metadata compatibility-lock semantic identity drifted.")
    if (
        lock.metadata_profile_table_hash != EXPECTED_METADATA_PROFILE_TABLE_HASH
    ) or (
        lock.compatibility_score_table_hash
        != EXPECTED_COMPATIBILITY_SCORE_TABLE_HASH
    ):
        raise ProtocolError("Metadata compatibility lock table identity drifted.")
    return lock


def read_compatibility_lock(path: str | Path) -> MetadataCompatibilityLock:
    source = Path(path)
    if source.is_dir():
        source = source / "manifests/compatibility_lock.json"
    payload = _json(source, "metadata compatibility lock")
    expected_fields = {
        "schema_version",
        "experiment_id",
        "claim_scope",
        "config_contract_hash",
        "input_artifact_id",
        "domain_mapping_sha256",
        "metadata_profile_lock_hash",
        "metadata_profile_table_hash",
        "compatibility_score_table_hash",
        "scoring_family",
        "scoring_namespace",
        "ordered_axes",
        "component_weights",
        "scorer_inputs",
        "center_or_domain_ids_passed_to_scorer",
        "directionality",
        "target_expert_excluded",
        "eligible_target_count",
        "sources_per_target",
        "ordered_score_count",
        "minimum_score",
        "maximum_score",
        "metadata_score_is_proxy_only",
        "ranking_performed",
        "selection_performed",
        "weighting_performed",
        "nelbo_computed",
        "true_utility_computed",
        "compatibility_lock_hash",
    }
    if set(payload) != expected_fields:
        raise ProtocolError("Metadata compatibility-lock schema drifted.")
    return MetadataCompatibilityLock(payload)


def read_metadata_profile_lock(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_dir():
        source = source / "manifests/metadata_profile_lock.json"
    payload = _json(source, "metadata profile lock")
    expected_fields = {
        "schema_version",
        "experiment_id",
        "claim_scope",
        "config_contract_hash",
        "input_artifact_id",
        "input_member",
        "domain_mapping_sha256",
        "domain_axis",
        "ordered_axes",
        "eligible_centers",
        "excluded_centers",
        "profile_count",
        "profiles",
        "metadata_profile_table_hash",
        "parsed_input_fields",
        "all_other_input_fields_ignored",
        "profile_values_only",
        "center_4_profile_emitted",
        "metadata_profile_lock_hash",
    }
    if set(payload) != expected_fields:
        raise ProtocolError("Metadata profile-lock schema drifted.")
    expected_semantics = {
        "schema_version": "midogpp_uniform_b_v2_metadata_profile_lock_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": EXPECTED_CONFIG_CONTRACT_HASH,
        "input_artifact_id": INPUT_ARTIFACT_ID,
        "input_member": DOMAIN_MAPPING_MEMBER,
        "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
        "domain_axis": DOMAIN_AXIS,
        "ordered_axes": list(ORDERED_AXES),
        "eligible_centers": list(ELIGIBLE_CENTERS),
        "excluded_centers": list(EXCLUDED_CENTERS),
        "profile_count": EXPECTED_PROFILE_COUNT,
        "metadata_profile_table_hash": EXPECTED_METADATA_PROFILE_TABLE_HASH,
        "parsed_input_fields": ["domain_axis", "domain_name_to_id"],
        "all_other_input_fields_ignored": True,
        "profile_values_only": True,
        "center_4_profile_emitted": False,
    }
    if any(payload.get(key) != value for key, value in expected_semantics.items()):
        raise ProtocolError("Metadata profile-lock semantic identity drifted.")
    observed = payload.get("metadata_profile_lock_hash")
    unhashed = {
        key: value for key, value in payload.items() if key != "metadata_profile_lock_hash"
    }
    if observed != stable_hash(unhashed):
        raise ProtocolError("Metadata profile-lock hash drifted.")
    if observed != EXPECTED_METADATA_PROFILE_LOCK_HASH:
        raise ProtocolError("Metadata profile-lock frozen identity drifted.")
    return json.loads(json.dumps(payload))


def _json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read {label}: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"{label.capitalize()} must be an object.")
    return payload


__all__ = (
    "build_compatibility_lock",
    "build_metadata_profile_lock",
    "read_compatibility_lock",
    "read_metadata_profile_lock",
)
