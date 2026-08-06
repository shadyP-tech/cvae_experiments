"""Deterministic manifest and report payloads for reconstruction validation."""

from __future__ import annotations

from typing import Mapping

from ....common.hashing import stable_hash
from .config import MetadataCompatibilityConfig
from .contracts import (
    CLAIM_SCOPE,
    COMPATIBILITY_DECISION,
    DOMAIN_MAPPING_SHA256,
    EXPERIMENT_ID,
    INPUT_ARTIFACT_ID,
    MetadataCompatibilityLock,
    ORDERED_AXES,
    PUBLICATION_STATE,
)


def protocol_manifest_payload(
    config: MetadataCompatibilityConfig,
    metadata_profile_lock: Mapping[str, object],
    compatibility_lock: MetadataCompatibilityLock,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_protocol_v1",
        "experiment_id": EXPERIMENT_ID,
        "claim_scope": CLAIM_SCOPE,
        "config_contract_hash": config.contract_hash,
        "input_artifact_ids": [INPUT_ARTIFACT_ID],
        "domain_mapping_sha256": DOMAIN_MAPPING_SHA256,
        "metadata_profile_lock_hash": metadata_profile_lock[
            "metadata_profile_lock_hash"
        ],
        "compatibility_lock_hash": compatibility_lock.compatibility_lock_hash,
        "metadata_profile_table_hash": (
            compatibility_lock.metadata_profile_table_hash
        ),
        "compatibility_score_table_hash": (
            compatibility_lock.compatibility_score_table_hash
        ),
        "ordered_axes": list(ORDERED_AXES),
        "parsed_input_fields": ["domain_axis", "domain_name_to_id"],
        "all_other_input_fields_ignored": True,
        "target_identity_binds_profile_only": True,
        "scorer_uses_profile_values_only": True,
        "center_or_domain_ids_passed_to_scorer": False,
        "all_ordered_target_excluded_pairs_scored": True,
        "metadata_score_is_proxy_only": True,
        "lock_only": True,
        "target_sample_rows_used": False,
        "target_support_used": False,
        "target_labels_used": False,
        "nelbo_computed": False,
        "true_utility_computed": False,
        "ranking_performed": False,
        "selection_performed": False,
        "weighting_performed": False,
        "routing_quality_claimed": False,
        "may_feed_deployable_selection": True,
    }
    payload["protocol_hash"] = stable_hash(payload)
    return payload


def compatibility_decision_payload(
    compatibility_lock: MetadataCompatibilityLock,
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_decision_v1",
        "decision": COMPATIBILITY_DECISION,
        "publication_state": PUBLICATION_STATE,
        "claim_scope": CLAIM_SCOPE,
        "compatibility_lock_hash": compatibility_lock.compatibility_lock_hash,
        "metadata_score_is_proxy_only": True,
        "deployable_policy_input": True,
        "routing_policy_emitted": False,
        "ranking_performed": False,
        "selection_performed": False,
        "weighting_performed": False,
        "routing_quality_claimed": False,
        "downstream_utility_claimed": False,
        "next_required_evidence": (
            "freeze_a_separate_stage60_policy_then_evaluate_true_utility_only_in_"
            "the_authorized_downstream_protocol"
        ),
    }


def leakage_report_payload() -> dict[str, object]:
    return {
        "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_leakage_v1",
        "status": "PASS",
        "sole_input_artifact_id": INPUT_ARTIFACT_ID,
        "parsed_input_fields": ["domain_axis", "domain_name_to_id"],
        "all_other_input_fields_ignored": True,
        "target_identity_binds_profile_only": True,
        "target_identity_used_by_scorer": False,
        "profile_values_used_by_scorer": True,
        "target_expert_excluded_in_every_score": True,
        "center_4_emitted": False,
        "sample_manifest_used": False,
        "sample_ids_used": False,
        "sample_paths_used": False,
        "domain_counts_used": False,
        "class_labels_used": False,
        "target_labels_used": False,
        "target_evaluation_labels_used": False,
        "support_set_used": False,
        "nelbo_used": False,
        "stage20_input_used": False,
        "stage50_input_used": False,
        "stage90_input_used": False,
        "ranking_performed": False,
        "selection_performed": False,
        "weighting_performed": False,
        "routing_policy_emitted": False,
        "true_utility_computed": False,
        "routing_quality_claimed": False,
        "downstream_utility_computed": False,
    }


def run_state_payload(status: str) -> dict[str, object]:
    if status not in {"RUNNING", "COMPLETE", "FAILED"}:
        raise ValueError(f"Invalid metadata compatibility run state: {status!r}.")
    return {
        "schema_version": "midogpp_uniform_b_v2_metadata_compatibility_run_state_v1",
        "status": status,
        "claim_scope": CLAIM_SCOPE,
    }


__all__ = (
    "compatibility_decision_payload",
    "leakage_report_payload",
    "protocol_manifest_payload",
    "run_state_payload",
)
