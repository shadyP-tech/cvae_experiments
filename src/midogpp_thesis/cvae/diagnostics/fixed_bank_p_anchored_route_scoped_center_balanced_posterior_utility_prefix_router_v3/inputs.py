"""Exact-six v3 admission and label-free consumed-test cache loading."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ....data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from ....data.features.stage70_test_cache.contracts import (
    CACHE_ARTIFACT_ID as UNDERLYING_CACHE_ARTIFACT_ID,
    CACHE_NAME as UNDERLYING_CACHE_NAME,
    REPRESENTATION_ID,
)
from ....data.features.stage70_test_cache.validation import (
    load_validated_stage70_test_cache,
)
from ...expert_bank.uniform_b_v2_promotion import (
    load_promotion_config,
    validate_promoted_bank,
)
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .constants import (
    CENTERS,
    EXPECTED_TOTAL_CASE_COUNT,
    FAILED_V2_EXPERIMENT_ID,
    FAILED_V2_OUTPUT_ARTIFACT_ID,
    QUARANTINED_V1_EXPERIMENT_ID,
    QUARANTINED_V1_OUTPUT_ARTIFACT_ID,
    REPAIR_BASE_COMMIT,
    REPAIR_CODE_IDENTITY,
    V1_FAILURE_EXCEPTION,
    V1_FAILURE_PHASE,
    V2_FAILURE_EXCEPTION,
    V2_FAILURE_PHASE,
)
from .experiment_contracts import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_SCOPE,
    AUTHORIZED_INPUT_ROLES,
    CLAIM_ROLE,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPERIMENT_ID,
    FORBIDDEN_INPUT_FRAGMENTS,
    FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_SCHEMA_VERSION,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    WORKSPACE_ALIAS_PLACEHOLDER_IDS,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .source_seal import validate_repair_source_seal
from .workspace_inputs import (
    validate_active_workspace_binding,
    validate_workspace_provenance,
)


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    parent_ledger: Mapping[str, object]
    ledger_amendment: Mapping[str, object]


def assert_input_fence(config: object) -> None:
    """Reject predecessor results, checkpoints, amendments, and scratch paths."""

    if (
        tuple(getattr(config, "input_artifact_ids")) != INPUT_ARTIFACT_IDS
        or len(INPUT_ARTIFACT_IDS) != 6
        or len(set(INPUT_ARTIFACT_IDS)) != 6
        or getattr(config, "experiment_id") != EXPERIMENT_ID
        or getattr(config, "output_artifact_id") != OUTPUT_ARTIFACT_ID
        or OUTPUT_ARTIFACT_ID in INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("CBPUPR requires exactly six fenced inputs.")
    values = (
        *(str(value) for value in getattr(config, "input_artifact_ids")),
        str(getattr(config, "expert_bank_root")),
        str(getattr(config, "generation_lock_root")),
        str(getattr(config, "test_cache_root")),
        str(getattr(config, "test_manifest_path")),
        str(getattr(config, "test_consumption_ledger_path")),
        str(getattr(config, "ledger_amendment_path")),
    )
    for value in values:
        folded = value.casefold()
        if any(fragment.casefold() in folded for fragment in FORBIDDEN_INPUT_FRAGMENTS):
            raise ProtocolError("CBPUPR rejected predecessor diagnostic input.")
        if any(
            token.casefold() in folded
            for token in FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS
        ):
            raise ProtocolError("CBPUPR rejected numbered-stage output.")


def load_label_free_test_frame(config: object) -> LabelFreeTestFrame:
    assert_input_fence(config)
    cache = load_validated_stage70_test_cache(Path(getattr(config, "test_cache_root")))
    summary = dict(cache.summary)
    if (
        summary.get("status") != "PASS"
        or summary.get("manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or summary.get("row_count") != EXPECTED_TEST_ROWS
        or summary.get("rows_by_center") != dict(EXPECTED_TEST_ROWS_BY_CENTER)
        or summary.get("content_hash") != EXPECTED_TEST_CACHE_CONTENT_HASH
        or summary.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or EXPECTED_TEST_CACHE_SEMANTIC_ID != UNDERLYING_CACHE_NAME
        or EXPECTED_TEST_CACHE_REPRESENTATION_ID != REPRESENTATION_ID
        or summary.get("fresh_evidence") is not False
    ):
        raise ProtocolError("CBPUPR consumed-test cache drifted.")
    rows: list[TestRowIdentity] = []
    embeddings: list[np.ndarray] = []
    by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
    shard_hashes: dict[str, str] = {}
    ordinal = 0
    for center in CENTERS:
        shard = cache.load_center(center)
        center_rows: list[TestRowIdentity] = []
        for row_id, manifest_index, case_id in zip(
            shard.evaluation_row_ids,
            shard.contract_row_indices,
            shard.case_ids,
            strict=True,
        ):
            identity = TestRowIdentity(
                ordinal, int(manifest_index), str(row_id), str(case_id), center
            )
            rows.append(identity)
            center_rows.append(identity)
            ordinal += 1
        embeddings.append(np.asarray(shard.embeddings, dtype=np.float32))
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard.shard_sha256
    if len({(row.center, row.case_id) for row in rows}) != EXPECTED_TOTAL_CASE_COUNT:
        raise ProtocolError("CBPUPR case coverage drifted.")
    binding = {
        "schema_version": "fixed_bank_cbpupr_test_cache_lineage_v1",
        "cache_alias_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "manifest_alias_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
        "underlying_cache_artifact_id": UNDERLYING_CACHE_ARTIFACT_ID,
        "underlying_cache_name": UNDERLYING_CACHE_NAME,
        "representation_id": REPRESENTATION_ID,
        "split": "test",
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "row_count": len(rows),
        "rows_by_center": dict(EXPECTED_TEST_ROWS_BY_CENTER),
        "feature_dim": COMMON_OUTPUT_DIM,
        "cache_content_hash": summary["content_hash"],
        "row_order_hash": summary["row_order_hash"],
        "shard_sha256_by_center": shard_hashes,
        "labels_persisted": False,
        "manifest_opened": False,
        "sample_paths_persisted": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed": False,
    }
    return LabelFreeTestFrame(
        np.ascontiguousarray(np.concatenate(embeddings), dtype=np.float32),
        tuple(rows),
        by_center,
        binding,
    )


def load_validated_locks(config: object) -> ValidatedLocks:
    assert_input_fence(config)
    generation_root = Path(getattr(config, "generation_lock_root"))
    generation_config = load_generation_lock_config(
        generation_root / "config.resolved.yaml"
    )
    validate_generation_bundle(generation_root, config=generation_config)
    generation = read_generation_lock(
        generation_root / "manifests/generation_lock.json"
    )
    if (
        generation.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or generation.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
    ):
        raise ProtocolError("CBPUPR GenerationLock drifted.")
    parent, amendment = _load_ledger_chain(config)
    return ValidatedLocks(generation, parent, amendment)


def validate_pre_gpu_firewall(
    config: object,
    frame: LabelFreeTestFrame,
    locks: ValidatedLocks | None = None,
) -> Mapping[str, object]:
    protocol = dict(getattr(config, "protocol"))
    source_seal = validate_repair_source_seal(
        expected_manifest_sha256=protocol.get("repair_source_manifest_sha256"),
        expected_tree_sha256=protocol.get("repair_source_tree_sha256"),
    )
    if (
        protocol.get("repair_source_manifest_member")
        != source_seal.get("repair_source_manifest_member")
        or protocol.get("repair_source_member_count")
        != source_seal.get("repair_source_member_count")
        or protocol.get("repair_source_manifest_checked_pre_gpu") is not True
        or protocol.get("repair_source_identity_persisted_in_protocol_manifest")
        is not True
    ):
        raise ProtocolError("CBPUPR repair source contract drifted.")
    validated = locks or load_validated_locks(config)
    bank_root = Path(getattr(config, "expert_bank_root"))
    promotion = load_promotion_config(bank_root / "config.resolved.yaml")
    checks = validate_promoted_bank(bank_root, config=promotion, allow_pending=False)
    amendment = validated.ledger_amendment
    forbidden = (
        "quarantined_v1_output_used",
        "quarantined_v1_scratch_or_checkpoint_used",
        "quarantined_v1_terminal_outputs_used",
        "prior_v1_label_capability_history_used",
        "prior_v1_amendment_used",
        "failed_v2_output_used",
        "failed_v2_scratch_or_checkpoint_used",
        "failed_v2_preterminal_outputs_used",
        "prior_v2_label_capability_history_used",
        "prior_v2_amendment_used",
        "prior_v2_execution_authorization_reused",
        "previous_stage90_outputs_used",
        "previous_stage90_amendments_used",
        "previous_prediction_surfaces_used",
        "previous_stage90_scratch_or_checkpoints_used",
    )
    if (
        checks.get("status") != "PASS"
        or checks.get("all_experts_source_only") is not True
        or sha256_file(Path(getattr(config, "test_manifest_path")))
        != EXPECTED_MANIFEST_SHA256
        or frame.cache_binding.get("manifest_opened") is not False
        or frame.cache_binding.get(
            "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed"
        )
        is not False
        or any(amendment.get(key) is not False for key in forbidden)
    ):
        raise ProtocolError("CBPUPR pre-GPU firewall failed.")
    return MappingProxyType(
        {
            "status": "PASS",
            "evaluation_split": "test",
            "test_split_previously_consumed": True,
            "fresh_evidence": False,
            "target_labels_opened": False,
            "target_expert_used": False,
            "predecessor_stage90_artifact_prediction_checkpoint_or_scratch_consumed": False,
            **dict(source_seal),
            "gpu_work_authorized": True,
        }
    )


def _load_ledger_chain(
    config: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    parent_path = Path(getattr(config, "test_consumption_ledger_path"))
    amendment_path = Path(getattr(config, "ledger_amendment_path"))
    parent = _read_json(parent_path)
    amendment = _read_json(amendment_path)
    protocol = dict(getattr(config, "protocol"))
    required_true = (
        "mechanical_repair_only",
        "scientific_protocol_unchanged_from_v1",
        "scientific_protocol_unchanged_from_v2",
        "canonical_row_order_repair_verified",
        "canonical_row_order_repair_from_v1_retained",
        "global_surface_lineage_repair_only",
        "global_surface_lineage_repair_verified",
        "outer_endpoint_job_carries_global_physical_surface_hash",
        "outer_endpoint_job_carries_center_surface_hash",
        "global_plan_hash_compared_only_to_global_job_hash",
        "center_surface_hash_compared_only_to_prepared_center_hash",
        "v1_output_quarantined",
        "v1_target_terminal_capability_had_opened",
        "v1_terminal_outputs_had_persisted",
        "v1_canonical_row_order_drift_recorded",
        "v2_failure_preterminal",
        "v2_global_surface_lineage_drift_recorded",
        "preterminal_closed_world_validation_required",
        "preterminal_parent_validation_required",
        "preterminal_validation_attested_before_terminal_labels",
        "terminal_label_loader_forbidden_before_preterminal_attestation",
        "execution_authorized",
        "authorization_is_separate_from_implementation_request",
        "single_use_execution_identity",
        "split_previously_consumed",
        "method_development_is_posthoc",
        "prior_consumed_test_findings_informed_method_design",
        "source_experts_frozen",
        "generation_lock_frozen",
        "physical_probability_surface_recomputed_from_original_inputs",
        "target_support_labels_used",
        "target_support_labels_are_non_deployable_consumed_test_support",
        "all_target_and_pseudo_candidates_sealed_before_pseudo_evaluation",
        "all_replays_and_calibrations_sealed_before_target_decisions",
        "all_target_decisions_and_aggregate_predictions_sealed_before_terminal_labels",
        "posterior_expected_utility_uses_posterior_augmented_center_denominators",
        "pseudo_outer_H_frozen_label_free_expert_fingerprint_covariates_present",
        "pseudo_outer_H_excluded_from_actionable_endpoint_source_selection",
        "pseudo_outer_H_and_J_excluded_from_donor_calibration",
        "exact_P_fallback_required",
        "structural_transport_lineage_is_authorization_gate",
        "zero_MAD_numeric_transport_division_forbidden",
        "scratch_reuse_forbidden",
        "cross_run_recovery_forbidden",
        "two_fresh_process_validation_required",
        "repair_source_manifest_required",
        "repair_source_manifest_checked_pre_gpu",
        "repair_source_identity_persisted_in_protocol_manifest",
    )
    required_false = (
        "fresh_evidence",
        "scientific_method_changed_from_v1",
        "scientific_method_changed_from_v2",
        "canonical_row_order_repair_only",
        "v1_failure_preterminal",
        "v1_final_validation_passed",
        "quarantined_v1_output_used",
        "quarantined_v1_scratch_or_checkpoint_used",
        "quarantined_v1_terminal_outputs_used",
        "prior_v1_label_capability_history_used",
        "prior_v1_amendment_used",
        "preexisting_v2_semantic_artifacts_used",
        "v2_target_terminal_access_intent_persisted",
        "v2_target_terminal_capability_had_opened",
        "v2_terminal_outputs_had_persisted",
        "v2_final_validation_passed",
        "failed_v2_output_used",
        "failed_v2_scratch_or_checkpoint_used",
        "failed_v2_preterminal_outputs_used",
        "prior_v2_label_capability_history_used",
        "prior_v2_amendment_used",
        "prior_v2_execution_authorization_reused",
        "prior_consumed_test_bytes_used_as_scientific_inputs",
        "previous_prediction_surfaces_used",
        "previous_stage90_outputs_used",
        "previous_stage90_amendments_used",
        "previous_stage90_scratch_or_checkpoints_used",
        "stage50_stage60_or_stage70_result_used",
        "target_support_labels_may_update_source_experts_or_shared_models",
        "pseudo_outer_H_support_rows_or_labels_enter_posterior_fit_or_normalization",
        "pseudo_posterior_is_outer_H_covariate_invariant",
        "outer_case_labels_enter_own_route",
        "pseudo_case_labels_enter_own_candidate",
        "target_expert_used",
        "shared_model_updated_with_target_labels",
        "target_evaluation_labels_used_before_route_seal",
        "terminal_labels_from_this_run_used_to_define_policy",
        "numeric_transport_is_authorization_gate",
        "finite_sample_conformal_coverage_claimed",
        "confidence_bound_claimed",
        "calibrated_uncertainty_claimed",
        "nominal_significance_claimed",
        "routing_success_claimed",
        "routing_quality_claimed",
        "downstream_utility_claimed",
        "nelbo_compatibility_claimed",
        "expert_selection_claimed",
        "deployment_claimed",
        "promotion_eligible",
        "may_authorize_routing",
        "may_authorize_policy_update",
        "may_authorize_promotion",
        "may_authorize_deployment",
        "may_feed_stage50",
        "may_feed_stage60",
        "may_feed_stage70",
        "may_feed_another_stage90",
        "may_feed_another_experiment",
        "generic_consumer_authorized",
        "raw_labels_may_be_persisted",
        "raw_sample_or_image_paths_may_be_persisted",
    )
    if (
        sha256_file(parent_path) != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or sha256_file(amendment_path)
        != str(getattr(config, "expected_ledger_amendment_sha256"))
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("schema_version")
        != LEDGER_AMENDMENT_SCHEMA_VERSION
        or amendment.get("parent_artifact_id")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or amendment.get("parent_member") != "reports/test_consumption_ledger.json"
        or amendment.get("parent_sha256") != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or amendment.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or amendment.get("authorization_scope") != AUTHORIZATION_SCOPE
        or amendment.get("authorization_basis") != AUTHORIZATION_BASIS
        or amendment.get("authorized_input_roles") != list(AUTHORIZED_INPUT_ROLES)
        or amendment.get("fresh_v3_workspace_aliases")
        != list(WORKSPACE_ALIAS_PLACEHOLDER_IDS)
        or amendment.get("repair_code_identity") != REPAIR_CODE_IDENTITY
        or amendment.get("repair_base_commit") != REPAIR_BASE_COMMIT
        or amendment.get("repair_source_manifest_member")
        != protocol.get("repair_source_manifest_member")
        or amendment.get("repair_source_manifest_sha256")
        != protocol.get("repair_source_manifest_sha256")
        or amendment.get("repair_source_tree_sha256")
        != protocol.get("repair_source_tree_sha256")
        or amendment.get("repair_source_member_count")
        != protocol.get("repair_source_member_count")
        or amendment.get("quarantined_v1_experiment_id")
        != QUARANTINED_V1_EXPERIMENT_ID
        or amendment.get("quarantined_v1_output_artifact_id")
        != QUARANTINED_V1_OUTPUT_ARTIFACT_ID
        or amendment.get("v1_failure_phase")
        != V1_FAILURE_PHASE
        or amendment.get("v1_failure_exception")
        != V1_FAILURE_EXCEPTION
        or amendment.get("failed_v2_experiment_id") != FAILED_V2_EXPERIMENT_ID
        or amendment.get("failed_v2_output_artifact_id")
        != FAILED_V2_OUTPUT_ARTIFACT_ID
        or amendment.get("v2_failure_phase") != V2_FAILURE_PHASE
        or amendment.get("v2_failure_exception") != V2_FAILURE_EXCEPTION
        or amendment.get("canonical_physical_row_order")
        != "lexicographic_case_id_then_sample_id"
        or amendment.get("posterior_prediction_and_model_row_order")
        != "lexicographic_case_id_then_sample_id"
        or amendment.get("outer_endpoint_global_surface_hash_field")
        != "physical_surface_hash"
        or amendment.get("outer_endpoint_center_surface_hash_field")
        != "prepared.surface.surface_hash"
        or amendment.get("preterminal_fresh_process_validation_count") != 2
        or amendment.get("final_fresh_process_validation_count") != 2
        or amendment.get("claim_role") != CLAIM_ROLE
        or amendment.get("held_unit_count") != EXPECTED_TOTAL_CASE_COUNT
        or amendment.get("outer_route_count") != EXPECTED_TOTAL_CASE_COUNT
        or amendment.get("ordered_H_J_pair_count") != 72
        or amendment.get("pseudo_route_scope")
        != (
            "J_minus_d_posterior_reuse_with_outer_H_role_exclusion_not_"
            "covariate_exclusion"
        )
        or amendment.get("primary_method_id") != "CBPUPR_UNIFIED_PREFIX"
        or amendment.get("protected_fallback_method_id") != "P_PROTECTED"
        or amendment.get("exact_P_fallback_storage_dtype") != "float32"
        or amendment.get("posterior_expected_utility_coordinates")
        != [
            "favorable_bacc_contribution",
            "favorable_brier_contribution",
            "favorable_log_loss_contribution",
        ]
        or any(amendment.get(key) is not True for key in required_true)
        or any(amendment.get(key) is not False for key in required_false)
    ):
        raise ProtocolError("CBPUPR consumption-ledger chain drifted.")
    return MappingProxyType(parent), MappingProxyType(amendment)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read CBPUPR JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("CBPUPR JSON must be an object.")
    return value


__all__ = (
    "LabelFreeTestFrame",
    "TestRowIdentity",
    "ValidatedLocks",
    "assert_input_fence",
    "load_label_free_test_frame",
    "load_validated_locks",
    "validate_active_workspace_binding",
    "validate_pre_gpu_firewall",
    "validate_workspace_provenance",
)
