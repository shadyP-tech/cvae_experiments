"""Exact-six admission and label-free consumed-test cache loading.

This module deliberately has no dependency on another Stage-90 diagnostic.
Only original input capabilities and the neutral generation/cache contracts are
reachable before the physical probability seal.
"""

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
from .constants import CENTERS, EXPECTED_CASE_COUNTS_BY_CENTER
from .experiment_contracts import (
    AUTHORIZATION_SCOPE,
    CLAIM_ROLE,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
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
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .workspace_inputs import (
    validate_active_diagnostic_workspace_binding,
    validate_workspace_provenance,
)


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    parent_ledger: Mapping[str, object]
    ledger_amendment: Mapping[str, object]


def assert_input_fence(config: object) -> None:
    """Reject every prior result, prediction, checkpoint, and scratch input."""

    if (
        tuple(getattr(config, "input_artifact_ids")) != INPUT_ARTIFACT_IDS
        or len(INPUT_ARTIFACT_IDS) != 6
        or len(set(INPUT_ARTIFACT_IDS)) != 6
        or getattr(config, "experiment_id") != EXPERIMENT_ID
        or getattr(config, "output_artifact_id") != OUTPUT_ARTIFACT_ID
        or OUTPUT_ARTIFACT_ID in INPUT_ARTIFACT_IDS
    ):
        raise ProtocolError("Case-directional requires exactly six fenced inputs.")
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
            raise ProtocolError("Case-directional rejected prior diagnostic input.")
        if any(
            token.casefold() in folded
            for token in FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS
        ):
            raise ProtocolError("Case-directional rejected numbered-stage output.")


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
        raise ProtocolError("Case-directional consumed-test cache drifted.")
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
                ordinal,
                int(manifest_index),
                str(row_id),
                str(case_id),
                center,
            )
            rows.append(identity)
            center_rows.append(identity)
            ordinal += 1
        embeddings.append(np.asarray(shard.embeddings, dtype=np.float32))
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard.shard_sha256
    case_counts = {
        center: len({row.case_id for row in by_center[center]}) for center in CENTERS
    }
    if case_counts != dict(EXPECTED_CASE_COUNTS_BY_CENTER):
        raise ProtocolError("Case-directional case coverage drifted.")
    binding = {
        "schema_version": "fixed_bank_cdca_test_cache_lineage_v1",
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
        "previous_stage90_output_prediction_checkpoint_or_scratch_consumed": False,
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
        raise ProtocolError("Case-directional GenerationLock drifted.")
    parent, amendment = _load_ledger_chain(config)
    return ValidatedLocks(generation, parent, amendment)


def validate_pre_gpu_firewall(
    config: object,
    frame: LabelFreeTestFrame,
    locks: ValidatedLocks | None = None,
) -> Mapping[str, object]:
    validated = locks or load_validated_locks(config)
    bank_root = Path(getattr(config, "expert_bank_root"))
    promotion = load_promotion_config(bank_root / "config.resolved.yaml")
    checks = validate_promoted_bank(bank_root, config=promotion, allow_pending=False)
    amendment = validated.ledger_amendment
    if (
        checks.get("status") != "PASS"
        or checks.get("all_experts_source_only") is not True
        or sha256_file(Path(getattr(config, "test_manifest_path")))
        != EXPECTED_MANIFEST_SHA256
        or frame.cache_binding.get("manifest_opened") is not False
        or frame.cache_binding.get(
            "previous_stage90_output_prediction_checkpoint_or_scratch_consumed"
        )
        is not False
        or amendment.get("previous_stage90_outputs_used") is not False
        or amendment.get("previous_stage90_amendments_used") is not False
        or amendment.get("previous_prediction_surfaces_used") is not False
        or amendment.get("previous_stage90_scratch_or_checkpoints_used") is not False
    ):
        raise ProtocolError("Case-directional pre-GPU firewall failed.")
    return {
        "status": "PASS",
        "evaluation_split": "test",
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "target_labels_opened": False,
        "target_expert_used": False,
        "previous_stage90_output_prediction_checkpoint_or_scratch_consumed": False,
        "gpu_work_authorized": True,
    }


def _load_ledger_chain(
    config: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    parent_path = Path(getattr(config, "test_consumption_ledger_path"))
    amendment_path = Path(getattr(config, "ledger_amendment_path"))
    parent = _read_json(parent_path)
    amendment = _read_json(amendment_path)
    required_true = (
        "all_physical_probabilities_globally_sealed_before_any_label_access",
        "label_free_held_case_features_sealed_before_support_labels",
        "role_scoped_label_capabilities_enforced",
        "all_72_donor_grants_complete_before_route_support",
        "route_scoped_support_grants_only",
        "route_labels_never_enter_own_fit_scaler_state_or_decision",
        "all_218_predictions_and_decisions_sealed_before_terminal_label_access",
        "all_aggregate_method_seals_complete_before_terminal_label_access",
    )
    required_false = (
        "fresh_evidence",
        "previous_stage90_outputs_used",
        "previous_stage90_amendments_used",
        "previous_prediction_surfaces_used",
        "previous_stage90_scratch_or_checkpoints_used",
        "target_expert_used",
        "shared_model_updated_with_target_labels",
        "promotion_eligible",
        "may_feed_another_experiment",
    )
    if (
        sha256_file(parent_path) != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or sha256_file(amendment_path) != EXPECTED_LEDGER_AMENDMENT_SHA256
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("parent_sha256") != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or amendment.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or amendment.get("authorization_scope") != AUTHORIZATION_SCOPE
        or amendment.get("claim_role") != CLAIM_ROLE
        or any(amendment.get(key) is not True for key in required_true)
        or any(amendment.get(key) is not False for key in required_false)
    ):
        raise ProtocolError("Case-directional consumption-ledger chain drifted.")
    return MappingProxyType(parent), MappingProxyType(amendment)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read case-directional JSON: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Case-directional JSON must be an object.")
    return value


__all__ = (
    "LabelFreeTestFrame",
    "TestRowIdentity",
    "ValidatedLocks",
    "assert_input_fence",
    "load_label_free_test_frame",
    "load_validated_locks",
    "validate_active_diagnostic_workspace_binding",
    "validate_pre_gpu_firewall",
    "validate_workspace_provenance",
)
