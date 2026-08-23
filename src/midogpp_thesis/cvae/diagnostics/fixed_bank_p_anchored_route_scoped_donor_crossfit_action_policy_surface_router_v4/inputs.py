"""Exact-six P-DCAPS v4 input admission and label-free cache loading."""

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
from ...expert_bank.uniform_b_v2_promotion.contracts import CENTERS
from ...generation import (
    load_generation_lock_config,
    read_generation_lock,
    validate_generation_bundle,
)
from ...generation.contracts import COMMON_OUTPUT_DIM, GenerationLock
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    AUTHORIZED_INPUT_ROLES,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    FORBIDDEN_INPUT_FRAGMENTS,
    INPUT_ARTIFACT_IDS,
    LEDGER_AMENDMENT_ARTIFACT_ID,
    LEDGER_AMENDMENT_SCHEMA_VERSION,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
    V1_EXPERIMENT_ID,
    V1_OUTPUT_ARTIFACT_ID,
    V2_EXPERIMENT_ID,
    V2_OUTPUT_ARTIFACT_ID,
    V3_EXPERIMENT_ID,
    V3_OUTPUT_ARTIFACT_ID,
)
from .identity import AUTHORIZATION_BASIS, AUTHORIZATION_SCOPE, EXPERIMENT_ID
from .input_contracts import (
    LabelFreeTestFrame,
    TestRowIdentity,
    validate_source_snapshot,
)
from .workspace_inputs import (
    validate_active_workspace_binding,
    validate_workspace_provenance,
)
from .source_seal import validate_combined_source_seal


_FALSE_PREDECESSOR_OR_HISTORY_FIELDS = (
    "v1_output_used",
    "v1_amendment_used",
    "v1_label_capability_history_used",
    "v1_scratch_or_checkpoint_used",
    "prior_v1_execution_authorization_reused",
    "prior_v2_output_used",
    "prior_v2_amendment_used",
    "prior_v2_label_capability_history_used",
    "prior_v2_scratch_or_checkpoint_used",
    "prior_v2_execution_authorization_reused",
    "v2_execution_attempted",
    "v2_run_history_used",
    "cross_run_recovery_used",
    "v3_output_used",
    "v3_amendment_used",
    "v3_label_capability_history_used",
    "v3_scratch_or_checkpoint_used",
    "prior_v3_execution_authorization_reused",
)


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    parent_ledger: Mapping[str, object]
    ledger_amendment: Mapping[str, object]


def assert_input_fence(config: object) -> None:
    """Reject v1 and every non-six-input diagnostic/result path."""

    input_ids = tuple(getattr(config, "input_artifact_ids", ()))
    if (
        input_ids != INPUT_ARTIFACT_IDS
        or len(input_ids) != 6
        or len(set(input_ids)) != 6
        or getattr(config, "experiment_id", None) != EXPERIMENT_ID
        or any(
            forbidden in input_ids
            for forbidden in (
                V1_OUTPUT_ARTIFACT_ID,
                V1_EXPERIMENT_ID,
                V2_OUTPUT_ARTIFACT_ID,
                V2_EXPERIMENT_ID,
                V3_OUTPUT_ARTIFACT_ID,
                V3_EXPERIMENT_ID,
            )
        )
    ):
        raise ProtocolError("P-DCAPS v4 requires exactly six fresh fenced inputs.")
    values = (
        *(str(value) for value in input_ids),
        str(getattr(config, "expert_bank_root", "")),
        str(getattr(config, "generation_lock_root", "")),
        str(getattr(config, "test_cache_root", "")),
        str(getattr(config, "test_manifest_path", "")),
        str(getattr(config, "test_consumption_ledger_path", "")),
        str(getattr(config, "ledger_amendment_path", "")),
    )
    for value in values:
        folded = value.casefold()
        if (
            any(
                forbidden.casefold() in folded
                for forbidden in (
                    V1_OUTPUT_ARTIFACT_ID,
                    V1_EXPERIMENT_ID,
                    V2_OUTPUT_ARTIFACT_ID,
                    V2_EXPERIMENT_ID,
                    V3_OUTPUT_ARTIFACT_ID,
                    V3_EXPERIMENT_ID,
                )
            )
            or any(fragment.casefold() in folded for fragment in FORBIDDEN_INPUT_FRAGMENTS)
        ):
            raise ProtocolError("P-DCAPS v4 rejected predecessor diagnostic input.")


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
        raise ProtocolError("P-DCAPS v4 consumed-test cache drifted.")
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
    if len(rows) != EXPECTED_TEST_ROWS:
        raise ProtocolError("P-DCAPS v4 test-row coverage drifted.")
    binding = {
        "schema_version": "pdcaps_v4_test_cache_lineage_v1",
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
        "v1_v2_v3_artifact_amendment_probability_capability_run_state_or_scratch_used": False,
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
        raise ProtocolError("P-DCAPS v4 GenerationLock drifted.")
    parent, amendment = _load_ledger_chain(config)
    return ValidatedLocks(generation, parent, amendment)


def validate_pre_gpu_firewall(
    config: object,
    frame: LabelFreeTestFrame,
    locks: ValidatedLocks | None = None,
) -> Mapping[str, object]:
    """Validate all label-free authority before either A5000 is allocated."""

    assert_input_fence(config)
    source = validate_source_snapshot(
        expected_manifest_sha256=getattr(
            config, "expected_source_snapshot_manifest_sha256"
        ),
        expected_tree_sha256=getattr(config, "expected_source_snapshot_tree_sha256"),
        expected_member_count=getattr(config, "expected_source_snapshot_member_count"),
    )
    combined_source = validate_combined_source_seal()
    validate_active_workspace_binding(config)
    validate_workspace_provenance(Path(getattr(config, "artifact_root")), config)
    validated = locks or load_validated_locks(config)
    bank_root = Path(getattr(config, "expert_bank_root"))
    promotion = load_promotion_config(bank_root / "config.resolved.yaml")
    checks = validate_promoted_bank(bank_root, config=promotion, allow_pending=False)
    if (
        checks.get("status") != "PASS"
        or checks.get("all_experts_source_only") is not True
        or sha256_file(Path(getattr(config, "test_manifest_path")))
        != EXPECTED_MANIFEST_SHA256
        or frame.cache_binding.get("manifest_opened") is not False
        or any(
            validated.ledger_amendment.get(key) is not False
            for key in _FALSE_PREDECESSOR_OR_HISTORY_FIELDS
        )
    ):
        raise ProtocolError("P-DCAPS v4 pre-GPU firewall failed.")
    return MappingProxyType(
        {
            "status": "PASS",
            "evaluation_split": "test",
            "test_split_previously_consumed": True,
            "fresh_evidence": False,
            "target_labels_opened": False,
            "target_expert_used": False,
            "v1_v2_v3_diagnostic_state_used": False,
            **dict(source),
            "combined_source_seal_sha256": (
                combined_source.combined_source_seal_sha256
            ),
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
    if (
        sha256_file(parent_path) != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or sha256_file(amendment_path)
        != str(getattr(config, "expected_ledger_amendment_sha256"))
        or amendment.get("schema_version") != LEDGER_AMENDMENT_SCHEMA_VERSION
        or amendment.get("amendment_id") != LEDGER_AMENDMENT_ARTIFACT_ID
        or amendment.get("parent_artifact_id")
        != "midogpp_uniform_b_test_consumption_ledger_v1"
        or amendment.get("parent_member") != "reports/test_consumption_ledger.json"
        or amendment.get("parent_sha256") != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or amendment.get("authorized_consumer_experiment_ids") != [EXPERIMENT_ID]
        or amendment.get("authorization_basis") != AUTHORIZATION_BASIS
        or amendment.get("authorization_scope") != AUTHORIZATION_SCOPE
        or amendment.get("execution_authorized") is not True
        or amendment.get("consumed_test_reuse_authorized") is not True
        or amendment.get("authorization_is_separate_from_implementation_request")
        is not True
        or amendment.get("single_use_execution_identity") is not True
        or amendment.get("authorization_exhausted") is not False
        or amendment.get("direct_input_artifact_ids") != list(INPUT_ARTIFACT_IDS)
        or amendment.get("authorized_input_roles") != list(AUTHORIZED_INPUT_ROLES)
        or amendment.get("fresh_v4_workspace_aliases") != list(INPUT_ARTIFACT_IDS[2:])
        or amendment.get("source_snapshot_manifest_sha256")
        != getattr(config, "expected_source_snapshot_manifest_sha256")
        or amendment.get("source_snapshot_tree_sha256")
        != getattr(config, "expected_source_snapshot_tree_sha256")
        or amendment.get("source_snapshot_member_count")
        != getattr(config, "expected_source_snapshot_member_count")
        or amendment.get("fresh_evidence") is not False
        or amendment.get("publication_status")
        != "POST_HOC_CONSUMED_TEST_SENSITIVITY"
        or amendment.get("terminal_decision")
        != "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
        or any(
            amendment.get(key) is not False
            for key in _FALSE_PREDECESSOR_OR_HISTORY_FIELDS
        )
    ):
        raise ProtocolError("P-DCAPS v4 consumption-ledger chain drifted.")
    return MappingProxyType(parent), MappingProxyType(amendment)


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("P-DCAPS v4 ledger member is absent or unsafe.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError("Cannot read P-DCAPS v4 ledger JSON.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("P-DCAPS v4 ledger JSON must be an object.")
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
