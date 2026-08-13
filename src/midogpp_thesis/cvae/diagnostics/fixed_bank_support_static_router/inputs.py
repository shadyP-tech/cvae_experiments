"""Fail-closed exact-six admission and label-free consumed-test loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from .experiment_contracts import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPERIMENT_ID,
    FORBIDDEN_INPUT_FRAGMENTS,
    FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    parent_ledger: Mapping[str, object]
    ledger_amendment: Mapping[str, object]


def assert_input_fence(config: object) -> None:
    """Admit only the six registered original inputs; reject prior diagnostics."""

    input_ids = tuple(getattr(config, "input_artifact_ids"))
    if (
        input_ids != INPUT_ARTIFACT_IDS
        or len(set(input_ids)) != 6
        or getattr(config, "experiment_id") != EXPERIMENT_ID
        or getattr(config, "output_artifact_id") != OUTPUT_ARTIFACT_ID
        or OUTPUT_ARTIFACT_ID in input_ids
    ):
        raise ProtocolError("S4 requires exactly its six fenced inputs.")
    values = (
        *(str(value) for value in input_ids),
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
            raise ProtocolError("S4 input fence rejected a prior diagnostic output.")
        if any(token.casefold() in folded for token in FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS):
            raise ProtocolError("S4 input fence rejected a numbered-stage output.")


def load_label_free_test_frame(config: object) -> LabelFreeTestFrame:
    """Load embeddings and opaque identities without opening the manifest."""

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
        raise ProtocolError("S4 consumed-test cache identity drifted.")

    rows: list[TestRowIdentity] = []
    arrays: list[np.ndarray] = []
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
            row = TestRowIdentity(
                ordinal,
                int(manifest_index),
                str(row_id),
                str(case_id),
                center,
            )
            rows.append(row)
            center_rows.append(row)
            ordinal += 1
        arrays.append(np.asarray(shard.embeddings, dtype=np.float32))
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = str(shard.shard_sha256)

    observed_case_counts = {
        center: len({row.case_id for row in by_center[center]}) for center in CENTERS
    }
    if observed_case_counts != dict(EXPECTED_CASE_COUNTS_BY_CENTER):
        raise ProtocolError("S4 consumed-test case counts drifted.")
    binding = {
        "schema_version": "midogpp_fixed_bank_support_static_router_test_cache_binding_v1",
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
        "previous_stage90_output_prediction_or_scratch_consumed": False,
    }
    return LabelFreeTestFrame(
        np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32),
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
        generation.bank_lock_hash != getattr(config, "expected_bank_lock_hash")
        or generation.generation_lock_hash
        != getattr(config, "expected_generation_lock_hash")
    ):
        raise ProtocolError("S4 GenerationLock lineage drifted.")
    from .ledger import load_validated_ledger_chain

    ledger = load_validated_ledger_chain(config)
    return ValidatedLocks(generation, ledger.parent, ledger.amendment)


def validate_pre_gpu_firewall(
    config: object,
    frame: LabelFreeTestFrame,
    locks: ValidatedLocks | None = None,
) -> Mapping[str, object]:
    """Revalidate every original upstream lock before any GPU work."""

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
            "previous_stage90_output_prediction_or_scratch_consumed"
        )
        is not False
        or amendment.get("previous_prediction_surfaces_used") is not False
        or amendment.get("previous_stage90_outputs_used") is not False
        or amendment.get("previous_stage90_scratch_or_checkpoints_used") is not False
        or amendment.get("fresh_evidence") is not False
    ):
        raise ProtocolError("S4 pre-GPU firewall failed.")
    return {
        "schema_version": "fixed_bank_support_static_router_admission_v1",
        "status": "PASS",
        "evaluation_split": "test",
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "target_labels_opened": False,
        "target_expert_used": False,
        "source_expert_updated": False,
        "previous_stage90_output_prediction_or_scratch_consumed": False,
        "gpu_work_authorized": True,
    }


__all__ = (
    "LabelFreeTestFrame",
    "TestRowIdentity",
    "ValidatedLocks",
    "assert_input_fence",
    "load_label_free_test_frame",
    "load_validated_locks",
    "validate_pre_gpu_firewall",
)
