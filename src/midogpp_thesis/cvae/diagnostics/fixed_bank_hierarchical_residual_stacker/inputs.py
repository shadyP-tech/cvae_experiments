"""Fail-closed original-six input fence for the terminal residual stacker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
)
from ....data.features.stage70_test_cache.contracts import (
    CACHE_ARTIFACT_ID as UNDERLYING_TEST_CACHE_ARTIFACT_ID,
    CACHE_NAME as UNDERLYING_TEST_CACHE_NAME,
    REPRESENTATION_ID,
)
from ....data.features.stage70_test_cache.validation import load_validated_stage70_test_cache
from ...expert_bank.uniform_b_v2_promotion import load_promotion_config, validate_promoted_bank
from ...generation import load_generation_lock_config, read_generation_lock, validate_generation_bundle
from ...generation.contracts import (
    COMMON_OUTPUT_DIM,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    GenerationLock,
)
from ...protocol import ProtocolError
from .experiment_contracts import (
    CENTERS,
    EXPERIMENT_ID,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS,
    FORBIDDEN_PRIOR_STAGE90_ARTIFACT_IDS,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .ledger import load_validated_ledger_chain
from .workspace_inputs import (
    validate_active_diagnostic_workspace_binding as _validate_workspace_binding,
    validate_workspace_provenance as _validate_workspace_provenance,
)


# Every value is checked, including paths and aliases.  These tokens cover every
# existing Stage-90 consumer/output family and both superseded stacker attempts.
_FORBIDDEN_INPUT_FRAGMENTS = (
    "50_all_candidate_utility_matrix",
    "60_routing_and_composition",
    "70_frozen_policy_downstream",
    "frozen_policy_downstream",
    "utility_aligned_",
    "fixed_bank_decision_audit",
    "fixed_bank_label_aware_case_oof_ceiling",
    "fixed_bank_pooled_bacc_case_oof_ceiling",
    "residual_topup",
    "case_aware_proxy",
    "ensemble_endpoint_proxy",
    "exact_tail_router",
    "historical",
    "quarantine",
    "/scratch/",
    "/checkpoints/",
)


class DiagnosticInputConfig(Protocol):
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: Sequence[str]
    expert_bank_root: Path
    generation_lock_root: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    test_cache_root: Path
    test_manifest_path: Path


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    test_consumption_ledger: Mapping[str, object]
    ledger_amendment: Mapping[str, object]


def assert_input_fence(config: DiagnosticInputConfig) -> None:
    values = (
        *(str(value) for value in config.input_artifact_ids),
        str(config.expert_bank_root),
        str(config.generation_lock_root),
        str(config.test_consumption_ledger_path),
        str(config.ledger_amendment_path),
        str(config.test_cache_root),
        str(config.test_manifest_path),
    )
    forbidden = [
        value for value in values
        if any(fragment in value.lower() for fragment in _FORBIDDEN_INPUT_FRAGMENTS)
        or any(
            artifact_id.lower() in value.lower()
            for artifact_id in FORBIDDEN_PRIOR_STAGE90_ARTIFACT_IDS
        )
        or any(
            token.lower() in value.lower()
            for token in FORBIDDEN_NUMBERED_STAGE_OUTPUT_TOKENS
        )
    ]
    if forbidden:
        raise ProtocolError(
            "Residual stacker cannot consume prior Stage-90, Stage-50/60/70 result, "
            f"scratch, label, metric, or prediction outputs: {forbidden}."
        )
    if tuple(config.input_artifact_ids) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Residual stacker requires its exact six fenced inputs.")
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or config.output_artifact_id in config.input_artifact_ids
    ):
        raise ProtocolError("Residual-stacker input/output identity drifted.")


def load_label_free_test_frame(config: DiagnosticInputConfig) -> LabelFreeTestFrame:
    assert_input_fence(config)
    cache = load_validated_stage70_test_cache(config.test_cache_root)
    summary = dict(cache.summary)
    expected_counts = dict(EXPECTED_TEST_ROWS_BY_CENTER)
    if (
        summary.get("status") != "PASS"
        or summary.get("manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or summary.get("row_count") != EXPECTED_TEST_ROWS
        or summary.get("rows_by_center") != expected_counts
        or summary.get("content_hash") != EXPECTED_TEST_CACHE_CONTENT_HASH
        or summary.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or EXPECTED_TEST_CACHE_SEMANTIC_ID != UNDERLYING_TEST_CACHE_NAME
        or EXPECTED_TEST_CACHE_REPRESENTATION_ID != REPRESENTATION_ID
        or summary.get("fresh_evidence") is not False
    ):
        raise ProtocolError("Residual-stacker consumed-test cache failed validation.")
    arrays: list[np.ndarray] = []
    rows: list[TestRowIdentity] = []
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
                row_ordinal=ordinal,
                manifest_row_index=int(manifest_index),
                evaluation_row_id=str(row_id),
                case_id=str(case_id),
                center=center,
            )
            rows.append(row)
            center_rows.append(row)
            ordinal += 1
        arrays.append(np.asarray(shard.embeddings, dtype=np.float32))
        by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard.shard_sha256
    binding = {
        "schema_version": "midogpp_stage90_residual_stacker_consumed_test_cache_binding_v1",
        "cache_alias_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "manifest_alias_artifact_id": TEST_MANIFEST_ARTIFACT_ID,
        "underlying_cache_artifact_id": UNDERLYING_TEST_CACHE_ARTIFACT_ID,
        "underlying_cache_name": UNDERLYING_TEST_CACHE_NAME,
        "representation_id": REPRESENTATION_ID,
        "split": "test",
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "row_count": len(rows),
        "rows_by_center": expected_counts,
        "feature_dim": COMMON_OUTPUT_DIM,
        "cache_content_hash": summary.get("content_hash"),
        "row_order_hash": summary.get("row_order_hash"),
        "shard_sha256_by_center": shard_hashes,
        "labels_persisted": False,
        "sample_paths_persisted": False,
        "manifest_opened": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "dedicated_alias_only": True,
        "prior_stage90_output_consumed": False,
        "previous_prediction_surface_consumed": False,
    }
    return LabelFreeTestFrame(
        embeddings=np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=by_center,
        cache_binding=binding,
    )


def load_validated_locks(config: DiagnosticInputConfig) -> ValidatedLocks:
    assert_input_fence(config)
    generation_config = load_generation_lock_config(
        config.generation_lock_root / "config.resolved.yaml"
    )
    validate_generation_bundle(config.generation_lock_root, config=generation_config)
    generation = read_generation_lock(
        config.generation_lock_root / "manifests/generation_lock.json"
    )
    if (
        generation.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or generation.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
    ):
        raise ProtocolError("Residual-stacker frozen-generation lineage drifted.")
    ledger = load_validated_ledger_chain(config)
    return ValidatedLocks(
        generation=generation,
        test_consumption_ledger=ledger.parent,
        ledger_amendment=ledger.amendment,
    )


def validate_pre_gpu_firewall(
    config: DiagnosticInputConfig,
    frame: LabelFreeTestFrame,
    locks: ValidatedLocks | None = None,
) -> Mapping[str, object]:
    assert_input_fence(config)
    validated = locks or load_validated_locks(config)
    promotion_config = load_promotion_config(config.expert_bank_root / "config.resolved.yaml")
    checks = validate_promoted_bank(
        config.expert_bank_root, config=promotion_config, allow_pending=False
    )
    bank_index = _json(config.expert_bank_root / "manifests/expert_bank_index.json")
    records = bank_index.get("records")
    amendment = validated.ledger_amendment
    if (
        checks.get("status") != "PASS"
        or checks.get("all_experts_source_only") is not True
        or not isinstance(records, list)
        or len(records) != 27
        or any(
            not isinstance(row, Mapping)
            or row.get("fresh_source_only_training") is not True
            for row in records
        )
        or frame.cache_binding.get("split") != "test"
        or frame.cache_binding.get("manifest_opened") is not False
        or _sha256_file(config.test_manifest_path) != EXPECTED_MANIFEST_SHA256
        or amendment.get("parent_sha256") != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or amendment.get("previous_stage90_outputs_used") is not False
        or amendment.get("previous_stage90_scratch_or_checkpoints_used") is not False
        or amendment.get("previous_prediction_surfaces_used") is not False
    ):
        raise ProtocolError("Residual-stacker pre-GPU firewall failed.")
    return {
        "status": "PASS",
        "bank_lock_hash": bank_index.get("bank_lock_hash"),
        "expert_count": len(records),
        "fresh_source_only_training": True,
        "evaluation_split": "test",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "test_split_previously_consumed": True,
        "ledger_parent_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        "ledger_amendment_sha256": EXPECTED_LEDGER_AMENDMENT_SHA256,
        "fresh_evidence": False,
        "previous_stage90_output_or_scratch_consumed": False,
        "previous_prediction_surface_consumed": False,
        "target_labels_opened": False,
        "target_expert_used": False,
        "gpu_work_authorized": True,
    }


def validate_workspace_provenance(
    root: Path, config: DiagnosticInputConfig
) -> dict[str, Mapping[str, object]]:
    assert_input_fence(config)
    return _validate_workspace_provenance(root, config)


def validate_active_diagnostic_workspace_binding(
    config: DiagnosticInputConfig,
) -> Mapping[str, object]:
    assert_input_fence(config)
    return _validate_workspace_binding(config)


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read residual-stacker JSON input: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Residual-stacker JSON input must be an object: {path}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash residual-stacker input: {path}.") from exc
    return digest.hexdigest()


__all__ = (
    "ValidatedLocks",
    "assert_input_fence",
    "load_label_free_test_frame",
    "load_validated_locks",
    "validate_active_diagnostic_workspace_binding",
    "validate_pre_gpu_firewall",
    "validate_workspace_provenance",
)
