"""Exact-eight read-only input validation for executable SCEPTRE v2."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ....data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
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
from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .experiment_contracts import (
    AUTHORIZED_INPUT_ROLES,
    EXECUTION_AMENDMENT_ARTIFACT_ID,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_CASE_CONFUSION_ROWS,
    EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS,
    EXPECTED_SOURCE_EVALUATION_ROW_COUNT,
    EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
    EXPECTED_SOURCE_POLICY_LOCK_HASH,
    EXPECTED_SOURCE_UTILITY_ROWS,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    FORBIDDEN_INPUT_FRAGMENTS,
    INPUT_ARTIFACT_IDS,
    SOURCE_INNER_ALIAS_ARTIFACT_ID,
    SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
    SOURCE_INNER_MEMBER_SHA256,
    SOURCE_INNER_MEMBERS,
    SOURCE_INNER_ORIGINAL_ARTIFACT_ID,
)
from .identity import (
    AUTHORIZATION_BASIS,
    AUTHORIZATION_DATE,
    AUTHORIZATION_SCOPE,
    EXECUTION_REVISION,
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    V1_EXPERIMENT_ID,
    V1_OUTPUT_ARTIFACT_ID,
    file_sha256,
    require_sha256,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity
from .protocol import PROTOCOL_SCHEMA


@dataclass(frozen=True, slots=True)
class SourceInnerInputReceipt:
    alias_artifact_id: str
    amendment_artifact_id: str
    amendment_sha256: str
    member_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            self.alias_artifact_id != SOURCE_INNER_ALIAS_ARTIFACT_ID
            or self.amendment_artifact_id != SOURCE_INNER_AMENDMENT_ARTIFACT_ID
            or self.amendment_sha256 != EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
            or dict(self.member_sha256) != SOURCE_INNER_MEMBER_SHA256
        ):
            raise ProtocolError("SCEPTRE v2 source-inner receipt drifted.")
        object.__setattr__(
            self, "member_sha256", MappingProxyType(dict(self.member_sha256))
        )


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    frame: LabelFreeTestFrame
    generation_lock: GenerationLock
    source_inner: SourceInnerInputReceipt
    bank_validation: Mapping[str, object]
    parent_ledger: Mapping[str, object]
    execution_amendment: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bank_validation", MappingProxyType(dict(self.bank_validation))
        )
        object.__setattr__(self, "parent_ledger", MappingProxyType(dict(self.parent_ledger)))
        object.__setattr__(
            self,
            "execution_amendment",
            MappingProxyType(dict(self.execution_amendment)),
        )


def assert_input_fence(config: object) -> None:
    input_ids = tuple(getattr(config, "input_artifact_ids", ()))
    if (
        input_ids != INPUT_ARTIFACT_IDS
        or len(input_ids) != 8
        or len(set(input_ids)) != 8
        or getattr(config, "experiment_id", None) != EXPERIMENT_ID
        or V1_OUTPUT_ARTIFACT_ID in input_ids
        or V1_EXPERIMENT_ID in input_ids
    ):
        raise ProtocolError("SCEPTRE v2 requires exactly eight fenced inputs.")
    values = (
        *(str(value) for value in input_ids),
        *(str(getattr(config, role, "")) for role in _INPUT_PATH_ROLES),
    )
    for value in values:
        folded = value.casefold()
        if any(fragment.casefold() in folded for fragment in FORBIDDEN_INPUT_FRAGMENTS):
            raise ProtocolError("SCEPTRE v2 rejected predecessor Stage-90 state.")


def load_label_free_test_frame(config: object) -> LabelFreeTestFrame:
    assert_input_fence(config)
    cache_root = _safe_directory(Path(getattr(config, "test_cache_root")), "test cache")
    try:
        cache = load_validated_stage70_test_cache(cache_root)
    except Exception as exc:  # canonical loader has its own typed contract error
        raise ProtocolError("SCEPTRE v2 consumed-test cache validation failed.") from exc
    summary = dict(cache.summary)
    if (
        summary.get("status") != "PASS"
        or summary.get("manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or summary.get("content_hash") != EXPECTED_TEST_CACHE_CONTENT_HASH
        or summary.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or summary.get("fresh_evidence") is not False
        or UNDERLYING_CACHE_NAME != EXPECTED_TEST_CACHE_SEMANTIC_ID
        or REPRESENTATION_ID != EXPECTED_TEST_CACHE_REPRESENTATION_ID
    ):
        raise ProtocolError("SCEPTRE v2 consumed-test cache identity drifted.")

    rows: list[TestRowIdentity] = []
    embeddings: list[np.ndarray] = []
    rows_by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
    cases_by_center: dict[str, tuple[str, ...]] = {}
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
                row_ordinal=ordinal,
                manifest_row_index=int(manifest_index),
                evaluation_row_id=str(row_id),
                case_id=str(case_id),
                center=str(center),
            )
            rows.append(identity)
            center_rows.append(identity)
            ordinal += 1
        embeddings.append(np.asarray(shard.embeddings, dtype=np.float32))
        rows_by_center[center] = tuple(center_rows)
        cases_by_center[center] = tuple(
            dict.fromkeys(row.case_id for row in center_rows)
        )
        shard_hashes[center] = shard.shard_sha256

    binding = {
        "schema_version": "sceptre_v2_test_cache_lineage_v1",
        "underlying_cache_artifact_id": UNDERLYING_CACHE_ARTIFACT_ID,
        "underlying_cache_name": UNDERLYING_CACHE_NAME,
        "representation_id": REPRESENTATION_ID,
        "split": "test",
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "row_count": len(rows),
        "case_count": len({(row.center, row.case_id) for row in rows}),
        "rows_by_center": {center: len(rows_by_center[center]) for center in CENTERS},
        "cases_by_center": {center: len(cases_by_center[center]) for center in CENTERS},
        "cache_content_hash": summary["content_hash"],
        "row_order_hash": summary["row_order_hash"],
        "shard_sha256_by_center": shard_hashes,
        "labels_persisted": False,
        "manifest_opened": False,
        "sample_paths_persisted": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
    }
    return LabelFreeTestFrame(
        embeddings=np.ascontiguousarray(np.concatenate(embeddings), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cases_by_center=cases_by_center,
        cache_binding=binding,
    )


def load_source_inner_inputs(config: object) -> SourceInnerInputReceipt:
    assert_input_fence(config)
    root = _safe_directory(Path(getattr(config, "source_inner_root")), "source-inner alias")
    observed: dict[str, str] = {}
    for member in SOURCE_INNER_MEMBERS:
        path = _safe_file(root / member, f"source-inner {member}")
        observed[member] = file_sha256(path)
    if observed != SOURCE_INNER_MEMBER_SHA256:
        raise ProtocolError("SCEPTRE v2 source-inner bytes drifted.")
    _require_csv_rows(root / SOURCE_INNER_MEMBERS[1], EXPECTED_SOURCE_UTILITY_ROWS)
    _require_csv_rows(root / SOURCE_INNER_MEMBERS[2], EXPECTED_SOURCE_CASE_CONFUSION_ROWS)
    _require_csv_rows(root / SOURCE_INNER_MEMBERS[5], EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS)
    _require_csv_rows(root / SOURCE_INNER_MEMBERS[6], EXPECTED_SOURCE_EVALUATION_ROW_COUNT)
    utility_lock = _read_json(root / SOURCE_INNER_MEMBERS[0], "source-inner utility lock")
    if utility_lock.get("policy_consumption_lock_hash") != EXPECTED_SOURCE_POLICY_LOCK_HASH:
        raise ProtocolError("SCEPTRE v2 source-inner policy lock drifted.")

    amendment_path = _safe_file(
        Path(getattr(config, "source_inner_amendment_path")),
        "source-inner amendment",
    )
    amendment_sha = file_sha256(amendment_path)
    if (
        amendment_sha != EXPECTED_SOURCE_INNER_AMENDMENT_SHA256
        or amendment_sha != getattr(config, "expected_source_inner_amendment_sha256")
    ):
        raise ProtocolError("SCEPTRE v2 source-inner amendment hash drifted.")
    _validate_source_inner_amendment(
        _read_json(amendment_path, "source-inner amendment")
    )
    return SourceInnerInputReceipt(
        alias_artifact_id=SOURCE_INNER_ALIAS_ARTIFACT_ID,
        amendment_artifact_id=SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
        amendment_sha256=amendment_sha,
        member_sha256=observed,
    )


def load_validated_inputs(config: object) -> ValidatedInputs:
    """Validate every direct input without opening a target label column."""

    assert_input_fence(config)
    bank_root = _safe_directory(Path(getattr(config, "expert_bank_root")), "expert bank")
    generation_root = _safe_directory(
        Path(getattr(config, "generation_lock_root")), "GenerationLock"
    )
    try:
        promotion = load_promotion_config(bank_root / "config.resolved.yaml")
        bank = validate_promoted_bank(bank_root, config=promotion, allow_pending=False)
        generation_config = load_generation_lock_config(
            generation_root / "config.resolved.yaml"
        )
        validate_generation_bundle(generation_root, config=generation_config)
        generation = read_generation_lock(
            generation_root / "manifests/generation_lock.json"
        )
    except Exception as exc:
        raise ProtocolError("SCEPTRE v2 frozen-bank validation failed.") from exc
    if (
        bank.get("status") != "PASS"
        or bank.get("all_experts_source_only") is not True
        or generation.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or generation.generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
    ):
        raise ProtocolError("SCEPTRE v2 frozen bank or GenerationLock drifted.")

    source_inner = load_source_inner_inputs(config)
    frame = load_label_free_test_frame(config)
    manifest_path = _safe_file(
        Path(getattr(config, "test_manifest_path")), "test manifest"
    )
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ProtocolError("SCEPTRE v2 role-scoped manifest bytes drifted.")
    parent, amendment = load_ledger_chain(config)
    return ValidatedInputs(
        frame=frame,
        generation_lock=generation,
        source_inner=source_inner,
        bank_validation=bank,
        parent_ledger=parent,
        execution_amendment=amendment,
    )


def canonical_execution_amendment_payload(config: object) -> dict[str, object]:
    """Return the exact authority object whose file bytes are separately pinned."""

    return {
        "schema_version": "midogpp_sceptre_test_execution_amendment_v2",
        "amendment_id": EXECUTION_AMENDMENT_ARTIFACT_ID,
        "experiment_id": EXPERIMENT_ID,
        "execution_revision": EXECUTION_REVISION,
        "protocol_schema": PROTOCOL_SCHEMA,
        "parent_artifact_id": "midogpp_uniform_b_test_consumption_ledger_v1",
        "parent_member": "reports/test_consumption_ledger.json",
        "parent_sha256": EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
        "authorized_consumer_experiment_ids": [EXPERIMENT_ID],
        "authorization_basis": AUTHORIZATION_BASIS,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "authorization_date": AUTHORIZATION_DATE,
        "execution_authorized": True,
        "consumed_test_reuse_authorized": True,
        "authorization_is_separate_from_implementation_request": True,
        "implementation_request_alone_authorizes_execution": False,
        "source_code_or_registration_alone_authorizes_execution": False,
        "single_use_execution_identity": True,
        "authorization_exhausted": False,
        "direct_input_artifact_ids": list(INPUT_ARTIFACT_IDS),
        "authorized_input_roles": list(AUTHORIZED_INPUT_ROLES),
        "source_inner_alias_artifact_id": SOURCE_INNER_ALIAS_ARTIFACT_ID,
        "source_inner_amendment_artifact_id": SOURCE_INNER_AMENDMENT_ARTIFACT_ID,
        "source_inner_amendment_sha256": EXPECTED_SOURCE_INNER_AMENDMENT_SHA256,
        "shared_runtime_dependencies_in_source_seal": False,
        "source_snapshot_scope": (
            "sceptre_owned_executable_and_inherited_scientific_python"
        ),
        "source_snapshot_manifest_sha256": getattr(
            config, "expected_source_snapshot_manifest_sha256"
        ),
        "source_snapshot_tree_sha256": getattr(
            config, "expected_source_snapshot_tree_sha256"
        ),
        "source_snapshot_member_count": getattr(
            config, "expected_source_snapshot_member_count"
        ),
        "v1_experiment_id": V1_EXPERIMENT_ID,
        "v1_output_artifact_id": V1_OUTPUT_ARTIFACT_ID,
        "v1_output_used": False,
        "v1_run_state_used": False,
        "v1_scratch_or_checkpoint_used": False,
        "previous_stage90_output_used": False,
        "previous_stage90_run_state_used": False,
        "previous_stage90_scratch_or_checkpoint_used": False,
        "cross_run_recovery_allowed": False,
        "target_support_and_evaluation_label_roles_scoped": True,
        "selection_and_calibration_labels_form_per_fold_policy": True,
        "route_policy_frozen_before_evaluation_labels": True,
        "decision_estimand": (
            "downstream_classifier_utility_sensitivity_not_CVAE_NELBO"
        ),
        "raw_labels_may_be_persisted": False,
        "publication_status": PUBLICATION_STATUS,
        "terminal_decision": TERMINAL_DECISION,
        "fresh_evidence": False,
        "routing_success_claimed": False,
        "downstream_utility_claimed": False,
        "nelbo_compatibility_claimed": False,
        "significance_claimed": False,
        "promotion_allowed": False,
        "deployment_claimed": False,
        "may_feed_another_experiment": False,
    }


def load_ledger_chain(
    config: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    parent_path = _safe_file(
        Path(getattr(config, "test_consumption_ledger_path")), "parent ledger"
    )
    amendment_path = _safe_file(
        Path(getattr(config, "execution_amendment_path")), "execution amendment"
    )
    parent = _read_json(parent_path, "parent ledger")
    amendment = _read_json(amendment_path, "execution amendment")
    expected_amendment_sha = require_sha256(
        getattr(config, "expected_execution_amendment_sha256"),
        "execution amendment hash",
    )
    if (
        file_sha256(parent_path) != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or file_sha256(parent_path)
        != getattr(config, "expected_test_consumption_ledger_sha256")
        or parent.get("status") != "CONSUMED_FOR_REPRESENTATION_ADOPTION"
        or parent.get("split") != "test"
        or file_sha256(amendment_path) != expected_amendment_sha
        or amendment != canonical_execution_amendment_payload(config)
    ):
        raise ProtocolError("SCEPTRE v2 consumption-ledger chain drifted.")
    return MappingProxyType(parent), MappingProxyType(amendment)


def _validate_source_inner_amendment(raw: Mapping[str, object]) -> None:
    binding = _mapping(raw, "source_binding")
    claims = _mapping(raw, "claim_boundary")
    development = _mapping(raw, "development_protocol")
    execution = _mapping(raw, "execution_authority")
    original = _mapping(raw, "original_contract_treatment")
    evaluation = _mapping(raw, "test_evaluation_protocol")
    if (
        raw.get("schema_version")
        != "midogpp_sceptre_source_inner_adaptive_reuse_amendment_v2"
        or raw.get("amendment_id") != SOURCE_INNER_AMENDMENT_ARTIFACT_ID
        or raw.get("authorized_consumer_experiment_id") != EXPERIMENT_ID
        or raw.get("authorized_input_alias_id") != SOURCE_INNER_ALIAS_ARTIFACT_ID
        or raw.get("authorization_date") != AUTHORIZATION_DATE
        or binding.get("original_artifact_id") != SOURCE_INNER_ORIGINAL_ARTIFACT_ID
        or binding.get("utility_lock_sha256")
        != SOURCE_INNER_MEMBER_SHA256[SOURCE_INNER_MEMBERS[0]]
        or binding.get("candidate_utility_csv_sha256")
        != SOURCE_INNER_MEMBER_SHA256[SOURCE_INNER_MEMBERS[1]]
        or binding.get("candidate_utility_row_count") != EXPECTED_SOURCE_UTILITY_ROWS
        or binding.get("case_confusions_csv_sha256")
        != SOURCE_INNER_MEMBER_SHA256[SOURCE_INNER_MEMBERS[2]]
        or binding.get("case_confusions_row_count")
        != EXPECTED_SOURCE_CASE_CONFUSION_ROWS
        or binding.get("candidate_predictions_npz_sha256")
        != SOURCE_INNER_MEMBER_SHA256[SOURCE_INNER_MEMBERS[3]]
        or binding.get("prediction_index_json_sha256")
        != SOURCE_INNER_MEMBER_SHA256[SOURCE_INNER_MEMBERS[4]]
        or binding.get("classifier_fits_csv_sha256")
        != SOURCE_INNER_MEMBER_SHA256[SOURCE_INNER_MEMBERS[5]]
        or binding.get("classifier_fit_row_count") != EXPECTED_SOURCE_CLASSIFIER_FIT_ROWS
        or binding.get("evaluation_rows_csv_sha256")
        != SOURCE_INNER_MEMBER_SHA256[SOURCE_INNER_MEMBERS[6]]
        or binding.get("evaluation_row_count") != EXPECTED_SOURCE_EVALUATION_ROW_COUNT
        or claims.get("publication_status") != PUBLICATION_STATUS
        or claims.get("terminal_decision") != TERMINAL_DECISION
        or claims.get("fresh_evidence") is not False
        or claims.get("routing_success_claim_allowed") is not False
        or claims.get("downstream_utility_claim_allowed") is not False
        or claims.get("nelbo_compatibility_claim_allowed", False) is not False
        or claims.get("promotion_allowed") is not False
        or development.get("complete_nested_lodo_required") is not True
        or development.get("outer_target_candidate_rows_excluded") is not True
        or development.get("outer_target_query_rows_excluded") is not True
        or development.get("seed_selection_allowed") is not False
        or execution.get("execution_authorized") is not False
        or execution.get("consumed_test_reuse_authorized") is not False
        or execution.get("separate_consumer_specific_execution_amendment_required")
        is not True
        or original.get("original_policy_consumption_lock_mutated") is not False
        or original.get("source_inner_policy_lock_hash")
        != EXPECTED_SOURCE_POLICY_LOCK_HASH
        or evaluation.get("this_amendment_authorizes_test_evaluation") is not False
        or evaluation.get(
            "prelabel_router_model_thresholds_and_G_frozen_before_test_label_access"
        ) is not True
        or evaluation.get(
            "per_fold_route_policy_formed_from_disjoint_selection_and_calibration_labels"
        ) is not True
        or evaluation.get("route_policy_frozen_before_evaluation_label_access")
        is not True
        or evaluation.get(
            "support_metrics_are_downstream_classifier_utility_not_nelbo"
        ) is not True
    ):
        raise ProtocolError("SCEPTRE v2 source-inner amendment drifted.")


def _safe_directory(path: Path, role: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ProtocolError(f"SCEPTRE v2 {role} is absent or unsafe.")
    return path


def _safe_file(path: Path, role: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProtocolError(f"SCEPTRE v2 {role} is absent or unsafe.")
    return path


def _read_json(path: Path, role: str) -> dict[str, object]:
    _safe_file(path, role)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read SCEPTRE v2 {role} JSON.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"SCEPTRE v2 {role} JSON must be an object.")
    return value


def _mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"SCEPTRE v2 amendment section absent: {key}.")
    return value


def _require_csv_rows(path: Path, expected: int) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader)
            observed = sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, StopIteration) as exc:
        raise ProtocolError("Cannot count SCEPTRE v2 source-inner CSV rows.") from exc
    if observed != expected:
        raise ProtocolError("SCEPTRE v2 source-inner CSV row count drifted.")


_INPUT_PATH_ROLES = (
    "expert_bank_root",
    "generation_lock_root",
    "source_inner_root",
    "source_inner_amendment_path",
    "test_cache_root",
    "test_manifest_path",
    "test_consumption_ledger_path",
    "execution_amendment_path",
)


__all__ = (
    "LabelFreeTestFrame",
    "SourceInnerInputReceipt",
    "TestRowIdentity",
    "ValidatedInputs",
    "assert_input_fence",
    "canonical_execution_amendment_payload",
    "load_label_free_test_frame",
    "load_ledger_chain",
    "load_source_inner_inputs",
    "load_validated_inputs",
)
