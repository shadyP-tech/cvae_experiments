"""Closed-world inputs for the consumed-test case-aware proxy audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
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
from ....data.features.stage70_test_cache.validation import (
    load_validated_stage70_test_cache,
)
from ....workspace.runtime import MidogppWorkspace
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
from ...generation.contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    GenerationLock,
)
from ...protocol import ProtocolError
from ...routing.metadata_compatibility import (
    derive_compatibility_scores,
    derive_metadata_profiles,
)
from ...routing.metadata_compatibility.contracts import (
    DOMAIN_MAPPING_MEMBER,
    DOMAIN_MAPPING_SHA256,
)
from .experiment_contracts import (
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    INPUT_ARTIFACT_IDS,
    METADATA_PROFILE_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TEST_CONSUMPTION_LEDGER_ARTIFACT_ID,
    TEST_MANIFEST_ARTIFACT_ID,
)
from .input_contracts import LabelFreeTestFrame, TestRowIdentity


EXPECTED_TEST_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)

_FORBIDDEN_INPUT_FRAGMENTS = (
    "50_all_candidate_utility_matrix",
    "60_routing_and_composition",
    "70_frozen_policy_downstream",
    "frozen_policy_downstream",
    "artifacts/midogpp/90_oracles_and_diagnostics",
    "midogpp_output_uniform_b_v2_consumed_validation",
    "midogpp_output_uniform_b_v2_consumed_test",
    "historical",
    "quarantine",
)


class DiagnosticInputConfig(Protocol):
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: Sequence[str]
    expert_bank_root: Path
    generation_lock_root: Path
    test_consumption_ledger_path: Path
    test_cache_root: Path
    test_manifest_path: Path
    metadata_profile_root: Path
    expected_manifest_sha256: str


@dataclass(frozen=True)
class ValidatedLocks:
    generation: GenerationLock
    test_consumption_ledger: Mapping[str, object]


def assert_input_fence(config: DiagnosticInputConfig) -> None:
    values = (
        *(str(value) for value in config.input_artifact_ids),
        str(config.expert_bank_root),
        str(config.generation_lock_root),
        str(config.test_consumption_ledger_path),
        str(config.test_cache_root),
        str(config.test_manifest_path),
        str(config.metadata_profile_root),
    )
    forbidden = sorted(
        {
            value
            for value in values
            if any(fragment in value.lower() for fragment in _FORBIDDEN_INPUT_FRAGMENTS)
        }
    )
    # The cache alias resolves to a derived-feature path whose historical name
    # contains Stage-70.  Only that exact path is admitted; no Stage-70 output
    # artifact or prediction/scoring product is allowed.
    allowed_cache_path = "uniform_b_v2_descriptive_test_cache_v1"
    forbidden = [
        value
        for value in forbidden
        if not (
            allowed_cache_path in value
            and value == str(config.test_cache_root)
        )
    ]
    if forbidden:
        raise ProtocolError(
            "Case-aware audit cannot consume Stage-50/60/70 outputs, prior "
            "Stage-90 outputs, or historical inputs: " + ", ".join(forbidden)
        )
    if tuple(config.input_artifact_ids) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Case-aware audit requires its exact six fenced inputs.")
    if config.output_artifact_id in config.input_artifact_ids:
        raise ProtocolError("Case-aware audit cannot consume its own output.")


def load_label_free_test_frame(config: DiagnosticInputConfig) -> LabelFreeTestFrame:
    """Validate and load the reused cache without opening the manifest."""

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
        raise ProtocolError("Case-aware consumed-test cache failed validation.")
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
        "schema_version": "midogpp_stage90_case_aware_consumed_test_cache_binding_v1",
        "cache_alias_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "underlying_cache_artifact_id": UNDERLYING_TEST_CACHE_ARTIFACT_ID,
        "underlying_cache_name": UNDERLYING_TEST_CACHE_NAME,
        "representation_id": REPRESENTATION_ID,
        "split": "test",
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "row_count": len(rows),
        "rows_by_center": expected_counts,
        "feature_dim": 3_840,
        "cache_content_hash": summary.get("content_hash"),
        "row_order_hash": summary.get("row_order_hash"),
        "shard_sha256_by_center": shard_hashes,
        "labels_persisted": False,
        "sample_ids_persisted": False,
        "manifest_opened": False,
        "test_split_previously_consumed": True,
        "repurposed_for_terminal_stage90_diagnostic": True,
        "fresh_evidence": False,
        "prior_stage90_output_consumed": False,
        "stage70_prediction_or_scoring_output_consumed": False,
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
        raise ProtocolError("Case-aware frozen generation lineage drifted.")
    ledger = _json(config.test_consumption_ledger_path)
    if (
        _sha256_file(config.test_consumption_ledger_path)
        != EXPECTED_TEST_LEDGER_SHA256
        or ledger
        != {
            "schema_version": "midogpp_uniform_b_test_consumption_ledger_v1",
            "status": "CONSUMED_FOR_REPRESENTATION_ADOPTION",
            "split": "test",
            "row_count": 9_928,
            "observed_centers": 9,
            "consumed_decision": ledger.get("consumed_decision"),
            "may_be_reused_as_fresh_representation_selection_evidence": False,
            "may_be_reused_for_descriptive_locked-model_scoring": True,
            "new_center_uncertainty_covered": False,
            "external_dataset_uncertainty_covered": False,
        }
    ):
        raise ProtocolError("Case-aware test-consumption ledger drifted.")
    return ValidatedLocks(
        generation=generation,
        test_consumption_ledger=MappingProxyType(dict(ledger)),
    )


def load_metadata_similarity(
    config: DiagnosticInputConfig,
) -> Mapping[str, Mapping[str, float]]:
    assert_input_fence(config)
    profiles = derive_metadata_profiles(
        config.metadata_profile_root / DOMAIN_MAPPING_MEMBER,
        expected_sha256=DOMAIN_MAPPING_SHA256,
    )
    result: dict[str, dict[str, float]] = {center: {} for center in CENTERS}
    for score in derive_compatibility_scores(profiles):
        result[score.target_center][score.source_center] = (
            float(score.exact_match_count) / 3.0
        )
    if any(
        set(result[center]) != set(CENTERS).difference({center})
        for center in CENTERS
    ):
        raise ProtocolError("Case-aware metadata surface coverage drifted.")
    return MappingProxyType(
        {center: MappingProxyType(dict(result[center])) for center in CENTERS}
    )


def validate_pre_gpu_firewall(
    config: DiagnosticInputConfig,
    frame: LabelFreeTestFrame,
) -> Mapping[str, object]:
    assert_input_fence(config)
    promotion_config = load_promotion_config(
        config.expert_bank_root / "config.resolved.yaml"
    )
    checks = validate_promoted_bank(
        config.expert_bank_root, config=promotion_config, allow_pending=False
    )
    bank_index = _json(config.expert_bank_root / "manifests/expert_bank_index.json")
    leakage = _json(config.expert_bank_root / "reports/leakage_report.json")
    source_evidence = _json(
        config.expert_bank_root / "manifests/source_evidence_lock.json"
    )
    records = bank_index.get("records")
    if (
        checks.get("status") != "PASS"
        or checks.get("all_experts_source_only") is not True
        or not isinstance(records, list)
        or len(records) != 27
        or any(
            not isinstance(row, Mapping)
            or row.get("fresh_source_only_training") is not True
            or row.get("parent_checkpoint_used") is not False
            for row in records
        )
        or leakage.get("status") != "PASS"
        or int(leakage.get("identity_overlap_failures", -1)) != 0
        or int(source_evidence.get("identity_overlap_failures", -1)) != 0
        or frame.cache_binding.get("split") != "test"
        or frame.cache_binding.get("manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or frame.cache_binding.get("labels_persisted") is not False
        or frame.cache_binding.get("manifest_opened") is not False
        or frame.cache_binding.get("test_split_previously_consumed") is not True
        or frame.cache_binding.get("prior_stage90_output_consumed") is not False
        or _sha256_file(config.test_manifest_path) != CANONICAL_MANIFEST_SHA256
    ):
        raise ProtocolError("Case-aware pre-GPU firewall failed.")
    return {
        "status": "PASS",
        "bank_lock_hash": str(bank_index.get("bank_lock_hash")),
        "expert_count": len(records),
        "fresh_source_only_training": True,
        "bank_identity_overlap_failures": 0,
        "evaluation_split": "test",
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "test_cache_label_fields_absent": True,
        "test_split_previously_consumed": True,
        "repurposed_for_method_development": True,
        "fresh_evidence": False,
        "prior_stage90_output_consumed": False,
        "stage50_output_consumed": False,
        "stage60_output_consumed": False,
        "stage70_prediction_or_scoring_output_consumed": False,
        "target_labels_opened": False,
        "gpu_work_authorized": True,
    }


def validate_workspace_provenance(
    root: Path,
    config: DiagnosticInputConfig,
) -> dict[str, Mapping[str, object]]:
    assert_input_fence(config)
    payload = _json(root / "provenance/input_artifacts.json")
    if (
        payload.get("schema_version") != "midogpp_input_artifacts_v2"
        or payload.get("dataset_id") != "midogpp"
        or payload.get("experiment_id") != config.experiment_id
        or payload.get("stage") != "90_oracles_and_diagnostics"
        or payload.get("claim_scope") != "diagnostic_only"
    ):
        raise ProtocolError("Case-aware workspace provenance header drifted.")
    raw_rows = payload.get("input_artifacts")
    if not isinstance(raw_rows, list) or not all(
        isinstance(row, Mapping) for row in raw_rows
    ):
        raise ProtocolError("Case-aware provenance rows are malformed.")
    actual_ids = tuple(str(row.get("artifact_id")) for row in raw_rows)
    if (
        len(set(actual_ids)) != len(actual_ids)
        or actual_ids != tuple(sorted(config.input_artifact_ids))
    ):
        raise ProtocolError("Case-aware workspace provenance order drifted.")
    by_id = {str(row.get("artifact_id")): row for row in raw_rows}
    expected_paths = {
        EXPERT_BANK_ARTIFACT_ID: config.expert_bank_root,
        GENERATION_LOCK_ARTIFACT_ID: config.generation_lock_root,
        TEST_CONSUMPTION_LEDGER_ARTIFACT_ID: config.test_consumption_ledger_path.parent.parent,
        TEST_CACHE_ARTIFACT_ID: config.test_cache_root,
        TEST_MANIFEST_ARTIFACT_ID: config.test_manifest_path.parent,
        METADATA_PROFILE_ARTIFACT_ID: config.metadata_profile_root,
    }
    for artifact_id in config.input_artifact_ids:
        row = by_id.get(artifact_id)
        if (
            row is None
            or Path(str(row.get("resolved_path", ""))).resolve()
            != expected_paths[artifact_id].resolve()
            or row.get("exists") is not True
            or not isinstance(row.get("semantic_identities"), Mapping)
            or not isinstance(row.get("file_integrity"), Mapping)
        ):
            raise ProtocolError(f"Case-aware provenance drifted: {artifact_id}.")
    return {artifact_id: by_id[artifact_id] for artifact_id in config.input_artifact_ids}


def validate_active_diagnostic_workspace_binding(
    config: DiagnosticInputConfig,
) -> Mapping[str, object]:
    assert_input_fence(config)
    try:
        workspace = MidogppWorkspace.load()
        workspace.validate()
        experiment = workspace.get_experiment(config.experiment_id)
        output = workspace.artifacts[config.output_artifact_id]
    except (KeyError, ValueError, OSError) as exc:
        raise ProtocolError("Case-aware canonical workspace binding failed.") from exc
    if (
        experiment.status != "diagnostic"
        or experiment.stage != "90_oracles_and_diagnostics"
        or experiment.claim_scope != "diagnostic_only"
        or experiment.output_artifact_id != config.output_artifact_id
        or experiment.input_artifact_ids != tuple(config.input_artifact_ids)
        or output.stage != "90_oracles_and_diagnostics"
        or output.claim_scope != "diagnostic_only"
    ):
        raise ProtocolError("Case-aware experiment binding drifted.")
    return {
        "status": "PASS",
        "experiment_id": experiment.experiment_id,
        "output_artifact_id": experiment.output_artifact_id,
        "stage": experiment.stage,
        "claim_scope": experiment.claim_scope,
    }


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read case-aware JSON input: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"Case-aware JSON input must be an object: {path}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ProtocolError(f"Cannot hash case-aware input: {path}.") from exc
    return digest.hexdigest()


__all__ = (
    "DiagnosticInputConfig",
    "ValidatedLocks",
    "assert_input_fence",
    "load_label_free_test_frame",
    "load_metadata_similarity",
    "load_validated_locks",
    "validate_active_diagnostic_workspace_binding",
    "validate_pre_gpu_firewall",
    "validate_workspace_provenance",
)
