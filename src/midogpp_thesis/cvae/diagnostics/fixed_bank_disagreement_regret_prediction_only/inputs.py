"""Fail-closed label-free source projection and post-model test-cache loader."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np

from ....data.contract.stage70_target_evaluation.contracts import (
    CANONICAL_MANIFEST_SHA256,
)
from ....data.features.cache_io import load_cache_rows
from ....data.features.stage70_test_cache.contracts import (
    CACHE_ARTIFACT_ID as UNDERLYING_TEST_CACHE_ARTIFACT_ID,
    CACHE_NAME as UNDERLYING_TEST_CACHE_NAME,
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
from ...generation.contracts import GenerationLock
from ...protocol import ProtocolError
from ...runtime.artifact_io import sha256_file
from .constants import (
    CENTERS,
    EXPECTED_SOURCE_ROWS,
    EXPECTED_SOURCE_ROWS_BY_CENTER,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FEATURE_DIM,
)
from .experiment_contracts import (
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_LEDGER_AMENDMENT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TEST_CACHE_CONTENT_HASH,
    EXPECTED_TEST_CACHE_REPRESENTATION_ID,
    EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
    EXPECTED_TEST_CACHE_SEMANTIC_ID,
    EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256,
    EXPECTED_TRAIN_CACHE_SHA256,
    EXPERIMENT_ID,
    FORBIDDEN_INPUT_FRAGMENTS,
    INPUT_ARTIFACT_IDS,
    OUTPUT_ARTIFACT_ID,
    TEST_CACHE_ARTIFACT_ID,
    TRAIN_CACHE_ARTIFACT_ID,
)
from .hashing import canonical_hash
from .input_contracts import (
    LabelFreeSourceFrame,
    LabelFreeTestFrame,
    SourceRowIdentity,
    TestInferenceAdmission,
    TestRowIdentity,
    opaque_source_row_id,
    row_identity_hash,
)
from .ledger import load_validated_ledger_chain
from .workspace_inputs import (
    validate_active_diagnostic_workspace_binding as _workspace_binding,
    validate_workspace_provenance as _workspace_provenance,
)


class PredictionOnlyInputConfig(Protocol):
    experiment_id: str
    output_artifact_id: str
    input_artifact_ids: Sequence[str]
    expert_bank_root: Path
    generation_lock_root: Path
    train_cache_root: Path
    test_cache_root: Path
    test_consumption_ledger_path: Path
    ledger_amendment_path: Path
    expected_bank_lock_hash: str
    expected_generation_lock_hash: str
    expected_train_cache_sha256: str
    expected_manifest_sha256: str
    expected_test_cache_semantic_id: str
    expected_test_cache_representation_id: str
    expected_test_cache_content_hash: str
    expected_test_cache_row_order_hash: str
    expected_test_consumption_ledger_sha256: str
    expected_ledger_amendment_sha256: str


@dataclass(frozen=True)
class ValidatedLocks:
    """Validated promoted-generation and consumed-test protocol locks."""

    generation: GenerationLock
    test_consumption_ledger: Mapping[str, object]
    ledger_amendment: Mapping[str, object]


def assert_input_fence(config: PredictionOnlyInputConfig) -> None:
    """Reject every non-canonical or previous-result input before file access."""

    values = (
        *(str(value) for value in config.input_artifact_ids),
        str(config.expert_bank_root),
        str(config.generation_lock_root),
        str(config.train_cache_root),
        str(config.test_cache_root),
        str(config.test_consumption_ledger_path),
        str(config.ledger_amendment_path),
    )
    forbidden = [
        value
        for value in values
        if any(fragment in value.lower() for fragment in FORBIDDEN_INPUT_FRAGMENTS)
    ]
    if forbidden:
        raise ProtocolError(
            "Prediction-only diagnostic cannot consume a prior Stage-90 result, "
            f"prediction surface, scratch tree, or checkpoint: {forbidden}."
        )
    if tuple(config.input_artifact_ids) != INPUT_ARTIFACT_IDS:
        raise ProtocolError("Prediction-only diagnostic requires its exact six inputs.")
    if (
        config.experiment_id != EXPERIMENT_ID
        or config.output_artifact_id != OUTPUT_ARTIFACT_ID
        or config.output_artifact_id in config.input_artifact_ids
        or config.expected_bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or config.expected_generation_lock_hash != EXPECTED_GENERATION_LOCK_HASH
        or config.expected_test_consumption_ledger_sha256
        != EXPECTED_TEST_CONSUMPTION_LEDGER_SHA256
        or config.expected_ledger_amendment_sha256
        != EXPECTED_LEDGER_AMENDMENT_SHA256
    ):
        raise ProtocolError("Prediction-only input/output identity drifted.")


def load_validated_locks(config: PredictionOnlyInputConfig) -> ValidatedLocks:
    """Validate the neutral promoted bank lineage and local ledger chain."""

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
        raise ProtocolError("Prediction-only frozen-generation lineage drifted.")
    ledger = load_validated_ledger_chain(config)
    return ValidatedLocks(
        generation=generation,
        test_consumption_ledger=ledger.parent,
        ledger_amendment=ledger.amendment,
    )


def validate_pre_gpu_firewall(
    config: PredictionOnlyInputConfig,
    source_frame: LabelFreeSourceFrame,
    locks: ValidatedLocks | None = None,
) -> Mapping[str, object]:
    """Authorize source-stream GPU work without opening the test cache or labels."""

    assert_input_fence(config)
    validated = locks or load_validated_locks(config)
    promotion_config = load_promotion_config(
        config.expert_bank_root / "config.resolved.yaml"
    )
    checks = validate_promoted_bank(
        config.expert_bank_root, config=promotion_config, allow_pending=False
    )
    bank_index = _json(config.expert_bank_root / "manifests/expert_bank_index.json")
    records = bank_index.get("records")
    binding = source_frame.cache_binding
    amendment = validated.ledger_amendment
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
        or bank_index.get("bank_lock_hash") != EXPECTED_BANK_LOCK_HASH
        or validated.generation.bank_lock_hash != EXPECTED_BANK_LOCK_HASH
        or validated.generation.generation_lock_hash
        != EXPECTED_GENERATION_LOCK_HASH
        or binding.get("split") != "train"
        or binding.get("row_count") != EXPECTED_SOURCE_ROWS
        or binding.get("feature_dim") != FEATURE_DIM
        or binding.get("cache_sha256") != EXPECTED_TRAIN_CACHE_SHA256
        or binding.get("labels_in_typed_frame") is not False
        or binding.get("historical_sample_ids_persisted") is not False
        or binding.get("source_label_field_accessed_by_projection_code") is not False
        or binding.get("source_labels_physically_present_in_input_metadata") is not True
        or binding.get("single_consumer_alias_only") is not True
        or amendment.get("parent_sha256")
        != config.expected_test_consumption_ledger_sha256
        or amendment.get("previous_stage90_outputs_used") is not False
        or amendment.get("previous_stage90_scratch_or_checkpoints_used") is not False
        or amendment.get("previous_prediction_surfaces_used") is not False
        or amendment.get("target_cache_is_label_free") is not True
        or amendment.get("no_target_label_capability_created") is not True
    ):
        raise ProtocolError("Prediction-only pre-GPU firewall failed.")
    return {
        "schema_version": "midogpp_prediction_only_pre_gpu_firewall_v1",
        "status": "PASS",
        "bank_lock_hash": EXPECTED_BANK_LOCK_HASH,
        "generation_lock_hash": EXPECTED_GENERATION_LOCK_HASH,
        "expert_count": len(records),
        "fresh_source_only_training": True,
        "source_split": "train",
        "source_row_count": EXPECTED_SOURCE_ROWS,
        "source_cache_sha256": EXPECTED_TRAIN_CACHE_SHA256,
        "source_labels_opened": False,
        "test_cache_opened": False,
        "test_manifest_opened": False,
        "test_labels_opened": False,
        "test_labels_available": False,
        "test_split_previously_consumed": True,
        "ledger_amendment_sha256": EXPECTED_LEDGER_AMENDMENT_SHA256,
        "prior_stage90_output_or_prediction_surface_consumed": False,
        "gpu_work_authorized_for_source_streams_only": True,
    }


def validate_workspace_provenance(
    root: Path, config: PredictionOnlyInputConfig
) -> dict[str, Mapping[str, object]]:
    assert_input_fence(config)
    return _workspace_provenance(root, config)


def validate_active_diagnostic_workspace_binding(
    config: PredictionOnlyInputConfig,
) -> Mapping[str, object]:
    assert_input_fence(config)
    return _workspace_binding(config)


def load_label_free_source_frame(
    config: PredictionOnlyInputConfig,
) -> LabelFreeSourceFrame:
    """Project the canonical train cache into an outcome-free typed frame.

    The historical torch payload contains source labels for legacy consumers.
    This function never reads their values and emits only neutral row IDs.  The
    label capability reopens only this train-only cache after the source seal.
    """

    _assert_input_constants(config)
    root = Path(config.train_cache_root)
    cache_path = root / "embeddings/train.pt"
    if sha256_file(cache_path) != EXPECTED_TRAIN_CACHE_SHA256:
        raise ProtocolError("Prediction-only source cache bytes drifted.")
    _validate_train_cache_envelope(root)
    loaded = load_cache_rows(cache_path, expected_dim=FEATURE_DIM)
    if loaded.cache_sha256 != EXPECTED_TRAIN_CACHE_SHA256:
        raise ProtocolError("Prediction-only source cache hash drifted after load.")
    metadata = tuple(loaded.metadata)
    values = np.asarray(loaded.embeddings, dtype=np.float32)
    if values.shape != (EXPECTED_SOURCE_ROWS, FEATURE_DIM) or len(metadata) != len(values):
        raise ProtocolError("Prediction-only source cache geometry drifted.")

    indices_by_center: dict[str, list[int]] = {center: [] for center in CENTERS}
    projected_by_index: dict[int, tuple[str, str, str]] = {}
    raw_ids_seen: set[str] = set()
    opaque_ids_seen: set[str] = set()
    for index, metadata_row in enumerate(metadata):
        # Deliberately access only identity/split fields.  In particular the
        # historical ``label`` value is not referenced in this phase.
        raw_id = str(metadata_row.get("sample_id", ""))
        case_id = str(metadata_row.get("case_id", ""))
        center = str(metadata_row.get("center", ""))
        split = str(metadata_row.get("split", ""))
        if (
            not raw_id
            or not case_id
            or center not in CENTERS
            or split != "train"
            or raw_id in raw_ids_seen
        ):
            raise ProtocolError("Prediction-only source cache identity drifted.")
        opaque_id = opaque_source_row_id(
            raw_id, cache_sha256=EXPECTED_TRAIN_CACHE_SHA256
        )
        if opaque_id in opaque_ids_seen:
            raise ProtocolError("Prediction-only opaque source identity collided.")
        raw_ids_seen.add(raw_id)
        opaque_ids_seen.add(opaque_id)
        projected_by_index[index] = (opaque_id, case_id, center)
        indices_by_center[center].append(index)
    counts = {center: len(indices_by_center[center]) for center in CENTERS}
    if counts != EXPECTED_SOURCE_ROWS_BY_CENTER:
        raise ProtocolError("Prediction-only source center coverage drifted.")

    rows: list[SourceRowIdentity] = []
    arrays: list[np.ndarray] = []
    rows_by_center: dict[str, tuple[SourceRowIdentity, ...]] = {}
    ordinal = 0
    for center in CENTERS:
        center_rows: list[SourceRowIdentity] = []
        center_indices = indices_by_center[center]
        for cache_index in center_indices:
            opaque_id, case_id, observed_center = projected_by_index[cache_index]
            row = SourceRowIdentity(
                row_ordinal=ordinal,
                cache_row_index=cache_index,
                source_row_id=opaque_id,
                case_id=case_id,
                center=observed_center,
            )
            rows.append(row)
            center_rows.append(row)
            ordinal += 1
        arrays.append(np.asarray(values[center_indices], dtype=np.float32))
        rows_by_center[center] = tuple(center_rows)
    binding = {
        "schema_version": "midogpp_prediction_only_label_free_source_binding_v1",
        "cache_alias_artifact_id": TRAIN_CACHE_ARTIFACT_ID,
        "underlying_cache_artifact_id": (
            "midogpp_virchow2_uniform_b_canonical_train_cache_seed42"
        ),
        "representation_id": "annotation_jpeg_fixed_center_b_v3",
        "split": "train",
        "row_count": EXPECTED_SOURCE_ROWS,
        "rows_by_center": counts,
        "feature_dim": FEATURE_DIM,
        "cache_sha256": EXPECTED_TRAIN_CACHE_SHA256,
        "row_identity_hash": row_identity_hash(rows),
        "labels_in_typed_frame": False,
        "historical_sample_ids_persisted": False,
        "outcome_bearing_sample_ids_persisted": False,
        "source_label_field_accessed_by_projection_code": False,
        "source_labels_physically_present_in_input_metadata": True,
        "source_labels_are_posthoc": True,
        "single_consumer_alias_only": True,
    }
    return LabelFreeSourceFrame(
        embeddings=np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cache_binding=binding,
    )


def load_label_free_test_frame(
    config: PredictionOnlyInputConfig,
    *,
    admission: TestInferenceAdmission,
) -> LabelFreeTestFrame:
    """Load all consumed-test rows only after the source-trained model seal."""

    _assert_input_constants(config)
    if not isinstance(admission, TestInferenceAdmission):
        raise ProtocolError("Prediction-only test cache lacks typed post-model admission.")
    cache = load_validated_stage70_test_cache(config.test_cache_root)
    summary = dict(cache.summary)
    if (
        summary.get("status") != "PASS"
        or summary.get("manifest_sha256") != CANONICAL_MANIFEST_SHA256
        or summary.get("row_count") != EXPECTED_TEST_ROWS
        or summary.get("rows_by_center") != EXPECTED_TEST_ROWS_BY_CENTER
        or summary.get("content_hash") != EXPECTED_TEST_CACHE_CONTENT_HASH
        or summary.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or EXPECTED_TEST_CACHE_SEMANTIC_ID != UNDERLYING_TEST_CACHE_NAME
        or EXPECTED_TEST_CACHE_REPRESENTATION_ID != REPRESENTATION_ID
        or summary.get("fresh_evidence") is not False
    ):
        raise ProtocolError("Prediction-only consumed-test cache failed validation.")
    arrays: list[np.ndarray] = []
    rows: list[TestRowIdentity] = []
    rows_by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
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
        rows_by_center[center] = tuple(center_rows)
        shard_hashes[center] = shard.shard_sha256
    binding = _test_cache_binding(
        admission=admission,
        cache_content_hash=str(summary.get("content_hash")),
        row_order_hash=str(summary.get("row_order_hash")),
        shard_hashes=shard_hashes,
    )
    return LabelFreeTestFrame(
        embeddings=np.ascontiguousarray(np.concatenate(arrays, axis=0), dtype=np.float32),
        rows=tuple(rows),
        rows_by_center=rows_by_center,
        cache_binding=binding,
        admission=admission,
    )


def expected_test_cache_binding_hash_from_provenance(
    config: PredictionOnlyInputConfig,
    *,
    admission: TestInferenceAdmission,
    provenance: Mapping[str, Mapping[str, object]],
) -> str:
    """Rebuild the label-free test binding without reopening the test cache."""

    _assert_input_constants(config)
    row = provenance.get(TEST_CACHE_ARTIFACT_ID)
    if not isinstance(row, Mapping):
        raise ProtocolError("Prediction-only test-cache provenance is absent.")
    identities = row.get("semantic_identities")
    integrity = row.get("file_integrity")
    files = integrity.get("files") if isinstance(integrity, Mapping) else None
    if (
        not isinstance(identities, Mapping)
        or identities.get("cache_name") != EXPECTED_TEST_CACHE_SEMANTIC_ID
        or identities.get("content_hash") != EXPECTED_TEST_CACHE_CONTENT_HASH
        or identities.get("row_order_hash") != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
        or identities.get("row_count") != str(EXPECTED_TEST_ROWS)
        or identities.get("feature_dim") != str(FEATURE_DIM)
        or identities.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
        or identities.get("representation_id")
        != EXPECTED_TEST_CACHE_REPRESENTATION_ID
        or identities.get("split") != "test"
        or identities.get("experiment_fenced") != "true"
        or identities.get("fresh_evidence") != "false"
        or identities.get("labels_persisted") != "false"
        or identities.get("labels_absent") != "true"
        or identities.get("sample_ids_persisted") != "false"
        or identities.get("image_paths_persisted") != "false"
        or identities.get("metadata_artifact_used") != "false"
        or identities.get("previous_stage90_output_used") != "false"
        or identities.get("authorized_consumer_experiment_ids") != EXPERIMENT_ID
        or not isinstance(files, list)
        or not all(isinstance(value, Mapping) for value in files)
    ):
        raise ProtocolError("Prediction-only test-cache provenance drifted.")
    by_path = {str(value.get("path")): value for value in files}
    shard_hashes: dict[str, str] = {}
    for center in CENTERS:
        member = f"embeddings/by_center/center_{center}.pt"
        file_row = by_path.get(member)
        computed = file_row.get("computed") if isinstance(file_row, Mapping) else None
        digest = computed.get("sha256") if isinstance(computed, Mapping) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or file_row.get("exists") is not True
        ):
            raise ProtocolError(
                "Prediction-only test-cache shard provenance drifted."
            )
        shard_hashes[center] = digest
    binding = _test_cache_binding(
        admission=admission,
        cache_content_hash=EXPECTED_TEST_CACHE_CONTENT_HASH,
        row_order_hash=EXPECTED_TEST_CACHE_ROW_ORDER_HASH,
        shard_hashes=shard_hashes,
    )
    return canonical_hash(binding)


def _test_cache_binding(
    *,
    admission: TestInferenceAdmission,
    cache_content_hash: str,
    row_order_hash: str,
    shard_hashes: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_prediction_only_consumed_test_binding_v1",
        "cache_alias_artifact_id": TEST_CACHE_ARTIFACT_ID,
        "underlying_cache_artifact_id": UNDERLYING_TEST_CACHE_ARTIFACT_ID,
        "underlying_cache_name": UNDERLYING_TEST_CACHE_NAME,
        "representation_id": REPRESENTATION_ID,
        "split": "test",
        "manifest_sha256": CANONICAL_MANIFEST_SHA256,
        "row_count": EXPECTED_TEST_ROWS,
        "rows_by_center": dict(EXPECTED_TEST_ROWS_BY_CENTER),
        "feature_dim": FEATURE_DIM,
        "cache_content_hash": cache_content_hash,
        "row_order_hash": row_order_hash,
        "shard_sha256_by_center": dict(shard_hashes),
        "test_inference_admission_hash": admission.admission_hash,
        "source_prediction_seal_hash": admission.source_prediction_seal_hash,
        "action_classifier_bank_seal_hash": admission.action_classifier_bank_seal_hash,
        "regret_model_bank_seal_hash": admission.regret_model_bank_seal_hash,
        "labels_persisted": False,
        "manifest_opened": False,
        "target_labels_available": False,
        "test_scoring_permitted": False,
        "classifier_refit_permitted": False,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "single_consumer_alias_only": True,
    }


def assert_train_test_disjoint(
    source_frame: LabelFreeSourceFrame,
    test_frame: LabelFreeTestFrame,
) -> Mapping[str, object]:
    source_cases = {row.case_id for row in source_frame.rows}
    test_cases = {row.case_id for row in test_frame.rows}
    overlap = source_cases.intersection(test_cases)
    source_ids = {row.source_row_id for row in source_frame.rows}
    test_ids = {row.evaluation_row_id for row in test_frame.rows}
    if overlap or source_ids.intersection(test_ids):
        raise ProtocolError("Prediction-only train/test identity overlap detected.")
    payload = {
        "schema_version": "midogpp_prediction_only_train_test_disjointness_v1",
        "status": "PASS",
        "source_row_count": len(source_frame.rows),
        "test_row_count": len(test_frame.rows),
        "source_case_count": len(source_cases),
        "test_case_count": len(test_cases),
        "case_overlap_count": 0,
        "opaque_row_identity_overlap_count": 0,
        "source_split": "train",
        "test_split": "test",
    }
    return {**payload, "audit_hash": canonical_hash(payload)}


def _assert_input_constants(config: PredictionOnlyInputConfig) -> None:
    if (
        str(config.expected_train_cache_sha256) != EXPECTED_TRAIN_CACHE_SHA256
        or str(config.expected_manifest_sha256) != EXPECTED_MANIFEST_SHA256
        or str(config.expected_test_cache_semantic_id)
        != EXPECTED_TEST_CACHE_SEMANTIC_ID
        or str(config.expected_test_cache_representation_id)
        != EXPECTED_TEST_CACHE_REPRESENTATION_ID
        or str(config.expected_test_cache_content_hash)
        != EXPECTED_TEST_CACHE_CONTENT_HASH
        or str(config.expected_test_cache_row_order_hash)
        != EXPECTED_TEST_CACHE_ROW_ORDER_HASH
    ):
        raise ProtocolError("Prediction-only cache identities drifted from config.")


def _validate_train_cache_envelope(root: Path) -> None:
    required = (
        "embeddings/train.pt",
        "manifests/frozen_cache_protocol.json",
        "manifests/content_index.json",
        "reports/cache_builder_report.json",
        "reports/validation_report.json",
    )
    if any(not (root / member).is_file() for member in required):
        raise ProtocolError("Prediction-only source cache bundle is incomplete.")
    frozen = _json(root / "manifests/frozen_cache_protocol.json")
    report = _json(root / "reports/cache_builder_report.json")
    validation = _json(root / "reports/validation_report.json")
    if (
        frozen.get("representation_id") != "annotation_jpeg_fixed_center_b_v3"
        or frozen.get("row_count") != EXPECTED_SOURCE_ROWS
        or frozen.get("feature_dim") != FEATURE_DIM
        or frozen.get("labels_used_for_feature_construction") is not False
        or frozen.get("test_rows_present") is not False
        or report.get("status") != "PASS"
        or report.get("numeric_transformation") != "none"
        or validation.get("status") != "PASS"
    ):
        raise ProtocolError("Prediction-only source cache protocol drifted.")
    content = _json(root / "manifests/content_index.json")
    rows = content.get("files")
    if not isinstance(rows, list):
        raise ProtocolError("Prediction-only source content index is malformed.")
    observed: set[str] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ProtocolError("Prediction-only source content row is malformed.")
        relative = str(raw.get("path", ""))
        member = root / relative
        if not member.is_file() or sha256_file(member) != raw.get("sha256"):
            raise ProtocolError("Prediction-only source content member drifted.")
        observed.add(relative)
    expected = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "content_index.json"
    }
    if observed != expected:
        raise ProtocolError("Prediction-only source content coverage drifted.")


def _json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read prediction-only input metadata: {path}.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("Prediction-only input metadata must be an object.")
    return value


__all__ = (
    "PredictionOnlyInputConfig",
    "ValidatedLocks",
    "assert_input_fence",
    "assert_train_test_disjoint",
    "expected_test_cache_binding_hash_from_provenance",
    "load_label_free_source_frame",
    "load_label_free_test_frame",
    "load_validated_locks",
    "validate_active_diagnostic_workspace_binding",
    "validate_pre_gpu_firewall",
    "validate_workspace_provenance",
)
