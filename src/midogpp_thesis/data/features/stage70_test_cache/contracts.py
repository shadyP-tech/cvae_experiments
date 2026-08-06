"""Frozen extraction and shard contracts for the descriptive Stage-70 cache."""

from __future__ import annotations

import re
from typing import Mapping

from midogpp_thesis.data.contract.stage70_target_evaluation.contracts import (
    AUTHORIZED_CONSUMER_EXPERIMENT_ID,
    ELIGIBLE_CENTERS,
    EVALUATION_SPLIT,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEST_ROWS_BY_CENTER,
    FRESH_EVIDENCE,
    PURPOSE,
    RESERVATION_ROW_FIELDS,
    semantic_sha256,
)


CACHE_NAME = "uniform_b_v2_descriptive_test_cache_v1"
CACHE_EXPERIMENT_ID = (
    "midogpp.frozen_policy_downstream.uniform_b_v2_descriptive_test_cache.v1"
)
CACHE_ARTIFACT_ID = "midogpp_virchow2_uniform_b_v2_descriptive_test_cache_seed42"
CANONICAL_OUTPUT_RELATIVE_ROOT = (
    "datasets/midogpp/derived/features/virchow2/"
    "uniform_b_v2_descriptive_test_cache_v1/seed42"
)
CACHE_SCHEMA_VERSION = "midogpp_stage70_descriptive_test_cache_v1"
FEATURE_EXTRACTOR_SCHEMA_VERSION = (
    "midogpp_stage70_descriptive_test_feature_extractor_v1"
)
REPRESENTATION_ID = "annotation_jpeg_fixed_center_b_v3"
FEATURE_DIM = 3_840
POOLING_ID = "fixed_center_rows6to9_cols6to9"
FIXED_WINDOW_START = (6, 6)
SHARD_METADATA_FIELDS = RESERVATION_ROW_FIELDS

MODEL_REF = "hf-hub:paige-ai/Virchow2"
MODEL_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
MODEL_CONFIG_SHA256 = (
    "7db445b996bb165e88fe70e826c2ebb530539a2b1d136aa16eeb847df5f1e3db"
)
CHECKPOINT_FILE_SHA256 = (
    "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
)
STATE_DICT_SHA256 = (
    "91084959869cb53bf76e5038e5dc8a8ddc1ef8359a886fa22c19b4e8c62e112a"
)
PREPROCESSING_CONFIG_HASH = "4fb7d9ab76d1da72"

# These names are forbidden in shard metadata regardless of spelling.  The
# validator also scans all metadata string values for historical filename
# encodings rather than trusting key names alone.
FORBIDDEN_METADATA_FIELDS = frozenset(
    {
        "label",
        "label_name",
        "sample_id",
        "image_path",
        "class",
        "class_id",
        "target",
        "target_value",
        "y",
        "y_true",
    }
)
LEGACY_OUTCOME_PATTERN = re.compile(r"(?:^|_)y[01](?=$|[^0-9])", re.IGNORECASE)


class Stage70TestCacheError(ValueError):
    """Raised when the descriptive test cache violates its frozen boundary."""


def stage70_extractor_protocol() -> dict[str, object]:
    """Return the predeclared cache identity authorized before extraction."""

    return {
        "schema_version": FEATURE_EXTRACTOR_SCHEMA_VERSION,
        "cache_name": CACHE_NAME,
        "cache_artifact_id": CACHE_ARTIFACT_ID,
        "cache_experiment_id": CACHE_EXPERIMENT_ID,
        "authorized_consumer_experiment_id": AUTHORIZED_CONSUMER_EXPERIMENT_ID,
        "purpose": PURPOSE,
        "fresh_evidence": FRESH_EVIDENCE,
        "evidence_status": "previously_consumed_test",
        "allowed_use": "descriptive_locked_model_scoring_only",
        "evaluation_split": EVALUATION_SPLIT,
        "eligible_centers": list(ELIGIBLE_CENTERS),
        "expected_row_count": EXPECTED_TEST_ROWS,
        "expected_rows_by_center": dict(EXPECTED_TEST_ROWS_BY_CENTER),
        "representation_id": REPRESENTATION_ID,
        "feature_dim": FEATURE_DIM,
        "pooling": POOLING_ID,
        "fixed_window_start": list(FIXED_WINDOW_START),
        "token_layout": {
            "cls_token_count": 1,
            "register_token_count": 4,
            "patch_grid_side": 16,
            "window_side": 4,
            "patch_order": "row-major",
            "token_width": 1280,
        },
        "model_identity": expected_model_identity(),
        "shard_metadata_fields": sorted(SHARD_METADATA_FIELDS),
        "source_location_access": "opaque_bound_row_bytes_only",
        "metric_computation": "absent",
    }


def stage70_extractor_protocol_hash() -> str:
    return semantic_sha256(stage70_extractor_protocol())


def expected_model_identity() -> dict[str, str]:
    """Return the immutable identity subset required from an extractor."""

    return {
        "schema_version": "midogpp_virchow2_pinned_identity_v1",
        "model_ref": MODEL_REF,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "checkpoint_file_sha256": CHECKPOINT_FILE_SHA256,
        "state_dict_sha256": STATE_DICT_SHA256,
        "preprocessing_config_hash": PREPROCESSING_CONFIG_HASH,
    }


def validate_model_identity(identity: Mapping[str, object]) -> dict[str, object]:
    """Validate the pinned identity while retaining safe preprocessing detail."""

    if not isinstance(identity, Mapping):
        raise Stage70TestCacheError(
            "Stage-70 feature extractor identity is missing."
        )
    expected = expected_model_identity()
    drift = {
        key: {"observed": identity.get(key), "expected": value}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if drift:
        raise Stage70TestCacheError(
            f"Stage-70 pinned Virchow2 identity drifted: {drift}."
        )
    normalized = dict(identity)
    if any(_contains_legacy_outcome(value) for value in normalized.values()):
        raise Stage70TestCacheError(
            "Stage-70 extractor identity contains a legacy outcome encoding."
        )
    return normalized


def _contains_legacy_outcome(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_legacy_outcome(key) or _contains_legacy_outcome(nested)
            for key, nested in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_legacy_outcome(item) for item in value)
    return isinstance(value, str) and LEGACY_OUTCOME_PATTERN.search(value) is not None


__all__ = (
    "CACHE_ARTIFACT_ID",
    "CACHE_EXPERIMENT_ID",
    "CACHE_NAME",
    "CACHE_SCHEMA_VERSION",
    "CANONICAL_OUTPUT_RELATIVE_ROOT",
    "CHECKPOINT_FILE_SHA256",
    "ELIGIBLE_CENTERS",
    "EVALUATION_SPLIT",
    "EXPECTED_TEST_ROWS",
    "EXPECTED_TEST_ROWS_BY_CENTER",
    "FEATURE_DIM",
    "FEATURE_EXTRACTOR_SCHEMA_VERSION",
    "FIXED_WINDOW_START",
    "FORBIDDEN_METADATA_FIELDS",
    "FRESH_EVIDENCE",
    "LEGACY_OUTCOME_PATTERN",
    "MODEL_CONFIG_SHA256",
    "MODEL_REF",
    "MODEL_REVISION",
    "POOLING_ID",
    "PREPROCESSING_CONFIG_HASH",
    "PURPOSE",
    "REPRESENTATION_ID",
    "SHARD_METADATA_FIELDS",
    "STATE_DICT_SHA256",
    "Stage70TestCacheError",
    "expected_model_identity",
    "stage70_extractor_protocol",
    "stage70_extractor_protocol_hash",
    "validate_model_identity",
)
