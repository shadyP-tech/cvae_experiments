"""Immutable identities and data contracts for HARP v21 input preparation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .identity import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    authorization_input_binding_payload,
)
from .input_surfaces import HarpConsumedCacheIdentity, V21_CACHE_IDENTITY


# Immutable all-test cache identity.  The compatibility names are retained for
# callers that predate the source-train/full-test v21 preparation contract.
CANONICAL_TEST_CACHE_CONTENT_HASH = (
    "df0bdbf64881ee000fe7c56bc486724313accf373ef8e90896344f8d03d187db"
)
CANONICAL_TEST_CACHE_ROW_ORDER_HASH = (
    "bd1a85b95496203500bfe2dc5232f8bfb383e73d222a8ba083e81b2c6b33c389"
)
CANONICAL_TEST_MANIFEST_SHA256 = (
    "db661ac7e3dbafde8e283528de6706ab35f2c26629b389706c4504e458cc5869"
)
CANONICAL_CACHE_CONTENT_HASH = CANONICAL_TEST_CACHE_CONTENT_HASH
CANONICAL_CACHE_ROW_ORDER_HASH = CANONICAL_TEST_CACHE_ROW_ORDER_HASH
CANONICAL_MANIFEST_SHA256 = CANONICAL_TEST_MANIFEST_SHA256

# The train tensor is the sole outcome-bearing source input.  The remaining
# members authenticate its construction protocol without treating any label as
# a feature.  The prepared cache persists only opaque source row identifiers.
CANONICAL_SOURCE_TRAIN_TENSOR_SHA256 = (
    "1ed7602f225c592a6f8103b24ebfc93f72dc6d5d0c27565566a8b2260783d1dc"
)
CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256 = (
    "a4faf27a427cfb424789e5592048aa748a057f37124566d46b8b6c557e2bfe69"
)
CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256 = (
    "307991668f11454da69e3798feb23a2e899e1a00c2ee5132b031e7f7fb9ab82e"
)
CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256 = (
    "3e3c40449196dc6db9fe0ab982defa86afb1094e3d958e944875396bc363b0ec"
)
CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256 = (
    "e8b69f557ea92ac8e70a20e504150aba1c947f2b47f735b34e3ca7147efcf6b7"
)
CANONICAL_PARENT_LEDGER_SHA256 = (
    "8b16eae7bfdb5d20945e8ba3e02447ec74ab857adae441e493b2e37114feab16"
)
CANONICAL_TEST_CACHE_NAME = "uniform_b_v2_descriptive_test_cache_v1"
CANONICAL_SOURCE_TRAIN_CACHE_NAME = "uniform_b_canonical_train_cache_v1"
CANONICAL_CACHE_NAME = CANONICAL_TEST_CACHE_NAME
# This is the immutable physical source-frame identity, not the HARP execution
# revision. HARP v21 owns a distinct derived cache while deliberately reusing
# the byte-identical Uniform-B source representation produced as v3.
CANONICAL_REPRESENTATION = "annotation_jpeg_fixed_center_b_v3"
CANONICAL_EXPERT_BANK_LOCK_HASH = "9972a41dcd4814cd"
CANONICAL_GENERATION_LOCK_HASH = "34e551425710362e"
EXPECTED_SOURCE_TRAIN_ROW_COUNT = 9648
EXPECTED_SOURCE_TRAIN_CASE_COUNT = 216
EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER = {
    "0": 1786,
    "1": 742,
    "2": 3404,
    "3": 764,
    "5": 626,
    "6": 366,
    "7": 498,
    "8": 1116,
    "9": 346,
}
EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER = {
    "0": 22,
    "1": 20,
    "2": 25,
    "3": 38,
    "5": 22,
    "6": 20,
    "7": 22,
    "8": 22,
    "9": 25,
}
EXPECTED_TARGET_TEST_ROW_COUNT = 9928
EXPECTED_TARGET_TEST_CASE_COUNT = 218
EXPECTED_TARGET_TEST_ROWS_BY_CENTER = {
    "0": 1532,
    "1": 866,
    "2": 3210,
    "3": 1278,
    "5": 628,
    "6": 742,
    "7": 282,
    "8": 726,
    "9": 664,
}
EXPECTED_TARGET_TEST_CASES_BY_CENTER = {
    "0": 23,
    "1": 20,
    "2": 24,
    "3": 39,
    "5": 23,
    "6": 23,
    "7": 21,
    "8": 22,
    "9": 23,
}
# Composite totals are used only for closed-world cache validation.  Scientific
# The single pooled policy uses every train case under its known center q; the
# 218 test cases never enter fitting, calibration, or selection.
EXPECTED_ROW_COUNT = EXPECTED_SOURCE_TRAIN_ROW_COUNT + EXPECTED_TARGET_TEST_ROW_COUNT
EXPECTED_CASE_COUNT = EXPECTED_SOURCE_TRAIN_CASE_COUNT + EXPECTED_TARGET_TEST_CASE_COUNT
EXPECTED_ROWS_BY_CENTER = EXPECTED_TARGET_TEST_ROWS_BY_CENTER
EXPECTED_CASES_BY_CENTER = EXPECTED_TARGET_TEST_CASES_BY_CENTER
PARTITION_NAMESPACE = "midogpp_harp_v21_source_train_development_full_test_roles_v1"
PREPARATION_RECEIPT = Path("manifests/harp_v21_consumed_test_preparation_receipt.json")
LABEL_FREE_BARRIER = Path("manifests/harp_v21_label_free_partition_barrier.json")
LABEL_FREE_CONTENT_INDEX = Path("manifests/harp_v21_label_free_content_index.json")
CASE_PARTITION = Path("manifests/harp_v21_case_partition.json")
METADATA_FIELDS = {
    "evaluation_row_id",
    "contract_row_index",
    "case_id",
    "center",
    "split",
}
LEGACY_LABEL = re.compile(r"(?:^|_)y[01](?=$|[^0-9])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class HarpPreparationIdentity:
    """Closed execution-revision identity for deterministic cache preparation."""

    experiment_id: str
    publication_status: str
    terminal_decision: str
    prepared_inputs_schema: str
    partition_schema: str
    preparation_receipt_schema: str
    label_free_barrier_schema: str
    cache_identity: HarpConsumedCacheIdentity
    preparation_receipt: Path = PREPARATION_RECEIPT
    label_free_barrier: Path = LABEL_FREE_BARRIER
    label_free_content_index: Path = LABEL_FREE_CONTENT_INDEX
    case_partition: Path = CASE_PARTITION


V21_PREPARATION_IDENTITY = HarpPreparationIdentity(
    experiment_id=EXPERIMENT_ID,
    publication_status=PUBLICATION_STATUS,
    terminal_decision=TERMINAL_DECISION,
    prepared_inputs_schema="midogpp_harp_source_train_support_full_test_prepared_inputs_v21",
    partition_schema="midogpp_harp_source_train_support_full_test_case_roles_v21",
    preparation_receipt_schema=(
        "midogpp_harp_source_train_support_full_test_preparation_receipt_v21"
    ),
    label_free_barrier_schema=(
        "midogpp_harp_source_train_support_full_test_label_free_barrier_v21"
    ),
    cache_identity=V21_CACHE_IDENTITY,
)


@dataclass(frozen=True, slots=True)
class HarpPreparedInputData:
    """Internal prepared-input receipt fields for this fenced package."""

    cache_root: Path
    development_manifest_path: Path
    evaluation_manifest_path: Path
    cache_content_sha256: str
    development_manifest_sha256: str
    evaluation_manifest_sha256: str
    parent_ledger_sha256: str
    partition_hash: str
    preparation_receipt_hash: str


@dataclass(frozen=True, slots=True)
class CanonicalFrameRow:
    center: str
    case_id: str
    sample_id: str
    contract_row_index: int
    center_row_index: int
    source_split: str


@dataclass(frozen=True, slots=True)
class CanonicalLabelBlindFrame:
    rows_by_center: Mapping[str, tuple[CanonicalFrameRow, ...]]
    embeddings_by_center: Mapping[str, np.ndarray]
    cache_content_hash: str
    row_order_hash: str
    source_member_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CanonicalLabelBlindCacheIdentity:
    """Closed-world byte identity of the immutable consumed-test cache."""

    root: Path
    content_hash: str
    member_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class HarpV21PreparedInputs:
    cache_root: Path
    development_manifest_path: Path
    evaluation_manifest_path: Path
    cache_content_sha256: str
    development_manifest_sha256: str
    evaluation_manifest_sha256: str
    parent_ledger_sha256: str
    partition_hash: str
    preparation_receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        amendment_binding = authorization_input_binding_payload(
            expert_bank_lock_hash=CANONICAL_EXPERT_BANK_LOCK_HASH,
            generation_lock_hash=CANONICAL_GENERATION_LOCK_HASH,
            test_cache_content_sha256=self.cache_content_sha256,
            development_manifest_sha256=self.development_manifest_sha256,
            evaluation_manifest_sha256=self.evaluation_manifest_sha256,
            parent_ledger_sha256=self.parent_ledger_sha256,
        )
        return {
            "schema_version": "midogpp_harp_source_train_support_full_test_prepared_inputs_v21",
            "cache_root": str(self.cache_root),
            "development_manifest_path": str(self.development_manifest_path),
            "evaluation_manifest_path": str(self.evaluation_manifest_path),
            "test_cache_content_sha256": self.cache_content_sha256,
            "development_manifest_sha256": self.development_manifest_sha256,
            "evaluation_manifest_sha256": self.evaluation_manifest_sha256,
            "evaluation_artifact_kind": "sealed_label_free_release_descriptor",
            "evaluation_truth_rows_published_during_preparation": False,
            "evaluation_release_requires_frozen_route_receipt": True,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "partition_hash": self.partition_hash,
            "preparation_receipt_hash": self.preparation_receipt_hash,
            "proposed_amendment_input_binding": amendment_binding,
            "execution_amendment_created": False,
            "execution_authorized": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }


__all__ = (
    "CANONICAL_CACHE_CONTENT_HASH",
    "CANONICAL_CACHE_ROW_ORDER_HASH",
    "CANONICAL_MANIFEST_SHA256",
    "CANONICAL_PARENT_LEDGER_SHA256",
    "CANONICAL_CACHE_NAME",
    "CANONICAL_TEST_CACHE_CONTENT_HASH",
    "CANONICAL_TEST_CACHE_ROW_ORDER_HASH",
    "CANONICAL_TEST_MANIFEST_SHA256",
    "CANONICAL_TEST_CACHE_NAME",
    "CANONICAL_SOURCE_TRAIN_TENSOR_SHA256",
    "CANONICAL_SOURCE_TRAIN_PROTOCOL_SHA256",
    "CANONICAL_SOURCE_TRAIN_CONTENT_INDEX_SHA256",
    "CANONICAL_SOURCE_TRAIN_BUILDER_REPORT_SHA256",
    "CANONICAL_SOURCE_TRAIN_VALIDATION_REPORT_SHA256",
    "CANONICAL_SOURCE_TRAIN_CACHE_NAME",
    "CANONICAL_REPRESENTATION",
    "CANONICAL_EXPERT_BANK_LOCK_HASH",
    "CANONICAL_GENERATION_LOCK_HASH",
    "EXPECTED_ROW_COUNT",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_ROWS_BY_CENTER",
    "EXPECTED_CASES_BY_CENTER",
    "EXPECTED_SOURCE_TRAIN_ROW_COUNT",
    "EXPECTED_SOURCE_TRAIN_CASE_COUNT",
    "EXPECTED_SOURCE_TRAIN_ROWS_BY_CENTER",
    "EXPECTED_SOURCE_TRAIN_CASES_BY_CENTER",
    "EXPECTED_TARGET_TEST_ROW_COUNT",
    "EXPECTED_TARGET_TEST_CASE_COUNT",
    "EXPECTED_TARGET_TEST_ROWS_BY_CENTER",
    "EXPECTED_TARGET_TEST_CASES_BY_CENTER",
    "PARTITION_NAMESPACE",
    "PREPARATION_RECEIPT",
    "LABEL_FREE_BARRIER",
    "LABEL_FREE_CONTENT_INDEX",
    "CASE_PARTITION",
    "METADATA_FIELDS",
    "LEGACY_LABEL",
    "HarpPreparationIdentity",
    "V21_PREPARATION_IDENTITY",
    "HarpPreparedInputData",
    "CanonicalFrameRow",
    "CanonicalLabelBlindFrame",
    "CanonicalLabelBlindCacheIdentity",
    "HarpV21PreparedInputs",
)
