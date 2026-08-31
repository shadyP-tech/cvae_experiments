"""Prepare the dedicated HARP v2 cache under the unchanged case partition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..fixed_bank_harp_router_v1.preparation import (
    CANONICAL_EXPERT_BANK_LOCK_HASH,
    CANONICAL_GENERATION_LOCK_HASH,
    CANONICAL_MANIFEST_SHA256,
    CANONICAL_PARENT_LEDGER_SHA256,
    PARTITION_NAMESPACE,
    HarpPreparationIdentity,
    build_case_partition_payload,
    deterministic_case_partition,
    prepare_harp_consumed_test_inputs_with_identity,
)
from .identity import (
    EXPERIMENT_ID,
    PUBLICATION_STATUS,
    TERMINAL_DECISION,
    authorization_input_binding_payload,
)
from .input_surfaces import V2_CACHE_IDENTITY


V2_PREPARATION_IDENTITY = HarpPreparationIdentity(
    experiment_id=EXPERIMENT_ID,
    publication_status=PUBLICATION_STATUS,
    terminal_decision=TERMINAL_DECISION,
    prepared_inputs_schema="midogpp_harp_consumed_test_prepared_inputs_v2",
    # The schema and namespace deliberately remain v1 so assignments are the
    # exact predeclared whole-case partition; only cache/receipt identity changes.
    partition_schema="midogpp_harp_consumed_test_case_partition_v1",
    preparation_receipt_schema="midogpp_harp_consumed_test_preparation_receipt_v2",
    label_free_barrier_schema="midogpp_harp_consumed_test_label_free_barrier_v2",
    cache_identity=V2_CACHE_IDENTITY,
    preparation_receipt=Path("manifests/harp_v2_consumed_test_preparation_receipt.json"),
)
PREPARATION_RECEIPT = V2_PREPARATION_IDENTITY.preparation_receipt
LABEL_FREE_BARRIER = V2_PREPARATION_IDENTITY.label_free_barrier
LABEL_FREE_CONTENT_INDEX = V2_PREPARATION_IDENTITY.label_free_content_index
CASE_PARTITION = V2_PREPARATION_IDENTITY.case_partition


@dataclass(frozen=True, slots=True)
class HarpV2PreparedInputs:
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
        binding = authorization_input_binding_payload(
            expert_bank_lock_hash=CANONICAL_EXPERT_BANK_LOCK_HASH,
            generation_lock_hash=CANONICAL_GENERATION_LOCK_HASH,
            test_cache_content_sha256=self.cache_content_sha256,
            development_manifest_sha256=self.development_manifest_sha256,
            evaluation_manifest_sha256=self.evaluation_manifest_sha256,
            parent_ledger_sha256=self.parent_ledger_sha256,
        )
        return {
            "schema_version": "midogpp_harp_consumed_test_prepared_inputs_v2",
            "experiment_id": EXPERIMENT_ID,
            "cache_artifact_id": V2_CACHE_IDENTITY.artifact_id,
            "cache_root": str(self.cache_root),
            "development_manifest_path": str(self.development_manifest_path),
            "evaluation_manifest_path": str(self.evaluation_manifest_path),
            "test_cache_content_sha256": self.cache_content_sha256,
            "development_manifest_sha256": self.development_manifest_sha256,
            "evaluation_manifest_sha256": self.evaluation_manifest_sha256,
            "parent_ledger_sha256": self.parent_ledger_sha256,
            "partition_namespace": PARTITION_NAMESPACE,
            "partition_hash": self.partition_hash,
            "preparation_receipt_hash": self.preparation_receipt_hash,
            "proposed_amendment_input_binding": binding,
            "execution_amendment_created": False,
            "execution_authorized": False,
            "publication_status": PUBLICATION_STATUS,
            "terminal_decision": TERMINAL_DECISION,
            "fresh_evidence": False,
        }


def prepare_harp_consumed_test_inputs_v2(
    *,
    canonical_cache_root: str | Path,
    canonical_manifest_path: str | Path,
    parent_ledger_path: str | Path,
    cache_root: str | Path,
    development_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    expected_manifest_sha256: str = CANONICAL_MANIFEST_SHA256,
    expected_parent_ledger_sha256: str = CANONICAL_PARENT_LEDGER_SHA256,
) -> HarpV2PreparedInputs:
    prepared = prepare_harp_consumed_test_inputs_with_identity(
        canonical_cache_root=canonical_cache_root,
        canonical_manifest_path=canonical_manifest_path,
        parent_ledger_path=parent_ledger_path,
        cache_root=cache_root,
        development_manifest_path=development_manifest_path,
        evaluation_manifest_path=evaluation_manifest_path,
        identity=V2_PREPARATION_IDENTITY,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_parent_ledger_sha256=expected_parent_ledger_sha256,
    )
    return HarpV2PreparedInputs(
        cache_root=prepared.cache_root,
        development_manifest_path=prepared.development_manifest_path,
        evaluation_manifest_path=prepared.evaluation_manifest_path,
        cache_content_sha256=prepared.cache_content_sha256,
        development_manifest_sha256=prepared.development_manifest_sha256,
        evaluation_manifest_sha256=prepared.evaluation_manifest_sha256,
        parent_ledger_sha256=prepared.parent_ledger_sha256,
        partition_hash=prepared.partition_hash,
        preparation_receipt_hash=prepared.preparation_receipt_hash,
    )


__all__ = (
    "CASE_PARTITION",
    "HarpV2PreparedInputs",
    "LABEL_FREE_BARRIER",
    "LABEL_FREE_CONTENT_INDEX",
    "PARTITION_NAMESPACE",
    "PREPARATION_RECEIPT",
    "V2_PREPARATION_IDENTITY",
    "build_case_partition_payload",
    "deterministic_case_partition",
    "prepare_harp_consumed_test_inputs_v2",
)
