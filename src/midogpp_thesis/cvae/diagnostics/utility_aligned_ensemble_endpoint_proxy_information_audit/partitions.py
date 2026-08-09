"""Audit-owned deterministic two-case source-inner support partitions."""

from __future__ import annotations

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import deterministic_case_partitions
from .contracts import (
    CENTERS,
    EXPECTED_TOTAL_CASE_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
)
from .input_contracts import (
    FixedPartitionSurface,
    LabelFreeValidationFrame,
    ValidationRowIdentity,
    row_identity_hash,
)


SUPPORT_PARTITION_COLUMNS = (
    "schema_version",
    "row_ordinal",
    "manifest_row_index",
    "sample_id",
    "case_id",
    "center",
    "split",
    "partition_role",
    "center_partition_hash",
    "support_split_seed",
    "support_partition_namespace",
    "label_present",
)


def build_fixed_partition_surface(
    frame: LabelFreeValidationFrame, *, config_contract_hash: str
) -> FixedPartitionSurface:
    support_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    evaluation_by_center: dict[str, tuple[ValidationRowIdentity, ...]] = {}
    table: list[dict[str, object]] = []
    centers_payload: dict[str, object] = {}
    all_cases: set[str] = set()
    all_samples: set[str] = set()
    for center in CENTERS:
        original = frame.rows_by_center[center]
        partition = deterministic_case_partitions(
            [row.sample_id for row in original],
            [row.case_id for row in original],
            target_center=center,
            support_case_count=FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
            namespace=SUPPORT_PARTITION_NAMESPACE,
            split_seed=SUPPORT_SPLIT_SEED,
        )
        support_indices = set(partition.support_indices)
        support: list[ValidationRowIdentity] = []
        evaluation: list[ValidationRowIdentity] = []
        for local_index, raw in enumerate(original):
            role = "support" if local_index in support_indices else "evaluation"
            row = ValidationRowIdentity(
                row_ordinal=raw.row_ordinal,
                manifest_row_index=raw.manifest_row_index,
                sample_id=raw.sample_id,
                case_id=raw.case_id,
                center=center,
                partition_role=role,
            )
            (support if role == "support" else evaluation).append(row)
            table.append(
                {
                    "schema_version": (
                        "midogpp_stage90_ensemble_endpoint_proxy_audit_partition_row_v1"
                    ),
                    **row.identity_payload(),
                    "center_partition_hash": partition.partition_hash,
                    "support_split_seed": SUPPORT_SPLIT_SEED,
                    "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
                    "label_present": False,
                }
            )
        support_cases = tuple(sorted({row.case_id for row in support}))
        evaluation_cases = tuple(sorted({row.case_id for row in evaluation}))
        if (
            len(support_cases) != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
            or not evaluation_cases
            or set(support_cases) & set(evaluation_cases)
        ):
            raise ProtocolError("Proxy-audit support/evaluation partition drifted.")
        for row in (*support, *evaluation):
            if row.sample_id in all_samples:
                raise ProtocolError("Proxy-audit sample IDs are not unique.")
            all_samples.add(row.sample_id)
        for case_id in (*support_cases, *evaluation_cases):
            if case_id in all_cases:
                raise ProtocolError("Proxy-audit case IDs are not unique.")
            all_cases.add(case_id)
        support_by_center[center] = tuple(support)
        evaluation_by_center[center] = tuple(evaluation)
        centers_payload[center] = {
            "partition_hash": partition.partition_hash,
            "support_case_ids": list(support_cases),
            "evaluation_case_ids": list(evaluation_cases),
            "support_row_identity_hash": row_identity_hash(support),
            "evaluation_row_identity_hash": row_identity_hash(evaluation),
            "support_row_count": len(support),
            "evaluation_row_count": len(evaluation),
        }
    if len(all_cases) != EXPECTED_TOTAL_CASE_COUNT:
        raise ProtocolError("Proxy-audit total-case geometry drifted.")
    unhashed = {
        "schema_version": (
            "midogpp_stage90_ensemble_endpoint_proxy_audit_partition_lock_v1"
        ),
        "status": "LOCKED_FROM_LABEL_FREE_CACHE_IDENTITIES",
        "config_contract_hash": config_contract_hash,
        "validation_cache_binding_hash": frame.cache_binding_hash,
        "fixed_support_case_count_per_center": FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
        "support_split_seed": SUPPORT_SPLIT_SEED,
        "support_partition_namespace": SUPPORT_PARTITION_NAMESPACE,
        "centers": centers_payload,
        "manifest_opened": False,
        "labels_used": False,
        "whole_case": True,
        "support_evaluation_case_disjoint": True,
        "support_evaluation_sample_disjoint": True,
        "seed_or_patch_rows_are_independent_units": False,
    }
    return FixedPartitionSurface(
        support_rows_by_center=support_by_center,
        evaluation_rows_by_center=evaluation_by_center,
        table_rows=tuple(table),
        lock_payload={
            **unhashed,
            "support_partition_lock_hash": stable_hash(unhashed),
        },
    )


__all__ = ("SUPPORT_PARTITION_COLUMNS", "build_fixed_partition_surface")
