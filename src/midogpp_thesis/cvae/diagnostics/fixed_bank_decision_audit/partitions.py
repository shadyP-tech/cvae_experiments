"""Fixed eight-case consumed-test support/evaluation partition."""

from __future__ import annotations

from ....common.hashing import stable_hash
from ...protocol import ProtocolError
from ...routing.dense_residual_soft_router import deterministic_case_partitions
from .experiment_contracts import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
)
from .input_contracts import (
    FixedTestPartitionSurface,
    LabelFreeTestFrame,
    TestRowIdentity,
    row_identity_hash,
)


SUPPORT_PARTITION_COLUMNS = (
    "schema_version",
    "row_ordinal",
    "manifest_row_index",
    "evaluation_row_id",
    "case_id",
    "center",
    "split",
    "partition_role",
    "center_partition_hash",
    "support_split_seed",
    "support_partition_namespace",
    "label_present",
)


def build_fixed_test_partition_surface(
    frame: LabelFreeTestFrame,
    *,
    config_contract_hash: str,
    support_case_count: int = FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    split_seed: int = SUPPORT_SPLIT_SEED,
    namespace: str = SUPPORT_PARTITION_NAMESPACE,
) -> FixedTestPartitionSurface:
    if (
        support_case_count != FIXED_SUPPORT_CASE_COUNT_PER_CENTER
        or split_seed != SUPPORT_SPLIT_SEED
        or namespace != SUPPORT_PARTITION_NAMESPACE
        or not config_contract_hash
    ):
        raise ProtocolError("Fixed-bank eight-case partition contract drifted.")
    support_by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
    evaluation_by_center: dict[str, tuple[TestRowIdentity, ...]] = {}
    table: list[dict[str, object]] = []
    centers_payload: dict[str, object] = {}
    all_cases: set[str] = set()
    all_rows: set[str] = set()
    support_case_total = 0
    evaluation_case_total = 0
    for center in CENTERS:
        original = frame.rows_by_center[center]
        available_cases = {row.case_id for row in original}
        expected_case_count = EXPECTED_CASE_COUNTS_BY_CENTER[center]
        if len(available_cases) != expected_case_count:
            raise ProtocolError(
                f"Center {center} case count drifted: "
                f"{len(available_cases)} != {expected_case_count}."
            )
        partition = deterministic_case_partitions(
            [row.evaluation_row_id for row in original],
            [row.case_id for row in original],
            target_center=center,
            support_case_count=support_case_count,
            namespace=namespace,
            split_seed=split_seed,
        )
        support_indices = set(partition.support_indices)
        support: list[TestRowIdentity] = []
        evaluation: list[TestRowIdentity] = []
        for local_index, raw in enumerate(original):
            role = "support" if local_index in support_indices else "evaluation"
            row = TestRowIdentity(
                row_ordinal=raw.row_ordinal,
                manifest_row_index=raw.manifest_row_index,
                evaluation_row_id=raw.evaluation_row_id,
                case_id=raw.case_id,
                center=center,
                partition_role=role,
            )
            (support if role == "support" else evaluation).append(row)
            table.append(
                {
                    "schema_version": "midogpp_stage90_fixed_bank_test_partition_row_v1",
                    **row.identity_payload(),
                    "center_partition_hash": partition.partition_hash,
                    "support_split_seed": split_seed,
                    "support_partition_namespace": namespace,
                    "label_present": False,
                }
            )
        support_cases = tuple(sorted({row.case_id for row in support}))
        evaluation_cases = tuple(sorted({row.case_id for row in evaluation}))
        if (
            len(support_cases) != support_case_count
            or len(evaluation_cases)
            != EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER[center]
            or set(support_cases).intersection(evaluation_cases)
        ):
            raise ProtocolError("Fixed-bank support/evaluation partition drifted.")
        for row in (*support, *evaluation):
            if row.evaluation_row_id in all_rows:
                raise ProtocolError("Fixed-bank test row identities are duplicated.")
            all_rows.add(row.evaluation_row_id)
        for case_id in (*support_cases, *evaluation_cases):
            if case_id in all_cases:
                raise ProtocolError("MIDOG++ case identity occurs in multiple centers.")
            all_cases.add(case_id)
        support_case_total += len(support_cases)
        evaluation_case_total += len(evaluation_cases)
        support_by_center[center] = tuple(support)
        evaluation_by_center[center] = tuple(evaluation)
        centers_payload[center] = {
            "partition_hash": partition.partition_hash,
            "available_case_count": len(available_cases),
            "support_case_ids": list(support_cases),
            "evaluation_case_ids": list(evaluation_cases),
            "support_row_identity_hash": row_identity_hash(support),
            "evaluation_row_identity_hash": row_identity_hash(evaluation),
            "support_row_count": len(support),
            "evaluation_row_count": len(evaluation),
        }
    if (
        len(all_cases) != EXPECTED_TOTAL_CASE_COUNT
        or support_case_total != EXPECTED_SUPPORT_CASE_COUNT
        or evaluation_case_total != EXPECTED_EVALUATION_CASE_COUNT
    ):
        raise ProtocolError("Fixed-bank global 72/146 case geometry drifted.")
    unhashed = {
        "schema_version": "midogpp_stage90_fixed_bank_test_partition_lock_v1",
        "status": "LOCKED_FROM_LABEL_FREE_CONSUMED_TEST_IDENTITIES",
        "config_contract_hash": config_contract_hash,
        "test_cache_binding_hash": frame.cache_binding_hash,
        "fixed_support_case_count_per_center": support_case_count,
        "support_split_seed": split_seed,
        "support_partition_namespace": namespace,
        "total_case_count": len(all_cases),
        "support_case_count_total": support_case_total,
        "evaluation_case_count_total": evaluation_case_total,
        "centers": centers_payload,
        "manifest_opened": False,
        "labels_used": False,
        "whole_case": True,
        "support_evaluation_case_disjoint": True,
        "support_evaluation_row_disjoint": True,
        "test_split_previously_consumed": True,
        "fresh_evidence": False,
        "seed_or_patch_rows_are_independent_units": False,
        "partition_owned_by_experiment": True,
    }
    return FixedTestPartitionSurface(
        support_rows_by_center=support_by_center,
        evaluation_rows_by_center=evaluation_by_center,
        table_rows=tuple(table),
        lock_payload={
            **unhashed,
            "support_partition_lock_hash": stable_hash(unhashed),
        },
    )


__all__ = ("SUPPORT_PARTITION_COLUMNS", "build_fixed_test_partition_surface")
