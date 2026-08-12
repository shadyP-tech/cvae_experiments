"""Label-free lexical whole-case partition for the consumed MIDOG++ test set."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import (
    CENTERS,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_CASE_COUNT,
    EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER,
    EXPECTED_EVALUATION_ROW_COUNT,
    EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER,
    EXPECTED_SUPPORT_CASE_COUNT,
    EXPECTED_SUPPORT_ROW_COUNT,
    EXPECTED_TEST_ROW_COUNT,
    EXPECTED_TOTAL_CASE_COUNT,
    SUPPORT_CASE_COUNT_PER_CENTER,
    SUPPORT_PARTITION_NAMESPACE,
)


@dataclass(frozen=True)
class LabelFreeCaseRow:
    """One opaque test-row identity; it deliberately has no label field."""

    row_ordinal: int
    manifest_row_index: int
    evaluation_row_id: str
    case_id: str
    center: str
    partition_role: str = "unassigned"
    split: str = "test"
    row_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.row_ordinal, bool)
            or not isinstance(self.row_ordinal, Integral)
            or int(self.row_ordinal) < 0
            or isinstance(self.manifest_row_index, bool)
            or not isinstance(self.manifest_row_index, Integral)
            or int(self.manifest_row_index) < 0
            or not _canonical_text(self.evaluation_row_id)
            or not _canonical_text(self.case_id)
            or self.center not in CENTERS
            or self.partition_role not in {"unassigned", "support", "evaluation"}
            or self.split != "test"
        ):
            raise ProtocolError("Consumed-test row identity is malformed.")
        object.__setattr__(self, "row_ordinal", int(self.row_ordinal))
        object.__setattr__(self, "manifest_row_index", int(self.manifest_row_index))
        object.__setattr__(
            self,
            "row_hash",
            canonical_sha256(
                {
                    "schema_version": "midogpp_consumed_test_label_free_row_v1",
                    **self.identity_payload(),
                }
            ),
        )

    @property
    def sample_id(self) -> str:
        return self.evaluation_row_id

    def identity_payload(self) -> dict[str, object]:
        return {
            "row_ordinal": self.row_ordinal,
            "manifest_row_index": self.manifest_row_index,
            "evaluation_row_id": self.evaluation_row_id,
            "case_id": self.case_id,
            "center": self.center,
            "split": self.split,
            "partition_role": self.partition_role,
        }


@dataclass(frozen=True)
class CenterCasePartition:
    center: str
    namespace: str
    support_case_ids: tuple[str, ...]
    evaluation_case_ids: tuple[str, ...]
    support_rows: tuple[LabelFreeCaseRow, ...]
    evaluation_rows: tuple[LabelFreeCaseRow, ...]
    partition_hash: str

    def __post_init__(self) -> None:
        if (
            self.center not in CENTERS
            or self.namespace != SUPPORT_PARTITION_NAMESPACE
            or len(self.support_case_ids) != SUPPORT_CASE_COUNT_PER_CENTER
            or len(self.evaluation_case_ids)
            != EXPECTED_EVALUATION_CASE_COUNTS_BY_CENTER[self.center]
            or set(self.support_case_ids) & set(self.evaluation_case_ids)
            or any(row.center != self.center for row in (*self.support_rows, *self.evaluation_rows))
            or any(row.partition_role != "support" for row in self.support_rows)
            or any(row.partition_role != "evaluation" for row in self.evaluation_rows)
        ):
            raise ProtocolError("Consumed-test center partition geometry drifted.")
        expected = canonical_sha256(
            _center_partition_payload(
                center=self.center,
                namespace=self.namespace,
                support_case_ids=self.support_case_ids,
                evaluation_case_ids=self.evaluation_case_ids,
                support_rows=self.support_rows,
                evaluation_rows=self.evaluation_rows,
            )
        )
        if self.partition_hash != expected:
            raise ProtocolError("Consumed-test center partition hash drifted.")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "midogpp_consumed_test_center_partition_v1",
            "center": self.center,
            "namespace": self.namespace,
            "support_case_ids": list(self.support_case_ids),
            "evaluation_case_ids": list(self.evaluation_case_ids),
            "support_case_count": len(self.support_case_ids),
            "evaluation_case_count": len(self.evaluation_case_ids),
            "support_row_count": len(self.support_rows),
            "evaluation_row_count": len(self.evaluation_rows),
            "support_row_identity_hash": _row_identity_hash(self.support_rows),
            "evaluation_row_identity_hash": _row_identity_hash(self.evaluation_rows),
            "partition_hash": self.partition_hash,
        }


@dataclass(frozen=True)
class ConsumedTestPartitionSurface:
    by_center: Mapping[str, CenterCasePartition]
    support_rows_by_center: Mapping[str, tuple[LabelFreeCaseRow, ...]]
    evaluation_rows_by_center: Mapping[str, tuple[LabelFreeCaseRow, ...]]
    table_rows: tuple[Mapping[str, object], ...]
    lock_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        partitions = {str(key): value for key, value in self.by_center.items()}
        support = {str(key): tuple(value) for key, value in self.support_rows_by_center.items()}
        evaluation = {
            str(key): tuple(value)
            for key, value in self.evaluation_rows_by_center.items()
        }
        if (
            tuple(partitions) != CENTERS
            or tuple(support) != CENTERS
            or tuple(evaluation) != CENTERS
            or any(partitions[c].support_rows != support[c] for c in CENTERS)
            or any(partitions[c].evaluation_rows != evaluation[c] for c in CENTERS)
            or sum(len(value) for value in support.values()) != EXPECTED_SUPPORT_ROW_COUNT
            or sum(len(value) for value in evaluation.values()) != EXPECTED_EVALUATION_ROW_COUNT
        ):
            raise ProtocolError("Consumed-test partition surface is incomplete.")
        payload = dict(self.lock_payload)
        lock_hash = payload.get("support_partition_lock_hash")
        unhashed = {key: value for key, value in payload.items() if key != "support_partition_lock_hash"}
        if lock_hash != canonical_sha256(unhashed):
            raise ProtocolError("Consumed-test partition lock hash drifted.")
        object.__setattr__(self, "by_center", MappingProxyType(partitions))
        object.__setattr__(self, "support_rows_by_center", MappingProxyType(support))
        object.__setattr__(self, "evaluation_rows_by_center", MappingProxyType(evaluation))
        object.__setattr__(
            self,
            "table_rows",
            tuple(MappingProxyType(dict(row)) for row in self.table_rows),
        )
        object.__setattr__(self, "lock_payload", MappingProxyType(payload))

    @property
    def lock_hash(self) -> str:
        return str(self.lock_payload["support_partition_lock_hash"])


def build_consumed_test_partitions(
    rows: Sequence[LabelFreeCaseRow],
    *,
    namespace: str = SUPPORT_PARTITION_NAMESPACE,
) -> ConsumedTestPartitionSurface:
    """Select the lexical first eight complete cases in every center.

    Membership is a pure function of label-free ``(center, case_id)`` values.
    There is intentionally no seed argument.
    """

    values = tuple(rows)
    if (
        namespace != SUPPORT_PARTITION_NAMESPACE
        or len(values) != EXPECTED_TEST_ROW_COUNT
        or any(not isinstance(row, LabelFreeCaseRow) for row in values)
        or any(row.partition_role != "unassigned" for row in values)
    ):
        raise ProtocolError("Consumed-test lexical partition input drifted.")
    ordinals = tuple(sorted(row.row_ordinal for row in values))
    manifest_indices = tuple(sorted(row.manifest_row_index for row in values))
    if (
        ordinals != tuple(range(EXPECTED_TEST_ROW_COUNT))
        or len(set(manifest_indices)) != EXPECTED_TEST_ROW_COUNT
        or len({row.evaluation_row_id for row in values}) != EXPECTED_TEST_ROW_COUNT
    ):
        raise ProtocolError("Consumed-test row identities are incomplete or duplicated.")

    partitions: dict[str, CenterCasePartition] = {}
    table: list[dict[str, object]] = []
    all_case_ids: set[str] = set()
    for center in CENTERS:
        center_rows = tuple(sorted((row for row in values if row.center == center), key=lambda row: row.row_ordinal))
        case_ids = tuple(sorted({row.case_id for row in center_rows}))
        if len(case_ids) != EXPECTED_CASE_COUNTS_BY_CENTER[center]:
            raise ProtocolError(f"Center {center} whole-case count drifted.")
        if all_case_ids & set(case_ids):
            raise ProtocolError("A consumed-test case identity occurs in multiple centers.")
        all_case_ids.update(case_ids)
        support_case_ids = case_ids[:SUPPORT_CASE_COUNT_PER_CENTER]
        evaluation_case_ids = case_ids[SUPPORT_CASE_COUNT_PER_CENTER:]
        support_set = set(support_case_ids)
        support = tuple(
            replace(row, partition_role="support")
            for row in center_rows
            if row.case_id in support_set
        )
        evaluation = tuple(
            replace(row, partition_role="evaluation")
            for row in center_rows
            if row.case_id not in support_set
        )
        if len(evaluation) != EXPECTED_EVALUATION_ROW_COUNTS_BY_CENTER[center]:
            raise ProtocolError(f"Center {center} evaluation row count drifted.")
        partition_unhashed = _center_partition_payload(
            center=center,
            namespace=namespace,
            support_case_ids=support_case_ids,
            evaluation_case_ids=evaluation_case_ids,
            support_rows=support,
            evaluation_rows=evaluation,
        )
        partition = CenterCasePartition(
            center=center,
            namespace=namespace,
            support_case_ids=support_case_ids,
            evaluation_case_ids=evaluation_case_ids,
            support_rows=support,
            evaluation_rows=evaluation,
            partition_hash=canonical_sha256(partition_unhashed),
        )
        partitions[center] = partition
        for row in (*support, *evaluation):
            table.append(
                {
                    "schema_version": "midogpp_consumed_test_partition_row_v1",
                    **row.identity_payload(),
                    "center_partition_hash": partition.partition_hash,
                    "support_partition_namespace": namespace,
                    "membership_seed": None,
                    "label_present": False,
                }
            )

    support_case_total = sum(len(value.support_case_ids) for value in partitions.values())
    evaluation_case_total = sum(len(value.evaluation_case_ids) for value in partitions.values())
    support_row_total = sum(len(value.support_rows) for value in partitions.values())
    evaluation_row_total = sum(len(value.evaluation_rows) for value in partitions.values())
    if (
        len(all_case_ids) != EXPECTED_TOTAL_CASE_COUNT
        or support_case_total != EXPECTED_SUPPORT_CASE_COUNT
        or evaluation_case_total != EXPECTED_EVALUATION_CASE_COUNT
        or support_row_total != EXPECTED_SUPPORT_ROW_COUNT
        or evaluation_row_total != EXPECTED_EVALUATION_ROW_COUNT
    ):
        raise ProtocolError("Consumed-test global case/row partition geometry drifted.")
    unhashed = {
        "schema_version": "midogpp_consumed_test_support_partition_lock_v1",
        "status": "LOCKED_FROM_LABEL_FREE_IDENTITIES",
        "centers": list(CENTERS),
        "support_partition_namespace": namespace,
        "membership_rule": "canonical_case_id_sort_then_first_eight",
        "membership_seed": None,
        "support_case_count_per_center": SUPPORT_CASE_COUNT_PER_CENTER,
        "support_case_count_total": support_case_total,
        "evaluation_case_count_total": evaluation_case_total,
        "support_row_count_total": support_row_total,
        "evaluation_row_count_total": evaluation_row_total,
        "total_case_count": len(all_case_ids),
        "total_row_count": len(values),
        "partition_hashes_by_center": {
            center: partitions[center].partition_hash for center in CENTERS
        },
        "whole_case": True,
        "support_evaluation_case_disjoint": True,
        "support_evaluation_row_disjoint": True,
        "all_cases_assigned_exactly_once": True,
        "labels_used": False,
        "consumed_test_data": True,
        "fresh_evidence": False,
    }
    lock_payload = {
        **unhashed,
        "support_partition_lock_hash": canonical_sha256(unhashed),
    }
    return ConsumedTestPartitionSurface(
        by_center=partitions,
        support_rows_by_center={
            center: partitions[center].support_rows for center in CENTERS
        },
        evaluation_rows_by_center={
            center: partitions[center].evaluation_rows for center in CENTERS
        },
        table_rows=tuple(sorted(table, key=lambda row: int(row["row_ordinal"]))),
        lock_payload=lock_payload,
    )


def _row_identity_hash(rows: Sequence[LabelFreeCaseRow]) -> str:
    return canonical_sha256([row.identity_payload() for row in rows])


def _center_partition_payload(
    *,
    center: str,
    namespace: str,
    support_case_ids: Sequence[str],
    evaluation_case_ids: Sequence[str],
    support_rows: Sequence[LabelFreeCaseRow],
    evaluation_rows: Sequence[LabelFreeCaseRow],
) -> dict[str, object]:
    return {
        "schema_version": "midogpp_consumed_test_center_partition_v1",
        "center": center,
        "namespace": namespace,
        "membership_rule": "canonical_case_id_sort_then_first_eight",
        "seed_used": False,
        "support_case_ids": list(support_case_ids),
        "evaluation_case_ids": list(evaluation_case_ids),
        "support_row_identity_hash": _row_identity_hash(support_rows),
        "evaluation_row_identity_hash": _row_identity_hash(evaluation_rows),
    }


def _canonical_text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.strip() == value


build_fixed_test_partition_surface = build_consumed_test_partitions


__all__ = (
    "CenterCasePartition",
    "ConsumedTestPartitionSurface",
    "LabelFreeCaseRow",
    "build_consumed_test_partitions",
    "build_fixed_test_partition_surface",
)
