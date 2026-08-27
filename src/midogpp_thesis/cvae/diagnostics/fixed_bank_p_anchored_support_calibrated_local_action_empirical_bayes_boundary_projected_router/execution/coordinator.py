"""Deterministic coarse-H process orchestration for SCALE-BP."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from typing import Sequence

from ..final_route_inventory import FinalRouteInventoryReceipt
from ..hashing import canonical_hash
from ..identity import CENTERS, EXPECTED_CASE_COUNT
from ..protocol import ProtocolError
from .dtos import OuterCenterResult, OuterCenterTask
from .workstation import assert_coordinator_process, initialize_cpu_outer_worker


def _planned_route_hash(task: OuterCenterTask, case_id: str) -> str:
    return canonical_hash(
        {
            "schema_version": "scale_bp_planned_route_inventory_v1",
            "target_center": task.target_center,
            "case_id": case_id,
            "task_hash": task.task_hash,
            "execution_authorized": False,
            "labels_opened": False,
        }
    )


def execute_outer_center_task(task: OuterCenterTask) -> OuterCenterResult:
    """Exercise the sealed task boundary without opening labels or arrays.

    The planned v1 is deliberately non-executable scientifically.  Its later
    authorized sibling will replace this pure inventory kernel with the sealed
    SCALE-BP engine while retaining this DTO and orchestration boundary.
    """

    route_hashes = tuple(_planned_route_hash(task, case_id) for case_id in task.case_ids)
    return OuterCenterResult(
        task.target_center,
        task.task_hash,
        task.final_route_inventory_hash,
        task.case_ids,
        route_hashes,
    )


def validate_outer_center_tasks(
    tasks: Sequence[OuterCenterTask],
    inventory_receipt: FinalRouteInventoryReceipt,
) -> tuple[OuterCenterTask, ...]:
    """Reject any task rectangle other than the sealed 9-center universe."""

    rows = tuple(tasks)
    if not isinstance(inventory_receipt, FinalRouteInventoryReceipt):
        raise ProtocolError("SCALE-BP final-route inventory receipt drifted.")
    if (
        len(rows) != len(CENTERS)
        or any(not isinstance(row, OuterCenterTask) for row in rows)
        or tuple(row.target_center for row in rows) != CENTERS
        or len({row.target_center for row in rows}) != len(CENTERS)
        or len({row.task_hash for row in rows}) != len(CENTERS)
        or len({row.protocol_hash for row in rows}) != 1
        or len({row.physical_bank.receipt_hash for row in rows}) != 1
        or any(
            row.final_route_inventory_hash != inventory_receipt.receipt_hash
            or row.case_ids != inventory_receipt.cases(row.target_center)
            or row.physical_bank.dataset_case_inventory_hash
            != inventory_receipt.dataset_case_inventory_hash
            or row.physical_bank.cache_content_hash
            != inventory_receipt.cache_content_hash
            or row.physical_bank.row_order_hash != inventory_receipt.row_order_hash
            for row in rows
        )
    ):
        raise ProtocolError("SCALE-BP outer task closed-world inventory drifted.")
    case_ids = tuple(case_id for row in rows for case_id in row.case_ids)
    expected = tuple(
        case_id
        for center in CENTERS
        for case_id in inventory_receipt.cases(center)
    )
    if (
        len(case_ids) != EXPECTED_CASE_COUNT
        or len(set(case_ids)) != EXPECTED_CASE_COUNT
        or case_ids != expected
    ):
        raise ProtocolError("SCALE-BP outer task case universe drifted.")
    return rows


def validate_outer_center_results(
    tasks: Sequence[OuterCenterTask],
    results: Sequence[OuterCenterResult],
    inventory_receipt: FinalRouteInventoryReceipt,
) -> tuple[OuterCenterResult, ...]:
    """Bind every returned route hash to its canonical center/case position."""

    task_rows = validate_outer_center_tasks(tasks, inventory_receipt)
    result_rows = tuple(results)
    if (
        len(result_rows) != len(task_rows)
        or any(not isinstance(row, OuterCenterResult) for row in result_rows)
        or len({row.result_hash for row in result_rows}) != len(result_rows)
        or any(
            result.target_center != task.target_center
            or result.task_hash != task.task_hash
            or result.final_route_inventory_hash != inventory_receipt.receipt_hash
            or result.case_ids != task.case_ids
            or len(result.route_hashes) != len(task.case_ids)
            or result.route_hashes
            != tuple(_planned_route_hash(task, case_id) for case_id in task.case_ids)
            for task, result in zip(task_rows, result_rows, strict=True)
        )
    ):
        raise ProtocolError("SCALE-BP outer result closed-world inventory drifted.")
    route_hashes = tuple(
        route_hash for result in result_rows for route_hash in result.route_hashes
    )
    if len(route_hashes) != EXPECTED_CASE_COUNT or len(set(route_hashes)) != len(
        route_hashes
    ):
        raise ProtocolError("SCALE-BP outer result route universe drifted.")
    return result_rows


def run_outer_center_tasks(
    tasks: Sequence[OuterCenterTask],
    *,
    inventory_receipt: FinalRouteInventoryReceipt,
    use_processes: bool,
) -> tuple[OuterCenterResult, ...]:
    """Run one task per H with stable serial/spawn ordering."""

    rows = validate_outer_center_tasks(tasks, inventory_receipt)
    if not use_processes:
        results = tuple(execute_outer_center_task(row) for row in rows)
        return validate_outer_center_results(rows, results, inventory_receipt)
    assert_coordinator_process()
    with ProcessPoolExecutor(
        max_workers=min(4, len(rows)),
        mp_context=mp.get_context("spawn"),
        initializer=initialize_cpu_outer_worker,
    ) as executor:
        results = tuple(executor.map(execute_outer_center_task, rows, chunksize=1))
    return validate_outer_center_results(rows, results, inventory_receipt)


__all__ = (
    "execute_outer_center_task",
    "run_outer_center_tasks",
    "validate_outer_center_results",
    "validate_outer_center_tasks",
)
