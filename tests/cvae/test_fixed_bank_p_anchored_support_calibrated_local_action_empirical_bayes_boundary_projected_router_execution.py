from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.case_inventory import (
    DatasetCaseInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.coordinator import (
    run_outer_center_tasks,
    validate_outer_center_results,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.dtos import (
    OuterCenterTask,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.physical_bank import (
    PhysicalBankCellSpec,
    PhysicalBankReceipt,
    build_physical_bank_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.final_route_inventory import (
    FinalRouteInventoryReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.physical.library import (
    build_physical_cell_inventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.route_identity import (
    RouteIdentityInventory,
    SampleIdentity,
    build_route_identity_inventory,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "b" * 64


def _inventory() -> DatasetCaseInventory:
    counts = dict(EXPECTED_CASE_COUNTS_BY_CENTER)
    return DatasetCaseInventory(
        SHA,
        SHA,
        SHA,
        tuple(
            (
                center,
                tuple(f"case-{center}-{index:03d}" for index in range(counts[center])),
            )
            for center in CENTERS
        ),
    )


def _receipt() -> FinalRouteInventoryReceipt:
    return FinalRouteInventoryReceipt.from_case_inventory(_inventory())


def _route_inventory() -> RouteIdentityInventory:
    inventory = _inventory()
    return build_route_identity_inventory(
        tuple(
            SampleIdentity(
                center,
                case,
                f"group-{case}",
                f"patient-{case}",
                f"slide-{case}",
                f"sample-{case}",
            )
            for center in CENTERS
            for case in inventory.cases(center)
        ),
        case_inventory=inventory,
    )


def _bank(root: Path, *, name: str = "physical-bank.f32") -> PhysicalBankReceipt:
    root.mkdir(parents=True, exist_ok=True)
    route_inventory = _route_inventory()
    row_count = len(route_inventory.case_bindings)
    cell_payload = bytes(row_count * 4)
    path = root / name
    path.write_bytes(cell_payload * 810)
    digest = hashlib.sha256(cell_payload).hexdigest()
    specs = tuple(
        PhysicalBankCellSpec(
            identity,
            name,
            (row_count,),
            index * len(cell_payload),
            digest,
        )
        for index, identity in enumerate(build_physical_cell_inventory())
    )
    return build_physical_bank_receipt(
        root,
        specs,
        route_identity_inventory=route_inventory,
    )


def _tasks(
    tmp_path: Path, receipt: FinalRouteInventoryReceipt | None = None
) -> tuple[OuterCenterTask, ...]:
    sealed = _receipt() if receipt is None else receipt
    bank = _bank(tmp_path)
    return tuple(
        OuterCenterTask(
            center,
            sealed.cases(center),
            bank,
            SHA,
            sealed.receipt_hash,
        )
        for center in CENTERS
    )


def test_serial_and_spawn_inventory_hashes_match(tmp_path: Path) -> None:
    receipt = _receipt()
    tasks = _tasks(tmp_path, receipt)
    serial = run_outer_center_tasks(
        tasks, inventory_receipt=receipt, use_processes=False
    )
    spawned = run_outer_center_tasks(
        tasks, inventory_receipt=receipt, use_processes=True
    )
    assert [row.result_hash for row in spawned] == [
        row.result_hash for row in serial
    ]
    assert tuple(row.target_center for row in spawned) == CENTERS
    assert sum(len(row.route_hashes) for row in spawned) == EXPECTED_CASE_COUNT
    assert all(
        row.final_route_inventory_hash == receipt.receipt_hash for row in spawned
    )


def test_outer_task_closed_world_rejects_partial_extra_duplicate_and_order_poison(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    tasks = _tasks(tmp_path, receipt)
    poisoned = (
        tasks[:-1],
        (*tasks, tasks[0]),
        tuple(reversed(tasks)),
        (replace(tasks[0], case_ids=tasks[0].case_ids[:-1]), *tasks[1:]),
        (
            replace(
                tasks[0],
                case_ids=tuple(sorted((*tasks[0].case_ids, "zz-extra-case"))),
            ),
            *tasks[1:],
        ),
        (replace(tasks[0], final_route_inventory_hash=SHA), *tasks[1:]),
        (replace(tasks[0], physical_bank=_bank(tmp_path / "other")), *tasks[1:]),
    )
    for rows in poisoned:
        with pytest.raises(ProtocolError, match="closed-world inventory"):
            run_outer_center_tasks(
                rows,
                inventory_receipt=receipt,
                use_processes=False,
            )
    with pytest.raises(ProtocolError, match="outer task topology"):
        replace(
            tasks[0],
            case_ids=(tasks[0].case_ids[0], tasks[0].case_ids[0]),
        )
    with pytest.raises(ProtocolError, match="outer task topology"):
        replace(tasks[0], case_ids=tuple(reversed(tasks[0].case_ids)))


def test_outer_result_closed_world_rejects_missing_extra_duplicate_and_reorder_poison(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    tasks = _tasks(tmp_path, receipt)
    results = run_outer_center_tasks(
        tasks, inventory_receipt=receipt, use_processes=False
    )
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        validate_outer_center_results(tasks, results[:-1], receipt)
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        validate_outer_center_results(tasks, (*results, results[0]), receipt)
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        validate_outer_center_results(tasks, tuple(reversed(results)), receipt)
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        validate_outer_center_results(
            tasks,
            (results[0], results[0], *results[2:]),
            receipt,
        )
    missing_case = replace(
        results[0],
        case_ids=results[0].case_ids[:-1],
        route_hashes=results[0].route_hashes[:-1],
    )
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        validate_outer_center_results(
            tasks, (missing_case, *results[1:]), receipt
        )
    extra_case = replace(
        results[0],
        case_ids=tuple(sorted((*results[0].case_ids, "zz-extra-case"))),
        route_hashes=(*results[0].route_hashes, "c" * 64),
    )
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        validate_outer_center_results(tasks, (extra_case, *results[1:]), receipt)
    reordered_routes = replace(
        results[0], route_hashes=tuple(reversed(results[0].route_hashes))
    )
    with pytest.raises(ProtocolError, match="closed-world inventory"):
        validate_outer_center_results(
            tasks, (reordered_routes, *results[1:]), receipt
        )
    with pytest.raises(ProtocolError, match="outer result topology"):
        replace(
            results[0],
            route_hashes=(
                results[0].route_hashes[0],
                results[0].route_hashes[0],
                *results[0].route_hashes[2:],
            ),
        )
    with pytest.raises(ProtocolError, match="outer result topology"):
        replace(results[0], case_ids=tuple(reversed(results[0].case_ids)))
