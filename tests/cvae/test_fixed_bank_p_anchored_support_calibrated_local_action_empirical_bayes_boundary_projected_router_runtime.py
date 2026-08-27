from __future__ import annotations

from dataclasses import replace
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.case_inventory import (
    DatasetCaseInventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.dtos import (
    MemmapReference,
    OuterCenterTask,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.memmap_contracts import (
    _issue_memmap_reference,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.memmaps import (
    open_readonly_memmap,
    row_index_hash,
    validate_row_index,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.physical_bank import (
    PhysicalBankCellSpec,
    PhysicalBankReceipt,
    build_physical_bank_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.execution.workstation import (
    OUTER_WORKER_ENV,
    assert_coordinator_process,
    build_workstation_plan,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.final_route_inventory import (
    FinalRouteInventoryReceipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.inputs.physical_memmap import (
    load_physical_cell_surface,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.inputs.manifest import (
    load_manifest_identity_receipt,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.identity import (
    CENTERS,
    EXPECTED_CASE_COUNT,
    EXPECTED_CASE_COUNTS_BY_CENTER,
    EXPECTED_TEST_ROW_COUNT,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.physical.library import (
    B_ACTION_ID,
    PhysicalCellIdentity,
    action_ids_for_target,
    build_physical_cell_inventory,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_boundary_projected_router.route_identity import (
    RouteScopeWitness,
    SampleIdentity,
    build_route_identity_inventory,
)
from midogpp_thesis.cvae.protocol import ProtocolError


SHA = "a" * 64


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


def _route_inventory(
    inventory: DatasetCaseInventory | None = None,
):
    case_inventory = _inventory() if inventory is None else inventory
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
            for case in case_inventory.cases(center)
        ),
        case_inventory=case_inventory,
    )


def _bank_material(
    root: Path,
    route_inventory,
    *,
    selected_identity: PhysicalCellIdentity | None = None,
    selected_row: int = 0,
    selected_value: float = 0.75,
) -> tuple[Path, tuple[PhysicalBankCellSpec, ...]]:
    root.mkdir(parents=True, exist_ok=True)
    cells = build_physical_cell_inventory()
    row_count = sum(row.row_count for row in route_inventory.case_bindings)
    values = np.full((len(cells), row_count), 0.5, dtype=np.float32)
    if selected_identity is not None:
        cell_index = tuple(row.cell_hash for row in cells).index(
            selected_identity.cell_hash
        )
        values[cell_index, selected_row] = selected_value
    path = root / "physical-bank.f32"
    payload = values.tobytes(order="C")
    path.write_bytes(payload)
    slice_bytes = row_count * 4
    specs = tuple(
        PhysicalBankCellSpec(
            identity,
            path.name,
            (row_count,),
            index * slice_bytes,
            hashlib.sha256(
                payload[index * slice_bytes : (index + 1) * slice_bytes]
            ).hexdigest(),
        )
        for index, identity in enumerate(cells)
    )
    return path, specs


def _physical_bank(
    root: Path,
    route_inventory=None,
    **kwargs: object,
) -> PhysicalBankReceipt:
    inventory = _route_inventory() if route_inventory is None else route_inventory
    _path, specs = _bank_material(root, inventory, **kwargs)
    return build_physical_bank_receipt(
        root,
        specs,
        route_identity_inventory=inventory,
    )


def _task(tmp_path: Path, center: str = "0") -> OuterCenterTask:
    receipt = FinalRouteInventoryReceipt.from_case_inventory(_inventory())
    cases = receipt.cases(center)
    return OuterCenterTask(
        center,
        cases,
        _physical_bank(tmp_path),
        SHA,
        receipt.receipt_hash,
    )


def test_worker_dtos_are_primitive_and_pickle_safe(tmp_path: Path) -> None:
    task = _task(tmp_path)
    restored = pickle.loads(pickle.dumps(task))
    assert restored == task
    assert restored.task_hash == task.task_hash
    assert len(restored.physical_bank.cells) == 810
    assert not hasattr(restored.physical_bank.references[0], "filename")


def test_final_route_inventory_receipt_is_exact_primitive_and_pickle_safe() -> None:
    inventory = _inventory()
    receipt = FinalRouteInventoryReceipt.from_case_inventory(inventory)
    restored = pickle.loads(pickle.dumps(receipt))
    assert restored == receipt
    assert restored.dataset_case_inventory_hash == inventory.inventory_hash
    assert restored.case_count == EXPECTED_CASE_COUNT
    assert tuple(center for center, _cases in restored.cases_by_center) == CENTERS
    assert not hasattr(restored, "case_inventory")


def test_workstation_plan_is_coarse_h_and_non_executable() -> None:
    plan = build_workstation_plan()
    assert plan.gpu_devices == ("cuda:0", "cuda:1")
    assert plan.persistent_gpu_workers == 2
    assert plan.physical_cell_count == 810
    assert plan.cpu_outer_workers == 4
    assert plan.blas_threads_per_worker == 1
    assert plan.support_fold_count == 4
    assert plan.nested_pools_allowed is False
    assert plan.execution_authorized is False


def test_physical_inventory_is_exact_810_and_target_excluded() -> None:
    cells = build_physical_cell_inventory()
    assert len(cells) == 810
    assert len({row.cell_hash for row in cells}) == 810
    for target in ("0", "9"):
        actions = action_ids_for_target(target)
        assert len(actions) == 10
        assert f"A1::source={target}" not in actions


def test_nested_pool_guard_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OUTER_WORKER_ENV, "1")
    with pytest.raises(ProtocolError, match="nested process pools"):
        assert_coordinator_process()


def test_memmap_loader_validates_slice_hash_and_row_identity(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "surface.f32"
    payload = values.tobytes(order="C")
    path.write_bytes(payload)
    rows = tuple(("0", "case", f"sample-{index}") for index in range(3))
    reference = _issue_memmap_reference(
        path=str(path.resolve()),
        dtype="float32",
        shape=values.shape,
        offset_bytes=0,
        sha256=hashlib.sha256(payload).hexdigest(),
        semantic_role="posterior_statistics",
        byte_length=len(payload),
        order="C",
        row_index_hash=row_index_hash(rows),
        cache_content_hash=SHA,
        row_order_hash=SHA,
    )
    loaded = open_readonly_memmap(reference)
    assert loaded.flags.writeable is False
    np.testing.assert_array_equal(loaded, values)
    assert validate_row_index(reference, rows) == reference.row_index_hash
    with pytest.raises(ProtocolError, match="row-index"):
        validate_row_index(reference, tuple(reversed(rows)))
    with pytest.raises(ProtocolError, match="not factory issued"):
        MemmapReference(
            str(path.resolve()),
            "float32",
            values.shape,
            0,
            hashlib.sha256(payload).hexdigest(),
            "posterior_statistics",
            len(payload),
            "C",
            row_index_hash(rows),
            SHA,
            SHA,
        )


def test_physical_surface_is_issued_from_exact_readonly_case_slice(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    identities = tuple(
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
    )
    route_inventory = build_route_identity_inventory(
        identities,
        case_inventory=inventory,
    )
    case_id = inventory.cases("0")[0]
    witness = RouteScopeWitness("0", case_id, route_inventory)
    keys = (("0", case_id, f"sample-{case_id}"),)
    identity = PhysicalCellIdentity("0", B_ACTION_ID, 17, 17)
    bank = _physical_bank(
        tmp_path / "bank",
        route_inventory,
        selected_identity=identity,
    )
    surface = load_physical_cell_surface(
        bank,
        identity=identity,
        route_witness=witness,
        ordered_sample_keys=keys,
    )
    reference = bank.reference_for(identity)
    assert surface.probabilities == pytest.approx((0.75,))
    assert surface.physical_bank_receipt_hash == bank.receipt_hash
    assert surface.memmap_reference_hash == reference.reference_hash
    with pytest.raises(ProtocolError, match="case lineage"):
        load_physical_cell_surface(
            bank,
            identity=identity,
            route_witness=witness,
            ordered_sample_keys=(("0", case_id, "poison"),),
        )


def test_physical_bank_rejects_wrong_cell_slice_symlink_missing_and_duplicate(
    tmp_path: Path,
) -> None:
    route_inventory = _route_inventory()
    root = tmp_path / "canonical"
    path, specs = _bank_material(root, route_inventory)
    bank = build_physical_bank_receipt(
        root,
        specs,
        route_identity_inventory=route_inventory,
    )
    identities = build_physical_cell_inventory()
    first_reference = bank.reference_for(identities[0])
    with pytest.raises(ProtocolError, match="cell/reference mapping"):
        open_readonly_memmap(
            first_reference,
            physical_bank=bank,
            physical_identity=identities[1],
        )
    with pytest.raises(ProtocolError, match="inventory drifted"):
        build_physical_bank_receipt(
            root,
            specs[:-1],
            route_identity_inventory=route_inventory,
        )
    duplicate = (specs[0], specs[0], *specs[2:])
    with pytest.raises(ProtocolError, match="inventory drifted"):
        build_physical_bank_receipt(
            root,
            duplicate,
            route_identity_inventory=route_inventory,
        )

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    (symlink_root / path.name).symlink_to(path)
    with pytest.raises(ProtocolError, match="symlink is forbidden"):
        build_physical_bank_receipt(
            symlink_root,
            specs,
            route_identity_inventory=route_inventory,
        )

    poisoned = bytearray(path.read_bytes())
    poisoned[0] ^= 1
    path.write_bytes(poisoned)
    with pytest.raises(ProtocolError, match="content hash drifted"):
        open_readonly_memmap(
            first_reference,
            physical_bank=bank,
            physical_identity=identities[0],
        )


def test_manifest_identity_inventory_is_derived_from_exact_csv_bytes(
    tmp_path: Path,
) -> None:
    base = _inventory()
    header = "center,case_id,group_id,patient_id,slide_id,sample_id\n"
    rows = []
    first_center = CENTERS[0]
    first_case = base.cases(first_center)[0]
    extra = EXPECTED_TEST_ROW_COUNT - EXPECTED_CASE_COUNT
    for center in CENTERS:
        for case in base.cases(center):
            sample_count = 1 + (extra if (center, case) == (first_center, first_case) else 0)
            rows.extend(
                f"{center},{case},group-{case},patient-{case},slide-{case},sample-{index:05d}-{case}\n"
                for index in range(sample_count)
            )
    content = (header + "".join(rows)).encode("utf-8")
    path = tmp_path / "manifest.csv"
    path.write_bytes(content)
    manifest_hash = hashlib.sha256(content).hexdigest()
    inventory = DatasetCaseInventory(
        base.cache_content_hash,
        base.row_order_hash,
        manifest_hash,
        base.cases_by_center,
    )
    receipt = load_manifest_identity_receipt(path, case_inventory=inventory)
    assert receipt.row_count == EXPECTED_TEST_ROW_COUNT
    assert receipt.manifest_sha256 == manifest_hash
    assert receipt.route_identity_inventory.case_inventory.inventory_hash == (
        inventory.inventory_hash
    )
    path.write_bytes(content + b"\n")
    with pytest.raises(ProtocolError, match="byte hash"):
        load_manifest_identity_receipt(path, case_inventory=inventory)
