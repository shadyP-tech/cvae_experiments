from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14 import runner
from midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_v14.execution.menu_roots import (
    build_center_menu_roots,
    validate_center_menu_root_bijection,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.runtime.harp_v14_execution.contracts import (
    ActionKind,
    LabelFreeActionBlock,
    LabelFreeOuterMenu,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.menu_root_binding import (
    CenterMenuRootBinding,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.stores import (
    CompactStoreReceipt,
    write_label_free_outer_menu,
)
from midogpp_thesis.cvae.runtime.harp_v14_execution.validation import (
    run_two_fresh_validations,
)


CENTERS = ("0", "1")


def _menu(center: str, *, shift: float = 0.0) -> LabelFreeOuterMenu:
    other = next(value for value in CENTERS if value != center)
    specifications = tuple(
        (
            surface_role,
            query,
            kind,
            source,
            value,
        )
        for surface_role, query in (
            ("development", other),
            ("target", center),
        )
        for kind, source, value in (
            (ActionKind.B, None, 0.5 + shift),
            (ActionKind.U, None, 0.6 + shift),
            *(
                ((ActionKind.HXE, other, 0.7 + shift),)
                if surface_role == "target"
                else ()
            ),
        )
    )
    blocks = tuple(
        sorted(
            (
                LabelFreeActionBlock(
                    surface_role=surface_role,
                    outer_target_id=center,
                    query_center_id=query,
                    action_kind=kind,
                    selected_source_id=source,
                    sample_ids=(f"sample-{query}",),
                    case_ids=(f"case-{query}",),
                    probabilities=np.asarray((value,), dtype=np.float32),
                    seed_dispersion=np.asarray((0.0,), dtype=np.float32),
                )
                for surface_role, query, kind, source, value in specifications
            ),
            key=lambda block: block.key,
        )
    )
    return LabelFreeOuterMenu(
        outer_target_id=center,
        blocks=blocks,
        lineage={"schema_version": "test_harp_v14_menu_lineage_v1"},
    )


def _durable_binding(
    tmp_path: Path,
) -> tuple[
    Path,
    tuple[LabelFreeOuterMenu, ...],
    Mapping[str, Path],
    tuple[CompactStoreReceipt, ...],
]:
    parent = (tmp_path / "stores/physical_menu").resolve()
    menus = tuple(_menu(center) for center in CENTERS)
    roots = build_center_menu_roots(parent, centers=CENTERS, menus=menus)
    receipts = tuple(
        write_label_free_outer_menu(roots[menu.outer_target_id], menu)
        for menu in menus
    )
    return parent, menus, roots, receipts


def test_center_menu_roots_are_an_explicit_durable_bijection(
    tmp_path: Path,
) -> None:
    parent, menus, roots, receipts = _durable_binding(tmp_path)

    checked = validate_center_menu_root_bijection(
        roots,
        physical_menu_parent=parent,
        centers=CENTERS,
        menus=menus,
        receipts=receipts,
    )

    assert isinstance(checked, Mapping)
    assert tuple(checked) == CENTERS
    assert checked == {
        center: parent / f"outer_{center}" for center in CENTERS
    }


def test_center_menu_root_builder_rejects_wrong_menu_order() -> None:
    with pytest.raises(ProtocolError, match="order/center coverage"):
        build_center_menu_roots(
            Path("/tmp/harp-v14-menu-roots"),
            centers=CENTERS,
            menus=tuple(_menu(center) for center in reversed(CENTERS)),
        )


def test_center_menu_root_binding_rejects_duplicate_or_missing_centers(
    tmp_path: Path,
) -> None:
    parent, menus, roots, receipts = _durable_binding(tmp_path)
    with pytest.raises(ProtocolError, match="center inventory"):
        build_center_menu_roots(
            parent,
            centers=("0", "0"),
            menus=(menus[0], menus[0]),
        )
    with pytest.raises(ProtocolError, match="mapping order/center coverage"):
        validate_center_menu_root_bijection(
            {"0": roots["0"]},
            physical_menu_parent=parent,
            centers=CENTERS,
            menus=menus,
            receipts=receipts,
        )


def test_center_menu_root_binding_rejects_noncanonical_path(
    tmp_path: Path,
) -> None:
    parent, menus, roots, receipts = _durable_binding(tmp_path)
    noncanonical = MappingProxyType(
        {
            "0": parent / "outer_0" / ".." / "outer_0",
            "1": roots["1"],
        }
    )

    with pytest.raises(ProtocolError, match="not canonical"):
        validate_center_menu_root_bijection(
            noncanonical,
            physical_menu_parent=parent,
            centers=CENTERS,
            menus=menus,
            receipts=receipts,
        )


def test_center_menu_root_binding_reads_back_exact_menu_hash(
    tmp_path: Path,
) -> None:
    parent, menus, roots, receipts = _durable_binding(tmp_path)
    replacement = _menu("0", shift=0.1)
    replacement_receipt = write_label_free_outer_menu(roots["0"], replacement)

    with pytest.raises(ProtocolError, match="durable content binding"):
        validate_center_menu_root_bijection(
            roots,
            physical_menu_parent=parent,
            centers=CENTERS,
            menus=menus,
            receipts=(replacement_receipt, receipts[1]),
        )


def test_runtime_validator_rejects_positional_roots_before_spawn(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolError, match="typed center-menu-root binding"):
        run_two_fresh_validations(
            tmp_path / "routes",
            (tmp_path / "outer_0", tmp_path / "outer_1"),  # type: ignore[arg-type]
            tmp_path / "development",
            tmp_path / "model",
            tmp_path / "actions",
            expected_center_ids=CENTERS,
            expected_config_hash="a" * 64,
            effective_menu_root=tmp_path / "effective",
        )


def test_production_branch_revalidates_mapping_before_validator_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, menus, roots, receipts = _durable_binding(tmp_path)
    binding = CenterMenuRootBinding.create(
        common_parent=parent,
        centers=CENTERS,
        menu_roots=roots,
        menus=menus,
        receipts=receipts,
    )
    observed: dict[str, object] = {}

    def fake_validator(_route_root, menu_binding, *_args, **_kwargs):
        observed["menu_binding"] = menu_binding
        return ({"validator_id": "A"}, {"validator_id": "B"})

    monkeypatch.setattr(runner, "run_two_fresh_validations", fake_validator)
    result = runner._run_two_bound_fresh_validations(
        tmp_path / "routes",
        binding,
        tmp_path / "development",
        tmp_path / "model",
        tmp_path / "actions",
        expected_center_ids=CENTERS,
        expected_config_hash="a" * 64,
        effective_menu_root=tmp_path / "effective",
    )

    assert tuple(row["validator_id"] for row in result) == ("A", "B")
    assert isinstance(observed["menu_binding"], CenterMenuRootBinding)
    assert observed["menu_binding"].ordered_center_ids == CENTERS

    observed.clear()
    with pytest.raises(ProtocolError, match="typed center-menu-root binding"):
        runner._run_two_bound_fresh_validations(
            tmp_path / "routes",
            tuple(roots.values()),  # type: ignore[arg-type]
            tmp_path / "development",
            tmp_path / "model",
            tmp_path / "actions",
            expected_center_ids=CENTERS,
            expected_config_hash="a" * 64,
            effective_menu_root=tmp_path / "effective",
        )
    assert observed == {}

    source = inspect.getsource(runner.run_harp_stage90_v14)
    assert source.count("build_center_menu_roots(") == 2
    assert source.count("validate_center_menu_root_bijection(") == 2
    assert source.count("CenterMenuRootBinding.create(") == 2
    assert source.index("_run_two_bound_fresh_validations(") > source.index(
        'ledger.advance("PRELABEL_ROUTES_DURABLE")'
    )
