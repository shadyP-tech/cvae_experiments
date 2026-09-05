"""Typed center-to-menu-root binding for HARP v19 validation.

The fresh validators consume a mapping because a positional root sequence can
silently detach an outer-center identity from its durable physical menu.  This
module makes that mapping explicit at the producer boundary and revalidates the
menu/receipt/path bijection immediately before validator processes are spawned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from ....protocol import ProtocolError
from ....runtime.harp_v19_execution.contracts import LabelFreeOuterMenu
from ....runtime.harp_v19_execution.stores import (
    CompactStoreReceipt,
    read_label_free_outer_menu,
)


def build_center_menu_roots(
    physical_menu_parent: Path,
    *,
    centers: Sequence[str],
    menus: Sequence[LabelFreeOuterMenu],
) -> Mapping[str, Path]:
    """Construct one canonical physical-menu root for each ordered center."""

    center_ids = _center_ids(centers)
    menu_rows = tuple(menus)
    if len(menu_rows) != len(center_ids) or any(
        not isinstance(menu, LabelFreeOuterMenu) for menu in menu_rows
    ):
        raise ProtocolError("HARP v19 physical-menu inventory is incomplete.")
    observed = tuple(menu.outer_target_id for menu in menu_rows)
    if observed != center_ids:
        raise ProtocolError(
            "HARP v19 physical-menu order/center coverage drifted."
        )

    parent = _canonical_absolute_path(
        physical_menu_parent, role="physical-menu parent"
    )
    roots = {
        center: _canonical_absolute_path(
            parent / f"center_{center}", role=f"physical-menu root {center}"
        )
        for center in center_ids
    }
    if len(set(roots.values())) != len(center_ids):
        raise ProtocolError("HARP v19 physical-menu roots are not a bijection.")
    return MappingProxyType(roots)


def validate_center_menu_root_bijection(
    menu_roots: Mapping[str, Path],
    *,
    physical_menu_parent: Path,
    centers: Sequence[str],
    menus: Sequence[LabelFreeOuterMenu],
    receipts: Sequence[CompactStoreReceipt],
) -> Mapping[str, Path]:
    """Validate exact center, menu, receipt, and durable-path correspondence."""

    if not isinstance(menu_roots, Mapping):
        raise ProtocolError(
            "HARP v19 fresh validation requires a center-keyed menu-root mapping."
        )
    center_ids = _center_ids(centers)
    if tuple(menu_roots) != center_ids:
        raise ProtocolError(
            "HARP v19 menu-root mapping order/center coverage drifted."
        )
    menu_rows = tuple(menus)
    receipt_rows = tuple(receipts)
    if (
        len(menu_rows) != len(center_ids)
        or len(receipt_rows) != len(center_ids)
        or any(not isinstance(menu, LabelFreeOuterMenu) for menu in menu_rows)
        or any(
            not isinstance(receipt, CompactStoreReceipt)
            for receipt in receipt_rows
        )
    ):
        raise ProtocolError(
            "HARP v19 physical-menu binding inventory is incomplete."
        )
    if tuple(menu.outer_target_id for menu in menu_rows) != center_ids:
        raise ProtocolError(
            "HARP v19 physical-menu order/center coverage drifted."
        )

    expected = build_center_menu_roots(
        physical_menu_parent, centers=center_ids, menus=menu_rows
    )
    checked: dict[str, Path] = {}
    for center, menu, receipt in zip(
        center_ids, menu_rows, receipt_rows, strict=True
    ):
        candidate = _canonical_absolute_path(
            menu_roots[center], role=f"physical-menu root {center}"
        )
        if candidate != expected[center] or receipt.root != candidate:
            raise ProtocolError(
                "HARP v19 physical-menu center/path binding drifted."
            )
        if (
            receipt.manifest_path != candidate / "manifest.json"
            or receipt.npz_path != candidate / "arrays.npz"
            or not receipt.manifest_path.is_file()
            or receipt.manifest_path.is_symlink()
            or not receipt.npz_path.is_file()
            or receipt.npz_path.is_symlink()
        ):
            raise ProtocolError(
                "HARP v19 physical-menu receipt path binding drifted."
            )
        if menu.outer_target_id != center:
            raise ProtocolError(
                "HARP v19 physical-menu center identity drifted."
            )
        reconstructed = read_label_free_outer_menu(candidate)
        if (
            reconstructed.outer_target_id != center
            or reconstructed.menu_hash != menu.menu_hash
        ):
            raise ProtocolError(
                "HARP v19 physical-menu durable content binding drifted."
            )
        checked[center] = candidate
    if len(set(checked.values())) != len(center_ids):
        raise ProtocolError("HARP v19 physical-menu roots are not a bijection.")
    return MappingProxyType(checked)


def _center_ids(values: Sequence[str]) -> tuple[str, ...]:
    centers = tuple(values)
    if (
        not centers
        or any(
            type(center) is not str
            or not center
            or center.strip() != center
            or "/" in center
            or "\\" in center
            or center in {".", ".."}
            for center in centers
        )
        or len(set(centers)) != len(centers)
    ):
        raise ProtocolError("HARP v19 center inventory is malformed.")
    return centers


def _canonical_absolute_path(value: Path, *, role: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve(strict=False)
    if not candidate.is_absolute() or candidate != resolved:
        raise ProtocolError(f"HARP v19 {role} is not canonical.")
    return candidate


__all__ = (
    "build_center_menu_roots",
    "validate_center_menu_root_bijection",
)
