"""Label-free recovery and compact-store persistence for the HARP v5 runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...protocol import ProtocolError
from ...runtime.artifact_io import read_json, sha256_file
from ...runtime.harp_v5_execution.compatibility_adapter import (
    bind_compatibility_artifact_to_outer_menus,
)
from ...runtime.harp_v5_execution.stores import (
    CompactStoreReceipt,
    read_artifact_value,
    read_label_free_outer_menu,
    write_artifact_value,
    write_label_free_outer_menu,
)
from .execution import validate_complete_physical_menus


@dataclass(frozen=True, slots=True)
class LabelFreeMenuBundle:
    """Authenticated physical menus and their durable store witnesses."""

    menus: tuple[Any, ...]
    roots: Mapping[str, Path]
    receipts: tuple[CompactStoreReceipt, ...]


@dataclass(frozen=True, slots=True)
class CompatibilityBundle:
    """Menu-bound compatibility state and its compact-store witness."""

    state: Any
    receipt: CompactStoreReceipt


def recover_or_materialize_label_free_menus(
    *,
    root: Path,
    centers: Sequence[str],
    journal: Any,
    pipeline: Any,
    config: Any,
    cache: Any,
    scratch: Path,
) -> LabelFreeMenuBundle:
    """Resume authenticated outer menus and produce only missing centers."""

    resumed: dict[str, Any] = {}
    pending: list[str] = []
    for outer_target in centers:
        durable = journal.require_resumable(outer_target)
        if durable is None:
            pending.append(outer_target)
            continue
        expected_root = root / "stores/physical_menu" / f"outer_{outer_target}"
        if tuple(path.resolve() for path in durable) != (
            (expected_root / "manifest.json").resolve(),
            (expected_root / "arrays.npz").resolve(),
        ):
            raise ProtocolError("HARP v5 recovery menu path escaped its output root.")
        resumed[outer_target] = read_label_free_outer_menu(expected_root)

    fresh = tuple(
        pipeline.materialize_label_free_outer_menus(
            config,
            cache,
            outer_targets=tuple(pending),
            scratch_root=scratch,
        )
    )
    if tuple(menu.outer_target_id for menu in fresh) != tuple(pending):
        raise ProtocolError(
            "HARP v5 materializer returned incomplete pending outer-H coverage."
        )
    produced_by_outer = {
        **resumed,
        **{menu.outer_target_id: menu for menu in fresh},
    }
    produced = tuple(produced_by_outer[outer_target] for outer_target in centers)
    if tuple(menu.outer_target_id for menu in produced) != tuple(centers):
        raise ProtocolError("HARP v5 materializer returned incomplete outer-H coverage.")
    validate_complete_physical_menus(produced, centers=tuple(centers))

    menu_roots: dict[str, Path] = {}
    menu_receipts: list[CompactStoreReceipt] = []
    for menu in produced:
        store_root = root / "stores/physical_menu" / f"outer_{menu.outer_target_id}"
        if menu.outer_target_id in resumed:
            receipt = existing_compact_store_receipt(store_root)
        else:
            receipt = write_label_free_outer_menu(store_root, menu)
        reconstructed = read_label_free_outer_menu(store_root)
        if reconstructed.menu_hash != menu.menu_hash:
            raise ProtocolError("HARP v5 compact physical menu changed identity.")
        if menu.outer_target_id not in resumed:
            journal.record(
                outer_target_id=menu.outer_target_id,
                menu_hash=menu.menu_hash,
                manifest_path=receipt.manifest_path,
                npz_path=receipt.npz_path,
            )
        menu_roots[menu.outer_target_id] = store_root
        menu_receipts.append(receipt)
    return LabelFreeMenuBundle(
        menus=produced,
        roots=menu_roots,
        receipts=tuple(menu_receipts),
    )


def recover_or_materialize_compatibility(
    *,
    root: Path,
    output_state: str,
    menus: Sequence[Any],
    pipeline: Any,
    cache: Any,
    config: Any,
    scratch: Path,
) -> CompatibilityBundle:
    """Recover or compute compatibility, then bind it to exact menu bytes."""

    store_root = root / "stores/label_free_support_compatibility"
    if output_state == "LABEL_FREE_RECOVERY" and store_root.exists():
        state = read_artifact_value(
            store_root,
            role="label_free_support_compatibility",
        )
        receipt = existing_compact_store_receipt(store_root)
    else:
        state = pipeline.materialize_label_free_support_compatibility(
            tuple(menus),
            cache,
            config=config,
            scratch_root=scratch,
        )
        receipt = write_artifact_value(
            store_root,
            state,
            role="label_free_support_compatibility",
        )
    # Opaque Python state is deliberately absent after reconstruction.  The
    # exact physical menus are the external witness that hydrates it safely.
    state = bind_compatibility_artifact_to_outer_menus(state, tuple(menus))
    return CompatibilityBundle(state=state, receipt=receipt)


def existing_compact_store_receipt(root: Path) -> CompactStoreReceipt:
    """Project an already authenticated label-free store into a receipt."""

    manifest_path = root / "manifest.json"
    npz_path = root / "arrays.npz"
    manifest = read_json(manifest_path)
    chunks = manifest.get("chunk_hashes")
    if (
        type(manifest.get("manifest_hash")) is not str
        or type(manifest.get("npz_sha256")) is not str
        or not isinstance(chunks, Mapping)
    ):
        raise ProtocolError("HARP v5 resumed compact store receipt is malformed.")
    return CompactStoreReceipt(
        root=root,
        manifest_path=manifest_path,
        npz_path=npz_path,
        manifest_hash=str(manifest["manifest_hash"]),
        manifest_sha256=sha256_file(manifest_path),
        npz_sha256=sha256_file(npz_path),
        chunk_hashes={str(key): str(value) for key, value in chunks.items()},
    )


__all__ = (
    "CompatibilityBundle",
    "LabelFreeMenuBundle",
    "existing_compact_store_receipt",
    "recover_or_materialize_compatibility",
    "recover_or_materialize_label_free_menus",
)
