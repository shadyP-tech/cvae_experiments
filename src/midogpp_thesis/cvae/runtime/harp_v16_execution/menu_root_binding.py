"""Authenticated center-to-menu bindings for fresh HARP v16 validators.

The binding is deliberately runtime-owned: the durable physical menus are an
input to each independently spawned validator, not an orchestration-only path
convention.  A binding therefore carries the ordered center universe, the
canonical common parent and roots, the reconstructed menu identities, and all
receipt identities needed to reject a path swap or post-seal byte mutation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash
from ..artifact_io import read_json, sha256_file
from .contracts import ActionKind, LabelFreeOuterMenu
from .hash_contracts import require_sha256
from .stores import CompactStoreReceipt, read_label_free_outer_menu


_SCHEMA = "midogpp_harp_v16_center_menu_root_binding_v1"
_ENTRY_KEYS = {
    "center_id",
    "menu_root",
    "menu_hash",
    "manifest_hash",
    "manifest_sha256",
    "npz_sha256",
    "chunk_hashes",
}
_PAYLOAD_KEYS = {
    "schema_version",
    "ordered_center_ids",
    "common_parent",
    "entries",
    "binding_hash",
}


def _center_id(value: object, *, role: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or "\x00" in value
    ):
        raise ProtocolError(f"HARP v16 {role} is not a canonical center identity.")
    return value


def _ordered_centers(values: Sequence[object]) -> tuple[str, ...]:
    centers = tuple(_center_id(value, role="menu-binding center") for value in values)
    if not centers or centers != tuple(sorted(set(centers))):
        raise ProtocolError("HARP v16 menu-binding center order/coverage drifted.")
    return centers


def _digest(value: object, *, role: str) -> str:
    return require_sha256(value, name=role)


def _canonical_existing_directory(value: object, *, role: str) -> Path:
    try:
        candidate = Path(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"HARP v16 {role} is not a path.") from exc
    if not candidate.is_absolute():
        raise ProtocolError(f"HARP v16 {role} is not canonical.")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"HARP v16 {role} is absent or unsafe.") from exc
    if candidate != resolved or candidate.is_symlink() or not candidate.is_dir():
        raise ProtocolError(f"HARP v16 {role} is absent, symlinked, or noncanonical.")
    return candidate


def _chunk_hashes(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise ProtocolError("HARP v16 menu receipt chunk hashes are malformed.")
    rows: list[tuple[str, str]] = []
    for raw_key, raw_digest in value.items():
        if (
            type(raw_key) is not str
            or not raw_key
            or raw_key.strip() != raw_key
            or "/" in raw_key
            or "\\" in raw_key
        ):
            raise ProtocolError("HARP v16 menu receipt chunk name is unsafe.")
        rows.append((raw_key, _digest(raw_digest, role="menu receipt chunk hash")))
    ordered = tuple(sorted(rows))
    if not ordered or len({name for name, _ in ordered}) != len(ordered):
        raise ProtocolError("HARP v16 menu receipt chunk inventory is malformed.")
    return ordered


def _chunk_hash_pairs(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (tuple, list)):
        raise ProtocolError("HARP v16 menu receipt chunk hashes are malformed.")
    try:
        rows = tuple(value)
        if any(not isinstance(row, (tuple, list)) or len(row) != 2 for row in rows):
            raise TypeError
        names = tuple(row[0] for row in rows)
        if len(set(names)) != len(names):
            raise TypeError
        return _chunk_hashes({row[0]: row[1] for row in rows})
    except (TypeError, ValueError) as exc:
        raise ProtocolError(
            "HARP v16 menu receipt chunk hashes are malformed."
        ) from exc


@dataclass(frozen=True, slots=True)
class CenterMenuRootEntry:
    """One center's immutable durable menu and receipt projection."""

    center_id: str
    menu_root: Path
    menu_hash: str
    manifest_hash: str
    manifest_sha256: str
    npz_sha256: str
    chunk_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        center = _center_id(self.center_id, role="menu-binding entry center")
        root = _canonical_existing_directory(
            self.menu_root, role=f"menu root for center {center}"
        )
        chunks = _chunk_hash_pairs(self.chunk_hashes)
        object.__setattr__(self, "center_id", center)
        object.__setattr__(self, "menu_root", root)
        object.__setattr__(
            self, "menu_hash", _digest(self.menu_hash, role="durable menu hash")
        )
        object.__setattr__(
            self,
            "manifest_hash",
            _digest(self.manifest_hash, role="menu receipt manifest hash"),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            _digest(self.manifest_sha256, role="menu receipt manifest SHA-256"),
        )
        object.__setattr__(
            self,
            "npz_sha256",
            _digest(self.npz_sha256, role="menu receipt NPZ SHA-256"),
        )
        object.__setattr__(self, "chunk_hashes", chunks)

    def payload(self) -> dict[str, object]:
        return {
            "center_id": self.center_id,
            "menu_root": str(self.menu_root),
            "menu_hash": self.menu_hash,
            "manifest_hash": self.manifest_hash,
            "manifest_sha256": self.manifest_sha256,
            "npz_sha256": self.npz_sha256,
            "chunk_hashes": dict(self.chunk_hashes),
        }


@dataclass(frozen=True, slots=True)
class CenterMenuRootBinding:
    """Canonical, spawn-serializable binding of centers to physical menus."""

    ordered_center_ids: tuple[str, ...]
    common_parent: Path
    entries: tuple[CenterMenuRootEntry, ...]
    binding_hash: str = field(init=False)

    def __post_init__(self) -> None:
        centers = _ordered_centers(self.ordered_center_ids)
        parent = _canonical_existing_directory(
            self.common_parent, role="physical-menu common parent"
        )
        entries = tuple(self.entries)
        if (
            len(entries) != len(centers)
            or any(not isinstance(entry, CenterMenuRootEntry) for entry in entries)
            or tuple(entry.center_id for entry in entries) != centers
            or len({entry.menu_root for entry in entries}) != len(entries)
        ):
            raise ProtocolError(
                "HARP v16 menu-binding entry order/coverage/bijection drifted."
            )
        for entry in entries:
            if (
                entry.menu_root.parent != parent
                or entry.menu_root.name != f"outer_{entry.center_id}"
            ):
                raise ProtocolError("HARP v16 menu root escaped its center binding.")
        body = self._body(centers=centers, parent=parent, entries=entries)
        object.__setattr__(self, "ordered_center_ids", centers)
        object.__setattr__(self, "common_parent", parent)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "binding_hash", canonical_hash(body))

    @classmethod
    def create(
        cls,
        *,
        common_parent: Path,
        centers: Sequence[str],
        menu_roots: Mapping[str, Path],
        menus: Sequence[LabelFreeOuterMenu],
        receipts: Sequence[CompactStoreReceipt],
    ) -> "CenterMenuRootBinding":
        """Bind the in-memory menu inventory to exact durable receipt bytes."""

        center_ids = _ordered_centers(tuple(centers))
        if (
            not isinstance(menu_roots, Mapping)
            or set(menu_roots) != set(center_ids)
            or len(menu_roots) != len(center_ids)
        ):
            raise ProtocolError(
                "HARP v16 menu binding requires exact center-keyed root coverage."
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
            or tuple(menu.outer_target_id for menu in menu_rows) != center_ids
        ):
            raise ProtocolError("HARP v16 menu binding inventory is incomplete.")
        entries: list[CenterMenuRootEntry] = []
        for center, menu, receipt in zip(
            center_ids, menu_rows, receipt_rows, strict=True
        ):
            try:
                root = Path(menu_roots[center])
            except TypeError as exc:
                raise ProtocolError("HARP v16 menu root is not a path.") from exc
            if (
                receipt.root != root
                or receipt.manifest_path != root / "manifest.json"
                or receipt.npz_path != root / "arrays.npz"
            ):
                raise ProtocolError("HARP v16 menu receipt/root binding drifted.")
            entries.append(
                CenterMenuRootEntry(
                    center_id=center,
                    menu_root=root,
                    menu_hash=menu.menu_hash,
                    manifest_hash=receipt.manifest_hash,
                    manifest_sha256=receipt.manifest_sha256,
                    npz_sha256=receipt.npz_sha256,
                    chunk_hashes=tuple(sorted(receipt.chunk_hashes.items())),
                )
            )
        binding = cls(
            ordered_center_ids=center_ids,
            common_parent=common_parent,
            entries=tuple(entries),
        )
        binding.validate_durable()
        return binding

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        validate_durable: bool = True,
    ) -> "CenterMenuRootBinding":
        """Reconstruct and fully revalidate a child-process binding payload."""

        if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS:
            raise ProtocolError("HARP v16 serialized menu binding is malformed.")
        if payload.get("schema_version") != _SCHEMA:
            raise ProtocolError("HARP v16 serialized menu binding schema drifted.")
        raw_centers = payload.get("ordered_center_ids")
        raw_entries = payload.get("entries")
        if not isinstance(raw_centers, list) or not isinstance(raw_entries, list):
            raise ProtocolError("HARP v16 serialized menu binding inventory is malformed.")
        entries: list[CenterMenuRootEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping) or set(raw) != _ENTRY_KEYS:
                raise ProtocolError("HARP v16 serialized menu entry is malformed.")
            entries.append(
                CenterMenuRootEntry(
                    center_id=raw["center_id"],  # type: ignore[arg-type]
                    menu_root=raw["menu_root"],  # type: ignore[arg-type]
                    menu_hash=raw["menu_hash"],  # type: ignore[arg-type]
                    manifest_hash=raw["manifest_hash"],  # type: ignore[arg-type]
                    manifest_sha256=raw["manifest_sha256"],  # type: ignore[arg-type]
                    npz_sha256=raw["npz_sha256"],  # type: ignore[arg-type]
                    chunk_hashes=_chunk_hashes(raw["chunk_hashes"]),
                )
            )
        binding = cls(
            ordered_center_ids=tuple(raw_centers),  # type: ignore[arg-type]
            common_parent=payload["common_parent"],  # type: ignore[arg-type]
            entries=tuple(entries),
        )
        supplied_hash = _digest(
            payload.get("binding_hash"), role="serialized menu binding hash"
        )
        if supplied_hash != binding.binding_hash:
            raise ProtocolError("HARP v16 serialized menu binding hash drifted.")
        if type(validate_durable) is not bool:
            raise ProtocolError("HARP v16 menu-binding validation mode is malformed.")
        if validate_durable:
            binding.validate_durable()
        return binding

    @property
    def menu_roots(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {entry.center_id: entry.menu_root for entry in self.entries}
        )

    @property
    def menu_hashes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {entry.center_id: entry.menu_hash for entry in self.entries}
        )

    @property
    def receipt_hashes(self) -> Mapping[str, str]:
        return MappingProxyType(
            {entry.center_id: entry.manifest_hash for entry in self.entries}
        )

    def to_payload(self) -> dict[str, object]:
        body = self._body(
            centers=self.ordered_center_ids,
            parent=self.common_parent,
            entries=self.entries,
        )
        return {**body, "binding_hash": self.binding_hash}

    def validate_durable(self) -> tuple[LabelFreeOuterMenu, ...]:
        """Verify paths, receipt bytes/semantics, menus, and action inventory."""

        parent = _canonical_existing_directory(
            self.common_parent, role="physical-menu common parent"
        )
        if parent != self.common_parent:
            raise ProtocolError("HARP v16 physical-menu parent identity drifted.")
        menus: list[LabelFreeOuterMenu] = []
        for entry in self.entries:
            root = _canonical_existing_directory(
                entry.menu_root, role=f"menu root for center {entry.center_id}"
            )
            if root.parent != parent or root.name != f"outer_{entry.center_id}":
                raise ProtocolError("HARP v16 durable menu root escaped its binding.")
            manifest_path = root / "manifest.json"
            npz_path = root / "arrays.npz"
            if (
                not manifest_path.is_file()
                or manifest_path.is_symlink()
                or not npz_path.is_file()
                or npz_path.is_symlink()
                or manifest_path.resolve(strict=True) != manifest_path
                or npz_path.resolve(strict=True) != npz_path
            ):
                raise ProtocolError("HARP v16 menu receipt members are absent or unsafe.")
            if sha256_file(manifest_path) != entry.manifest_sha256:
                raise ProtocolError("HARP v16 menu receipt manifest bytes drifted.")
            if sha256_file(npz_path) != entry.npz_sha256:
                raise ProtocolError("HARP v16 menu receipt NPZ bytes drifted.")
            manifest = read_json(manifest_path)
            body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
            if (
                manifest.get("schema_version")
                != "midogpp_harp_v16_outer_menu_compact_store_v2"
                or manifest.get("manifest_hash") != entry.manifest_hash
                or canonical_hash(body) != entry.manifest_hash
                or manifest.get("outer_target_id") != entry.center_id
                or manifest.get("menu_hash") != entry.menu_hash
                or manifest.get("npz_member") != "arrays.npz"
                or manifest.get("npz_sha256") != entry.npz_sha256
                or manifest.get("chunk_hashes") != dict(entry.chunk_hashes)
                or manifest.get("labels_consumed") is not False
                or manifest.get("physical_expert_weight") != 1.0
            ):
                raise ProtocolError("HARP v16 menu receipt semantics drifted.")
            menu = read_label_free_outer_menu(root)
            if (
                menu.outer_target_id != entry.center_id
                or menu.menu_hash != entry.menu_hash
            ):
                raise ProtocolError("HARP v16 durable menu identity drifted.")
            _validate_complete_candidate_inventory(
                menu, centers=self.ordered_center_ids
            )
            menus.append(menu)
        return tuple(menus)

    @staticmethod
    def _body(
        *,
        centers: tuple[str, ...],
        parent: Path,
        entries: tuple[CenterMenuRootEntry, ...],
    ) -> dict[str, object]:
        return {
            "schema_version": _SCHEMA,
            "ordered_center_ids": list(centers),
            "common_parent": str(parent),
            "entries": [entry.payload() for entry in entries],
        }


def _validate_complete_candidate_inventory(
    menu: LabelFreeOuterMenu, *, centers: tuple[str, ...]
) -> None:
    outer = menu.outer_target_id
    expected_contexts = {("support", outer), ("target", outer)}
    by_context: dict[tuple[str, str], list[object]] = {}
    for block in menu.blocks:
        by_context.setdefault((block.surface_role, block.query_center_id), []).append(
            block
        )
    if set(by_context) != expected_contexts:
        raise ProtocolError("HARP v16 durable menu context inventory is incomplete.")
    for (_role, _query), raw_blocks in by_context.items():
        blocks = tuple(raw_blocks)
        expected_sources = tuple(center for center in centers if center != outer)
        controls = tuple(
            block.action_kind
            for block in blocks
            if block.action_kind is not ActionKind.HXE
        )
        sources = tuple(
            sorted(
                block.selected_source_id
                for block in blocks
                if block.action_kind is ActionKind.HXE
            )
        )
        if (
            controls.count(ActionKind.B) != 1
            or controls.count(ActionKind.U) != 1
            or len(controls) != 2
            or sources != expected_sources
            or len(blocks) != len(expected_sources) + 2
        ):
            raise ProtocolError(
                "HARP v16 durable menu candidate-set inventory is incomplete."
            )


def validate_serialized_center_menu_root_binding(
    payload: Mapping[str, object]
) -> str:
    """Spawn-safe entry point used by tests and fresh validator workers."""

    return CenterMenuRootBinding.from_payload(payload).binding_hash


__all__ = (
    "CenterMenuRootBinding",
    "CenterMenuRootEntry",
    "validate_serialized_center_menu_root_binding",
)
