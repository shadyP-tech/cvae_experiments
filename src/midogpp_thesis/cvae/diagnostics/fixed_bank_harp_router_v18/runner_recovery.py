"""Durable recovery and persistence helpers for the HARP v18 runner.

This module owns the mechanics that are intentionally independent from the
runner's scientific phase ordering: authenticated label-free menu recovery,
compact-store receipt projection, idempotent artifact/JSON persistence, stable
preflight hashing, and phase announcements.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import numpy as np

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_bytes, canonical_hash
from ...runtime.artifact_io import atomic_json, read_json, sha256_file
from ...runtime.harp_v18_execution.contracts import (
    ArtifactValue,
    HarpV18Pipeline,
    LabelFreeOuterMenu,
)
from ...runtime.harp_v18_execution.journal import LabelFreeProgressJournal
from ...runtime.harp_v18_execution.stores import (
    CompactStoreReceipt,
    read_artifact_value,
    read_label_free_outer_menu,
    write_artifact_value,
    write_label_free_outer_menu,
)
from .config import HarpStage90V18Config
from .execution import validate_complete_physical_menus
from .source_train_label_access_fence import source_train_label_access_has_begun


def recover_or_materialize_menus(
    *,
    root: Path,
    centers: tuple[str, ...],
    journal: LabelFreeProgressJournal,
    pipeline: HarpV18Pipeline,
    config: HarpStage90V18Config,
    cache: object,
    scratch: Path,
) -> tuple[tuple[LabelFreeOuterMenu, ...], tuple[CompactStoreReceipt, ...]]:
    """Resume only authenticated label-free center stores."""

    if source_train_label_access_has_begun(root):
        raise ProtocolError(
            "HARP v18 forbids physical-menu recovery after source-train label access."
        )
    parent = root / "stores/physical_menu"
    existing: dict[str, tuple[LabelFreeOuterMenu, CompactStoreReceipt]] = {}
    completed = journal.completed()
    for center in centers:
        menu_root = parent / f"center_{center}"
        manifest = menu_root / "manifest.json"
        arrays = menu_root / "arrays.npz"
        if manifest.exists() != arrays.exists():
            raise ProtocolError("HARP v18 physical menu is only partially durable.")
        if manifest.exists():
            journaled = journal.require_resumable(center) if center in completed else None
            menu = read_label_free_outer_menu(menu_root)
            receipt = compact_store_receipt(menu_root)
            if menu.outer_target_id != center:
                raise ProtocolError("HARP v18 recovered menu center drifted.")
            if journaled is None:
                journal.record(
                    outer_target_id=center,
                    menu_hash=menu.menu_hash,
                    manifest_path=receipt.manifest_path,
                    npz_path=receipt.npz_path,
                )
            elif completed[center].get("menu_hash") != menu.menu_hash:
                raise ProtocolError("HARP v18 recovered menu hash drifted.")
            existing[center] = (menu, receipt)
    if len(existing) != len(centers):
        pending_centers = tuple(center for center in centers if center not in existing)
        produced = tuple(
            pipeline.materialize_label_free_outer_menus(
                config,
                cache,
                outer_targets=pending_centers,
                scratch_root=scratch,
            )
        )
        validate_complete_physical_menus(
            produced,
            centers=centers,
            expected_context_ids=pending_centers,
        )
        if tuple(menu.outer_target_id for menu in produced) != pending_centers:
            raise ProtocolError("HARP v18 materialized menu order drifted.")
        for menu in produced:
            menu_root = parent / f"center_{menu.outer_target_id}"
            prior = existing.get(menu.outer_target_id)
            if prior is not None:
                if prior[0].menu_hash != menu.menu_hash:
                    raise ProtocolError("HARP v18 recovered and rebuilt menus differ.")
                continue
            receipt = write_label_free_outer_menu(menu_root, menu)
            journal.record(
                outer_target_id=menu.outer_target_id,
                menu_hash=menu.menu_hash,
                manifest_path=receipt.manifest_path,
                npz_path=receipt.npz_path,
            )
            existing[menu.outer_target_id] = (menu, receipt)
    rows = tuple(existing[center] for center in centers)
    validate_complete_physical_menus(
        tuple(row[0] for row in rows), centers=centers
    )
    return tuple(row[0] for row in rows), tuple(row[1] for row in rows)


def compact_store_receipt(root: Path) -> CompactStoreReceipt:
    """Project an authenticated compact store into its durable receipt."""

    manifest_path = root / "manifest.json"
    npz_path = root / "arrays.npz"
    payload = read_json(manifest_path)
    chunks = payload.get("chunk_hashes")
    if not isinstance(chunks, Mapping):
        raise ProtocolError("HARP v18 compact-store receipt lacks chunk hashes.")
    receipt = CompactStoreReceipt(
        root=root.resolve(),
        manifest_path=manifest_path.resolve(),
        npz_path=npz_path.resolve(),
        manifest_hash=str(payload.get("manifest_hash")),
        manifest_sha256=sha256_file(manifest_path),
        npz_sha256=sha256_file(npz_path),
        chunk_hashes={str(key): str(value) for key, value in chunks.items()},
    )
    if payload.get("npz_sha256") != receipt.npz_sha256:
        raise ProtocolError("HARP v18 compact-store receipt NPZ hash drifted.")
    return receipt


def write_or_validate_artifact(
    root: Path, value: ArtifactValue, *, role: str
) -> CompactStoreReceipt:
    """Write an opaque artifact once or prove an existing copy is identical."""

    manifest_path = root / "manifest.json"
    npz_path = root / "arrays.npz"
    if manifest_path.exists() != npz_path.exists():
        raise ProtocolError("HARP v18 opaque artifact is only partially durable.")
    if not manifest_path.exists():
        return write_artifact_value(root, value, role=role)
    observed = read_artifact_value(root, role=role)
    if canonical_bytes(observed.manifest) != canonical_bytes(value.manifest):
        raise ProtocolError("HARP v18 recovered opaque artifact manifest drifted.")
    if set(observed.arrays) != set(value.arrays):
        raise ProtocolError("HARP v18 recovered opaque artifact arrays drifted.")
    for name in observed.arrays:
        left = np.asarray(observed.arrays[name])
        right = np.asarray(value.arrays[name])
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise ProtocolError("HARP v18 recovered opaque artifact bytes drifted.")
    return compact_store_receipt(root)


def persist_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    """Persist JSON once and reject any later semantic drift."""

    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("HARP v18 durable JSON path is unsafe.")
        if canonical_bytes(read_json(path)) != canonical_bytes(payload):
            raise ProtocolError("HARP v18 refuses to overwrite drifted durable JSON.")
        return
    atomic_json(path, payload)


def stable_preflight_hash(preflight: Mapping[str, object]) -> str:
    """Bind science/topology while excluding volatile capacity observations."""

    stable = dict(preflight)
    raw_live = stable.get("live_workstation")
    if isinstance(raw_live, Mapping):
        live = dict(raw_live)
        live.pop("scratch_free_bytes", None)
        live.pop("scratch_probe_path", None)
        raw_gpus = live.get("gpus")
        if isinstance(raw_gpus, list):
            live["gpus"] = [
                {
                    key: value
                    for key, value in row.items()
                    if key != "memory_free_mib"
                }
                if isinstance(row, Mapping)
                else row
                for row in raw_gpus
            ]
        stable["live_workstation"] = live
    return canonical_hash(stable)


def announce(phase: str) -> None:
    """Emit one immediately visible phase transition to stderr."""

    print(f"[harp-v18] phase={phase}", file=sys.stderr, flush=True)


__all__ = (
    "announce",
    "compact_store_receipt",
    "persist_or_validate_json",
    "recover_or_materialize_menus",
    "stable_preflight_hash",
    "write_or_validate_artifact",
)
