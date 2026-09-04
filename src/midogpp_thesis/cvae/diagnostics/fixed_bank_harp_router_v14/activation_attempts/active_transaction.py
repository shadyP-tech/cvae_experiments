"""Crash-safe revocation transaction for active unclaimed v14 authority."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from pathlib import Path

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_bytes
from .. import authorization
from ..activation_lock import activation_lock
from ..activation_paths import RepositoryBoundary
from ..activation_transaction import TRANSACTION_RELATIVE_PATH
from .active_audit import build_active_supersession_plan
from .admin_snapshot import require_exact_snapshot_tree
from .audit import require_exact_regular
from .contracts import (
    ACTIVE_SUPERSESSION_CONFIRMATION,
    ACTIVE_SUPERSESSION_RECEIPT,
    ARCHIVED_ADMIN_CONTENT,
    ARCHIVED_ADMIN_MANIFEST,
    ARCHIVED_AMENDMENT,
    ARCHIVED_FINAL_CATALOG,
    ARCHIVED_FINAL_CONFIG,
    ARCHIVED_FINAL_REGISTRY,
    ARCHIVED_JOURNAL,
    ARCHIVED_RETIREMENT_FENCE,
    HarpV14ActiveActivationSupersessionPlan,
    HarpV14ActiveActivationSupersessionReceipt,
    RETIRED_ADMIN_OUTPUT,
    active_receipt_payload,
    retirement_fence_payload,
)


FaultInjector = Callable[[str], None]


def supersede_harp_v14_active_activation(
    plan: HarpV14ActiveActivationSupersessionPlan,
    *,
    confirmation: str,
    _fault_injector: FaultInjector | None = None,
) -> HarpV14ActiveActivationSupersessionReceipt:
    """Archive and revoke exact active authority that never claimed its lease."""

    if type(plan) is not HarpV14ActiveActivationSupersessionPlan:
        raise ProtocolError("HARP v14 active supersession requires a typed plan.")
    if confirmation != ACTIVE_SUPERSESSION_CONFIRMATION:
        raise ProtocolError(
            "HARP v14 active supersession confirmation is absent or drifted."
        )
    boundary = RepositoryBoundary.open(plan.repository_root)
    with activation_lock(boundary):
        current = build_active_supersession_plan(boundary)
        if current.supersession_plan_hash != plan.supersession_plan_hash:
            raise ProtocolError(
                "HARP v14 active supersession state changed after planning."
            )
        _persist_active_archive(current)
        _inject(_fault_injector, "archive_durable")
        _claim_or_validate_retirement_fence(current)
        _inject(_fault_injector, "retirement_fence_durable")

        journal = current.journal
        # Registry is the runnable gate and is deliberately closed first.
        _restore_metadata_idempotent(
            journal.registry_path,
            expected=journal.final_registry_bytes,
            replacement=journal.original_registry_bytes,
            token=current.supersession_plan_hash + ".registry",
            label="activated registry",
        )
        _inject(_fault_injector, "registry_restored")
        _restore_metadata_idempotent(
            journal.catalog_path,
            expected=journal.final_catalog_bytes,
            replacement=journal.original_catalog_bytes,
            token=current.supersession_plan_hash + ".catalog",
            label="activated catalog",
        )
        _inject(_fault_injector, "catalog_restored")
        _restore_metadata_idempotent(
            journal.config_path,
            expected=journal.final_config_bytes,
            replacement=journal.original_config_bytes,
            token=current.supersession_plan_hash + ".config",
            label="activated config",
        )
        _inject(_fault_injector, "config_restored")

        _archive_live_admin_output_idempotent(current)
        _inject(_fault_injector, "admin_output_retired")
        _retire_amendment_idempotent(current)
        _inject(_fault_injector, "active_amendment_retired")

        receipt = active_receipt_payload(current)
        _persist_or_validate(
            current.archive_root / ACTIVE_SUPERSESSION_RECEIPT,
            canonical_bytes(receipt) + b"\n",
        )
        _fsync_directories((current.archive_root, current.archive_root.parent))
        _inject(_fault_injector, "terminal_receipt_durable")
        _archive_retirement_fence_idempotent(current)
        _inject(_fault_injector, "retirement_fence_archived")

        # The journal is the last live recovery locator. No fallible operation
        # or injected fault follows its retirement.
        journal_path = boundary.member(
            TRANSACTION_RELATIVE_PATH,
            label="activation transaction journal",
            kind="file",
        )
        require_exact_regular(
            journal_path,
            journal.to_bytes(),
            label="active activation journal",
        )
        journal_path.unlink()
        _fsync_directories((journal_path.parent,))
        return HarpV14ActiveActivationSupersessionReceipt(payload=receipt)


def _persist_active_archive(plan: HarpV14ActiveActivationSupersessionPlan) -> None:
    root = plan.archive_root
    _ensure_archive_directory(root)
    journal = plan.journal
    for name, raw in (
        (ARCHIVED_JOURNAL, journal.to_bytes()),
        (ARCHIVED_AMENDMENT, journal.amendment_bytes),
        (ARCHIVED_FINAL_CONFIG, journal.final_config_bytes),
        (ARCHIVED_FINAL_REGISTRY, journal.final_registry_bytes),
        (ARCHIVED_FINAL_CATALOG, journal.final_catalog_bytes),
    ):
        _persist_or_validate(root / name, raw)
    _persist_admin_snapshot_content(plan)
    # The manifest is the completeness marker for its content and is therefore
    # committed only after every content byte and directory is durable.
    _persist_or_validate(
        root / ARCHIVED_ADMIN_MANIFEST,
        canonical_bytes(plan.admin_snapshot_manifest) + b"\n",
    )
    _fsync_directories((root, root.parent))


def _ensure_archive_directory(root: Path) -> None:
    parent = root.parent
    if not os.path.lexists(parent):
        parent.mkdir(mode=0o700)
        _fsync_directories((parent.parent,))
    elif not parent.is_dir() or parent.is_symlink():
        raise ProtocolError("HARP v14 active supersession archive parent is unsafe.")
    if not os.path.lexists(root):
        root.mkdir(mode=0o700)
        _fsync_directories((parent,))
    elif not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v14 active supersession archive root is unsafe.")


def _persist_admin_snapshot_content(
    plan: HarpV14ActiveActivationSupersessionPlan,
) -> None:
    destination = plan.archive_root / ARCHIVED_ADMIN_CONTENT
    state = plan.admin_snapshot_manifest.get("state")
    if state == "ABSENT":
        if os.path.lexists(destination):
            raise ProtocolError(
                "HARP v14 absent admin snapshot unexpectedly has archived content."
            )
        return
    if state != "WORKSPACE_ADMIN_PRISTINE":
        raise ProtocolError("HARP v14 admin snapshot state is malformed.")
    _ensure_owned_directory(destination, label="admin snapshot content")
    directories = plan.admin_snapshot_manifest.get("directories")
    if not isinstance(directories, list) or any(
        type(item) is not str for item in directories
    ):
        raise ProtocolError("HARP v14 admin snapshot directory inventory is malformed.")
    for relative in directories:
        path = destination / relative
        _ensure_owned_directory(path, label="admin snapshot directory")
    for relative, raw in sorted(plan.admin_snapshot_files.items()):
        _persist_or_validate(destination / relative, raw)
    require_exact_snapshot_tree(
        destination,
        directories=tuple(directories),
        files=plan.admin_snapshot_files,
    )


def _ensure_owned_directory(path: Path, *, label: str) -> None:
    if not os.path.lexists(path):
        path.mkdir(mode=0o700)
        _fsync_directories((path.parent,))
    elif not path.is_dir() or path.is_symlink():
        raise ProtocolError(f"HARP v14 {label} is unsafe.")


def _claim_or_validate_retirement_fence(
    plan: HarpV14ActiveActivationSupersessionPlan,
) -> None:
    live = authorization.lease_path(plan.repository_root)
    archived = plan.archive_root / ARCHIVED_RETIREMENT_FENCE
    raw = canonical_bytes(retirement_fence_payload(plan)) + b"\n"
    if os.path.lexists(archived):
        require_exact_regular(archived, raw, label="archived retirement fence")
        if os.path.lexists(live):
            raise ProtocolError("HARP v14 retirement fence location is ambiguous.")
        return
    if os.path.lexists(live):
        require_exact_regular(live, raw, label="live retirement fence")
        return
    # O_EXCL at the scientific lease path is the cross-process arbitration
    # primitive: either this fence wins, or claim_authorization's mkdir wins.
    _persist_or_validate(live, raw)
    _fsync_directories((live.parent,))


def _archive_live_admin_output_idempotent(
    plan: HarpV14ActiveActivationSupersessionPlan,
) -> None:
    state = plan.admin_snapshot_manifest.get("state")
    retired = plan.archive_root / RETIRED_ADMIN_OUTPUT
    if state == "ABSENT":
        if os.path.lexists(plan.output_root) or os.path.lexists(retired):
            raise ProtocolError("HARP v14 absent admin output changed after planning.")
        return
    directories = plan.admin_snapshot_manifest.get("directories")
    if not isinstance(directories, list) or any(
        type(item) is not str for item in directories
    ):
        raise ProtocolError("HARP v14 admin snapshot directory inventory is malformed.")
    expected_directories = tuple(directories)
    if os.path.lexists(retired):
        require_exact_snapshot_tree(
            retired,
            directories=expected_directories,
            files=plan.admin_snapshot_files,
        )
        if os.path.lexists(plan.output_root):
            raise ProtocolError("HARP v14 admin output recovery is ambiguous.")
        return
    require_exact_snapshot_tree(
        plan.output_root,
        directories=expected_directories,
        files=plan.admin_snapshot_files,
    )
    os.replace(plan.output_root, retired)
    _fsync_directories((plan.output_root.parent, plan.archive_root))
    require_exact_snapshot_tree(
        retired,
        directories=expected_directories,
        files=plan.admin_snapshot_files,
    )


def _restore_metadata_idempotent(
    path: Path,
    *,
    expected: bytes,
    replacement: bytes,
    token: str,
    label: str,
) -> None:
    if path.is_file() and not path.is_symlink() and path.read_bytes() == replacement:
        return
    require_exact_regular(path, expected, label=label)
    temporary = path.parent / f".{path.name}.harp-v14-active-supersession-{token}.tmp"
    if os.path.lexists(temporary):
        if (
            not temporary.is_file()
            or temporary.is_symlink()
            or temporary.read_bytes() != replacement
        ):
            raise ProtocolError("HARP v14 active supersession staging path is unsafe.")
    else:
        _persist_or_validate(temporary, replacement)
    os.replace(temporary, path)
    _fsync_directories((path.parent,))
    require_exact_regular(path, replacement, label=f"restored {label}")


def _retire_amendment_idempotent(
    plan: HarpV14ActiveActivationSupersessionPlan,
) -> None:
    live = plan.journal.amendment_path
    archived = plan.archive_root / ARCHIVED_AMENDMENT
    require_exact_regular(
        archived,
        plan.journal.amendment_bytes,
        label="archived amendment",
    )
    if os.path.lexists(live):
        require_exact_regular(
            live,
            plan.journal.amendment_bytes,
            label="active amendment",
        )
        live.unlink()
        _fsync_directories((live.parent,))


def _archive_retirement_fence_idempotent(
    plan: HarpV14ActiveActivationSupersessionPlan,
) -> None:
    live = authorization.lease_path(plan.repository_root)
    archived = plan.archive_root / ARCHIVED_RETIREMENT_FENCE
    raw = canonical_bytes(retirement_fence_payload(plan)) + b"\n"
    if os.path.lexists(archived):
        require_exact_regular(archived, raw, label="archived retirement fence")
        if os.path.lexists(live):
            raise ProtocolError("HARP v14 retirement fence location is ambiguous.")
        return
    require_exact_regular(live, raw, label="live retirement fence")
    os.replace(live, archived)
    _fsync_directories((live.parent, plan.archive_root))
    require_exact_regular(archived, raw, label="archived retirement fence")


def _persist_or_validate(path: Path, raw: bytes) -> None:
    if os.path.lexists(path):
        require_exact_regular(path, raw, label="supersession archive member")
        return
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise ProtocolError("HARP v14 supersession archive write failed.") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    require_exact_regular(path, raw, label="supersession archive member")


def _fsync_directories(paths: Sequence[Path]) -> None:
    for path in dict.fromkeys(paths):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _inject(injector: FaultInjector | None, point: str) -> None:
    if injector is not None:
        injector(point)


__all__ = ("supersede_harp_v14_active_activation",)
