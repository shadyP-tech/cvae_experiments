"""Crash-safe archive transaction for one rolled-back HARP v10 attempt."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path

from ....protocol import ProtocolError
from ....routing.harp_protocol import canonical_bytes
from ..activation_paths import RepositoryBoundary
from ..activation_transaction import (
    TRANSACTION_RELATIVE_PATH,
    activation_lock,
)
from .audit import build_supersession_plan, require_exact_regular
from .contracts import (
    ARCHIVED_AMENDMENT,
    ARCHIVED_JOURNAL,
    HarpV10ActivationSupersessionPlan,
    HarpV10ActivationSupersessionReceipt,
    SUPERSESSION_CONFIRMATION,
    SUPERSESSION_RECEIPT,
    receipt_payload,
)


FaultInjector = Callable[[str], None]


def supersede_harp_v10_activation(
    plan: HarpV10ActivationSupersessionPlan,
    *,
    confirmation: str,
    _fault_injector: FaultInjector | None = None,
) -> HarpV10ActivationSupersessionReceipt:
    """Durably archive exact prior authority bytes, then retire live copies."""

    if type(plan) is not HarpV10ActivationSupersessionPlan:
        raise ProtocolError("HARP v10 supersession requires a typed plan.")
    if confirmation != SUPERSESSION_CONFIRMATION:
        raise ProtocolError("HARP v10 supersession confirmation is absent or drifted.")
    boundary = RepositoryBoundary.open(plan.repository_root)
    with activation_lock(boundary):
        current = build_supersession_plan(boundary)
        if current.supersession_plan_hash != plan.supersession_plan_hash:
            raise ProtocolError("HARP v10 supersession state changed after planning.")
        receipt = _persist_archive(current)
        _inject(_fault_injector, "archive_durable")

        amendment = current.journal.amendment_path
        if os.path.lexists(amendment):
            require_exact_regular(
                amendment,
                current.journal.amendment_bytes,
                label="active amendment",
            )
            amendment.unlink()
            _fsync_directories((amendment.parent,))
        _inject(_fault_injector, "active_amendment_retired")

        journal_path = boundary.member(
            TRANSACTION_RELATIVE_PATH,
            label="activation transaction journal",
            kind="file",
        )
        require_exact_regular(
            journal_path,
            current.journal.to_bytes(),
            label="active activation journal",
        )
        journal_path.unlink()
        _fsync_directories((journal_path.parent,))
        _inject(_fault_injector, "active_journal_retired")
        return HarpV10ActivationSupersessionReceipt(payload=receipt)


def _persist_archive(
    plan: HarpV10ActivationSupersessionPlan,
) -> Mapping[str, object]:
    root = plan.archive_root
    parent = root.parent
    if not os.path.lexists(parent):
        parent.mkdir(mode=0o700)
        _fsync_directories((parent.parent,))
    elif not parent.is_dir() or parent.is_symlink():
        raise ProtocolError("HARP v10 supersession archive parent is unsafe.")
    if not os.path.lexists(root):
        root.mkdir(mode=0o700)
        _fsync_directories((parent,))
    elif not root.is_dir() or root.is_symlink():
        raise ProtocolError("HARP v10 supersession archive root is unsafe.")

    _persist_or_validate(root / ARCHIVED_JOURNAL, plan.journal.to_bytes())
    _persist_or_validate(root / ARCHIVED_AMENDMENT, plan.journal.amendment_bytes)
    receipt = receipt_payload(plan)
    _persist_or_validate(
        root / SUPERSESSION_RECEIPT,
        canonical_bytes(receipt) + b"\n",
    )
    _fsync_directories((root, parent))
    return receipt


def _persist_or_validate(path: Path, raw: bytes) -> None:
    if os.path.lexists(path):
        require_exact_regular(path, raw, label="supersession archive member")
        return
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise ProtocolError("HARP v10 supersession archive write failed.") from exc
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


__all__ = ("supersede_harp_v10_activation",)
