"""Capability-neutral atomic publication for one newly built directory."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
from typing import Iterator
from uuid import uuid4


@contextmanager
def staged_directory(final_path: Path) -> Iterator[Path]:
    """Build at a sibling path and atomically publish it on successful exit.

    Both the final path and the deterministic staging sibling must be absent
    when the transaction starts. The final path is checked again immediately
    before the single publication rename. Any staged bytes left by an
    exception are renamed to a unique quarantine sibling for inspection.
    """

    final = Path(final_path)
    parent = final.parent
    staging = staging_sibling(final)
    _require_publishable_parent(parent)
    _refuse_existing(final, role="final")
    _refuse_existing(staging, role="staging")
    staging.mkdir()
    try:
        yield staging
        _refuse_existing(final, role="final")
        if not staging.is_dir() or staging.is_symlink():
            raise FileNotFoundError(
                f"Staging directory is absent or no longer a directory: {staging}"
            )
        staging.rename(final)
    except BaseException as exc:
        try:
            quarantine_staging(staging, final_path=final)
        except Exception as quarantine_error:  # Keep the triggering failure primary.
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"Additionally failed to quarantine {staging}: {quarantine_error}"
                )
        raise


@contextmanager
def staged_existing_directory(final_path: Path) -> Iterator[Path]:
    """Adopt an existing prepared directory into one publication transaction.

    The caller is responsible for validating that ``final_path`` is an
    incomplete workspace-preparation bundle rather than an already published
    result. This helper moves those prepared bytes to the deterministic sibling
    before yielding, then publishes or quarantines exactly like
    :func:`staged_directory`.
    """

    final = Path(final_path)
    parent = final.parent
    staging = staging_sibling(final)
    _require_publishable_parent(parent)
    if not final.is_dir() or final.is_symlink():
        raise FileNotFoundError(
            f"Prepared final directory is absent or invalid: {final}"
        )
    _refuse_existing(staging, role="staging")
    final.rename(staging)
    try:
        yield staging
        _refuse_existing(final, role="final")
        if not staging.is_dir() or staging.is_symlink():
            raise FileNotFoundError(
                f"Staging directory is absent or no longer a directory: {staging}"
            )
        staging.rename(final)
    except BaseException as exc:
        try:
            quarantine_staging(staging, final_path=final)
        except Exception as quarantine_error:  # Keep the triggering failure primary.
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"Additionally failed to quarantine {staging}: {quarantine_error}"
                )
        raise


def staging_sibling(final_path: Path) -> Path:
    """Return the deterministic same-parent staging path for ``final_path``."""

    final = Path(final_path)
    if not final.name:
        raise ValueError("final_path must name a directory.")
    return final.with_name(f".{final.name}.staging")


def quarantine_staging(
    staging_path: Path,
    *,
    final_path: Path | None = None,
) -> Path | None:
    """Move an extant staging path to a unique same-parent quarantine sibling."""

    staging = Path(staging_path)
    if not _lexists(staging):
        return None
    final = Path(final_path) if final_path is not None else staging
    if final.parent != staging.parent:
        raise ValueError("Staging and quarantine paths must share the final parent.")
    quarantine = quarantine_sibling(final)
    staging.rename(quarantine)
    return quarantine


def quarantine_sibling(final_path: Path) -> Path:
    """Return a unique same-parent quarantine path for ``final_path``."""

    final = Path(final_path)
    if not final.name:
        raise ValueError("final_path must name a directory.")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    token = uuid4().hex[:12]
    return final.with_name(
        f".{final.name}.quarantine-{stamp}-{os.getpid()}-{token}"
    )


def _require_publishable_parent(parent: Path) -> None:
    if not parent.exists():
        raise FileNotFoundError(f"Publication parent does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Publication parent is not a directory: {parent}")


def _refuse_existing(path: Path, *, role: str) -> None:
    if _lexists(path):
        raise FileExistsError(f"Refusing to overwrite existing {role} path: {path}")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)
