"""Closed-world path validation for HARP v17 prepared inputs."""

from __future__ import annotations

from pathlib import Path

from ...protocol import ProtocolError


def safe_existing_member(root: Path, relative: str, *, role: str) -> Path:
    """Return one regular member without following any in-root symlink.

    Resolving first is insufficient because a symbolic link can resolve back to
    a regular file and hide that the declared inventory is reachable through a
    mutable alias.  Every declared component is therefore checked first.
    """

    value = Path(relative)
    if (
        not relative
        or value.is_absolute()
        or value.parts in {(), (".",)}
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise ProtocolError(f"HARP v17 {role} member path is unsafe.")
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError(f"HARP v17 {role} root is unsafe.")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError(f"HARP v17 {role} root is absent.") from exc

    candidate = resolved_root
    try:
        for part in value.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ProtocolError(
                    f"HARP v17 {role} member traverses a symbolic link."
                )
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except ProtocolError:
        raise
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"HARP v17 {role} member escaped or is absent.") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ProtocolError(f"HARP v17 {role} member is not a regular file.")
    return resolved


__all__ = ("safe_existing_member",)
