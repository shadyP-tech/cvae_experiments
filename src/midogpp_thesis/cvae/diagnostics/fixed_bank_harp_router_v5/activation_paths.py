"""Lexical and resolved repository boundary for HARP v5 activation.

Activation handles authority-bearing paths.  Those paths are checked twice:
first as written (before any symlink resolution), and then after resolution.
This prevents a repository-local lexical name from escaping through an
intermediate symlink.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from ...protocol import ProtocolError


_PREDECESSOR_PATH_FRAGMENTS = (
    "fixed_bank_sceptre",
    "source_inner_candidate_utility",
    "fixed_bank_harp_router/v1",
    "fixed_bank_harp_router/v2",
    "fixed_bank_harp_router/v3",
    "fixed_bank_harp_router/v4",
    "harp_router_v1",
    "harp_router_v2",
    "harp_router_v3",
    "harp_router_v4",
    "harp_consumed_test_cache_v1",
    "harp_consumed_test_cache_v2",
    "harp_consumed_test_cache_v3",
    "harp_consumed_test_cache_v4",
)


@dataclass(frozen=True, slots=True)
class RepositoryBoundary:
    """A symlink-free lexical root paired with its resolved identity."""

    lexical_root: Path
    resolved_root: Path

    @classmethod
    def open(cls, value: str | Path) -> "RepositoryBoundary":
        lexical = _absolute_without_resolving(Path(value))
        _reject_symlink_components(lexical, label="repository")
        if not lexical.is_dir():
            raise ProtocolError("HARP v5 activation repository is absent or unsafe.")
        resolved = lexical.resolve(strict=True)
        if not resolved.is_dir() or resolved.is_symlink():
            raise ProtocolError("HARP v5 activation repository is absent or unsafe.")
        return cls(lexical_root=lexical, resolved_root=resolved)

    def path(
        self,
        value: str | Path,
        *,
        label: str,
        kind: str,
        reject_predecessor: bool = True,
    ) -> Path:
        """Validate one path without resolving it before lexical checks.

        ``kind`` is one of ``file``, ``directory``, ``absent``, ``optional``,
        or ``future``. Optional members may be absent but, when present, must
        be regular files. Future members must be absent and may have not-yet
        created repository-local parents.
        """

        raw = Path(value)
        if any(part == ".." for part in raw.parts):
            raise ProtocolError(f"HARP v5 {label} contains lexical traversal.")
        candidate = raw if raw.is_absolute() else self.lexical_root / raw
        candidate = _absolute_without_resolving(candidate)
        if not candidate.is_relative_to(self.lexical_root) or candidate == self.lexical_root:
            raise ProtocolError(f"HARP v5 {label} is outside the repository.")
        relative = candidate.relative_to(self.lexical_root).as_posix()
        if reject_predecessor:
            reject_predecessor_path(relative, label=label)
        _reject_symlink_components(candidate, label=label, start=self.lexical_root)
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.resolved_root) or resolved == self.resolved_root:
            raise ProtocolError(f"HARP v5 {label} resolves outside the repository.")

        exists = os.path.lexists(candidate)
        if kind == "file":
            if not exists or not candidate.is_file() or candidate.is_symlink():
                raise ProtocolError(f"HARP v5 {label} is absent or unsafe.")
        elif kind == "directory":
            if not exists or not candidate.is_dir() or candidate.is_symlink():
                raise ProtocolError(f"HARP v5 {label} is absent or unsafe.")
        elif kind == "absent":
            if exists:
                raise ProtocolError(f"HARP v5 {label} already exists or is unsafe.")
            self._require_safe_parent(candidate, label=label)
        elif kind == "optional":
            if exists and (not candidate.is_file() or candidate.is_symlink()):
                raise ProtocolError(f"HARP v5 {label} is unsafe.")
            if not exists:
                self._require_safe_parent(candidate, label=label)
        elif kind == "future":
            if exists:
                raise ProtocolError(f"HARP v5 {label} already exists or is unsafe.")
            self._require_safe_ancestor(candidate, label=label)
        else:
            raise ProtocolError("HARP v5 path validator received an unknown kind.")
        return resolved

    def member(self, relative: str | Path, *, label: str, kind: str) -> Path:
        raw = Path(relative)
        if raw.is_absolute():
            raise ProtocolError(f"HARP v5 {label} must be repository-relative.")
        return self.path(raw, label=label, kind=kind)

    def _require_safe_parent(self, candidate: Path, *, label: str) -> None:
        parent = candidate.parent
        if not parent.is_dir() or parent.is_symlink():
            raise ProtocolError(f"HARP v5 {label} parent is absent or unsafe.")
        resolved_parent = parent.resolve(strict=True)
        if not resolved_parent.is_relative_to(self.resolved_root):
            raise ProtocolError(f"HARP v5 {label} parent escapes the repository.")

    def _require_safe_ancestor(self, candidate: Path, *, label: str) -> None:
        ancestor = candidate.parent
        while not os.path.lexists(ancestor):
            if ancestor == self.lexical_root:
                break
            ancestor = ancestor.parent
        if not ancestor.is_dir() or ancestor.is_symlink():
            raise ProtocolError(f"HARP v5 {label} ancestor is absent or unsafe.")
        if not ancestor.resolve(strict=True).is_relative_to(self.resolved_root):
            raise ProtocolError(f"HARP v5 {label} ancestor escapes the repository.")


def reject_predecessor_path(value: str | Path, *, label: str) -> None:
    normalized = Path(value).as_posix().lower()
    if any(fragment in normalized for fragment in _PREDECESSOR_PATH_FRAGMENTS):
        raise ProtocolError(f"HARP v5 {label} references a predecessor path.")


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(
    path: Path,
    *,
    label: str,
    start: Path | None = None,
) -> None:
    """Reject every existing lexical symlink through ``path``."""

    absolute = _absolute_without_resolving(path)
    if start is None:
        current = Path(absolute.anchor)
        parts = absolute.parts[1:]
    else:
        root = _absolute_without_resolving(start)
        if not absolute.is_relative_to(root):
            raise ProtocolError(f"HARP v5 {label} is outside the repository.")
        current = root
        parts = absolute.relative_to(root).parts
        if current.is_symlink():
            raise ProtocolError(f"HARP v5 {label} crosses a lexical symlink.")
    for part in parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ProtocolError(f"HARP v5 {label} crosses a lexical symlink.")


__all__ = ("RepositoryBoundary", "reject_predecessor_path")
