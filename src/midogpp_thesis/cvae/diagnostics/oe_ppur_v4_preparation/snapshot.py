"""Read-only workspace snapshotting for OE-PPUR v4 authorization."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess

from ...protocol import ProtocolError
from .contracts import ExecutionTopologyContract, ExcludedWorkspaceSurface
from .hashing import bytes_sha256, payload_sha256, require_sha256


@dataclass(frozen=True, slots=True)
class FileSeal:
    role: str
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if type(self.role) is not str or not self.role:
            raise ProtocolError("OE-PPUR v4 file-seal role is malformed.")
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or self.path != Path(os.path.normpath(self.path.as_posix()))
            or ".." in self.path.parts
        ):
            raise ProtocolError("OE-PPUR v4 file-seal path is not absolute.")
        object.__setattr__(self, "sha256", require_sha256(self.sha256, self.role))
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ProtocolError("OE-PPUR v4 file-seal size is malformed.")

    def to_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path.as_posix(),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceSealSpec:
    repository_root: Path
    sealed_allowlist: tuple[Path, ...]
    registry_path: Path
    catalog_path: Path
    config_path: Path
    helper_path: Path
    topology: ExecutionTopologyContract

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repository_root, Path)
            or not self.repository_root.is_absolute()
            or self.repository_root
            != Path(os.path.normpath(self.repository_root.as_posix()))
            or self.repository_root != self.topology.repository_root
            or self.helper_path != self.topology.helper_path
        ):
            raise ProtocolError("OE-PPUR v4 workspace-seal topology drifted.")
        allowlist = tuple(self.sealed_allowlist)
        if (
            type(self.sealed_allowlist) is not tuple
            or not allowlist
            or allowlist != tuple(sorted(set(allowlist), key=Path.as_posix))
            or not all(
                isinstance(path, Path)
                and path.is_absolute()
                and path == Path(os.path.normpath(path.as_posix()))
                and ".." not in path.parts
                for path in allowlist
            )
        ):
            raise ProtocolError("OE-PPUR v4 sealed allowlist is not canonical.")
        required_repository_files = (
            self.registry_path,
            self.catalog_path,
            self.config_path,
        )
        if (
            len(set((*required_repository_files, self.helper_path))) != 4
            or any(
                not isinstance(path, Path)
                or not path.is_absolute()
                or path != Path(os.path.normpath(path.as_posix()))
                or ".." in path.parts
                or not path.is_relative_to(self.repository_root)
                or path not in allowlist
                for path in required_repository_files
            )
        ):
            raise ProtocolError(
                "OE-PPUR v4 registry/catalog/config are not sealed allowlist members."
            )
        if self.helper_path.is_relative_to(self.repository_root) and (
            self.helper_path not in allowlist
        ):
            raise ProtocolError("OE-PPUR v4 in-repository helper is not allowlisted.")
        exclusions = self.topology.workspace_exclusions()
        if any(_path_in_surface(path, exclusions) for path in allowlist):
            raise ProtocolError("OE-PPUR v4 allowlist intersects an excluded surface.")


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    repository_root: Path
    git_head: str
    git_head_tree: str
    git_index: FileSeal
    repository_dirty: bool
    repository_status_sha256: str
    allowlist: tuple[FileSeal, ...]
    allowlist_sha256: str
    registry: FileSeal
    catalog: FileSeal
    config: FileSeal
    helper: FileSeal
    exclusions: tuple[ExcludedWorkspaceSurface, ...]
    snapshot_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repository_root, Path)
            or not self.repository_root.is_absolute()
            or self.repository_root
            != Path(os.path.normpath(self.repository_root.as_posix()))
            or type(self.git_head) is not str
            or len(self.git_head) not in {40, 64}
            or self.git_head != self.git_head.lower()
            or any(character not in "0123456789abcdef" for character in self.git_head)
            or type(self.git_head_tree) is not str
            or len(self.git_head_tree) not in {40, 64}
            or self.git_head_tree != self.git_head_tree.lower()
            or any(
                character not in "0123456789abcdef"
                for character in self.git_head_tree
            )
            or self.git_index.role != "git_index"
            or type(self.repository_dirty) is not bool
        ):
            raise ProtocolError("OE-PPUR v4 workspace revision is malformed.")
        object.__setattr__(
            self,
            "repository_status_sha256",
            require_sha256(self.repository_status_sha256, "repository status"),
        )
        object.__setattr__(
            self,
            "allowlist_sha256",
            require_sha256(self.allowlist_sha256, "allowlist"),
        )
        if (
            type(self.allowlist) is not tuple
            or not self.allowlist
            or tuple(row.path for row in self.allowlist)
            != tuple(sorted((row.path for row in self.allowlist), key=Path.as_posix))
            or len({row.path for row in self.allowlist}) != len(self.allowlist)
            or payload_sha256(
                {
                    "schema_version": "oe_ppur_v4_workspace_allowlist_v1",
                    "files": [row.to_payload() for row in self.allowlist],
                }
            )
            != self.allowlist_sha256
            or tuple(row.role for row in self.exclusions)
            != ("amendment", "output", "lease", "scratch_receipts")
        ):
            raise ProtocolError("OE-PPUR v4 workspace snapshot topology drifted.")
        by_path = {row.path: row for row in self.allowlist}
        for role, row in (
            ("registry", self.registry),
            ("catalog", self.catalog),
            ("config", self.config),
        ):
            allowlisted = by_path.get(row.path)
            if (
                row.role != role
                or allowlisted is None
                or (allowlisted.path, allowlisted.sha256, allowlisted.size_bytes)
                != (row.path, row.sha256, row.size_bytes)
            ):
                raise ProtocolError(f"OE-PPUR v4 exact {role} seal drifted.")
        if self.helper.role != "helper":
            raise ProtocolError("OE-PPUR v4 exact helper seal drifted.")
        object.__setattr__(self, "snapshot_hash", payload_sha256(self.to_payload()))

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_workspace_snapshot_v1",
            "repository_root": self.repository_root.as_posix(),
            "git_head": self.git_head,
            "git_head_tree": self.git_head_tree,
            "git_index": self.git_index.to_payload(),
            "repository_dirty": self.repository_dirty,
            "repository_status_sha256": self.repository_status_sha256,
            "sealed_allowlist": [row.to_payload() for row in self.allowlist],
            "sealed_allowlist_sha256": self.allowlist_sha256,
            "registry": self.registry.to_payload(),
            "catalog": self.catalog.to_payload(),
            "config": self.config.to_payload(),
            "helper": self.helper.to_payload(),
            "excluded_workspace_surfaces": [
                row.to_payload() for row in self.exclusions
            ],
            "filesystem_mutation_performed": False,
            "target_labels_opened": False,
        }


def capture_workspace_snapshot(spec: WorkspaceSealSpec) -> WorkspaceSnapshot:
    """Capture all authorized workspace bytes without creating any path."""

    if type(spec) is not WorkspaceSealSpec:
        raise ProtocolError("OE-PPUR v4 workspace seal spec is untyped.")
    root = spec.repository_root
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError("OE-PPUR v4 repository root is unavailable or unsafe.")
    allowlist = tuple(
        _seal_file(path, role=f"allowlist:{path.relative_to(root).as_posix()}")
        for path in spec.sealed_allowlist
    )
    allowlist_hash = payload_sha256(
        {
            "schema_version": "oe_ppur_v4_workspace_allowlist_v1",
            "files": [row.to_payload() for row in allowlist],
        }
    )
    by_path = {row.path: row for row in allowlist}
    registry = _rerole(by_path[spec.registry_path], "registry")
    catalog = _rerole(by_path[spec.catalog_path], "catalog")
    config = _rerole(by_path[spec.config_path], "config")
    helper = _seal_file(spec.helper_path, role="helper")
    head = _git(root, ("rev-parse", "HEAD")).decode("ascii").strip()
    head_tree = _git(root, ("rev-parse", "HEAD^{tree}")).decode("ascii").strip()
    index_raw_path = _git(root, ("rev-parse", "--git-path", "index")).decode(
        "utf-8"
    ).strip()
    index_path = Path(index_raw_path)
    if not index_path.is_absolute():
        index_path = root / index_path
    git_index = _seal_file(index_path, role="git_index")
    status_rows = _git_status_rows(
        root,
        spec.topology.workspace_exclusions(),
        allowlisted_paths=frozenset(spec.sealed_allowlist),
    )
    status_hash = payload_sha256(
        {
            "schema_version": "oe_ppur_v4_repository_status_v1",
            "rows": status_rows,
        }
    )
    return WorkspaceSnapshot(
        repository_root=root,
        git_head=head,
        git_head_tree=head_tree,
        git_index=git_index,
        repository_dirty=bool(status_rows),
        repository_status_sha256=status_hash,
        allowlist=allowlist,
        allowlist_sha256=allowlist_hash,
        registry=registry,
        catalog=catalog,
        config=config,
        helper=helper,
        exclusions=spec.topology.workspace_exclusions(),
    )


def validate_workspace_snapshot(
    expected: WorkspaceSnapshot,
    observed: WorkspaceSnapshot,
) -> WorkspaceSnapshot:
    if type(expected) is not WorkspaceSnapshot or type(observed) is not WorkspaceSnapshot:
        raise ProtocolError("OE-PPUR v4 workspace snapshot is untyped.")
    if observed != expected or observed.snapshot_hash != expected.snapshot_hash:
        raise ProtocolError("OE-PPUR v4 workspace snapshot drifted.")
    return observed


def _rerole(row: FileSeal, role: str) -> FileSeal:
    return FileSeal(role, row.path, row.sha256, row.size_bytes)


def _seal_file(path: Path, *, role: str) -> FileSeal:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path != Path(os.path.normpath(path.as_posix()))
        or ".." in path.parts
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} path is not absolute.")
    _assert_no_symlink_chain(path)
    try:
        metadata = path.stat()
        raw = path.read_bytes()
    except OSError as exc:
        raise ProtocolError(f"OE-PPUR v4 {role} could not be read.") from exc
    if not path.is_file() or path.is_symlink() or metadata.st_size != len(raw):
        raise ProtocolError(f"OE-PPUR v4 {role} is not a stable regular file.")
    after = path.stat()
    if (
        after.st_dev != metadata.st_dev
        or after.st_ino != metadata.st_ino
        or after.st_size != metadata.st_size
        or after.st_mtime_ns != metadata.st_mtime_ns
    ):
        raise ProtocolError(f"OE-PPUR v4 {role} changed while it was read.")
    return FileSeal(role, path, bytes_sha256(raw), len(raw))


def _assert_no_symlink_chain(path: Path) -> None:
    candidates = (path, *path.parents)
    if any(candidate.is_symlink() for candidate in candidates):
        raise ProtocolError("OE-PPUR v4 sealed path contains a symlink.")


def _git(root: Path, arguments: tuple[str, ...]) -> bytes:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ("git", *arguments),
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError("OE-PPUR v4 read-only Git snapshot failed.") from exc


def _git_status_rows(
    root: Path,
    exclusions: tuple[ExcludedWorkspaceSurface, ...],
    *,
    allowlisted_paths: frozenset[Path],
) -> list[dict[str, object]]:
    raw = _git(root, ("status", "--porcelain=v1", "-z", "--untracked-files=all"))
    tokens = raw.split(b"\0")
    rows: list[dict[str, object]] = []
    index = 0
    while index < len(tokens) and tokens[index]:
        token = tokens[index]
        index += 1
        if len(token) < 4 or token[2:3] != b" ":
            raise ProtocolError("OE-PPUR v4 Git status payload is malformed.")
        status = token[:2].decode("ascii")
        paths = [token[3:].decode("utf-8", errors="surrogateescape")]
        if "R" in status or "C" in status:
            if index >= len(tokens) or not tokens[index]:
                raise ProtocolError("OE-PPUR v4 Git rename status is malformed.")
            paths.append(tokens[index].decode("utf-8", errors="surrogateescape"))
            index += 1
        absolute_paths = tuple(root / Path(value) for value in paths)
        if all(_path_in_surface(path, exclusions) for path in absolute_paths):
            continue
        if not all(path in allowlisted_paths for path in absolute_paths):
            raise ProtocolError(
                "OE-PPUR v4 repository has unsealed non-excluded status bytes."
            )
        rows.append({"status": status, "paths": paths})
    return sorted(rows, key=lambda row: (str(row["status"]), tuple(row["paths"])))


def _path_in_surface(
    path: Path,
    exclusions: tuple[ExcludedWorkspaceSurface, ...],
) -> bool:
    return any(path == row.path or path.is_relative_to(row.path) for row in exclusions)


__all__ = (
    "FileSeal",
    "WorkspaceSealSpec",
    "WorkspaceSnapshot",
    "capture_workspace_snapshot",
    "validate_workspace_snapshot",
)
