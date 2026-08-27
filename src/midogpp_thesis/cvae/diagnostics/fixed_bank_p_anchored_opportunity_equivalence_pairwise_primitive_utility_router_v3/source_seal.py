"""Closed source seal for the OE-PPUR v3 adapter and neutral science core."""

from __future__ import annotations

import ast
from dataclasses import InitVar, dataclass, field
import hashlib
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import PACKAGE_NAME


_ADAPTER_RELATIVE_ROOT = "src/midogpp_thesis/cvae/diagnostics/" + PACKAGE_NAME
_NEUTRAL_RELATIVE_ROOT = "src/midogpp_thesis/cvae/routing/pairwise_primitive_utility"
_SHARED_PROTOCOL_RELATIVE_PATH = "src/midogpp_thesis/cvae/protocol.py"
_CURRENT_PREFIX = "midogpp_thesis.cvae.diagnostics." + PACKAGE_NAME
_NEUTRAL_PREFIX = "midogpp_thesis.cvae.routing.pairwise_primitive_utility"
_PROTOCOL_MODULE = "midogpp_thesis.cvae.protocol"
_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class SourceSealReceipt:
    repository_root: str
    adapter_member_count: int
    adapter_tree_sha256: str
    neutral_member_count: int
    neutral_tree_sha256: str
    shared_protocol_sha256: str
    combined_source_sha256: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        if _factory_token is not _FACTORY_TOKEN:
            raise ProtocolError("OE-PPUR v3 source seal bypassed source admission.")
        root = Path(self.repository_root)
        if (
            not root.is_absolute()
            or root.is_symlink()
            or type(self.adapter_member_count) is not int
            or self.adapter_member_count <= 0
            or type(self.neutral_member_count) is not int
            or self.neutral_member_count <= 0
        ):
            raise ProtocolError("OE-PPUR v3 source-seal topology drifted.")
        for role in (
            "adapter_tree_sha256",
            "neutral_tree_sha256",
            "shared_protocol_sha256",
            "combined_source_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v3_source_seal_receipt_v1",
            "adapter_relative_root": _ADAPTER_RELATIVE_ROOT,
            "adapter_member_count": self.adapter_member_count,
            "adapter_tree_sha256": self.adapter_tree_sha256,
            "neutral_relative_root": _NEUTRAL_RELATIVE_ROOT,
            "neutral_member_count": self.neutral_member_count,
            "neutral_tree_sha256": self.neutral_tree_sha256,
            "shared_protocol_relative_path": _SHARED_PROTOCOL_RELATIVE_PATH,
            "shared_protocol_sha256": self.shared_protocol_sha256,
            "combined_member_count": self.adapter_member_count
            + self.neutral_member_count
            + 1,
            "combined_source_sha256": self.combined_source_sha256,
            "predecessor_imports_present": False,
            "unsealed_project_imports_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def build_source_seal(
    repository_root: str | Path | None = None,
) -> SourceSealReceipt:
    root = _repository_root(repository_root)
    adapter = _source_members(root / _ADAPTER_RELATIVE_ROOT, root=root)
    neutral = _source_members(root / _NEUTRAL_RELATIVE_ROOT, root=root)
    for relative, _ in (*adapter, *neutral):
        _reject_unsealed_project_imports(root / relative, repository_root=root)
    shared_hash = _file_hash(root / _SHARED_PROTOCOL_RELATIVE_PATH)
    adapter_tree = _tree_hash(adapter, role="v3_adapter")
    neutral_tree = _tree_hash(neutral, role="neutral_pairwise_core")
    combined = canonical_hash(
        {
            "schema_version": "oe_ppur_v3_combined_source_tree_v1",
            "adapter_tree_sha256": adapter_tree,
            "adapter_member_count": len(adapter),
            "neutral_tree_sha256": neutral_tree,
            "neutral_member_count": len(neutral),
            "shared_protocol_sha256": shared_hash,
        }
    )
    return SourceSealReceipt(
        repository_root=str(root),
        adapter_member_count=len(adapter),
        adapter_tree_sha256=adapter_tree,
        neutral_member_count=len(neutral),
        neutral_tree_sha256=neutral_tree,
        shared_protocol_sha256=shared_hash,
        combined_source_sha256=combined,
        _factory_token=_FACTORY_TOKEN,
    )


def validate_source_seal(
    value: object,
    *,
    expected_source_hash: str | None = None,
) -> SourceSealReceipt:
    if type(value) is not SourceSealReceipt:
        raise ProtocolError("OE-PPUR v3 source seal is untyped.")
    rebuilt = build_source_seal(value.repository_root)
    if rebuilt != value:
        raise ProtocolError("OE-PPUR v3 source bytes drifted after sealing.")
    if expected_source_hash is not None:
        expected = require_sha256(expected_source_hash, "expected source seal hash")
        if expected not in {rebuilt.combined_source_sha256, rebuilt.receipt_hash}:
            raise ProtocolError("OE-PPUR v3 expected source seal drifted.")
    return rebuilt


def _repository_root(value: str | Path | None) -> Path:
    root = Path(value) if value is not None else next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "src/midogpp_thesis").is_dir()
        ),
        Path("."),
    )
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 repository root is absent.") from exc
    if (
        not resolved.is_dir()
        or resolved.is_symlink()
        or not (resolved / _ADAPTER_RELATIVE_ROOT).is_dir()
        or not (resolved / _NEUTRAL_RELATIVE_ROOT).is_dir()
    ):
        raise ProtocolError("OE-PPUR v3 repository topology drifted.")
    return resolved


def _source_members(directory: Path, *, root: Path) -> tuple[tuple[str, str], ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("OE-PPUR v3 source root is unsafe.")
    rows: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("OE-PPUR v3 source member is unsafe.")
        rows.append((path.relative_to(root).as_posix(), _file_hash(path)))
    if not rows:
        raise ProtocolError("OE-PPUR v3 source seal is empty.")
    return tuple(rows)


def _file_hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v3 source member is unsafe.")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v3 source member is unreadable.") from exc


def _tree_hash(rows: tuple[tuple[str, str], ...], *, role: str) -> str:
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v3_source_tree_v1",
            "role": role,
            "members": [
                {"path": path, "sha256": digest} for path, digest in rows
            ],
        }
    )


def _reject_unsealed_project_imports(path: Path, *, repository_root: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ProtocolError("OE-PPUR v3 source could not be parsed.") from exc
    package = _module_package(path, repository_root=repository_root)
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = _resolve_import_from(node, package_name=package)
        for name in names:
            if name == "midogpp_thesis" or name.startswith("midogpp_thesis."):
                if name == _PROTOCOL_MODULE or any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in (_CURRENT_PREFIX, _NEUTRAL_PREFIX)
                ):
                    continue
                raise ProtocolError(
                    "OE-PPUR v3 source imports an unsealed project module."
                )


def _module_package(path: Path, *, repository_root: Path) -> str:
    try:
        relative = path.relative_to(repository_root / "src").with_suffix("")
    except ValueError as exc:
        raise ProtocolError("OE-PPUR v3 source escaped the source root.") from exc
    parts = list(relative.parts)
    if path.name == "__init__.py":
        parts.pop()
        package = parts
    else:
        package = parts[:-1]
    if not package:
        raise ProtocolError("OE-PPUR v3 source module topology drifted.")
    return ".".join(package)


def _resolve_import_from(node: ast.ImportFrom, *, package_name: str) -> tuple[str, ...]:
    if node.level == 0:
        return () if node.module is None else (node.module,)
    package_parts = package_name.split(".")
    retained = len(package_parts) - (node.level - 1)
    if retained <= 0:
        raise ProtocolError("OE-PPUR v3 relative import escaped its package.")
    base = package_parts[:retained]
    if node.module is not None:
        return (".".join((*base, *node.module.split("."))),)
    return tuple(".".join((*base, alias.name)) for alias in node.names)


__all__ = ("SourceSealReceipt", "build_source_seal", "validate_source_seal")
