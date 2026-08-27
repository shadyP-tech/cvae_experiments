"""Closed-world source seal for the OE-PPUR v2 execution identity.

The seal covers the executable adapter and the neutral pairwise primitive-
utility core.  It deliberately rejects semantic imports from any sibling
diagnostic package so a predecessor run cannot become an undeclared seventh
input through Python source reuse.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import PACKAGE_NAME


_ADAPTER_RELATIVE_ROOT = (
    "src/midogpp_thesis/cvae/diagnostics/" + PACKAGE_NAME
)
_NEUTRAL_RELATIVE_ROOT = (
    "src/midogpp_thesis/cvae/routing/pairwise_primitive_utility"
)
_SHARED_PROTOCOL_RELATIVE_PATH = "src/midogpp_thesis/cvae/protocol.py"
_DIAGNOSTIC_IMPORT_PREFIX = "midogpp_thesis.cvae.diagnostics."
_CURRENT_IMPORT_PREFIX = _DIAGNOSTIC_IMPORT_PREFIX + PACKAGE_NAME
_PROTOCOL_IMPORT_PREFIX = "midogpp_thesis.cvae.protocol"
_NEUTRAL_IMPORT_PREFIX = "midogpp_thesis.cvae.routing.pairwise_primitive_utility"
_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class SourceContractReceipt:
    """Hash-only receipt for the exact executable source closure."""

    repository_root: str
    adapter_member_count: int
    adapter_tree_sha256: str
    neutral_member_count: int
    neutral_tree_sha256: str
    shared_protocol_sha256: str
    combined_source_sha256: str
    _factory_token: object = field(repr=False, compare=False)
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self._factory_token is not _FACTORY_TOKEN:
            raise ProtocolError("OE-PPUR v2 source receipt bypassed source admission.")
        root = Path(self.repository_root)
        if (
            not root.is_absolute()
            or root.is_symlink()
            or type(self.adapter_member_count) is not int
            or self.adapter_member_count <= 0
            or type(self.neutral_member_count) is not int
            or self.neutral_member_count <= 0
        ):
            raise ProtocolError("OE-PPUR v2 source receipt topology drifted.")
        for role in (
            "adapter_tree_sha256",
            "neutral_tree_sha256",
            "shared_protocol_sha256",
            "combined_source_sha256",
        ):
            object.__setattr__(
                self,
                role,
                require_sha256(getattr(self, role), role.replace("_", " ")),
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v2_source_contract_receipt_v2",
            "adapter_relative_root": _ADAPTER_RELATIVE_ROOT,
            "adapter_member_count": self.adapter_member_count,
            "adapter_tree_sha256": self.adapter_tree_sha256,
            "neutral_relative_root": _NEUTRAL_RELATIVE_ROOT,
            "neutral_member_count": self.neutral_member_count,
            "neutral_tree_sha256": self.neutral_tree_sha256,
            "shared_protocol_relative_path": _SHARED_PROTOCOL_RELATIVE_PATH,
            "shared_protocol_sha256": self.shared_protocol_sha256,
            "combined_member_count": (
                self.adapter_member_count + self.neutral_member_count + 1
            ),
            "combined_source_sha256": self.combined_source_sha256,
            "sibling_diagnostic_imports_present": False,
            "predecessor_semantic_runtime_imports_present": False,
            "project_local_import_allowlist_enforced": True,
            "unsealed_project_local_imports_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def build_source_contract_receipt(
    repository_root: str | Path | None = None,
) -> SourceContractReceipt:
    """Parse, hash, and seal the complete permitted Python source closure."""

    root = _repository_root(repository_root)
    adapter = _source_members(root / _ADAPTER_RELATIVE_ROOT, root=root)
    neutral = _source_members(root / _NEUTRAL_RELATIVE_ROOT, root=root)
    shared_protocol_sha256 = _source_file_sha256(
        root / _SHARED_PROTOCOL_RELATIVE_PATH
    )
    for path, _ in (*adapter, *neutral):
        _reject_unsealed_project_imports(root / path, repository_root=root)
    adapter_tree = _tree_hash(adapter, role="adapter")
    neutral_tree = _tree_hash(neutral, role="neutral_core")
    combined = canonical_hash(
        {
            "schema_version": "oe_ppur_v2_combined_source_tree_v2",
            "adapter_tree_sha256": adapter_tree,
            "adapter_member_count": len(adapter),
            "neutral_tree_sha256": neutral_tree,
            "neutral_member_count": len(neutral),
            "shared_protocol_sha256": shared_protocol_sha256,
        }
    )
    return SourceContractReceipt(
        repository_root=str(root),
        adapter_member_count=len(adapter),
        adapter_tree_sha256=adapter_tree,
        neutral_member_count=len(neutral),
        neutral_tree_sha256=neutral_tree,
        shared_protocol_sha256=shared_protocol_sha256,
        combined_source_sha256=combined,
        _factory_token=_FACTORY_TOKEN,
    )


def validate_source_contract_receipt(
    value: SourceContractReceipt,
    *,
    expected_source_contract_hash: str | None = None,
) -> SourceContractReceipt:
    """Recompute source bytes and require an exact receipt match."""

    if not isinstance(value, SourceContractReceipt):
        raise ProtocolError("OE-PPUR v2 source receipt is untyped.")
    rebuilt = build_source_contract_receipt(value.repository_root)
    if rebuilt.to_payload() != value.to_payload():
        raise ProtocolError("OE-PPUR v2 source contract drifted after sealing.")
    if expected_source_contract_hash is not None:
        expected = require_sha256(
            expected_source_contract_hash, "expected source contract hash"
        )
        if expected not in {rebuilt.combined_source_sha256, rebuilt.receipt_hash}:
            raise ProtocolError("OE-PPUR v2 authorized source hash drifted.")
    return rebuilt


def _repository_root(value: str | Path | None) -> Path:
    if value is not None:
        root = Path(value)
    else:
        root = next(
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
        raise ProtocolError("OE-PPUR v2 repository root is absent.") from exc
    if (
        not resolved.is_dir()
        or resolved.is_symlink()
        or not (resolved / _ADAPTER_RELATIVE_ROOT).is_dir()
        or not (resolved / _NEUTRAL_RELATIVE_ROOT).is_dir()
    ):
        raise ProtocolError("OE-PPUR v2 repository root drifted.")
    return resolved


def _source_members(directory: Path, *, root: Path) -> tuple[tuple[str, str], ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("OE-PPUR v2 source root is unsafe.")
    rows: list[tuple[str, str]] = []
    for path in sorted(directory.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("OE-PPUR v2 source member is unsafe.")
        relative = path.relative_to(root).as_posix()
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise ProtocolError("OE-PPUR v2 source member is unreadable.") from exc
        rows.append((relative, hashlib.sha256(payload).hexdigest()))
    if not rows:
        raise ProtocolError("OE-PPUR v2 source closure is empty.")
    return tuple(rows)


def _source_file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError("OE-PPUR v2 shared source member is unsafe.")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v2 shared source member is unreadable.") from exc
    return hashlib.sha256(payload).hexdigest()


def _tree_hash(rows: tuple[tuple[str, str], ...], *, role: str) -> str:
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v2_source_tree_v1",
            "role": role,
            "members": [
                {"path": path, "sha256": digest} for path, digest in rows
            ],
        }
    )


def _reject_unsealed_project_imports(
    path: Path,
    *,
    repository_root: Path,
) -> None:
    """Reject every statically imported project module outside the seal.

    Both absolute and relative imports are resolved to their full module name.
    The only project-local dependency outside the two sealed trees is the
    shared protocol-error module, which is deliberately narrow and label-free.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ProtocolError("OE-PPUR v2 source could not be parsed.") from exc
    _, package_name = _source_module_and_package(
        path,
        repository_root=repository_root,
    )
    for node in ast.walk(tree):
        names: tuple[str, ...] = ()
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = _resolve_import_from_names(
                node,
                package_name=package_name,
            )
        for name in names:
            if name == "midogpp_thesis" or name.startswith("midogpp_thesis."):
                if name == _PROTOCOL_IMPORT_PREFIX or any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in (
                        _CURRENT_IMPORT_PREFIX,
                        _NEUTRAL_IMPORT_PREFIX,
                    )
                ):
                    continue
                raise ProtocolError(
                    "OE-PPUR v2 source imports an unsealed project module."
                )


def _source_module_and_package(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[str, str]:
    try:
        relative = path.relative_to(repository_root / "src")
    except ValueError as exc:
        raise ProtocolError("OE-PPUR v2 source escaped the Python source root.") from exc
    parts = list(relative.with_suffix("").parts)
    if path.name == "__init__.py":
        parts.pop()
        module_parts = parts
        package_parts = parts
    else:
        module_parts = parts
        package_parts = parts[:-1]
    if not module_parts or not package_parts:
        raise ProtocolError("OE-PPUR v2 source module topology drifted.")
    return ".".join(module_parts), ".".join(package_parts)


def _resolve_import_from_names(
    node: ast.ImportFrom,
    *,
    package_name: str,
) -> tuple[str, ...]:
    if node.level == 0:
        return () if node.module is None else (node.module,)
    package_parts = package_name.split(".")
    retained = len(package_parts) - (node.level - 1)
    if retained <= 0:
        raise ProtocolError("OE-PPUR v2 relative source import escaped its package.")
    base_parts = package_parts[:retained]
    if node.module is not None:
        return (".".join((*base_parts, *node.module.split("."))),)
    return tuple(".".join((*base_parts, alias.name)) for alias in node.names)


__all__ = (
    "SourceContractReceipt",
    "build_source_contract_receipt",
    "validate_source_contract_receipt",
)
