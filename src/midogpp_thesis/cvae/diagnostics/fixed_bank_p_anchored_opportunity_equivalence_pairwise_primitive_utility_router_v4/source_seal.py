"""Recursive source seal for the executable OE-PPUR v4 scientific adapter."""

from __future__ import annotations

import ast
from dataclasses import InitVar, dataclass, field
import hashlib
from pathlib import Path

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import PACKAGE_NAME


_ADAPTER_RELATIVE_ROOT = "src/midogpp_thesis/cvae/diagnostics/" + PACKAGE_NAME
_PREPARATION_PREFIX = "midogpp_thesis.cvae.diagnostics.oe_ppur_v4_preparation"
_NEUTRAL_RELATIVE_ROOT = "src/midogpp_thesis/cvae/routing/pairwise_primitive_utility"
_SHARED_PROTOCOL_RELATIVE_PATH = "src/midogpp_thesis/cvae/protocol.py"
_PRODUCTION_RELATIVE_MEMBERS = (
    "src/midogpp_thesis/common/hashing.py",
    "src/midogpp_thesis/common/midogpp.py",
    "src/midogpp_thesis/cvae/block_frame.py",
    "src/midogpp_thesis/cvae/expert_bank/uniform_b_v2_promotion/contracts.py",
    "src/midogpp_thesis/cvae/expert_bank/uniform_b_v2_promotion/serialization.py",
    "src/midogpp_thesis/cvae/geco.py",
    "src/midogpp_thesis/cvae/generation/contracts.py",
    "src/midogpp_thesis/cvae/generation/generation.py",
    "src/midogpp_thesis/cvae/generation_samplers.py",
    "src/midogpp_thesis/cvae/keyed_training.py",
    "src/midogpp_thesis/cvae/latent_mixture_prior.py",
    "src/midogpp_thesis/cvae/latent_priors.py",
    "src/midogpp_thesis/cvae/models/__init__.py",
    "src/midogpp_thesis/cvae/models/cvae.py",
    "src/midogpp_thesis/cvae/models/learned_conditional_prior.py",
    "src/midogpp_thesis/cvae/models/mixture_prior.py",
    "src/midogpp_thesis/cvae/preservation/independent_source/__init__.py",
    "src/midogpp_thesis/cvae/preservation/independent_source/crossfit.py",
    "src/midogpp_thesis/cvae/preservation/independent_source/frame.py",
    "src/midogpp_thesis/cvae/preservation/splits.py",
    "src/midogpp_thesis/cvae/preservation/uniform_b_optimized_prior/config.py",
    "src/midogpp_thesis/cvae/preservation/uniform_b_optimized_prior/contracts.py",
    "src/midogpp_thesis/cvae/preservation/uniform_b_optimized_prior/core.py",
    "src/midogpp_thesis/cvae/preservation/uniform_b_task_geometry/frame.py",
    "src/midogpp_thesis/cvae/preservation/uniform_b_task_geometry/generation.py",
    "src/midogpp_thesis/cvae/runtime/artifact_io.py",
    "src/midogpp_thesis/cvae/runtime/fixed_bank_a1_prediction_contracts.py",
    "src/midogpp_thesis/cvae/runtime/fixed_bank_a1_prediction_planning.py",
    "src/midogpp_thesis/cvae/runtime/fixed_bank_a1_prediction_store.py",
    "src/midogpp_thesis/cvae/runtime/fixed_bank_a1_prediction_worker.py",
    "src/midogpp_thesis/cvae/runtime/frozen_source_streams.py",
    "src/midogpp_thesis/cvae/schedules.py",
    "src/midogpp_thesis/real_features/classifier_reference/artifacts/__init__.py",
    "src/midogpp_thesis/real_features/classifier_reference/classifiers.py",
    "src/midogpp_thesis/real_features/classifier_reference/protocol.py",
    "src/midogpp_thesis/real_features/classifier_reference/real_feature_frame.py",
    "src/midogpp_thesis/real_features/classifier_reference/schemas/__init__.py",
    "src/midogpp_thesis/real_features/classifier_reference/schemas/midogpp.py",
)
_CURRENT_PREFIX = "midogpp_thesis.cvae.diagnostics." + PACKAGE_NAME
_NEUTRAL_PREFIX = "midogpp_thesis.cvae.routing.pairwise_primitive_utility"
_PROTOCOL_MODULE = "midogpp_thesis.cvae.protocol"
_PRODUCTION_MODULES = frozenset(
    relative.removeprefix("src/")
    .removesuffix("/__init__.py")
    .removesuffix(".py")
    .replace("/", ".")
    for relative in _PRODUCTION_RELATIVE_MEMBERS
)
_PRODUCTION_IMPORT_SYMBOLS = frozenset(
    {
        "midogpp_thesis.real_features.classifier_reference.schemas.DIAGNOSTIC_ONLY",
        "midogpp_thesis.real_features.classifier_reference.schemas.SELECTION_ELIGIBLE",
    }
)
_FORBIDDEN_PREDECESSOR_FRAGMENTS = (
    "opportunity_equivalence_pairwise_primitive_utility_router_v2",
    "opportunity_equivalence_pairwise_primitive_utility_router_v3",
    "oe_ppur_v3_preparation",
)
_FACTORY_TOKEN = object()


@dataclass(frozen=True, slots=True)
class SourceSealReceipt:
    repository_root: str
    adapter_member_count: int
    adapter_tree_sha256: str
    neutral_member_count: int
    neutral_tree_sha256: str
    production_member_count: int
    production_tree_sha256: str
    shared_protocol_sha256: str
    combined_source_sha256: str
    _factory_token: InitVar[object | None] = None
    receipt_hash: str = field(init=False)

    def __post_init__(self, _factory_token: object | None) -> None:
        root = Path(self.repository_root)
        if (
            _factory_token is not _FACTORY_TOKEN
            or not root.is_absolute()
            or root.is_symlink()
            or self.adapter_member_count <= 0
            or self.neutral_member_count <= 0
            or self.production_member_count != len(_PRODUCTION_RELATIVE_MEMBERS)
        ):
            raise ProtocolError("OE-PPUR v4 source-seal topology drifted.")
        for role in (
            "adapter_tree_sha256",
            "neutral_tree_sha256",
            "production_tree_sha256",
            "shared_protocol_sha256",
            "combined_source_sha256",
        ):
            object.__setattr__(
                self, role, require_sha256(getattr(self, role), role.replace("_", " "))
            )
        object.__setattr__(self, "receipt_hash", canonical_hash(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": "oe_ppur_v4_source_seal_receipt_v1",
            "adapter_relative_root": _ADAPTER_RELATIVE_ROOT,
            "adapter_member_count": self.adapter_member_count,
            "adapter_tree_sha256": self.adapter_tree_sha256,
            "neutral_relative_root": _NEUTRAL_RELATIVE_ROOT,
            "neutral_member_count": self.neutral_member_count,
            "neutral_tree_sha256": self.neutral_tree_sha256,
            "production_members": list(_PRODUCTION_RELATIVE_MEMBERS),
            "production_member_count": self.production_member_count,
            "production_tree_sha256": self.production_tree_sha256,
            "shared_protocol_relative_path": _SHARED_PROTOCOL_RELATIVE_PATH,
            "shared_protocol_sha256": self.shared_protocol_sha256,
            "combined_source_sha256": self.combined_source_sha256,
            "predecessor_imports_present": False,
            "unsealed_project_imports_present": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {**self._payload(), "receipt_hash": self.receipt_hash}


def build_source_seal(repository_root: str | Path | None = None) -> SourceSealReceipt:
    root = _repository_root(repository_root)
    adapter = _source_members(root / _ADAPTER_RELATIVE_ROOT, root=root)
    neutral = _source_members(root / _NEUTRAL_RELATIVE_ROOT, root=root)
    production = _exact_source_members(root)
    for relative, _digest in (*adapter, *neutral):
        _validate_project_imports(root / relative, repository_root=root)
    adapter_tree = _tree_hash(adapter, role="v4_adapter")
    neutral_tree = _tree_hash(neutral, role="neutral_pairwise_core")
    production_tree = _tree_hash(production, role="fixed_bank_probability_runtime")
    protocol_hash = _file_hash(root / _SHARED_PROTOCOL_RELATIVE_PATH)
    combined = canonical_hash(
        {
            "schema_version": "oe_ppur_v4_combined_source_tree_v1",
            "adapter_tree_sha256": adapter_tree,
            "adapter_member_count": len(adapter),
            "neutral_tree_sha256": neutral_tree,
            "neutral_member_count": len(neutral),
            "production_tree_sha256": production_tree,
            "production_member_count": len(production),
            "shared_protocol_sha256": protocol_hash,
        }
    )
    return SourceSealReceipt(
        repository_root=root.as_posix(),
        adapter_member_count=len(adapter),
        adapter_tree_sha256=adapter_tree,
        neutral_member_count=len(neutral),
        neutral_tree_sha256=neutral_tree,
        production_member_count=len(production),
        production_tree_sha256=production_tree,
        shared_protocol_sha256=protocol_hash,
        combined_source_sha256=combined,
        _factory_token=_FACTORY_TOKEN,
    )


def validate_source_seal(
    value: object,
    *,
    expected_source_hash: str | None = None,
) -> SourceSealReceipt:
    if type(value) is not SourceSealReceipt:
        raise ProtocolError("OE-PPUR v4 source seal is untyped.")
    rebuilt = build_source_seal(value.repository_root)
    if rebuilt != value:
        raise ProtocolError("OE-PPUR v4 source bytes drifted after sealing.")
    if expected_source_hash is not None:
        expected = require_sha256(expected_source_hash, "expected source seal")
        if expected not in {rebuilt.combined_source_sha256, rebuilt.receipt_hash}:
            raise ProtocolError("OE-PPUR v4 expected source seal drifted.")
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
        raise ProtocolError("OE-PPUR v4 repository root is absent.") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise ProtocolError("OE-PPUR v4 repository topology drifted.")
    return resolved


def _source_members(directory: Path, *, root: Path) -> tuple[tuple[str, str], ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ProtocolError("OE-PPUR v4 source root is unsafe.")
    paths = tuple(sorted(directory.rglob("*.py")))
    if not paths:
        raise ProtocolError("OE-PPUR v4 source seal is empty.")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ProtocolError("OE-PPUR v4 source member is unsafe.")
    return tuple(
        (path.relative_to(root).as_posix(), _file_hash(path)) for path in paths
    )


def _exact_source_members(root: Path) -> tuple[tuple[str, str], ...]:
    if len(set(_PRODUCTION_RELATIVE_MEMBERS)) != len(_PRODUCTION_RELATIVE_MEMBERS):
        raise ProtocolError("OE-PPUR v4 production allowlist is duplicated.")
    rows = []
    for relative in _PRODUCTION_RELATIVE_MEMBERS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ProtocolError("OE-PPUR v4 production member is unsafe.")
        rows.append((relative, _file_hash(path)))
    return tuple(rows)


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ProtocolError("OE-PPUR v4 source member is unreadable.") from exc


def _tree_hash(rows: tuple[tuple[str, str], ...], *, role: str) -> str:
    return canonical_hash(
        {
            "schema_version": "oe_ppur_v4_source_tree_v1",
            "role": role,
            "members": [
                {"path": path, "sha256": digest} for path, digest in rows
            ],
        }
    )


def _validate_project_imports(path: Path, *, repository_root: Path) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ProtocolError("OE-PPUR v4 source could not be parsed.") from exc
    package = _module_package(path, repository_root=repository_root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = tuple(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names = _resolve_import_from(node, package_name=package)
        else:
            continue
        for name in names:
            if any(fragment in name for fragment in _FORBIDDEN_PREDECESSOR_FRAGMENTS):
                raise ProtocolError("OE-PPUR v4 imports predecessor runtime code.")
            if not (name == "midogpp_thesis" or name.startswith("midogpp_thesis.")):
                continue
            allowed_prefixes = (_CURRENT_PREFIX, _NEUTRAL_PREFIX, _PREPARATION_PREFIX)
            if (
                name == _PROTOCOL_MODULE
                or any(name == prefix or name.startswith(prefix + ".") for prefix in allowed_prefixes)
                or name in _PRODUCTION_MODULES
                or name in _PRODUCTION_IMPORT_SYMBOLS
            ):
                continue
            raise ProtocolError("OE-PPUR v4 source imports an unsealed project module.")


def _module_package(path: Path, *, repository_root: Path) -> str:
    try:
        parts = list(path.relative_to(repository_root / "src").with_suffix("").parts)
    except ValueError as exc:
        raise ProtocolError("OE-PPUR v4 source escaped the source root.") from exc
    if path.name == "__init__.py":
        parts.pop()
        package = parts
    else:
        package = parts[:-1]
    return ".".join(package)


def _resolve_import_from(node: ast.ImportFrom, *, package_name: str) -> tuple[str, ...]:
    if node.level == 0:
        return () if node.module is None else (node.module,)
    parts = package_name.split(".")
    retained = len(parts) - (node.level - 1)
    if retained <= 0:
        raise ProtocolError("OE-PPUR v4 relative import escaped its package.")
    base = parts[:retained]
    if node.module is not None:
        return (".".join((*base, *node.module.split("."))),)
    return tuple(".".join((*base, alias.name)) for alias in node.names)


__all__ = ("SourceSealReceipt", "build_source_seal", "validate_source_seal")
