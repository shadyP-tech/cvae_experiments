"""Scoped source snapshot for HARP v20 execution.

The v20 scientific roots are followed transitively and must be predecessor
free.  The shared CLI/workspace dispatch modules are a deliberate exception:
their bytes are sealed, but their imports are not traversed because those
modules also contain the closed dispatch branches of unrelated experiments.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash


SOURCE_SNAPSHOT_SCHEMA = "midogpp_harp_stage90_source_snapshot_v20"
SOURCE_TREE_SCHEMA = "midogpp_harp_stage90_source_tree_v20"
SOURCE_CLOSURE_DESCRIPTION = (
    "v20_scientific_roots_transitive_python_imports_to_shared_dispatch_boundary"
)
SOURCE_DISPATCH_LEAF_POLICY = (
    "shared_dispatch_entrypoints_byte_sealed_without_import_traversal"
)
SOURCE_ROOT_PATTERNS = (
    "midogpp_thesis/cvae/diagnostics/fixed_bank_harp_router_v20/**/*.py",
    "midogpp_thesis/cvae/diagnostics/harp_v20_cli.py",
)
# These launch/admission files import every registered experiment through
# closed dispatch branches.  Seal their bytes as execution leaves, but do not
# traverse unrelated experiment branches into this v20 scientific closure.
SOURCE_ENTRYPOINT_PATTERNS = (
    "midogpp_thesis/__main__.py",
    "midogpp_thesis/cli.py",
    "midogpp_thesis/cvae/diagnostics/cli.py",
    "midogpp_thesis/workspace/preparation_authority.py",
    "midogpp_thesis/workspace/runtime.py",
    "midogpp_thesis/workspace/cli.py",
)

# A successor source seal must never make an exhausted predecessor part of its
# executable closure.  Shared, identity-neutral modules are allowed; every
# versioned HARP lifecycle, runtime, and routing module below is not.
FORBIDDEN_PREDECESSOR_MODULE_PREFIXES = tuple(
    prefix
    for version in (
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
    )
    for prefix in (
        "midogpp_thesis.cvae.diagnostics.fixed_bank_harp_router_" + version,
        "midogpp_thesis.cvae.runtime.harp_" + version + "_execution",
        "midogpp_thesis.cvae.routing.harp_" + version,
    )
) + (
    # The v7-v17 science packages use architecture names outside the generic
    # ``harp_vN`` convention. Name them explicitly so the v20 closure can never
    # execute or seal an exhausted predecessor router.
    "midogpp_thesis.cvae.routing.source_active_selective_router_v7",
    "midogpp_thesis.cvae.routing.baseline_inclusive_action_safe_router_v8",
    "midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v9",
    "midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v10",
    "midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v11",
    "midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v12",
    "midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v13",
    "midogpp_thesis.cvae.routing.policy_calibrated_residual_router_v14",
    "midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v15",
    "midogpp_thesis.cvae.routing.hierarchical_support_action_risk_router_v16",
    "midogpp_thesis.cvae.routing.pooled_pairwise_selected_policy_router_v17",
    "midogpp_thesis.cvae.routing.case_conditional_composite_router_v18",
    "midogpp_thesis.cvae.routing.safe_winner_router_v19",
    "midogpp_thesis.cvae.routing.dense_residual_soft_router",
    "midogpp_thesis.cvae.routing.compatibility_conditioned_directional_router",
    "midogpp_thesis.cvae.runtime.harp_probability_menu.routing",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def source_members(root: Path | None = None) -> tuple[Path, ...]:
    repository = repository_root() if root is None else Path(root).resolve()
    source_root = repository / "src"
    if source_root.is_symlink() or not source_root.is_dir():
        raise ProtocolError("HARP v20 source root is absent or unsafe.")
    roots: set[Path] = set()
    for pattern in SOURCE_ROOT_PATTERNS:
        candidates = tuple(source_root.glob(pattern))
        if any(path.is_symlink() for path in candidates):
            raise ProtocolError("HARP v20 source inventory contains a symlink.")
        members = tuple(
            path for path in candidates
            if path.is_file() and path.suffix == ".py" and "__pycache__" not in path.parts
        )
        if not members:
            raise ProtocolError(f"HARP v20 sealed source root is empty: {pattern}.")
        roots.update(members)
    entrypoints: set[Path] = set()
    for pattern in SOURCE_ENTRYPOINT_PATTERNS:
        candidates = tuple(source_root.glob(pattern))
        if any(path.is_symlink() for path in candidates):
            raise ProtocolError("HARP v20 execution entrypoint inventory contains a symlink.")
        members = tuple(
            path
            for path in candidates
            if path.is_file() and path.suffix == ".py" and "__pycache__" not in path.parts
        )
        if not members:
            raise ProtocolError(f"HARP v20 sealed execution entrypoint is empty: {pattern}.")
        entrypoints.update(members)
    closure = _transitive_local_import_closure(source_root, roots, leaves=entrypoints)
    if not closure:
        raise ProtocolError("HARP v20 source inventory is empty.")
    return tuple(
        sorted(
            closure | entrypoints,
            key=lambda path: path.relative_to(source_root).as_posix(),
        )
    )


def build_source_snapshot_payload(root: Path | None = None) -> Mapping[str, object]:
    repository = repository_root() if root is None else Path(root).resolve()
    source_root = repository / "src"
    rows = tuple(
        {"relative_path": path.relative_to(source_root).as_posix(),
         "sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
        for path in source_members(repository)
    )
    manifest_sha256 = canonical_hash(
        {"schema_version": SOURCE_SNAPSHOT_SCHEMA, "members": list(rows)}
    )
    tree_sha256 = canonical_hash(
        {"schema_version": SOURCE_TREE_SCHEMA,
         "member_sha256": [[row["relative_path"], row["sha256"]] for row in rows]}
    )
    return MappingProxyType({
        "source_snapshot_schema": SOURCE_SNAPSHOT_SCHEMA,
        "source_snapshot_manifest_sha256": manifest_sha256,
        "source_snapshot_tree_sha256": tree_sha256,
        "source_snapshot_member_count": len(rows),
        "source_snapshot_member_pattern": (
            "roots="
            + "|".join(SOURCE_ROOT_PATTERNS)
            + ";entrypoint_leaves="
            + "|".join(SOURCE_ENTRYPOINT_PATTERNS)
            + f";scientific_closure={SOURCE_CLOSURE_DESCRIPTION}"
            + f";dispatch_leaf_policy={SOURCE_DISPATCH_LEAF_POLICY}"
        ),
        "source_snapshot_excludes_bytecode_and_cache": True,
        "v20_scientific_import_closure_to_dispatch_boundary_sealed": True,
        "v20_scientific_import_closure_to_dispatch_boundary_predecessor_free": True,
        "shared_dispatch_entrypoint_bytes_sealed": True,
        "shared_dispatch_entrypoint_imports_traversed": False,
        "shared_dispatch_entrypoints_outside_predecessor_free_closure": True,
        "full_dispatch_transitive_local_import_closure_claimed": False,
        "members": list(rows),
    })


def source_snapshot_identity(root: Path | None = None) -> Mapping[str, object]:
    payload = build_source_snapshot_payload(root)
    return MappingProxyType({
        key: payload[key] for key in (
            "source_snapshot_schema", "source_snapshot_manifest_sha256",
            "source_snapshot_tree_sha256", "source_snapshot_member_count",
            "source_snapshot_member_pattern", "source_snapshot_excludes_bytecode_and_cache",
            "v20_scientific_import_closure_to_dispatch_boundary_sealed",
            "v20_scientific_import_closure_to_dispatch_boundary_predecessor_free",
            "shared_dispatch_entrypoint_bytes_sealed",
            "shared_dispatch_entrypoint_imports_traversed",
            "shared_dispatch_entrypoints_outside_predecessor_free_closure",
            "full_dispatch_transitive_local_import_closure_claimed",
        )
    })


def validate_source_snapshot(expected: Mapping[str, object], root: Path | None = None) -> Mapping[str, object]:
    observed = source_snapshot_identity(root)
    if dict(expected) != dict(observed):
        raise ProtocolError("HARP v20 source snapshot drifted.")
    return observed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProtocolError("Cannot hash HARP v20 source member.") from exc
    return digest.hexdigest()


def _transitive_local_import_closure(
    source_root: Path, roots: set[Path], *, leaves: set[Path] | None = None
) -> set[Path]:
    members = set(roots)
    pending = list(roots)
    entrypoint_leaves = set() if leaves is None else set(leaves)
    while pending:
        path = pending.pop()
        for module in _imported_modules(path, source_root):
            _reject_predecessor_import(module)
            for candidate in _module_members(module, source_root):
                if candidate.is_symlink():
                    raise ProtocolError("HARP v20 source import closure contains a symlink.")
                if candidate not in members:
                    members.add(candidate)
                    if candidate not in entrypoint_leaves:
                        pending.append(candidate)
    return members


def _reject_predecessor_import(module: str) -> None:
    if any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in FORBIDDEN_PREDECESSOR_MODULE_PREFIXES
    ):
        raise ProtocolError(
            "HARP v20 source closure imports an exhausted predecessor module."
        )


def _imported_modules(path: Path, source_root: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise ProtocolError("Cannot inspect HARP v20 source imports.") from exc
    relative = path.relative_to(source_root).with_suffix("")
    package = relative.parts[:-1]
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                drop = node.level - 1
                if drop > len(package):
                    continue
                anchor = package[: len(package) - drop]
                suffix = tuple((node.module or "").split(".")) if node.module else ()
                base = ".".join((*anchor, *suffix))
            else:
                base = node.module or ""
            if base:
                modules.add(base)
            for alias in node.names:
                if base and alias.name != "*":
                    modules.add(f"{base}.{alias.name}")
    return tuple(sorted(modules))


def _module_members(module: str, source_root: Path) -> tuple[Path, ...]:
    if not module.startswith("midogpp_thesis"):
        return ()
    parts = tuple(part for part in module.split(".") if part)
    module_path = source_root.joinpath(*parts)
    candidates: set[Path] = set()
    for member in (module_path.with_suffix(".py"), module_path / "__init__.py"):
        if member.is_file():
            candidates.add(member)
    for depth in range(1, len(parts) + 1):
        package_init = source_root.joinpath(*parts[:depth]) / "__init__.py"
        if package_init.is_file():
            candidates.add(package_init)
    return tuple(sorted(candidates, key=lambda path: path.relative_to(source_root).as_posix()))


__all__ = (
    "FORBIDDEN_PREDECESSOR_MODULE_PREFIXES",
    "SOURCE_CLOSURE_DESCRIPTION", "SOURCE_DISPATCH_LEAF_POLICY",
    "SOURCE_ENTRYPOINT_PATTERNS",
    "SOURCE_ROOT_PATTERNS", "SOURCE_SNAPSHOT_SCHEMA",
    "build_source_snapshot_payload", "repository_root", "source_members",
    "source_snapshot_identity", "validate_source_snapshot",
)
