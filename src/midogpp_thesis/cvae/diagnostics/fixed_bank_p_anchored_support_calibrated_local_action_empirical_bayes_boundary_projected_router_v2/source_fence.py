"""Closed-world AST import and predecessor-input fence for SCALE-BP v2."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path

from .hashing import canonical_hash
from .identity import (
    CANONICAL_OUTPUT_RELATIVE_ROOT,
    CANONICAL_SCRATCH_ROOT,
    DIRECT_INPUT_ARTIFACT_IDS,
    GovernanceError,
    OUTPUT_ARTIFACT_ID,
    PACKAGE_NAME,
)


SOURCE_FENCE_SCHEMA = "scale_bp_v2_closed_world_source_fence_v1"
_PACKAGE_PREFIX = f"midogpp_thesis.cvae.diagnostics.{PACKAGE_NAME}"
_DIAGNOSTICS_PREFIX = "midogpp_thesis.cvae.diagnostics"
_CVAE_PREFIX = "midogpp_thesis.cvae"
_RUNTIME_PREFIX = "midogpp_thesis.cvae.runtime"
_MIDOGPP_PREFIX = "midogpp_thesis"


@dataclass(frozen=True, slots=True)
class SourceFenceReceipt:
    schema_version: str
    package_name: str
    member_count: int
    import_count: int
    runtime_import_count: int
    dynamic_import_count: int
    artifact_literal_count: int
    source_tree_hash: str
    receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "package_name": self.package_name,
            "member_count": self.member_count,
            "import_count": self.import_count,
            "runtime_import_count": self.runtime_import_count,
            "dynamic_import_count": self.dynamic_import_count,
            "artifact_literal_count": self.artifact_literal_count,
            "source_tree_hash": self.source_tree_hash,
            "sibling_diagnostic_import_count": 0,
            "predecessor_artifact_literal_count": 0,
            "receipt_hash": self.receipt_hash,
        }


def package_source_root() -> Path:
    return Path(__file__).resolve().parent


def validate_source_fence(package_root: str | Path | None = None) -> SourceFenceReceipt:
    """Reject all nonlocal thesis imports except neutral ``cvae.runtime``."""

    root = package_source_root() if package_root is None else Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise GovernanceError("SCALE-BP v2 source root is absent or unsafe.")
    members = tuple(
        sorted(
            (path for path in root.rglob("*.py") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not members or any(path.is_symlink() for path in members):
        raise GovernanceError("SCALE-BP v2 source fence found unsafe members.")

    import_count = 0
    runtime_import_count = 0
    dynamic_import_count = 0
    artifact_literal_count = 0
    tree_members: list[dict[str, object]] = []

    for path in members:
        try:
            source = path.read_bytes()
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise GovernanceError(
                f"SCALE-BP v2 source fence cannot parse {path.name}."
            ) from exc
        tree_members.append(
            {
                "member": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(source).hexdigest(),
                "size_bytes": len(source),
            }
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_count += 1
                    runtime_import_count += int(
                        _validate_import_target(alias.name, path)
                    )
            elif isinstance(node, ast.ImportFrom):
                import_count += 1
                target = _resolve_from_import(node, path, root)
                runtime_import_count += int(_validate_import_target(target, path))
            elif isinstance(node, ast.Call):
                dynamic_target = _literal_dynamic_import(node)
                if dynamic_target is not None:
                    dynamic_import_count += 1
                    runtime_import_count += int(
                        _validate_import_target(dynamic_target, path)
                    )
            elif (
                path.name != "source_fence.py"
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _looks_like_artifact_location(node.value)
            ):
                artifact_literal_count += 1
                _validate_artifact_literal(node.value, path)

    source_tree_hash = canonical_hash(
        {
            "schema_version": "scale_bp_v2_source_tree_for_fence_v1",
            "members": tree_members,
        }
    )
    body = {
        "schema_version": SOURCE_FENCE_SCHEMA,
        "package_name": PACKAGE_NAME,
        "member_count": len(members),
        "import_count": import_count,
        "runtime_import_count": runtime_import_count,
        "dynamic_import_count": dynamic_import_count,
        "artifact_literal_count": artifact_literal_count,
        "source_tree_hash": source_tree_hash,
        "sibling_diagnostic_import_count": 0,
        "predecessor_artifact_literal_count": 0,
    }
    return SourceFenceReceipt(
        schema_version=SOURCE_FENCE_SCHEMA,
        package_name=PACKAGE_NAME,
        member_count=len(members),
        import_count=import_count,
        runtime_import_count=runtime_import_count,
        dynamic_import_count=dynamic_import_count,
        artifact_literal_count=artifact_literal_count,
        source_tree_hash=source_tree_hash,
        receipt_hash=canonical_hash(body),
    )


def _resolve_from_import(node: ast.ImportFrom, path: Path, root: Path) -> str:
    module = node.module or ""
    if node.level == 0:
        return module
    relative = path.relative_to(root)
    package_parts = _PACKAGE_PREFIX.split(".") + list(relative.parent.parts)
    ascend = node.level - 1
    if ascend >= len(package_parts):
        raise GovernanceError(
            f"SCALE-BP v2 relative import escapes its namespace in {path.name}."
        )
    target_parts = package_parts[: len(package_parts) - ascend]
    if module:
        target_parts.extend(module.split("."))
    return ".".join(target_parts)


def _validate_import_target(module: str, path: Path) -> bool:
    if module == _PACKAGE_PREFIX or module.startswith(f"{_PACKAGE_PREFIX}."):
        return False
    if module == _RUNTIME_PREFIX or module.startswith(f"{_RUNTIME_PREFIX}."):
        return True
    if module == _DIAGNOSTICS_PREFIX or module.startswith(
        f"{_DIAGNOSTICS_PREFIX}."
    ):
        raise GovernanceError(
            f"SCALE-BP v2 imports sibling diagnostics in {path.name}: {module}."
        )
    if module == _CVAE_PREFIX or module.startswith(f"{_CVAE_PREFIX}."):
        raise GovernanceError(
            f"SCALE-BP v2 imports non-neutral cvae code in {path.name}: {module}."
        )
    if module == _MIDOGPP_PREFIX or module.startswith(f"{_MIDOGPP_PREFIX}."):
        raise GovernanceError(
            f"SCALE-BP v2 imports thesis code outside its closed world in "
            f"{path.name}: {module}."
        )
    return False


def _literal_dynamic_import(node: ast.Call) -> str | None:
    function = node.func
    is_import = isinstance(function, ast.Name) and function.id == "__import__"
    is_import_module = (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
    )
    if not (is_import or is_import_module) or not node.args:
        return None
    value = node.args[0]
    if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        raise GovernanceError("SCALE-BP v2 uses a nonliteral dynamic import.")
    if value.value.startswith("."):
        raise GovernanceError("SCALE-BP v2 uses a relative dynamic import.")
    return value.value


def _looks_like_artifact_location(value: str) -> bool:
    lowered = value.casefold()
    if lowered in {"artifact://", "output://", "/data/local/"}:
        return False
    return (
        "artifact://" in lowered
        or "output://" in lowered
        or "artifacts/midogpp/90_oracles_and_diagnostics/" in lowered
        or "/data/local/" in lowered
    )


def _validate_artifact_literal(value: str, path: Path) -> None:
    literal = value.strip()
    if literal.startswith("artifact://"):
        artifact_id = literal[len("artifact://") :].split("/", 1)[0]
        if artifact_id in DIRECT_INPUT_ARTIFACT_IDS:
            return
    elif literal.startswith("output://"):
        if literal == f"output://{OUTPUT_ARTIFACT_ID}":
            return
    elif literal == CANONICAL_SCRATCH_ROOT:
        return
    elif CANONICAL_OUTPUT_RELATIVE_ROOT in literal:
        return
    raise GovernanceError(
        f"SCALE-BP v2 references a forbidden artifact location in {path.name}."
    )


__all__ = (
    "SOURCE_FENCE_SCHEMA",
    "SourceFenceReceipt",
    "package_source_root",
    "validate_source_fence",
)
