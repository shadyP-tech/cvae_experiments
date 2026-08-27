"""AST import and predecessor-artifact fence for the SCALE-BP package."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from ...protocol import ProtocolError
from .identity import PACKAGE_NAME


_DIAGNOSTICS_PREFIX = "midogpp_thesis.cvae.diagnostics."
_CURRENT_PREFIX = f"{_DIAGNOSTICS_PREFIX}{PACKAGE_NAME}"
FORBIDDEN_ARTIFACT_PATH_FRAGMENTS = (
    "artifacts/midogpp/90_oracles_and_diagnostics/",
    "donor_crossfit_action_policy_surface_router",
    "directional_signed_utility_router",
    "crossfit_sample_influence_router",
    "boundary_projected_pcsi",
    "center_balanced_posterior_utility_prefix_router",
    "/pdcaps/",
    "/pdsur/",
    "/pcsi/",
    "/cbpupr/",
)


@dataclass(frozen=True)
class SourceFenceReceipt:
    schema_version: str
    member_count: int
    import_count: int
    artifact_path_literal_count: int


def package_source_root() -> Path:
    return Path(__file__).resolve().parent


def validate_source_fence(package_root: str | Path | None = None) -> SourceFenceReceipt:
    """Reject sibling diagnostic imports and predecessor Stage-90 paths."""

    root = package_source_root() if package_root is None else Path(package_root)
    members = tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))
    if not members:
        raise ProtocolError("SCALE-BP source fence found no Python members.")
    import_count = 0
    artifact_path_literal_count = 0
    for path in members:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise ProtocolError("SCALE-BP source fence could not parse source.") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_count += 1
                    _validate_absolute_import(alias.name, path)
            elif isinstance(node, ast.ImportFrom):
                import_count += 1
                _validate_from_import(node, path, root)
            elif (
                path.name != "source_fence.py"
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
            ):
                literal = node.value.lower()
                if _looks_like_artifact_path(literal):
                    artifact_path_literal_count += 1
                    if (
                        any(
                            fragment in literal
                            for fragment in FORBIDDEN_ARTIFACT_PATH_FRAGMENTS
                        )
                        and not _is_own_output_or_scratch_identity(literal)
                    ):
                        raise ProtocolError(
                            "SCALE-BP source references a forbidden predecessor "
                            f"artifact path in {path.name}."
                        )
    return SourceFenceReceipt(
        schema_version="scale_bp_v1_source_fence_receipt_v1",
        member_count=len(members),
        import_count=import_count,
        artifact_path_literal_count=artifact_path_literal_count,
    )


def _validate_absolute_import(module: str, path: Path) -> None:
    if module.startswith(_DIAGNOSTICS_PREFIX) and not (
        module == _CURRENT_PREFIX or module.startswith(f"{_CURRENT_PREFIX}.")
    ):
        raise ProtocolError(
            f"SCALE-BP source imports sibling diagnostics in {path.name}."
        )


def _validate_from_import(node: ast.ImportFrom, path: Path, root: Path) -> None:
    module = node.module or ""
    if node.level == 0:
        _validate_absolute_import(module, path)
        return
    relative_parent_depth = len(path.relative_to(root).parent.parts)
    ascend = node.level - 1
    if ascend <= relative_parent_depth:
        # The resolved target remains inside the current SCALE-BP package.
        return
    if node.level == 3 and module == "protocol" and relative_parent_depth == 0:
        # The one common exception is cvae.protocol.ProtocolError.
        return
    if node.level == 3 and (
        module == "runtime" or module.startswith("runtime.")
    ) and relative_parent_depth == 0:
        # Neutral cvae runtime mechanics carry no Stage-90 semantic identity.
        return
    if ascend == relative_parent_depth + 1:
        raise ProtocolError(
            f"SCALE-BP source imports sibling diagnostics in {path.name}."
        )
    raise ProtocolError(f"SCALE-BP source escapes its allowed namespace in {path.name}.")


def _looks_like_artifact_path(literal: str) -> bool:
    return (
        "artifact://" in literal
        or "output://" in literal
        or "artifacts/" in literal
        or "/90_oracles_and_diagnostics/" in literal
    )


def _is_own_output_or_scratch_identity(literal: str) -> bool:
    if PACKAGE_NAME not in literal:
        return False
    return (
        literal.endswith(f"{PACKAGE_NAME}/v1")
        or literal.endswith(f"{PACKAGE_NAME}_v1")
    )


__all__ = (
    "FORBIDDEN_ARTIFACT_PATH_FRAGMENTS",
    "SourceFenceReceipt",
    "package_source_root",
    "validate_source_fence",
)
