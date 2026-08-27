"""Closed-world, two-scope source provenance for planned OE-PPUR v1.

The diagnostic adapter and its stage-neutral scientific core are deliberately
sealed as separate source scopes. This prevents an apparently clean adapter
receipt from concealing a drifted core, while keeping the receipt plain and
pickle-safe for a future spawn-based workstation run.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath

from ...protocol import ProtocolError
from .hashing import canonical_hash, require_sha256
from .identity import PACKAGE_NAME


_DIAGNOSTICS_NAMESPACE = "midogpp_thesis.cvae.diagnostics"
_CURRENT_PREFIX = f"{_DIAGNOSTICS_NAMESPACE}.{PACKAGE_NAME}"
_ROUTING_NAMESPACE = "midogpp_thesis.cvae.routing"
_NEUTRAL_CORE_PREFIX = f"{_ROUTING_NAMESPACE}.pairwise_primitive_utility"
_SOURCE_SCOPE_ROLES = ("diagnostic_adapter", "neutral_scientific_core")

# The neutral core is a pure scientific library. Process, filesystem, network,
# serialization, and accelerator orchestration belong in the adapter only.
_CORE_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "concurrent",
        "http",
        "joblib",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "threading",
        "torch",
        "urllib",
        "yaml",
    }
)
_FORBIDDEN_MUTATION_CALLS = frozenset(
    {
        "chmod",
        "dump",
        "hardlink_to",
        "makedirs",
        "mkdir",
        "remove",
        "removedirs",
        "rename",
        "renames",
        "rmdir",
        "save",
        "savez",
        "savez_compressed",
        "symlink",
        "symlink_to",
        "to_csv",
        "to_feather",
        "to_json",
        "to_parquet",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    }
)
FORBIDDEN_PREDECESSOR_FRAGMENTS = (
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router_v2",
    "fixed_bank_p_anchored_support_calibrated_local_action_empirical_bayes_"
    "boundary_projected_router",
    "scale_bp_v2",
    "scale-bp-v2",
    "scale_bp_v1",
    "scale-bp-v1",
)


@dataclass(frozen=True, slots=True)
class SourceScopeReceipt:
    """Hash receipt for exactly one independently scanned source scope."""

    schema_version: str
    role: str
    member_count: int
    import_count: int
    literal_count: int
    mutation_call_count: int
    tree_sha256: str
    receipt_hash: str

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "role": self.role,
            "member_count": self.member_count,
            "import_count": self.import_count,
            "literal_count": self.literal_count,
            "mutation_call_count": self.mutation_call_count,
            "tree_sha256": self.tree_sha256,
            "receipt_hash": self.receipt_hash,
        }


@dataclass(frozen=True, slots=True)
class SourceFenceReceipt:
    """Spawn-safe combined provenance for the adapter and neutral core."""

    schema_version: str
    adapter: SourceScopeReceipt
    core: SourceScopeReceipt
    member_count: int
    import_count: int
    literal_count: int
    adapter_tree_sha256: str
    core_tree_sha256: str
    combined_source_seal_hash: str
    receipt_hash: str

    @property
    def tree_sha256(self) -> str:
        """Compatibility alias for callers that consumed the old receipt."""

        return self.combined_source_seal_hash

    @property
    def adapter_member_count(self) -> int:
        return self.adapter.member_count

    @property
    def core_member_count(self) -> int:
        return self.core.member_count

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "adapter": self.adapter.to_payload(),
            "core": self.core.to_payload(),
            "source_scopes_are_disjoint": True,
            "member_count": self.member_count,
            "import_count": self.import_count,
            "literal_count": self.literal_count,
            "adapter_tree_sha256": self.adapter_tree_sha256,
            "core_tree_sha256": self.core_tree_sha256,
            "combined_source_seal_hash": self.combined_source_seal_hash,
            "receipt_hash": self.receipt_hash,
        }


def package_source_root() -> Path:
    return Path(__file__).resolve().parent


def neutral_core_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "routing" / "pairwise_primitive_utility"


def assert_not_predecessor_reference(value: object, *, role: str) -> str:
    """Reject a lexical predecessor reference without resolving its path."""

    text = str(value)
    lowered = text.lower().replace("\\", "/")
    if any(fragment in lowered for fragment in FORBIDDEN_PREDECESSOR_FRAGMENTS):
        raise ProtocolError(f"OE-PPUR {role} references a forbidden predecessor.")
    return text


def build_source_fence_receipt(
    package_root: str | Path | None = None,
    *,
    core_root: str | Path | None = None,
) -> SourceFenceReceipt:
    """Build independent adapter/core seals after both AST fences pass."""

    adapter_path = package_source_root() if package_root is None else Path(package_root)
    core_path = neutral_core_source_root() if core_root is None else Path(core_root)
    _assert_disjoint_roots(adapter_path, core_path)
    adapter = _scan_scope(adapter_path, role="diagnostic_adapter")
    core = _scan_scope(core_path, role="neutral_scientific_core")
    seal_body = {
        "schema_version": "oe_ppur_v1_combined_source_seal_v2",
        "adapter_receipt_hash": adapter.receipt_hash,
        "adapter_tree_sha256": adapter.tree_sha256,
        "adapter_member_count": adapter.member_count,
        "core_receipt_hash": core.receipt_hash,
        "core_tree_sha256": core.tree_sha256,
        "core_member_count": core.member_count,
        "source_scopes_are_disjoint": True,
    }
    combined = canonical_hash(seal_body)
    receipt_body = {
        **seal_body,
        "combined_source_seal_hash": combined,
        "member_count": adapter.member_count + core.member_count,
        "import_count": adapter.import_count + core.import_count,
        "literal_count": adapter.literal_count + core.literal_count,
    }
    return SourceFenceReceipt(
        schema_version=str(seal_body["schema_version"]),
        adapter=adapter,
        core=core,
        member_count=int(receipt_body["member_count"]),
        import_count=int(receipt_body["import_count"]),
        literal_count=int(receipt_body["literal_count"]),
        adapter_tree_sha256=adapter.tree_sha256,
        core_tree_sha256=core.tree_sha256,
        combined_source_seal_hash=combined,
        receipt_hash=canonical_hash(receipt_body),
    )


def validate_source_fence(
    package_root: str | Path | None = None,
    *,
    core_root: str | Path | None = None,
    expected_adapter_tree_sha256: object | None = None,
    expected_core_tree_sha256: object | None = None,
    expected_combined_source_seal_hash: object | None = None,
) -> SourceFenceReceipt:
    """Validate both scopes and optional externally pinned source identities.

    The optional expected values are the non-recursive pinning hook: a frozen
    experiment config can bind these hashes without embedding self-referential
    constants in either scanned source tree.
    """

    receipt = validate_source_fence_receipt(
        build_source_fence_receipt(package_root, core_root=core_root)
    )
    checks = (
        (expected_adapter_tree_sha256, receipt.adapter_tree_sha256, "adapter tree"),
        (expected_core_tree_sha256, receipt.core_tree_sha256, "core tree"),
        (
            expected_combined_source_seal_hash,
            receipt.combined_source_seal_hash,
            "combined source seal",
        ),
    )
    for expected, observed, role in checks:
        if expected is None:
            continue
        if require_sha256(expected, role) != observed:
            raise ProtocolError(f"OE-PPUR {role} drifted.")
    return receipt


def validate_source_fence_receipt(
    receipt: SourceFenceReceipt,
) -> SourceFenceReceipt:
    """Revalidate a receipt after a spawn or persistence boundary."""

    if not isinstance(receipt, SourceFenceReceipt):
        raise ProtocolError("OE-PPUR combined source receipt type drifted.")
    adapter = receipt.adapter
    core = receipt.core
    if (
        adapter.role != "diagnostic_adapter"
        or core.role != "neutral_scientific_core"
        or adapter.schema_version != "oe_ppur_v1_source_scope_receipt_v2"
        or core.schema_version != "oe_ppur_v1_source_scope_receipt_v2"
        or adapter.member_count <= 0
        or core.member_count <= 0
        or adapter.mutation_call_count != 0
        or core.mutation_call_count != 0
    ):
        raise ProtocolError("OE-PPUR source scope receipt topology drifted.")
    for scope in (adapter, core):
        require_sha256(scope.tree_sha256, f"{scope.role} tree")
        expected_scope_hash = canonical_hash(
            {
                "schema_version": scope.schema_version,
                "role": scope.role,
                "member_count": scope.member_count,
                "import_count": scope.import_count,
                "literal_count": scope.literal_count,
                "mutation_call_count": scope.mutation_call_count,
                "tree_sha256": scope.tree_sha256,
            }
        )
        if (
            require_sha256(scope.receipt_hash, f"{scope.role} receipt")
            != expected_scope_hash
        ):
            raise ProtocolError("OE-PPUR source scope receipt hash drifted.")
    seal_body = {
        "schema_version": "oe_ppur_v1_combined_source_seal_v2",
        "adapter_receipt_hash": adapter.receipt_hash,
        "adapter_tree_sha256": adapter.tree_sha256,
        "adapter_member_count": adapter.member_count,
        "core_receipt_hash": core.receipt_hash,
        "core_tree_sha256": core.tree_sha256,
        "core_member_count": core.member_count,
        "source_scopes_are_disjoint": True,
    }
    combined = canonical_hash(seal_body)
    receipt_body = {
        **seal_body,
        "combined_source_seal_hash": combined,
        "member_count": adapter.member_count + core.member_count,
        "import_count": adapter.import_count + core.import_count,
        "literal_count": adapter.literal_count + core.literal_count,
    }
    if (
        receipt.schema_version != seal_body["schema_version"]
        or receipt.adapter_tree_sha256 != adapter.tree_sha256
        or receipt.core_tree_sha256 != core.tree_sha256
        or receipt.member_count != receipt_body["member_count"]
        or receipt.import_count != receipt_body["import_count"]
        or receipt.literal_count != receipt_body["literal_count"]
        or require_sha256(
            receipt.combined_source_seal_hash, "combined source seal"
        )
        != combined
        or require_sha256(receipt.receipt_hash, "combined source receipt")
        != canonical_hash(receipt_body)
    ):
        raise ProtocolError("OE-PPUR combined source receipt drifted.")
    return receipt


def _assert_disjoint_roots(adapter_root: Path, core_root: Path) -> None:
    try:
        adapter = adapter_root.resolve(strict=True)
        core = core_root.resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("OE-PPUR source scope root is absent.") from exc
    if adapter == core or adapter in core.parents or core in adapter.parents:
        raise ProtocolError("OE-PPUR adapter and core source scopes overlap.")


def _scan_scope(root: Path, *, role: str) -> SourceScopeReceipt:
    if role not in _SOURCE_SCOPE_ROLES:
        raise ProtocolError("OE-PPUR source scope role is unknown.")
    if root.is_symlink() or not root.is_dir():
        raise ProtocolError(f"OE-PPUR {role} source root is absent or unsafe.")
    members = tuple(
        sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    )
    if not members:
        raise ProtocolError(f"OE-PPUR {role} source fence found no Python members.")
    rows: list[tuple[str, str]] = []
    import_count = 0
    literal_count = 0
    mutation_call_count = 0
    for path in members:
        relative = path.relative_to(root).as_posix()
        _validate_member_name(relative, role)
        if path.is_symlink() or not path.is_file():
            raise ProtocolError(f"OE-PPUR {role} source member is unsafe.")
        try:
            payload = path.read_bytes()
            tree = ast.parse(payload.decode("utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise ProtocolError(f"OE-PPUR {role} source could not be parsed.") from exc
        rows.append((relative, hashlib.sha256(payload).hexdigest()))
        is_fence_implementation = path.resolve() == Path(__file__).resolve()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_count += 1
                    _validate_absolute_import(alias.name, path, scope_role=role)
            elif isinstance(node, ast.ImportFrom):
                import_count += 1
                _validate_from_import(node, path, root, scope_role=role)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                literal_count += 1
                if not (
                    is_fence_implementation
                    and node.value in FORBIDDEN_PREDECESSOR_FRAGMENTS
                ):
                    assert_not_predecessor_reference(node.value, role=path.name)
            elif isinstance(node, ast.Call) and _is_forbidden_mutation_call(node):
                mutation_call_count += 1
                raise ProtocolError(
                    f"OE-PPUR {role} source contains a forbidden mutation API in {path.name}."
                )
    tree_sha256 = canonical_hash(rows)
    body = {
        "schema_version": "oe_ppur_v1_source_scope_receipt_v2",
        "role": role,
        "member_count": len(members),
        "import_count": import_count,
        "literal_count": literal_count,
        "mutation_call_count": mutation_call_count,
        "tree_sha256": tree_sha256,
    }
    return SourceScopeReceipt(
        schema_version=str(body["schema_version"]),
        role=role,
        member_count=len(members),
        import_count=import_count,
        literal_count=literal_count,
        mutation_call_count=mutation_call_count,
        tree_sha256=tree_sha256,
        receipt_hash=canonical_hash(body),
    )


def _validate_member_name(value: str, role: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or path.as_posix() != value
        or any(part in {"", ".", "..", "__pycache__"} for part in path.parts)
        or path.suffix != ".py"
    ):
        raise ProtocolError(f"OE-PPUR {role} source member name is unsafe.")


def _validate_absolute_import(module: str, path: Path, *, scope_role: str) -> None:
    assert_not_predecessor_reference(module, role=f"import in {path.name}")
    if scope_role == "neutral_scientific_core":
        root_name = module.split(".", 1)[0]
        if root_name in _CORE_FORBIDDEN_IMPORT_ROOTS:
            raise ProtocolError(
                f"OE-PPUR neutral core imports forbidden runtime capability in {path.name}."
            )
        if module == _DIAGNOSTICS_NAMESPACE or module.startswith(
            f"{_DIAGNOSTICS_NAMESPACE}."
        ):
            raise ProtocolError(
                f"OE-PPUR neutral core imports a diagnostic adapter in {path.name}."
            )
        if (
            module == _ROUTING_NAMESPACE
            or module.startswith(f"{_ROUTING_NAMESPACE}.")
        ) and not (
            module == _NEUTRAL_CORE_PREFIX
            or module.startswith(f"{_NEUTRAL_CORE_PREFIX}.")
        ):
            raise ProtocolError(
                f"OE-PPUR neutral core imports another routing core in {path.name}."
            )
        return
    if (
        module == _DIAGNOSTICS_NAMESPACE
        or module.startswith(f"{_DIAGNOSTICS_NAMESPACE}.")
    ) and not (
        module == _CURRENT_PREFIX or module.startswith(f"{_CURRENT_PREFIX}.")
    ):
        raise ProtocolError(f"OE-PPUR source imports diagnostic sibling in {path.name}.")
    if (
        module == _ROUTING_NAMESPACE
        or module.startswith(f"{_ROUTING_NAMESPACE}.")
    ) and not (
        module == _NEUTRAL_CORE_PREFIX
        or module.startswith(f"{_NEUTRAL_CORE_PREFIX}.")
    ):
        raise ProtocolError(f"OE-PPUR source imports unapproved routing core in {path.name}.")


def _validate_from_import(
    node: ast.ImportFrom,
    path: Path,
    root: Path,
    *,
    scope_role: str,
) -> None:
    module = node.module or ""
    assert_not_predecessor_reference(module, role=f"import in {path.name}")
    if node.level == 0:
        _validate_absolute_import(module, path, scope_role=scope_role)
        return
    relative_parent_depth = len(path.relative_to(root).parent.parts)
    ascend = node.level - 1
    if ascend <= relative_parent_depth:
        return
    # Both source scopes may use the shared CVAE ProtocolError contract. The
    # number of dots grows with a member's nesting depth.
    if node.level == relative_parent_depth + 3 and module == "protocol":
        return
    if (
        scope_role == "diagnostic_adapter"
        and node.level == relative_parent_depth + 3
        and (
            module == "routing.pairwise_primitive_utility"
            or module.startswith("routing.pairwise_primitive_utility.")
        )
    ):
        return
    raise ProtocolError(
        f"OE-PPUR {scope_role} source escapes its allowed namespace in {path.name}."
    )


def _is_forbidden_mutation_call(node: ast.Call) -> bool:
    function = node.func
    name = (
        function.id
        if isinstance(function, ast.Name)
        else function.attr
        if isinstance(function, ast.Attribute)
        else ""
    )
    if name in _FORBIDDEN_MUTATION_CALLS:
        return True
    if name != "open":
        return False
    # A literal read-only open is permitted; dynamic modes fail closed.
    mode: object = "r"
    if len(node.args) >= 2:
        mode_node = node.args[1]
        mode = mode_node.value if isinstance(mode_node, ast.Constant) else None
    for keyword in node.keywords:
        if keyword.arg == "mode":
            mode = (
                keyword.value.value
                if isinstance(keyword.value, ast.Constant)
                else None
            )
    return mode not in {"r", "rb", "rt"}


__all__ = (
    "FORBIDDEN_PREDECESSOR_FRAGMENTS",
    "SourceFenceReceipt",
    "SourceScopeReceipt",
    "assert_not_predecessor_reference",
    "build_source_fence_receipt",
    "neutral_core_source_root",
    "package_source_root",
    "validate_source_fence",
    "validate_source_fence_receipt",
)
