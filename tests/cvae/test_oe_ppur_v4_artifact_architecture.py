from __future__ import annotations

import ast
from pathlib import Path

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4 import (
    artifact_index,
    completion_transaction,
    output_persistence,
)
from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4.artifact import (
    completion,
    contracts,
    schema,
)


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/midogpp_thesis/cvae/diagnostics/"
    "fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4"
)

ARTIFACT_LIFECYCLE_MODULES = {
    "artifact.completion",
    "artifact.contracts",
    "artifact.schema",
    "artifact.semantics",
    "artifact_index",
    "artifact_io",
    "complete_artifact_validation",
    "completion_transaction",
    "output_artifact",
    "output_persistence",
    "output_validation",
}


def _module_paths() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        if parts:
            result[".".join(parts)] = path
    return result


def _relative_import_target(module: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return None
    parent = module.split(".")[:-1]
    ascents = node.level - 1
    if ascents > len(parent):
        return None
    prefix = parent[: len(parent) - ascents] if ascents else parent
    suffix = tuple(part for part in (node.module or "").split(".") if part)
    return ".".join((*prefix, *suffix))


def _artifact_lifecycle_graph() -> dict[str, set[str]]:
    paths = _module_paths()
    graph = {module: set() for module in ARTIFACT_LIFECYCLE_MODULES}
    for module in graph:
        tree = ast.parse(paths[module].read_text(encoding="utf-8"), filename=module)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _relative_import_target(module, node)
            if target in graph:
                graph[module].add(target)
    return graph


def test_artifact_lifecycle_import_graph_is_acyclic() -> None:
    graph = _artifact_lifecycle_graph()
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visiting:
            cycle = " -> ".join((*visiting[visiting.index(module) :], module))
            raise AssertionError(f"artifact lifecycle import cycle: {cycle}")
        if module in visited:
            return
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)


def test_artifact_contract_and_schema_layers_do_not_import_writers() -> None:
    graph = _artifact_lifecycle_graph()
    assert graph["artifact.contracts"] == set()
    assert graph["artifact.schema"] == {"artifact.contracts"}
    assert "completion_transaction" not in graph["artifact.semantics"]
    assert "output_artifact" not in graph["completion_transaction"]
    assert graph["output_persistence"] == {"artifact.contracts"}


def test_legacy_facades_preserve_typed_contract_identity() -> None:
    assert (
        artifact_index.CompleteArtifactSealReceipt
        is contracts.CompleteArtifactSealReceipt
    )
    assert (
        completion_transaction.CompletionCommitReceipt
        is contracts.CompletionCommitReceipt
    )
    assert (
        completion_transaction.InterruptedCompletionReceipt
        is contracts.InterruptedCompletionReceipt
    )
    assert (
        completion_transaction.discover_completion_commit
        is completion.discover_completion_commit
    )
    assert artifact_index.build_complete_index_payload is schema.build_complete_index_payload
    assert (
        output_persistence.COMPLETE_CATALOG_MEMBERS
        is contracts.COMPLETE_CATALOG_MEMBERS
    )
    assert (
        output_persistence.COMPLETE_INTERNAL_MEMBERS
        is contracts.COMPLETE_INTERNAL_MEMBERS
    )
