from __future__ import annotations

from pathlib import Path
import ast

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v3.source_seal import (
    SOURCE_MEMBER_PATTERN,
    SOURCE_NAMESPACES,
    SOURCE_ROOT_ROLE,
    SOURCE_SNAPSHOT_SCHEMA,
    SOURCE_TREE_SCHEMA,
    build_source_snapshot_payload,
    source_snapshot_identity,
    validate_source_snapshot,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _fixture_roots(tmp_path: Path, *, amendment_hash: str = "a" * 64) -> dict[str, Path]:
    cvae = tmp_path / "midogpp_thesis/cvae"
    roots = {
        SOURCE_NAMESPACES[0]: cvae
        / "diagnostics/fixed_bank_sceptre_router_v3",
        SOURCE_NAMESPACES[1]: cvae / "diagnostics/sceptre_runtime",
        SOURCE_NAMESPACES[2]: cvae / "diagnostics/fixed_bank_sceptre_router",
        SOURCE_NAMESPACES[3]: cvae / "routing/sceptre",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    (roots[SOURCE_NAMESPACES[0]] / "experiment_contracts.py").write_text(
        "EXPECTED_EXECUTION_AMENDMENT_SHA256 = "
        f"{amendment_hash!r}\n",
        encoding="utf-8",
    )
    (roots[SOURCE_NAMESPACES[0]] / "runner.py").write_text(
        "RUNNER = True\n", encoding="utf-8"
    )
    (roots[SOURCE_NAMESPACES[1]] / "worker_lifecycle.py").write_text(
        "SPAWN = True\n", encoding="utf-8"
    )
    (roots[SOURCE_NAMESPACES[2]] / "model_freeze.py").write_text(
        "MODEL = 'frozen'\n", encoding="utf-8"
    )
    (roots[SOURCE_NAMESPACES[3]] / "ranking.py").write_text(
        "RANKING = 'source-only'\n", encoding="utf-8"
    )
    return roots


def test_production_seal_covers_all_four_namespace_prefixed_trees() -> None:
    payload = build_source_snapshot_payload()
    members = tuple(str(row["member"]) for row in payload["members"])

    assert payload["schema_version"] == SOURCE_SNAPSHOT_SCHEMA
    assert payload["source_root_role"] == SOURCE_ROOT_ROLE
    assert payload["member_pattern"] == SOURCE_MEMBER_PATTERN
    assert payload["source_namespaces"] == list(SOURCE_NAMESPACES)
    assert payload["source_namespace_count"] == 4
    assert payload["member_count"] == len(members) == len(set(members))
    assert tuple(sorted(members)) == members
    assert all(
        any(member.startswith(f"{namespace}/") for member in members)
        for namespace in SOURCE_NAMESPACES
    )
    assert (
        f"{SOURCE_NAMESPACES[0]}/runner.py" in members
        and f"{SOURCE_NAMESPACES[1]}/worker_lifecycle.py" in members
        and f"{SOURCE_NAMESPACES[2]}/model_freeze.py" in members
        and f"{SOURCE_NAMESPACES[3]}/ranking.py" in members
    )
    assert payload["normalized_external_anchor_members"] == {
        f"{SOURCE_NAMESPACES[0]}/experiment_contracts.py": [
            "EXPECTED_EXECUTION_AMENDMENT_SHA256"
        ]
    }


def test_injected_roots_are_deterministic_and_normalize_only_external_hash(
    tmp_path: Path,
) -> None:
    roots = _fixture_roots(tmp_path)
    first = build_source_snapshot_payload(namespace_roots=roots)
    inferred = build_source_snapshot_payload(package_root=roots[SOURCE_NAMESPACES[0]])
    assert inferred == first

    (roots[SOURCE_NAMESPACES[0]] / "experiment_contracts.py").write_text(
        f"EXPECTED_EXECUTION_AMENDMENT_SHA256 = {'b' * 64!r}\n",
        encoding="utf-8",
    )
    normalized = build_source_snapshot_payload(namespace_roots=roots)
    assert normalized == first

    (roots[SOURCE_NAMESPACES[3]] / "ranking.py").write_text(
        "RANKING = 'changed'\n", encoding="utf-8"
    )
    changed = build_source_snapshot_payload(namespace_roots=roots)
    assert changed["tree_sha256"] != first["tree_sha256"]
    assert changed["manifest_sha256"] != first["manifest_sha256"]


def test_optional_roots_require_exact_namespaces_and_reject_symlinks(
    tmp_path: Path,
) -> None:
    roots = _fixture_roots(tmp_path)
    incomplete = dict(roots)
    incomplete.pop(SOURCE_NAMESPACES[-1])
    with pytest.raises(ProtocolError, match="namespace inventory"):
        build_source_snapshot_payload(namespace_roots=incomplete)

    target = roots[SOURCE_NAMESPACES[-1]] / "target.txt"
    target.write_text("not Python\n", encoding="utf-8")
    (roots[SOURCE_NAMESPACES[-1]] / "unsafe.py").symlink_to(target)
    with pytest.raises(ProtocolError, match="symlink"):
        build_source_snapshot_payload(namespace_roots=roots)


def test_validation_recomputes_all_namespaces_from_optional_roots(
    tmp_path: Path,
) -> None:
    roots = _fixture_roots(tmp_path)
    identity = source_snapshot_identity(namespace_roots=roots)
    receipt = validate_source_snapshot(
        expected_manifest_sha256=identity["source_snapshot_manifest_sha256"],
        expected_tree_sha256=identity["source_snapshot_tree_sha256"],
        expected_member_count=identity["source_snapshot_member_count"],
        namespace_roots=roots,
    )
    assert receipt["status"] == "PASS"
    assert receipt["source_snapshot_schema"] == SOURCE_SNAPSHOT_SCHEMA

    (roots[SOURCE_NAMESPACES[2]] / "model_freeze.py").write_text(
        "MODEL = 'tampered'\n", encoding="utf-8"
    )
    with pytest.raises(ProtocolError, match="bytes or inventory"):
        validate_source_snapshot(
            expected_manifest_sha256=identity["source_snapshot_manifest_sha256"],
            expected_tree_sha256=identity["source_snapshot_tree_sha256"],
            expected_member_count=identity["source_snapshot_member_count"],
            namespace_roots=roots,
        )


def test_schema_constants_are_versioned_for_multi_namespace_seal() -> None:
    assert SOURCE_TREE_SCHEMA == "sceptre_v3_execution_source_tree_v1"
    assert SOURCE_SNAPSHOT_SCHEMA == "sceptre_v3_execution_source_snapshot_v1"
    assert SOURCE_ROOT_ROLE == (
        "sceptre_v3_executable_neutral_worker_runtime_and_inherited_"
        "scientific_python"
    )
    assert SOURCE_MEMBER_PATTERN == "|".join(
        f"{namespace}/**/*.py" for namespace in SOURCE_NAMESPACES
    )


def test_v3_source_inner_adapter_uses_only_public_reader_contract() -> None:
    repository = Path(__file__).resolve().parents[2]
    adapter = repository / (
        "src/midogpp_thesis/cvae/diagnostics/"
        "fixed_bank_sceptre_router_v3/source_inner_surfaces.py"
    )
    tree = ast.parse(adapter.read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert "load_authorized_source_inner_surfaces" in imported
    assert not any(name.startswith("_") for name in imported)


def test_v3_execution_package_never_imports_v2_execution_modules() -> None:
    package = Path(__file__).resolve().parents[2] / (
        "src/midogpp_thesis/cvae/diagnostics/fixed_bank_sceptre_router_v3"
    )
    forbidden = "fixed_bank_sceptre_router_v2"
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imported_modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(forbidden in module for module in imported_modules), source
