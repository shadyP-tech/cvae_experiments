from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4.source_seal import (
    SOURCE_ROOT_PATTERNS,
    build_source_snapshot_payload,
    source_members,
    source_snapshot_identity,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def _fixture_repository(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    relative_roots = tuple(
        Path(pattern.split("/**/*.py", 1)[0]) for pattern in SOURCE_ROOT_PATTERNS
    )
    files = []
    for ordinal, relative_root in enumerate(relative_roots):
        root = tmp_path / "src" / relative_root
        root.mkdir(parents=True)
        source = root / f"member_{ordinal}.py"
        source.write_text(f"VALUE = {ordinal}\n", encoding="utf-8")
        files.append(source)
    return tmp_path, tuple(files)


def test_source_seal_covers_every_namespace_and_is_deterministic(
    tmp_path: Path,
) -> None:
    repository, files = _fixture_repository(tmp_path)
    first = dict(build_source_snapshot_payload(repository))
    second = dict(build_source_snapshot_payload(repository))

    assert first == second
    assert first["source_snapshot_member_count"] == len(SOURCE_ROOT_PATTERNS)
    assert tuple(source_members(repository)) == tuple(
        sorted(files, key=lambda path: path.relative_to(repository / "src").as_posix())
    )
    assert dict(source_snapshot_identity(repository)) == {
        key: first[key]
        for key in (
            "source_snapshot_schema",
            "source_snapshot_manifest_sha256",
            "source_snapshot_tree_sha256",
            "source_snapshot_member_count",
            "source_snapshot_member_pattern",
            "source_snapshot_excludes_bytecode_and_cache",
        )
    }

    files[-1].write_text("VALUE = 'changed'\n", encoding="utf-8")
    changed = dict(build_source_snapshot_payload(repository))
    assert changed["source_snapshot_manifest_sha256"] != first[
        "source_snapshot_manifest_sha256"
    ]
    assert changed["source_snapshot_tree_sha256"] != first[
        "source_snapshot_tree_sha256"
    ]


def test_source_seal_rejects_symlinks_and_empty_namespaces(tmp_path: Path) -> None:
    repository, files = _fixture_repository(tmp_path)
    link = files[0].parent / "unsafe.py"
    link.symlink_to(files[0])
    with pytest.raises(ProtocolError, match="symlink"):
        source_members(repository)
    link.unlink()

    files[-1].unlink()
    with pytest.raises(ProtocolError, match="namespace is empty"):
        source_members(repository)


def test_v4_owned_source_rejects_predecessor_executable_imports(
    tmp_path: Path,
) -> None:
    repository, files = _fixture_repository(tmp_path)
    files[0].write_text(
        "from midogpp_thesis.cvae.diagnostics."
        "fixed_bank_sceptre_router_v3 import runner\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="predecessor executable"):
        source_members(repository)
