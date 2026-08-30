from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v5 import source_seal
from midogpp_thesis.cvae.protocol import ProtocolError


def test_v5_source_inventory_fences_every_executable_predecessor() -> None:
    assert source_seal.FORBIDDEN_EXECUTABLE_IMPORT_FRAGMENTS == (
        "fixed_bank_sceptre_router_v1",
        "fixed_bank_sceptre_router_v2",
        "fixed_bank_sceptre_router_v3",
        "fixed_bank_sceptre_router_v4",
    )
    members = source_seal.source_members()
    assert any("fixed_bank_sceptre_router_v5" in path.parts for path in members)
    assert not any(
        fragment in path.as_posix()
        for path in members
        for fragment in source_seal.FORBIDDEN_EXECUTABLE_IMPORT_FRAGMENTS
    )


def test_v5_source_fence_rejects_a_predecessor_closure_member(
    tmp_path: Path,
) -> None:
    predecessor = tmp_path / "fixed_bank_sceptre_router_v4" / "runner.py"
    predecessor.parent.mkdir(parents=True)
    predecessor.write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="closure contains predecessor"):
        source_seal._validate_predecessor_import_fence((predecessor,))


def test_v5_source_fence_rejects_a_v4_executable_import(tmp_path: Path) -> None:
    member = tmp_path / "fixed_bank_sceptre_router_v5" / "forbidden.py"
    member.parent.mkdir(parents=True)
    member.write_text(
        "from midogpp_thesis.cvae.diagnostics.fixed_bank_sceptre_router_v4 "
        "import runner\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolError, match="imports predecessor executable code"):
        source_seal._validate_predecessor_import_fence((member,))
