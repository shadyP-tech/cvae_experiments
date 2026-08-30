from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.oe_ppur_v4_preparation.publish import (
    _fsync_directory,
    _write_exclusive,
)
from midogpp_thesis.oe_ppur_v4 import main


def test_exclusive_amendment_member_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "amendment.json"
    _write_exclusive(path, b"first\n")
    _fsync_directory(tmp_path)
    assert path.read_bytes() == b"first\n"
    with pytest.raises(FileExistsError):
        _write_exclusive(path, b"second\n")
    assert path.read_bytes() == b"first\n"


def test_run_command_remains_closed_without_launch_authority(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="outside the current authorization"):
        main(["run", "--repository-root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []
