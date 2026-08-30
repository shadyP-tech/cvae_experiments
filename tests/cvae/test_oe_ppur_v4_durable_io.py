from __future__ import annotations

import os
from pathlib import Path

import pytest

from midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v4 import durable_io
from midogpp_thesis.cvae.protocol import ProtocolError


def test_stable_read_refuses_symlink_leaf(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"sealed")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)

    with pytest.raises(ProtocolError, match="symlink"):
        durable_io.read_regular_bytes_nofollow(alias, role="test member")


def test_exclusive_write_never_overwrites_existing_member(tmp_path: Path) -> None:
    destination = tmp_path / "member.bin"
    durable_io.write_bytes_exclusive(
        destination,
        b"first",
        role="test member",
    )

    with pytest.raises(FileExistsError):
        durable_io.write_bytes_exclusive(
            destination,
            b"second",
            role="test member",
        )
    assert destination.read_bytes() == b"first"


def test_read_and_write_descriptors_are_opened_nofollow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        pytest.skip("host does not expose O_NOFOLLOW")
    observed: list[tuple[Path, int]] = []
    real_open = os.open

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        observed.append((Path(path), flags))
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(durable_io.os, "open", recording_open)
    destination = tmp_path / "nofollow.bin"
    durable_io.write_bytes_exclusive(
        destination,
        b"sealed",
        role="test member",
    )

    assert observed
    assert all(flags & nofollow for _path, flags in observed)
    assert any(path == destination for path, flags in observed if flags & os.O_CREAT)
    assert any(path == destination for path, flags in observed if not flags & os.O_CREAT)


def test_exclusive_write_fsyncs_exact_parent_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []
    real_fsync_directory = durable_io.fsync_directory

    def recording_fsync_directory(path: Path) -> None:
        observed.append(Path(path))
        real_fsync_directory(path)

    monkeypatch.setattr(durable_io, "fsync_directory", recording_fsync_directory)
    parent = tmp_path / "durable"
    parent.mkdir()
    durable_io.write_bytes_exclusive(
        parent / "member.bin",
        b"sealed",
        role="test member",
    )

    assert observed == [parent]


def test_exclusive_write_fails_closed_on_readback_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "member.bin"

    def drifted_readback(path: Path, *, role: str):
        del role
        return b"drifted", os.stat(path)

    monkeypatch.setattr(
        durable_io,
        "read_regular_bytes_nofollow",
        drifted_readback,
    )
    with pytest.raises(ProtocolError, match="read-back drifted"):
        durable_io.write_bytes_exclusive(
            destination,
            b"sealed",
            role="test member",
        )
    assert destination.read_bytes() == b"sealed"


def test_canonical_json_write_has_one_stable_file_representation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "receipt.json"
    durable_io.write_canonical_json_exclusive(
        destination,
        {"z": 1, "a": 2},
        role="test receipt",
    )

    assert destination.read_bytes() == b'{"a":2,"z":1}\n'
    assert durable_io.read_json_regular_nofollow(
        destination,
        role="test receipt",
    ) == {"a": 2, "z": 1}


def test_atomic_canonical_json_replace_is_exact_and_refuses_symlink(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "state.json"
    durable_io.write_canonical_json_exclusive(
        destination,
        {"phase": "ADMITTED"},
        role="test state",
    )
    durable_io.replace_canonical_json_atomic(
        destination,
        {"phase": "INPUTS_SEALED"},
        role="test state",
    )
    assert destination.read_bytes() == b'{"phase":"INPUTS_SEALED"}\n'
    assert not tuple(tmp_path.glob(f".{destination.name}.*"))

    referent = tmp_path / "referent.json"
    referent.write_bytes(b"unchanged")
    alias = tmp_path / "alias.json"
    alias.symlink_to(referent)
    with pytest.raises(ProtocolError, match="target is unsafe"):
        durable_io.replace_canonical_json_atomic(
            alias,
            {"phase": "COMPLETE"},
            role="test state",
        )
    assert referent.read_bytes() == b"unchanged"
