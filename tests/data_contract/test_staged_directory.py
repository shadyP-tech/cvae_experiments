from __future__ import annotations

from pathlib import Path

import pytest

from midogpp_thesis.common.staged_directory import (
    staged_directory,
    staged_existing_directory,
    staging_sibling,
)


def test_staged_directory_publishes_with_one_final_rename(tmp_path: Path) -> None:
    final = tmp_path / "bundle"

    with staged_directory(final) as stage:
        assert stage == staging_sibling(final)
        assert not final.exists()
        (stage / "manifest.json").write_text('{"status":"PASS"}\n', encoding="utf-8")

    assert (final / "manifest.json").is_file()
    assert not staging_sibling(final).exists()


def test_staged_directory_quarantines_failure_without_final_path(
    tmp_path: Path,
) -> None:
    final = tmp_path / "bundle"

    with pytest.raises(RuntimeError, match="injected"):
        with staged_directory(final) as stage:
            (stage / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("injected")

    assert not final.exists()
    assert not staging_sibling(final).exists()
    quarantines = tuple(tmp_path.glob(".bundle.quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "partial.bin").read_bytes() == b"partial"


def test_staged_existing_directory_carries_prepared_bytes_into_publication(
    tmp_path: Path,
) -> None:
    final = tmp_path / "bundle"
    final.mkdir()
    (final / "prepared.txt").write_text("prepared", encoding="utf-8")

    with staged_existing_directory(final) as stage:
        assert not final.exists()
        assert (stage / "prepared.txt").read_text(encoding="utf-8") == "prepared"
        (stage / "result.txt").write_text("complete", encoding="utf-8")

    assert (final / "result.txt").read_text(encoding="utf-8") == "complete"


def test_staged_existing_directory_quarantines_prepared_and_partial_bytes(
    tmp_path: Path,
) -> None:
    final = tmp_path / "bundle"
    final.mkdir()
    (final / "prepared.txt").write_text("prepared", encoding="utf-8")

    with pytest.raises(RuntimeError, match="injected"):
        with staged_existing_directory(final) as stage:
            (stage / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("injected")

    assert not final.exists()
    quarantines = tuple(tmp_path.glob(".bundle.quarantine-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "prepared.txt").is_file()
    assert (quarantines[0] / "partial.txt").is_file()
