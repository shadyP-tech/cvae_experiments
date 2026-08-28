from __future__ import annotations

import os
from pathlib import Path
import stat

import numpy as np
import pytest

import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.preterminal_persistence as persistence
import midogpp_thesis.cvae.diagnostics.fixed_bank_p_anchored_opportunity_equivalence_pairwise_primitive_utility_router_v3.execution.fresh_attestation as fresh_attestation
from midogpp_thesis.cvae.protocol import ProtocolError


def _observe_fsync_kinds(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    observed: list[str] = []
    original = os.fsync

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        observed.append(
            "directory" if stat.S_ISDIR(mode) else "regular_file"
        )
        original(descriptor)

    monkeypatch.setattr(persistence.os, "fsync", recording_fsync)
    return observed


def test_exclusive_matrix_write_fsyncs_file_and_containing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "arrays"
    parent.mkdir()
    path = parent / "matrix.npy"
    observed = _observe_fsync_kinds(monkeypatch)

    persistence._write_npy_exclusive(
        path,
        np.asarray([[0.25, 0.75]], dtype="<f4"),
    )

    assert observed == ["regular_file", "directory"]
    with pytest.raises(FileExistsError):
        persistence._write_npy_exclusive(
            path,
            np.asarray([[0.5, 0.5]], dtype="<f4"),
        )


def test_exclusive_manifest_write_fsyncs_file_and_containing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "manifests"
    parent.mkdir()
    path = parent / "result.json"
    observed = _observe_fsync_kinds(monkeypatch)

    persistence._write_json_exclusive(path, {"sealed": True})

    assert observed == ["regular_file", "directory"]
    with pytest.raises(FileExistsError):
        persistence._write_json_exclusive(path, {"sealed": False})


def test_attestation_write_fsyncs_file_parent_and_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    (root / "reports").mkdir(parents=True)
    observed = _observe_fsync_kinds(monkeypatch)

    path = persistence.persist_attestation_json_exclusive(
        root,
        persistence.PRETERMINAL_ATTESTATION_MEMBER,
        {"schema_version": "test_attestation_v1", "labels_opened": False},
    )

    assert path == root / persistence.PRETERMINAL_ATTESTATION_MEMBER
    assert observed == ["regular_file", "directory", "directory"]
    with pytest.raises(ProtocolError, match="member identity drifted"):
        persistence.persist_attestation_json_exclusive(
            root,
            "reports/not_an_attestation.json",
            {},
        )


def test_preterminal_tree_barrier_fsyncs_both_files_parents_and_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    matrix = root / persistence.MATRIX_MEMBER
    manifest = root / persistence.MANIFEST_MEMBER
    matrix.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    matrix.write_bytes(b"matrix")
    manifest.write_bytes(b"manifest")
    observed = _observe_fsync_kinds(monkeypatch)

    persistence._fsync_preterminal_tree(
        root,
        matrix_path=matrix,
        manifest_path=manifest,
    )

    assert observed == [
        "regular_file",
        "directory",
        "regular_file",
        "directory",
        "directory",
    ]


def test_validator_runtime_hash_binds_both_split_implementation_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[Path] = []

    def fake_source_hash(path: Path) -> str:
        observed.append(path)
        return str(len(observed)) * 64

    monkeypatch.setattr(
        fresh_attestation,
        "_sha256_regular_file",
        fake_source_hash,
    )

    runtime_hash = fresh_attestation._validator_runtime_sha256()

    assert len(runtime_hash) == 64
    assert [path.name for path in observed] == [
        "fresh_attestation.py",
        "preterminal_persistence.py",
    ]
