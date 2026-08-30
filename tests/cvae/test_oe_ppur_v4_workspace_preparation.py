from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from midogpp_thesis.cvae.diagnostics.oe_ppur_v4_preparation import (
    ExecutionTopologyContract,
)
from midogpp_thesis.cvae.protocol import ProtocolError
from midogpp_thesis.cvae.diagnostics.oe_ppur_v4_preparation import host
from midogpp_thesis.oe_ppur_v4 import build_parser


def test_dedicated_preflight_and_authorize_commands_are_registered() -> None:
    parser = build_parser()
    preflight = parser.parse_args(
        ["preflight", "--repository-root", "/work/repo"]
    )
    authorize = parser.parse_args(
        [
            "authorize",
            "--repository-root",
            "/work/repo",
            "--preflight-receipt",
            "/tmp/preflight.json",
        ]
    )
    assert preflight.command == "preflight"
    assert authorize.command == "authorize"


def test_filesystem_probe_accepts_autofs_fronting_one_effective_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="autofs\nnfs4\n"),
    )
    assert host._filesystem_type(tmp_path) == "nfs4"


def test_filesystem_probe_rejects_multiple_effective_mount_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="autofs\nnfs4\next4\n"),
    )
    with pytest.raises(ProtocolError, match="filesystem topology is ambiguous"):
        host._filesystem_type(tmp_path)


def test_topology_binds_resolved_templates_and_rejects_scratch_overlap(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    output_parent = root / "artifacts/router"
    output = output_parent / "v4"
    helper = root / "publish.py"
    helper.write_text("# helper\n")
    with pytest.raises(ProtocolError, match="topology paths drifted"):
        ExecutionTopologyContract(
            host_id="workstation",
            mode="NFS_SAFE_IN_PLACE_COMMIT",
            repository_root=root,
            canonical_output_parent=output_parent,
            output_root=output,
            resolved_config_path=output / "config.resolved.yaml",
            input_manifest_path=output / "provenance/input_artifacts.json",
            envelope_path=output / "preparation/final_authorization_envelope.json",
            commit_marker_path=output / "COMMITTED",
            amendment_path=root / "contracts/v4/amendment.json",
            lease_path=output_parent / ".lease",
            scratch_root=root / "scratch",
            scratch_receipt_root=root / "scratch/receipts",
            topology_receipt_path=root / "scratch/receipts/topology.json",
            helper_path=helper,
            commit_protocol=(
                "EXCLUSIVE_FINAL_ROOT",
                "O_EXCL_MEMBERS",
                "COMMIT_MARKER_LAST",
            ),
        )
