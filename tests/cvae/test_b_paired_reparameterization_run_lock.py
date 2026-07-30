from __future__ import annotations

import json
import os
from pathlib import Path
import socket

import pytest

from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit.run_lock import (
    LOCK_SCHEMA,
    exclusive_artifact_lock,
)
from midogpp_thesis.cvae.protocol import ProtocolError


def test_run_lock_rejects_live_local_owner(tmp_path: Path) -> None:
    lock = tmp_path / ".run.lock"
    lock.write_text(
        json.dumps(
            {
                "schema_version": LOCK_SCHEMA,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "purpose": "existing",
                "token": "existing-token",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProtocolError, match="Another Stage-90 process owns"):
        with exclusive_artifact_lock(tmp_path, purpose="contender"):
            pass

    assert lock.is_file()


def test_run_lock_recovers_dead_local_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from midogpp_thesis.cvae.diagnostics.b_paired_reparameterization_audit import (
        run_lock,
    )

    lock = tmp_path / ".run.lock"
    lock.write_text(
        json.dumps(
            {
                "schema_version": LOCK_SCHEMA,
                "pid": 999_999,
                "hostname": socket.gethostname(),
                "purpose": "crashed",
                "token": "stale-token",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_lock, "_pid_is_alive", lambda _pid: False)

    with exclusive_artifact_lock(tmp_path, purpose="replacement"):
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["purpose"] == "replacement"
        assert payload["token"] != "stale-token"

    assert not lock.exists()


def test_run_lock_does_not_delete_replacement_owner(tmp_path: Path) -> None:
    lock = tmp_path / ".run.lock"

    with exclusive_artifact_lock(tmp_path, purpose="initial"):
        lock.unlink()
        replacement = {
            "schema_version": LOCK_SCHEMA,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "purpose": "replacement",
            "token": "replacement-token",
        }
        lock.write_text(json.dumps(replacement), encoding="utf-8")

    assert json.loads(lock.read_text(encoding="utf-8"))["token"] == "replacement-token"
