"""Crash-recoverable exclusive lock for the endpoint-router artifact root."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import socket
from typing import Iterator, Mapping

from ...protocol import ProtocolError


LOCK_SCHEMA = "midogpp_consumed_test_endpoint_router_run_lock_v1"


@contextmanager
def exclusive_run_lock(artifact_root: str | Path) -> Iterator[None]:
    """Own ``.run.lock`` and recover only a dead same-host owner."""

    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ProtocolError(f"Endpoint-router artifact root does not exist: {root}")
    lock_path = root / ".run.lock"
    token = secrets.token_hex(16)
    payload = {
        "schema_version": LOCK_SCHEMA,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "token": token,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    owner_stat: os.stat_result | None = None
    for attempt in range(2):
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as exc:
            if attempt == 0 and _remove_dead_local_owner(lock_path):
                continue
            raise ProtocolError(
                f"Another endpoint-router process owns {lock_path}."
            ) from exc
        try:
            serialized = (
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            os.write(descriptor, serialized)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        owner_stat = lock_path.stat()
        break
    if owner_stat is None:
        raise ProtocolError(f"Could not acquire endpoint-router lock: {lock_path}")

    try:
        yield
    finally:
        _release_if_owned(lock_path, token=token, expected_stat=owner_stat)


def _remove_dead_local_owner(lock_path: Path) -> bool:
    """Remove a stable lock only when its same-host PID is no longer alive."""

    try:
        before = lock_path.stat()
        payload = _read_payload(lock_path)
    except (FileNotFoundError, OSError, ProtocolError):
        return False
    if (
        payload.get("schema_version") != LOCK_SCHEMA
        or payload.get("hostname") != socket.gethostname()
    ):
        return False
    try:
        pid = int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    if pid <= 0 or _pid_is_alive(pid):
        return False
    try:
        after = lock_path.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            return False
        lock_path.unlink()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _release_if_owned(
    lock_path: Path, *, token: str, expected_stat: os.stat_result
) -> None:
    """Never delete a replacement lock created after this process acquired it."""

    try:
        observed_stat = lock_path.stat()
        payload = _read_payload(lock_path)
    except (FileNotFoundError, OSError, ProtocolError):
        return
    if (
        (observed_stat.st_dev, observed_stat.st_ino)
        != (expected_stat.st_dev, expected_stat.st_ino)
        or payload.get("token") != token
    ):
        return
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _read_payload(lock_path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Malformed endpoint-router run lock: {lock_path}") from exc
    if not isinstance(payload, Mapping):
        raise ProtocolError(f"Malformed endpoint-router run lock: {lock_path}")
    return payload


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


__all__ = ("LOCK_SCHEMA", "exclusive_run_lock")
