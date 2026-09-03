"""Process-wide serialization for HARP v13 authority transitions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import fcntl
import os

from ...protocol import ProtocolError
from .activation_paths import RepositoryBoundary


LOCK_RELATIVE_PATH = (
    "experiments/midogpp/stages/90_oracles_and_diagnostics/contracts/"
    "harp_router_v13/.harp_v13_activation.lock"
)


@contextmanager
def activation_lock(boundary: RepositoryBoundary) -> Iterator[None]:
    """Serialize every transition that owns v13 activation authority."""

    path = boundary.member(
        LOCK_RELATIVE_PATH,
        label="activation lock",
        kind="optional",
    )
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError("HARP v13 activation is already in progress.") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


__all__ = ("LOCK_RELATIVE_PATH", "activation_lock")
