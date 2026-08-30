"""One-shot, nonserializable target-outcome capability for HARP replay."""

from __future__ import annotations

import re
from typing import Mapping

from ...protocol import ProtocolError
from .sealing import FrozenHarpPredictionSeal


_CTOR_KEY = object()
_SHA256 = re.compile(r"[0-9a-f]{64}")


class HarpReplayCapability:
    __slots__ = ("_truth", "_seal_hash", "_authorization_hash", "_consumed")

    def __init__(self, key: object, *, truth: Mapping[tuple[str, str, str], int], seal_hash: str, authorization_hash: str) -> None:
        if key is not _CTOR_KEY:
            raise ProtocolError("HARP replay capabilities must be issued after a durable seal.")
        self._truth = dict(truth)
        self._seal_hash = seal_hash
        self._authorization_hash = authorization_hash
        self._consumed = False

    def consume(self, seal: FrozenHarpPredictionSeal) -> dict[tuple[str, str, str], int]:
        if not isinstance(seal, FrozenHarpPredictionSeal) or seal.seal_hash != self._seal_hash:
            raise ProtocolError("HARP replay capability is bound to a different prediction seal.")
        if self._consumed:
            raise ProtocolError("HARP replay capability has already been consumed.")
        self._consumed = True
        return dict(self._truth)

    def __reduce__(self) -> object:
        raise ProtocolError("HARP replay capability is deliberately nonserializable.")

    def __getstate__(self) -> object:
        raise ProtocolError("HARP replay capability is deliberately nonserializable.")


def issue_harp_replay_capability(
    seal: FrozenHarpPredictionSeal,
    *,
    target_truth: Mapping[tuple[str, str, str], int],
    authorization_hash: str,
) -> HarpReplayCapability:
    """The sole HARP target-truth ingress, available only after sealing."""

    if not isinstance(seal, FrozenHarpPredictionSeal):
        raise ProtocolError("Target outcomes cannot open before the durable prediction seal.")
    if type(authorization_hash) is not str or _SHA256.fullmatch(authorization_hash) is None:
        raise ProtocolError("Replay authorization must be a SHA-256 identity.")
    truth = dict(target_truth)
    expected = {row.row_key for row in seal.decisions}
    if set(truth) != expected or any(type(value) is not int or value not in (0, 1) for value in truth.values()):
        raise ProtocolError("Target truth must be binary and exactly aligned to sealed rows.")
    return HarpReplayCapability(_CTOR_KEY, truth=truth, seal_hash=seal.seal_hash, authorization_hash=authorization_hash)


__all__ = ("HarpReplayCapability", "issue_harp_replay_capability")
