"""The sole one-shot evaluation-label boundary for fresh HARP."""

from __future__ import annotations

from typing import Mapping

from ...protocol import ProtocolError
from ...runtime.harp_probability_menu.hashing import require_sha256
from .sealing import HarpFreshPrelabelSeal


_CTOR_KEY = object()


class HarpFreshEvaluationCapability:
    __slots__ = ("_labels", "_seal_hash", "_authorization_hash", "_consumed")

    def __init__(
        self,
        key: object,
        *,
        labels: Mapping[tuple[str, str, str], int],
        seal_hash: str,
        authorization_hash: str,
    ) -> None:
        if key is not _CTOR_KEY:
            raise ProtocolError("Fresh HARP label capabilities require a prelabel seal.")
        self._labels = dict(labels)
        self._seal_hash = seal_hash
        self._authorization_hash = authorization_hash
        self._consumed = False

    def consume(
        self, seal: HarpFreshPrelabelSeal
    ) -> dict[tuple[str, str, str], int]:
        if not isinstance(seal, HarpFreshPrelabelSeal) or seal.seal_hash != self._seal_hash:
            raise ProtocolError("Fresh HARP label capability is bound to another seal.")
        if self._consumed:
            raise ProtocolError("Fresh HARP label capability was already consumed.")
        self._consumed = True
        labels = dict(self._labels)
        self._labels.clear()
        return labels

    def __reduce__(self) -> object:
        raise ProtocolError("Fresh HARP label capabilities are nonserializable.")

    def __getstate__(self) -> object:
        raise ProtocolError("Fresh HARP label capabilities are nonserializable.")


def issue_harp_fresh_evaluation_capability(
    seal: HarpFreshPrelabelSeal,
    *,
    labels_by_row_key: Mapping[tuple[str, str, str], int],
    reservation_hash: str,
    target_cache_hash: str,
    authorization_hash: str,
) -> HarpFreshEvaluationCapability:
    """Open target labels only after every route and vector is durable."""

    if not isinstance(seal, HarpFreshPrelabelSeal):
        raise ProtocolError("Fresh HARP target labels cannot open before route sealing.")
    if (
        require_sha256(reservation_hash, name="reservation hash")
        != seal.reservation_hash
        or require_sha256(target_cache_hash, name="target-cache hash")
        != seal.target_cache_hash
    ):
        raise ProtocolError("Fresh HARP label manifest binding drifted.")
    authorization = require_sha256(authorization_hash, name="evaluation authorization")
    if not isinstance(labels_by_row_key, Mapping):
        raise ProtocolError("Fresh HARP labels must be row keyed.")
    labels = dict(labels_by_row_key)
    if (
        tuple(labels) != seal.row_keys
        or any(type(value) is not int or value not in (0, 1) for value in labels.values())
    ):
        raise ProtocolError("Fresh HARP labels must exactly cover sealed rows in order.")
    return HarpFreshEvaluationCapability(
        _CTOR_KEY,
        labels=labels,
        seal_hash=seal.seal_hash,
        authorization_hash=authorization,
    )


__all__ = (
    "HarpFreshEvaluationCapability",
    "issue_harp_fresh_evaluation_capability",
)
