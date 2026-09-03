"""Typed semantic hashing for HARP v12 worker-visible projections."""

from __future__ import annotations

from collections.abc import Mapping

from ....common.hashing import stable_hash
from .hash_contracts import require_stable_hash


def projection_semantic_hash(
    body: Mapping[str, object], *, name: str
) -> str:
    """Return the repository-standard 16-hex identity for a projection body."""

    return require_stable_hash(stable_hash(dict(body)), name=name)


__all__ = ("projection_semantic_hash",)
