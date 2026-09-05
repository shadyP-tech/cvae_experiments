"""Strict JSON-boundary normalization for HARP v20 runtime artifacts.

Runtime contracts intentionally expose immutable ``Mapping`` views.  Those
views remain useful in memory, but Python's JSON encoder does not recognize a
``mappingproxy`` nested inside an otherwise ordinary dictionary.  Normalize
only at the durable JSON boundary so scientific hashes continue to bind the
original semantic payload while persisted bytes contain plain JSON values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ...protocol import ProtocolError


_JSON_SCALARS = (str, int, float, bool, type(None))


def plain_json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Return a recursively plain, JSON-compatible copy of ``value``.

    Mapping keys must already be canonical strings.  Arbitrary iterables,
    NumPy objects, dataclasses, and opaque values are rejected instead of
    silently changing their representation.
    """

    normalized = _plain_json_value(value, path="$", allow_mapping=True)
    if not isinstance(normalized, dict):  # pragma: no cover - defensive guard
        raise ProtocolError("HARP v20 JSON payload root is not an object.")
    return normalized


def _plain_json_value(
    value: object,
    *,
    path: str,
    allow_mapping: bool = False,
) -> object:
    if isinstance(value, Mapping):
        output: dict[str, object] = {}
        for key, member in value.items():
            if type(key) is not str or not key:
                raise ProtocolError(
                    f"HARP v20 JSON payload has a non-canonical key at {path}."
                )
            output[key] = _plain_json_value(member, path=f"{path}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _plain_json_value(member, path=f"{path}[{index}]")
            for index, member in enumerate(value)
        ]
    if type(value) in _JSON_SCALARS:
        return value
    if allow_mapping:  # pragma: no cover - root type is statically constrained
        raise ProtocolError("HARP v20 JSON payload root is not an object.")
    raise ProtocolError(
        f"HARP v20 JSON payload contains an unsupported value at {path}."
    )


__all__ = ("plain_json_mapping",)
