"""Canonical JSON, hashes and case-set checks for policy admission."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from ...routing.residual_topup.hashing import canonical_sha256
from .contracts import CENTERS


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Cannot read utility-aligned JSON: {path}.") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Utility-aligned JSON must be a mapping.")
    return payload


def require_hash(payload: Mapping[str, object], key: str, role: str) -> None:
    observed = payload.get(key)
    unhashed = {name: value for name, value in payload.items() if name != key}
    if not sha256_like(observed) or observed != canonical_sha256(unhashed):
        raise ProtocolError(f"Utility-aligned {role} hash drifted.")


def sha256_like(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) == 64 and all(char in "0123456789abcdef" for char in rendered)


def upstream_hash_like(value: object) -> bool:
    rendered = str(value or "")
    return len(rendered) in {16, 64} and all(
        char in "0123456789abcdef" for char in rendered
    )


def case_mapping(
    value: object,
    *,
    role: str,
    minimum_count: int | None = None,
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or {str(key) for key in value} != set(CENTERS):
        raise ProtocolError(f"Utility-aligned {role} cases must cover all targets.")
    normalized = {str(key): raw for key, raw in value.items()}
    output: dict[str, tuple[str, ...]] = {}
    for target in CENTERS:
        raw = normalized[target]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ProtocolError(f"Utility-aligned {role} cases are malformed.")
        cases = tuple(str(case) for case in raw)
        minimum = (
            int(minimum_count)
            if minimum_count is not None
            else (8 if "support" in role else 1)
        )
        if minimum < 1:
            raise ProtocolError("Utility-aligned case minimum must be positive.")
        if len(cases) < minimum or len(cases) != len(set(cases)) or any(not case for case in cases):
            raise ProtocolError(f"Utility-aligned {role} cases are malformed.")
        output[target] = cases
    return MappingProxyType(output)


def require_disjoint_cases(
    support: Mapping[str, tuple[str, ...]],
    evaluation: Mapping[str, tuple[str, ...]],
) -> None:
    support_values = tuple(case for target in CENTERS for case in support[target])
    evaluation_values = tuple(case for target in CENTERS for case in evaluation[target])
    if (
        len(support_values) != len(set(support_values))
        or len(evaluation_values) != len(set(evaluation_values))
        or set(support_values).intersection(evaluation_values)
    ):
        raise ProtocolError("Utility-aligned support/evaluation cases overlap.")


__all__ = (
    "case_mapping", "read_json", "require_disjoint_cases", "require_hash",
    "sha256_like", "upstream_hash_like",
)
