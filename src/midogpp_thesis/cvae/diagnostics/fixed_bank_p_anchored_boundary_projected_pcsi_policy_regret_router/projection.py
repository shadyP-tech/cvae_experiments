"""P-anchored action geometries and projected-action equivalence classes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ...protocol import ProtocolError
from .constants import (
    ACTION_GEOMETRY_IDS,
    ALTERNATIVE_METHOD_IDS,
    DIRECTION_IDS,
    LEGACY_GEOMETRY_ID,
    PORTFOLIO_METHOD_ID,
    PROJECTION_GEOMETRY_ID,
    SIGN_PRESERVING_SHRINKAGE,
    UNPROJECTED_GEOMETRY_ID,
)
from .contracts import EndpointCasePrediction
from .hashing import canonical_hash, require_sha256
from .projection_lattice import (
    THRESHOLD,
    THRESHOLD_PREDECESSOR,
    as_binary32,
    canonical_bytes,
)


_ALTERNATIVE_ORDER = MappingProxyType(
    {name: index for index, name in enumerate(ALTERNATIVE_METHOD_IDS)}
)


@dataclass(frozen=True, order=True)
class ActionEquivalenceClass:
    """One emitted case/direction vector after deterministic equivalence collapse."""

    target_center: str
    case_id: str
    direction: str
    geometry_id: str
    representative: str
    members: tuple[str, ...]
    sample_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    crossing_sample_ids: tuple[str, ...]
    endpoint_prediction_hash: str
    probability_bytes_sha256: str = field(init=False, compare=True)
    action_hash: str = field(init=False, compare=True)

    def __post_init__(self) -> None:
        members = tuple(str(value) for value in self.members)
        samples = tuple(str(value) for value in self.sample_ids)
        crossings = tuple(str(value) for value in self.crossing_sample_ids)
        probabilities = as_binary32(self.probabilities, name="action class")
        if (
            self.direction not in DIRECTION_IDS
            or self.geometry_id not in ACTION_GEOMETRY_IDS
            or self.representative not in ALTERNATIVE_METHOD_IDS
            or not members
            or tuple(sorted(members, key=_ALTERNATIVE_ORDER.__getitem__)) != members
            or self.representative != members[0]
            or any(value not in ALTERNATIVE_METHOD_IDS for value in members)
            or len(members) != len(set(members))
            or not samples
            or len(samples) != len(set(samples))
            or len(probabilities) != len(samples)
            or len(crossings) != len(set(crossings))
            or any(value not in samples for value in crossings)
        ):
            raise ProtocolError("PCSI-PARC action equivalence identity drifted.")
        require_sha256(self.endpoint_prediction_hash, "endpoint_prediction_hash")
        digest = hashlib.sha256(canonical_bytes(probabilities)).hexdigest()
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(
            self, "probabilities", tuple(float(value) for value in probabilities)
        )
        object.__setattr__(self, "crossing_sample_ids", crossings)
        object.__setattr__(self, "probability_bytes_sha256", digest)
        object.__setattr__(self, "action_hash", canonical_hash(self._unhashed()))

    @property
    def crossing_count(self) -> int:
        return len(self.crossing_sample_ids)

    @property
    def structural_zero(self) -> bool:
        return self.crossing_count == 0

    @property
    def key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.target_center,
            self.case_id,
            self.geometry_id,
            self.direction,
            self.representative,
            self.probability_bytes_sha256,
        )

    def _unhashed(self) -> dict[str, object]:
        return {
            "schema_version": "fixed_bank_pcsi_parc_action_equivalence_v1",
            "target_center": self.target_center,
            "case_id": self.case_id,
            "direction": self.direction,
            "geometry_id": self.geometry_id,
            "representative": self.representative,
            "members": list(self.members),
            "sample_ids": list(self.sample_ids),
            "probability_bytes_sha256": self.probability_bytes_sha256,
            "crossing_sample_ids": list(self.crossing_sample_ids),
            "crossing_count": self.crossing_count,
            "structural_zero": self.structural_zero,
            "endpoint_prediction_hash": self.endpoint_prediction_hash,
            "probability_dtype": "float32_le",
            "labels_used": False,
        }

    def to_payload(self) -> dict[str, object]:
        return {
            **self._unhashed(),
            "probabilities": list(self.probabilities),
            "action_hash": self.action_hash,
        }


def _crossing_mask(
    portfolio: np.ndarray,
    alternative: np.ndarray,
    direction: str,
) -> np.ndarray:
    p_hard = portfolio >= THRESHOLD
    a_hard = alternative >= THRESHOLD
    if direction == DIRECTION_IDS[0]:
        return (~p_hard) & a_hard
    if direction == DIRECTION_IDS[1]:
        return p_hard & (~a_hard)
    raise ProtocolError("PCSI-PARC requested an unknown direction.")


def emit_directional_action(
    endpoint: EndpointCasePrediction,
    alternative: str,
    direction: str,
    *,
    geometry_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Emit one action without consulting any label or predecessor artifact."""

    if alternative not in ALTERNATIVE_METHOD_IDS or geometry_id not in ACTION_GEOMETRY_IDS:
        raise ProtocolError("PCSI-PARC action geometry identity drifted.")
    portfolio = as_binary32(
        endpoint.probabilities[PORTFOLIO_METHOD_ID], name="portfolio endpoint"
    )
    candidate = as_binary32(
        endpoint.probabilities[alternative], name="candidate endpoint"
    )
    if len(portfolio) != len(candidate):
        raise ProtocolError("PCSI-PARC endpoint lengths drifted.")
    mask = _crossing_mask(portfolio, candidate, direction)
    output = portfolio.copy()
    if geometry_id == PROJECTION_GEOMETRY_ID:
        projected = THRESHOLD if direction == DIRECTION_IDS[0] else THRESHOLD_PREDECESSOR
        output[mask] = projected
    elif geometry_id == UNPROJECTED_GEOMETRY_ID:
        output[mask] = candidate[mask]
    else:
        output[mask] = np.float32(0.5) + np.float32(
            SIGN_PRESERVING_SHRINKAGE
        ) * (candidate[mask] - np.float32(0.5))

    p_hard = portfolio >= THRESHOLD
    emitted_hard = output >= THRESHOLD
    if (
        not np.array_equal(output[~mask].view(np.uint32), portfolio[~mask].view(np.uint32))
        or np.any(emitted_hard[mask] == p_hard[mask])
    ):
        raise ProtocolError("PCSI-PARC action failed its P-byte or flip contract.")
    if geometry_id == PROJECTION_GEOMETRY_ID and np.any(mask):
        lower = np.minimum(portfolio[mask], candidate[mask])
        upper = np.maximum(portfolio[mask], candidate[mask])
        if np.any((output[mask] < lower) | (output[mask] > upper)):
            raise ProtocolError("PCSI-PARC projection escaped the P-to-action segment.")
        expected = THRESHOLD if direction == DIRECTION_IDS[0] else THRESHOLD_PREDECESSOR
        if not np.all(output[mask] == expected):
            raise ProtocolError("PCSI-PARC projection is not lattice-minimal.")
    return as_binary32(output, name="emitted action"), mask


def build_action_equivalence_classes(
    endpoint: EndpointCasePrediction,
    *,
    geometry_id: str,
    collapse_equivalent: bool | None = None,
) -> tuple[ActionEquivalenceClass, ...]:
    """Build both directions, collapsing projected vectors before modeling."""

    collapse = geometry_id == PROJECTION_GEOMETRY_ID if collapse_equivalent is None else bool(collapse_equivalent)
    output: list[ActionEquivalenceClass] = []
    for direction in DIRECTION_IDS:
        raw: list[tuple[str, np.ndarray, np.ndarray]] = []
        for alternative in ALTERNATIVE_METHOD_IDS:
            probabilities, mask = emit_directional_action(
                endpoint,
                alternative,
                direction,
                geometry_id=geometry_id,
            )
            raw.append((alternative, probabilities, mask))
        groups: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
        for alternative, probabilities, mask in raw:
            digest = hashlib.sha256(canonical_bytes(probabilities)).hexdigest()
            key = digest if collapse else f"{alternative}::{digest}"
            groups.setdefault(key, []).append((alternative, probabilities, mask))
        for rows in groups.values():
            ordered = tuple(sorted((row[0] for row in rows), key=_ALTERNATIVE_ORDER.__getitem__))
            representative = ordered[0]
            selected = next(row for row in rows if row[0] == representative)
            probabilities, mask = selected[1], selected[2]
            crossing_ids = tuple(
                endpoint.sample_ids[int(index)] for index in np.flatnonzero(mask)
            )
            output.append(
                ActionEquivalenceClass(
                    endpoint.center,
                    endpoint.case_id,
                    direction,
                    geometry_id,
                    representative,
                    ordered,
                    endpoint.sample_ids,
                    tuple(float(value) for value in probabilities),
                    crossing_ids,
                    endpoint.prediction_hash,
                )
            )
    result = tuple(
        sorted(
            output,
            key=lambda row: (
                DIRECTION_IDS.index(row.direction),
                _ALTERNATIVE_ORDER[row.representative],
                row.probability_bytes_sha256,
            ),
        )
    )
    if not result or len({row.key for row in result}) != len(result):
        raise ProtocolError("PCSI-PARC action equivalence surface drifted.")
    return result


def index_action_classes(
    rows: Sequence[ActionEquivalenceClass],
) -> Mapping[str, ActionEquivalenceClass]:
    indexed = {row.action_hash: row for row in rows}
    if len(indexed) != len(tuple(rows)):
        raise ProtocolError("PCSI-PARC action hashes are duplicated.")
    return MappingProxyType(indexed)


__all__ = (
    "ActionEquivalenceClass",
    "build_action_equivalence_classes",
    "emit_directional_action",
    "index_action_classes",
)
