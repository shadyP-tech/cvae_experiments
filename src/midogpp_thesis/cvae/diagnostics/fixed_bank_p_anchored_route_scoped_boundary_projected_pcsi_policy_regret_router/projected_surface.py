"""Materialization helpers for complete in-run action-geometry surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from ...protocol import ProtocolError
from .constants import ACTION_GEOMETRY_IDS, CENTERS
from .contracts import EndpointCasePrediction
from .projection import ActionEquivalenceClass, build_action_equivalence_classes


def build_geometry_surface(
    predictions_by_center: Mapping[str, Sequence[EndpointCasePrediction]],
    *,
    geometry_id: str,
) -> Mapping[str, tuple[ActionEquivalenceClass, ...]]:
    if geometry_id not in ACTION_GEOMETRY_IDS or set(predictions_by_center) != set(CENTERS):
        raise ProtocolError("PCSI-RACR geometry surface scope drifted.")
    output = {
        center: tuple(
            action
            for prediction in predictions_by_center[center]
            for action in build_action_equivalence_classes(
                prediction,
                geometry_id=geometry_id,
            )
        )
        for center in CENTERS
    }
    if any(len({row.key for row in rows}) != len(rows) for rows in output.values()):
        raise ProtocolError("PCSI-RACR geometry surface contains duplicate cells.")
    return MappingProxyType(output)


__all__ = ("build_geometry_surface",)
