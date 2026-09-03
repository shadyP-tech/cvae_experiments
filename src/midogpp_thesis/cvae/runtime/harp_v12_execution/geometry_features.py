"""Shared 6/7/8-source geometry features for source and target menus."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

from ...protocol import ProtocolError
from ...routing.harp_protocol import canonical_hash


GEOMETRY_FEATURE_NAMES = (
    "geometry_candidate_count",
    "geometry_prediction_or_target_indicator",
    "geometry_six_source_calibration_indicator",
    "geometry_action_maximum_source_weight",
    "geometry_action_effective_source_count",
    "geometry_density_excess_over_quarter",
    "geometry_effective_sources_shortfall_from_six",
)

_GEOMETRY = {
    6: (168, 126),
    7: (144, 126),
    8: (128, 128),
}


def geometry_feature_values(
    *, candidate_count: int, action_kind: str, context_kind: str
) -> tuple[float, ...]:
    """Return the identical ordered schema for 6/7/8-source contexts."""

    if candidate_count not in _GEOMETRY:
        raise ProtocolError("HARP v12 geometry candidate count is outside 6/7/8.")
    if action_kind not in {"B", "U", "HXE"}:
        raise ProtocolError("HARP v12 geometry action kind is unknown.")
    if context_kind not in {"source_calibration", "source_prediction", "target"}:
        raise ProtocolError("HARP v12 geometry context kind is unknown.")
    base, topup = _GEOMETRY[candidate_count]
    final = candidate_count * base + topup
    if action_kind in {"B", "U"}:
        maximum = 1.0 / float(candidate_count)
        effective = float(candidate_count)
    else:
        selected = base + topup
        weights = (selected / final,) + tuple(
            base / final for _ in range(candidate_count - 1)
        )
        maximum = max(weights)
        effective = 1.0 / sum(value * value for value in weights)
    return (
        float(candidate_count),
        float(context_kind in {"source_prediction", "target"}),
        float(candidate_count == 6),
        maximum,
        effective,
        max(maximum - 0.25, 0.0),
        max(6.0 - effective, 0.0),
    )


def geometry_feature_audit() -> Mapping[str, object]:
    rows = []
    for count in (6, 7, 8):
        for kind in ("B", "U", "HXE"):
            values = geometry_feature_values(
                candidate_count=count,
                action_kind=kind,
                context_kind=(
                    "source_calibration" if count == 6 else "source_prediction"
                ),
            )
            rows.append(
                {
                    "candidate_count": count,
                    "action_kind": kind,
                    "maximum_source_weight": values[3],
                    "effective_source_count": values[4],
                }
            )
    body = {
        "schema_version": "midogpp_harp_v12_shared_geometry_features_v1",
        "feature_names": list(GEOMETRY_FEATURE_NAMES),
        "six_source_context": "source_calibration_C_minus_H_minus_q_minus_r",
        "seven_source_context": "source_prediction_C_minus_H_minus_q",
        "eight_source_context": "target_C_minus_H",
        "rows": rows,
        "generic_quarter_six_bound_claimed_for_six_source_HXE": False,
    }
    return MappingProxyType({**body, "geometry_feature_hash": canonical_hash(body)})


__all__ = (
    "GEOMETRY_FEATURE_NAMES",
    "geometry_feature_audit",
    "geometry_feature_values",
)
