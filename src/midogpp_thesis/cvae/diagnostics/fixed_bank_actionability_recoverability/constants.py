"""Frozen scientific constants for the actionability/recoverability diagnostic.

The two action geometries are parallel, predeclared diagnostic surfaces.  No
constant or helper in this module provides a way to select between them.
"""

from __future__ import annotations

from ...protocol import ProtocolError


MIDOGPP_CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
BINARY_CLASSES = (0, 1)
SEED_PAIR_ORDINALS = tuple(range(9))

B_ACTION_ID = "B"
U_ACTION_ID = "U"
GEOMETRY_IDS = ("A0", "A1")

PRE_SUPPORT_METHOD_IDS = ("B", "U", "G", "R", "P")
SUPPORT_METHOD_ID = "S_y"
TERMINAL_ORACLE_METHOD_IDS = ("O_static", "O_case")
GEOMETRY_METHOD_IDS = ("U", "G", "R", "P", "S_y")

RIDGE_ALPHA = 1.0
HARD_THRESHOLD = 0.5
NEAR_THRESHOLD_HALF_WIDTH = 0.1
PROBABILITY_EPSILON = 1.0e-4
STANDARDIZATION_SCALE_FLOOR = 1.0e-3

B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
SELECTED_COUNT_PER_CLASS = 256
OTHER_COUNT_PER_CLASS = 128
A1_SELECTED_SAMPLE_WEIGHT = 23.0 / 16.0
A1_OTHER_SAMPLE_WEIGHT = 7.0 / 8.0

CASE_ACTION_FEATURE_NAMES = (
    "intercept",
    "baseline_probability_mean",
    "baseline_probability_sd",
    "uniform_delta_mean",
    "uniform_delta_abs_mean",
    "candidate_delta_mean",
    "candidate_delta_abs_mean",
    "candidate_delta_sd",
    "candidate_disagreement_vs_b",
    "candidate_disagreement_vs_u",
    "candidate_entropy_mean",
    "candidate_near_threshold_rate",
    "candidate_seed_sd_mean",
)


def _center(value: object, role: str) -> str:
    center = str(value)
    if center not in MIDOGPP_CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ {role} center: {center}.")
    return center


def candidate_sources(target: object) -> tuple[str, ...]:
    """Return the exact eight-source pool, never the held-out target expert."""

    target_center = _center(target, "target")
    return tuple(center for center in MIDOGPP_CENTERS if center != target_center)


def geometry_action_id(geometry: object, source: object) -> str:
    """Build an unambiguous action identifier for a frozen geometry/source."""

    geometry_id = str(geometry)
    if geometry_id not in GEOMETRY_IDS:
        raise ProtocolError(f"Unknown action geometry: {geometry_id}.")
    source_center = _center(source, "source")
    return f"{geometry_id}::source={source_center}"


def geometry_action_ids(target: object, geometry: object) -> tuple[str, ...]:
    return tuple(
        geometry_action_id(geometry, source) for source in candidate_sources(target)
    )


__all__ = (
    "A1_OTHER_SAMPLE_WEIGHT",
    "A1_SELECTED_SAMPLE_WEIGHT",
    "BINARY_CLASSES",
    "B_ACTION_ID",
    "B_COUNT_PER_SOURCE_CLASS",
    "CASE_ACTION_FEATURE_NAMES",
    "GEOMETRY_IDS",
    "GEOMETRY_METHOD_IDS",
    "HARD_THRESHOLD",
    "MIDOGPP_CENTERS",
    "NEAR_THRESHOLD_HALF_WIDTH",
    "OTHER_COUNT_PER_CLASS",
    "PRE_SUPPORT_METHOD_IDS",
    "PROBABILITY_EPSILON",
    "RIDGE_ALPHA",
    "SEED_PAIR_ORDINALS",
    "SELECTED_COUNT_PER_CLASS",
    "STANDARDIZATION_SCALE_FLOOR",
    "SUPPORT_METHOD_ID",
    "TERMINAL_ORACLE_METHOD_IDS",
    "U_ACTION_ID",
    "U_COUNT_PER_SOURCE_CLASS",
    "candidate_sources",
    "geometry_action_id",
    "geometry_action_ids",
)
