"""Canonical identities and fixed scientific geometry for the core audit."""

from __future__ import annotations

from itertools import product
from typing import Sequence

from ...protocol import ProtocolError
from .experiment_contracts import (
    CENTERS,
    CLAIM_SCOPE,
    DATASET_FAMILY,
    EXPERIMENT_ID,
    FIXED_SUPPORT_CASE_COUNT_PER_CENTER,
    GENERATION_SEEDS,
    PUBLICATION_STATUS,
    ROUTING_STATUS,
    STAGE_ID,
    TRAINING_SEEDS,
)


SEED_PAIRS = tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))
EXACT_SEED_PAIR_COUNT = len(SEED_PAIRS)
SEED_PAIR_COUNT = EXACT_SEED_PAIR_COUNT
MIN_SUPPORT_CASE_COUNT_PER_CENTER = FIXED_SUPPORT_CASE_COUNT_PER_CENTER
RIDGE_ALPHA = 1.0
CYCLIC_PERMUTATION_SEED = 90_902_026
CYCLIC_PERMUTATION_SHIFT = 1
OUTER_INFERENCE_UNIT_COUNT = len(CENTERS)
STUDENT_T_975_DF8 = 2.306004135204166

EXACT_BACC_DELTA = "exact_bacc_delta"
SMOOTH_BACC_DELTA = "smooth_bacc_delta"
RESPONSE_NAMES = (EXACT_BACC_DELTA, SMOOTH_BACC_DELTA)
PRIMARY_RESPONSE_NAME = EXACT_BACC_DELTA
DIAGNOSTIC_RESPONSE_NAMES = (SMOOTH_BACC_DELTA,)

EQUAL_UNION_NULL = "equal_union_null"
METADATA_ONLY_CONTROL = "metadata_only_control"
POOLED_ROW_WEIGHTED_SHIFT_CONTROL = "pooled_row_weighted_shift_control"
CASE_BALANCED_SHIFT_COMPACT = "case_balanced_shift_compact"
CASE_BALANCED_RICH_COMPACT = "case_balanced_rich_compact"
CASE_AWARE_HYBRID_COMPACT = "case_aware_hybrid_compact"
CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL = (
    "cyclic_directional_permutation_control"
)

# Compatibility aliases never change the serialized family IDs.
NULL_CONTROL = EQUAL_UNION_NULL
METADATA_CONTROL = METADATA_ONLY_CONTROL
POOLED_ROW_WEIGHTED_ABS_SHIFT_CONTROL = POOLED_ROW_WEIGHTED_SHIFT_CONTROL
NULL_FAMILY = EQUAL_UNION_NULL
METADATA_FAMILY = METADATA_ONLY_CONTROL
POOLED_ABS_SHIFT_CONTROL = POOLED_ROW_WEIGHTED_SHIFT_CONTROL
CYCLIC_CONTROL = CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL

FAMILY_IDS = (
    EQUAL_UNION_NULL,
    METADATA_ONLY_CONTROL,
    POOLED_ROW_WEIGHTED_SHIFT_CONTROL,
    CASE_BALANCED_SHIFT_COMPACT,
    CASE_BALANCED_RICH_COMPACT,
    CASE_AWARE_HYBRID_COMPACT,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
)
SCREENING_FAMILY_IDS = (
    CASE_BALANCED_SHIFT_COMPACT,
    CASE_BALANCED_RICH_COMPACT,
    CASE_AWARE_HYBRID_COMPACT,
)
CONTROL_FAMILY_IDS = (
    EQUAL_UNION_NULL,
    METADATA_ONLY_CONTROL,
    POOLED_ROW_WEIGHTED_SHIFT_CONTROL,
    CYCLIC_DIRECTIONAL_PERMUTATION_CONTROL,
)

FEATURE_ROW_SCHEMA = "midogpp_stage90_case_aware_proxy_feature_row_v1"
RESPONSE_ROW_SCHEMA = "midogpp_stage90_case_aware_proxy_response_row_v1"

EXPECTED_FEATURE_ROW_COUNT = len(CENTERS) * (len(CENTERS) - 1) * (
    len(CENTERS) - 2
)
EXPECTED_PROXY_FEATURE_ROW_COUNT = EXPECTED_FEATURE_ROW_COUNT
EXPECTED_ENSEMBLE_UTILITY_RESPONSE_COUNT = EXPECTED_FEATURE_ROW_COUNT
EXPECTED_QUERY_COUNT = len(CENTERS) * (len(CENTERS) - 1)
EXPECTED_OUTER_COUNT = len(CENTERS)


def candidate_sources(
    outer_target_id: object,
    query_id: object,
    *,
    centers: Sequence[str] = CENTERS,
) -> tuple[str, ...]:
    """Return legal candidates in canonical order for one ``(H, q)``."""

    frozen = _validated_centers(centers)
    outer = _center(outer_target_id, "outer_target_id", frozen)
    query = _center(query_id, "query_id", frozen)
    if outer == query:
        raise ProtocolError("Case-aware audit requires distinct H and q domains.")
    return tuple(value for value in frozen if value not in {outer, query})


def expected_strict_training_row_count(
    centers: int | Sequence[object] = CENTERS,
) -> int:
    """Count ordered H/q/e rows after excluding all three predicted roles."""

    count = centers if type(centers) is int else len(tuple(centers))
    if type(count) is not int or count < 6:
        raise ProtocolError("Strict H/q/e crossfit requires at least six centers.")
    remaining = count - 3
    return remaining * (remaining - 1) * (remaining - 2)


EXPECTED_STRICT_CROSSFIT_TRAINING_ROW_COUNT = expected_strict_training_row_count()


def expected_row_keys() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (outer, query, source)
        for outer in CENTERS
        for query in CENTERS
        if query != outer
        for source in candidate_sources(outer, query)
    )


def _validated_centers(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) < 3 or len(set(result)) != len(result):
        raise ProtocolError("Center geometry must contain distinct domain IDs.")
    if any(type(value) is not str or not value for value in result):
        raise ProtocolError("Center IDs must be nonempty strings.")
    return result


def _center(value: object, name: str, centers: Sequence[str]) -> str:
    if type(value) is not str or value not in centers:
        raise ProtocolError(f"{name} is outside the frozen center geometry.")
    return value


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "candidate_sources",
    "expected_row_keys",
    "expected_strict_training_row_count",
)
