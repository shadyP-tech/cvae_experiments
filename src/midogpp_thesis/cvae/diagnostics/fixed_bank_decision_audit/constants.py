"""Frozen scientific identities for the fixed-bank decision-audit core."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ...protocol import ProtocolError
from .experiment_contracts import (
    CENTERS,
    EXPECTED_CANDIDATE_COUNT_PER_QUERY,
    EXPECTED_FEATURE_ROW_COUNT,
    EXPECTED_QUERY_COUNT,
    EXPECTED_RESPONSE_ROW_COUNT,
    EXPECTED_STRICT_TRAINING_ROW_COUNT,
)


RIDGE_ALPHA = 1.0
CONFIDENCE_MULTIPLIER = 1.96
MINIMUM_ROUTE_GAIN = 0.0
OUTER_INFERENCE_UNIT_COUNT = len(CENTERS)
STUDENT_T_975_DF8 = 2.306004135204166
BLOCKED_PERMUTATION_SHIFT = 1

EXACT_BACC_DELTA = "exact_bacc_delta"
SMOOTH_BACC_DELTA = "smooth_bacc_delta"

NULL_TIED_EXACT_CONTROL = "null_tied_exact_control"
GLOBAL_SOURCE_EXACT_CONTROL = "global_source_exact_control"
POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL = (
    "pooled_row_weighted_shift_exact_control"
)
CASE_BALANCED_SHIFT_EXACT = "case_balanced_shift_exact"
CASE_BALANCED_SHIFT_BLOCKED_PERMUTATION_CONTROL = (
    "case_balanced_shift_blocked_permutation_control"
)
CASE_BALANCED_RICH_EXACT = "case_balanced_rich_exact"
CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL = (
    "case_balanced_rich_blocked_permutation_control"
)
CASE_AWARE_BOUNDARY_EXACT = "case_aware_boundary_exact"
CASE_AWARE_BOUNDARY_BLOCKED_PERMUTATION_CONTROL = (
    "case_aware_boundary_blocked_permutation_control"
)

EXACT_FAMILY_IDS = (
    NULL_TIED_EXACT_CONTROL,
    GLOBAL_SOURCE_EXACT_CONTROL,
    POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL,
    CASE_BALANCED_SHIFT_EXACT,
    CASE_BALANCED_SHIFT_BLOCKED_PERMUTATION_CONTROL,
    CASE_BALANCED_RICH_EXACT,
    CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL,
    CASE_AWARE_BOUNDARY_EXACT,
    CASE_AWARE_BOUNDARY_BLOCKED_PERMUTATION_CONTROL,
)
PRIMARY_R_FAMILY_ID = CASE_BALANCED_RICH_EXACT
SECONDARY_CHALLENGER_FAMILY_IDS = (
    CASE_BALANCED_SHIFT_EXACT,
    CASE_AWARE_BOUNDARY_EXACT,
)
PERMUTATION_CONTROL_FAMILY_IDS = (
    CASE_BALANCED_SHIFT_BLOCKED_PERMUTATION_CONTROL,
    CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL,
    CASE_AWARE_BOUNDARY_BLOCKED_PERMUTATION_CONTROL,
)
CONTROL_FAMILY_IDS = (
    NULL_TIED_EXACT_CONTROL,
    GLOBAL_SOURCE_EXACT_CONTROL,
    POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL,
    *PERMUTATION_CONTROL_FAMILY_IDS,
)

SHIFT_PREDICTORS = (
    "equal_case_abs_shift",
    "case_abs_shift_sd",
    "equal_case_signed_margin",
)
RICH_PREDICTORS = (
    "case_balanced_reconstruction_z",
    "case_balanced_kl_z",
    "case_balanced_log_mmd_z",
)
BOUNDARY_PREDICTORS = (
    "equal_case_signed_margin",
    "case_balanced_flip_rate",
    "case_balanced_entropy_change",
)

EXACT_FAMILY_PREDICTORS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        NULL_TIED_EXACT_CONTROL: (),
        GLOBAL_SOURCE_EXACT_CONTROL: (),
        POOLED_ROW_WEIGHTED_SHIFT_EXACT_CONTROL: (
            "pooled_row_weighted_abs_shift",
        ),
        CASE_BALANCED_SHIFT_EXACT: SHIFT_PREDICTORS,
        CASE_BALANCED_SHIFT_BLOCKED_PERMUTATION_CONTROL: SHIFT_PREDICTORS,
        CASE_BALANCED_RICH_EXACT: RICH_PREDICTORS,
        CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL: RICH_PREDICTORS,
        CASE_AWARE_BOUNDARY_EXACT: BOUNDARY_PREDICTORS,
        CASE_AWARE_BOUNDARY_BLOCKED_PERMUTATION_CONTROL: BOUNDARY_PREDICTORS,
    }
)
PERMUTATION_PARENT: Mapping[str, str] = MappingProxyType(
    {
        CASE_BALANCED_SHIFT_BLOCKED_PERMUTATION_CONTROL: (
            CASE_BALANCED_SHIFT_EXACT
        ),
        CASE_BALANCED_RICH_BLOCKED_PERMUTATION_CONTROL: CASE_BALANCED_RICH_EXACT,
        CASE_AWARE_BOUNDARY_BLOCKED_PERMUTATION_CONTROL: (
            CASE_AWARE_BOUNDARY_EXACT
        ),
    }
)

SMOOTH_SHIFT_DESCRIPTIVE = "case_balanced_shift_smooth_descriptive"
SMOOTH_RICH_DESCRIPTIVE = "case_balanced_rich_smooth_descriptive"
SMOOTH_BOUNDARY_DESCRIPTIVE = "case_aware_boundary_smooth_descriptive"
SMOOTH_DESCRIPTIVE_FAMILY_IDS = (
    SMOOTH_SHIFT_DESCRIPTIVE,
    SMOOTH_RICH_DESCRIPTIVE,
    SMOOTH_BOUNDARY_DESCRIPTIVE,
)
SMOOTH_FAMILY_PREDICTORS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        SMOOTH_SHIFT_DESCRIPTIVE: SHIFT_PREDICTORS,
        SMOOTH_RICH_DESCRIPTIVE: RICH_PREDICTORS,
        SMOOTH_BOUNDARY_DESCRIPTIVE: BOUNDARY_PREDICTORS,
    }
)

EXPECTED_EXACT_PREDICTION_COUNT = len(EXACT_FAMILY_IDS) * EXPECTED_FEATURE_ROW_COUNT
EXPECTED_EXACT_FOLD_COUNT = len(EXACT_FAMILY_IDS) * EXPECTED_QUERY_COUNT
EXPECTED_SMOOTH_PREDICTION_COUNT = (
    len(SMOOTH_DESCRIPTIVE_FAMILY_IDS) * EXPECTED_FEATURE_ROW_COUNT
)
EXPECTED_SMOOTH_FOLD_COUNT = len(SMOOTH_DESCRIPTIVE_FAMILY_IDS) * EXPECTED_QUERY_COUNT

SCHEMA_PREFIX = "midogpp_fixed_bank_decision_audit"
FEATURE_ROW_SCHEMA = f"{SCHEMA_PREFIX}_feature_row_v1"
RESPONSE_ROW_SCHEMA = f"{SCHEMA_PREFIX}_response_row_v1"
DATASET_SCHEMA = f"{SCHEMA_PREFIX}_dataset_v1"
FAMILY_SPEC_SCHEMA = f"{SCHEMA_PREFIX}_family_spec_v1"
FAMILY_DESIGN_SCHEMA = f"{SCHEMA_PREFIX}_family_design_v1"
EXACT_FOLD_SCHEMA = f"{SCHEMA_PREFIX}_exact_fold_v1"
EXACT_PREDICTION_SCHEMA = f"{SCHEMA_PREFIX}_exact_prediction_v1"
EXACT_CROSSFIT_SCHEMA = f"{SCHEMA_PREFIX}_exact_crossfit_v1"
SMOOTH_FOLD_SCHEMA = f"{SCHEMA_PREFIX}_smooth_fold_v1"
SMOOTH_PREDICTION_SCHEMA = f"{SCHEMA_PREFIX}_smooth_prediction_v1"
SMOOTH_CROSSFIT_SCHEMA = f"{SCHEMA_PREFIX}_smooth_crossfit_v1"
QUERY_METRIC_SCHEMA = f"{SCHEMA_PREFIX}_query_metric_v1"
OUTER_METRIC_SCHEMA = f"{SCHEMA_PREFIX}_outer_metric_v1"
FAMILY_SUMMARY_SCHEMA = f"{SCHEMA_PREFIX}_family_summary_v1"
ABSTENTION_DECISION_SCHEMA = f"{SCHEMA_PREFIX}_abstention_decision_v1"
ABSTENTION_SUMMARY_SCHEMA = f"{SCHEMA_PREFIX}_abstention_summary_v1"
AUDIT_RESULT_SCHEMA = f"{SCHEMA_PREFIX}_audit_result_v1"


def candidate_sources(
    outer_target_id: object,
    query_id: object,
    *,
    centers: Sequence[str] = CENTERS,
) -> tuple[str, ...]:
    """Return the fixed-bank candidates, excluding held ``H`` and ``q``."""

    frozen = tuple(centers)
    if (
        len(frozen) < 6
        or len(set(frozen)) != len(frozen)
        or any(type(value) is not str or not value for value in frozen)
    ):
        raise ProtocolError("Fixed-bank centers must be distinct nonempty IDs.")
    if outer_target_id not in frozen or query_id not in frozen:
        raise ProtocolError("H/q is outside the fixed-bank center geometry.")
    if outer_target_id == query_id:
        raise ProtocolError("Fixed-bank H and q must be distinct.")
    values = tuple(
        value for value in frozen if value not in {outer_target_id, query_id}
    )
    if len(values) != len(frozen) - 2:
        raise ProtocolError("Fixed-bank candidate geometry drifted.")
    return values


def expected_row_keys() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (outer, query, source)
        for outer in CENTERS
        for query in CENTERS
        if query != outer
        for source in candidate_sources(outer, query)
    )


def expected_training_row_count(center_count: int = len(CENTERS)) -> int:
    """Rows left by strict all-role exclusion of held ``H`` and ``q``."""

    if type(center_count) is not int or center_count < 6:
        raise ProtocolError("Fixed-bank crossfit requires at least six centers.")
    remaining = center_count - 2
    return remaining * (remaining - 1) * (remaining - 2)


if (
    EXPECTED_FEATURE_ROW_COUNT != len(expected_row_keys())
    or EXPECTED_RESPONSE_ROW_COUNT != EXPECTED_FEATURE_ROW_COUNT
    or EXPECTED_CANDIDATE_COUNT_PER_QUERY != len(CENTERS) - 2
    or expected_training_row_count() != EXPECTED_STRICT_TRAINING_ROW_COUNT
):
    raise RuntimeError("Fixed-bank experiment and scientific geometry drifted.")


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "candidate_sources",
    "expected_row_keys",
    "expected_training_row_count",
)
