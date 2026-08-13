"""Frozen scientific constants for the directional-shrinkage diagnostic.

This module is intentionally import-light.  It defines the complete executable
grid and action identities, but it contains no data access, fitting, or CUDA
initialization.
"""

from __future__ import annotations

from fractions import Fraction

from ...protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
MIDOGPP_CENTERS = CENTERS
EXCLUDED_CENTER = "4"
BINARY_CLASSES = (0, 1)
SEED_PAIR_COUNT = 9

B_ACTION_ID = "B"
U_ACTION_ID = "U"
OFF_ACTION_ID = "OFF"
A1_PREFIX = "A1::source="

DIRECTION_IDS = ("zero_to_one", "one_to_zero")
K_GRID = (4, 5, 6)
W_RATIONAL_GRID = ((1, 2), (3, 5), (7, 10))
W_FRACTION_GRID = tuple(
    Fraction(numerator, denominator)
    for numerator, denominator in W_RATIONAL_GRID
)
W_GRID = tuple(float(value) for value in W_FRACTION_GRID)
ARM_IDS = tuple(
    f"K{k}::w={numerator}/{denominator}"
    for k in K_GRID
    for numerator, denominator in W_RATIONAL_GRID
)

METHOD_IDS = (
    "B",
    "U",
    "DCSE_LOO",
    "G_directional_matched",
    "DLOO_raw",
    "LOO_frequency_committee",
    "O_directional_static",
    "O_case_directional",
)
PRE_TERMINAL_METHOD_IDS = METHOD_IDS[:6]
TERMINAL_ORACLE_IDS = METHOD_IDS[6:]

HARD_THRESHOLD = 0.5
TIE_TOLERANCE = 1.0e-12

B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
SOURCE_PREFIX_ROWS_PER_CLASS = 270
SELECTED_COUNT_PER_CLASS = 256
OTHER_COUNT_PER_CLASS = 128
A1_SELECTED_SAMPLE_WEIGHT = 23.0 / 16.0
A1_OTHER_SAMPLE_WEIGHT = 7.0 / 8.0
A1_EFFECTIVE_ROWS_PER_CLASS = 1_152

ACTION_COUNT_PER_TARGET = 10
TARGET_TASK_COUNT = len(CENTERS) * SEED_PAIR_COUNT
TARGET_PROBABILITY_CELL_COUNT = TARGET_TASK_COUNT * ACTION_COUNT_PER_TARGET
EXPECTED_TOTAL_CASE_COUNT = 218
EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_CASE_COUNTS_BY_CENTER = {
    "0": 23,
    "1": 20,
    "2": 24,
    "3": 39,
    "5": 23,
    "6": 23,
    "7": 21,
    "8": 22,
    "9": 23,
}

NULL_REPLICATES = 10_000
NULL_SEED = 20_260_813

DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
CLAIM_ROLE = (
    "posthoc_fixed_bank_whole_case_loo_directional_shrinkage_ensemble_diagnostic"
)
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"


def _center(value: object, role: str) -> str:
    center = str(value)
    if center not in CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ {role} center: {center}.")
    return center


def candidate_sources(target: object) -> tuple[str, ...]:
    """Return all and only the eight legal non-target sources."""

    target_center = _center(target, "target")
    return tuple(center for center in CENTERS if center != target_center)


def a1_action_id(source: object) -> str:
    return f"{A1_PREFIX}{_center(source, 'source')}"


def physical_action_ids(target: object) -> tuple[str, ...]:
    """Return B, U, and the eight frozen A1 actions in canonical order."""

    return (
        B_ACTION_ID,
        U_ACTION_ID,
        *(a1_action_id(source) for source in candidate_sources(target)),
    )


def source_from_action(action_id: object) -> str:
    action = str(action_id)
    if not action.startswith(A1_PREFIX):
        raise ProtocolError("Only A1 actions identify a selected source.")
    return _center(action[len(A1_PREFIX) :], "action source")


def arm_id(k: int, weight: Fraction | tuple[int, int] | float) -> str:
    """Return one of the nine executable arm IDs and reject all other grids."""

    if isinstance(weight, tuple):
        fraction = Fraction(*weight)
    else:
        fraction = Fraction(weight).limit_denominator()
    if k not in K_GRID or fraction not in W_FRACTION_GRID:
        raise ProtocolError(
            "Directional-shrinkage arm must use K in {4,5,6} and "
            "w in {1/2,3/5,7/10}."
        )
    return f"K{k}::w={fraction.numerator}/{fraction.denominator}"


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "a1_action_id",
    "arm_id",
    "candidate_sources",
    "physical_action_ids",
    "source_from_action",
)
