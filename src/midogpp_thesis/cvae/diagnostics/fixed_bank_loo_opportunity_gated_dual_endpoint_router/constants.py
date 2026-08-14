"""Frozen scientific constants for the opportunity-gated dual endpoint."""

from __future__ import annotations

from fractions import Fraction

from ...protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
MIDOGPP_CENTERS = CENTERS
EXCLUDED_CENTER = "4"
B_ACTION_ID = "B"
U_ACTION_ID = "U"
OFF_ACTION_ID = "OFF"
A1_PREFIX = "A1::source="
DIRECTION_IDS = ("zero_to_one", "one_to_zero")
SEED_PAIR_COUNT = 9
BINARY_CLASSES = (0, 1)
HARD_THRESHOLD = 0.5
TIE_TOLERANCE = 1.0e-12
EXACT_TIE_TOLERANCE = Fraction(1, 10**12)

FEATURE_NAMES = (
    "directional_flip_rate",
    "baseline_abs_margin_on_directional_flips",
    "candidate_abs_margin_on_directional_flips",
    "directional_probability_shift_on_flips",
    "seed_directional_flip_robustness",
    "candidate_seed_disagreement_on_directional_flips",
)
RIDGE_ALPHA = 1.0
IRLS_MAX_ITERATIONS = 50
IRLS_CONVERGENCE_TOLERANCE = 1.0e-12
IRLS_ETA_CLIP = 30.0
IRLS_PROBABILITY_CLIP = 1.0e-12

IDENTIFICATION_CASE_WEIGHT = Fraction(4, 5)
IDENTIFICATION_DONOR_WEIGHT = Fraction(1, 5)
PORTFOLIO_IDENTIFICATION_WEIGHT = Fraction(3, 5)
PORTFOLIO_ROBUST_WEIGHT = Fraction(2, 5)
K_GRID = (4, 5, 6)
W_RATIONAL_GRID = ((1, 2), (3, 5), (7, 10))
W_FRACTION_GRID = tuple(Fraction(n, d) for n, d in W_RATIONAL_GRID)
ARM_IDS = tuple(
    f"K{k}::w={n}/{d}" for k in K_GRID for n, d in W_RATIONAL_GRID
)

CANDIDATE_FEATURE_PERMUTATION_SEED = 20_260_814
CANDIDATE_FEATURE_PERMUTATION_ALGORITHM = (
    "splitmix64_route_direction_candidate_block_permutation_v1"
)

PRIMARY_METHOD_IDS = (
    B_ACTION_ID,
    U_ACTION_ID,
    "I_OPPORTUNITY_GATED",
    "R_NINE_ARM_ROBUST",
    "OGDE_PORTFOLIO",
)
CONTROL_METHOD_IDS = (
    "CALIBRATION_ONLY_B_R",
    "I_FEATURE_BLOCK_PERMUTED",
    "OGDE_FEATURE_BLOCK_PERMUTED",
    "I_GATE_ONLY",
    "I_SOURCE_ONLY",
    "G_DIRECTIONAL_MATCHED",
)
PRE_TERMINAL_METHOD_IDS = (*PRIMARY_METHOD_IDS, *CONTROL_METHOD_IDS)
TERMINAL_ORACLE_IDS = ("O_DIRECTIONAL_STATIC", "O_CASE_DIRECTIONAL")
METHOD_IDS = (*PRE_TERMINAL_METHOD_IDS, *TERMINAL_ORACLE_IDS)

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
ACTION_COUNT_PER_TARGET = 10
TARGET_PROBABILITY_CELL_COUNT = 810
B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
A1_SELECTED_ROWS_PER_CLASS = 256
A1_OTHER_ROWS_PER_CLASS = 128
A1_SELECTED_ROW_WEIGHT = 23.0 / 16.0
A1_OTHER_ROW_WEIGHT = 7.0 / 8.0
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"


def _center(value: object, role: str) -> str:
    center = str(value)
    if center not in CENTERS:
        raise ProtocolError(f"Unknown OGDE {role} center: {center}.")
    return center


def candidate_sources(target_center: object) -> tuple[str, ...]:
    target = _center(target_center, "target")
    return tuple(center for center in CENTERS if center != target)


def a1_action_id(source: object) -> str:
    return f"{A1_PREFIX}{_center(source, 'source')}"


def source_from_action(action_id: object) -> str:
    action = str(action_id)
    if not action.startswith(A1_PREFIX):
        raise ProtocolError("OGDE action does not identify an A1 source.")
    return _center(action[len(A1_PREFIX) :], "action source")


def physical_action_ids(target_center: object) -> tuple[str, ...]:
    target = _center(target_center, "target")
    return (
        B_ACTION_ID,
        U_ACTION_ID,
        *(a1_action_id(source) for source in candidate_sources(target)),
    )


def arm_id(k: int, weight: Fraction | tuple[int, int]) -> str:
    fraction = Fraction(*weight) if isinstance(weight, tuple) else Fraction(weight)
    if k not in K_GRID or fraction not in W_FRACTION_GRID:
        raise ProtocolError("OGDE robust arm lies outside the frozen nine-arm grid.")
    return f"K{k}::w={fraction.numerator}/{fraction.denominator}"


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "a1_action_id",
    "arm_id",
    "candidate_sources",
    "physical_action_ids",
    "source_from_action",
)
