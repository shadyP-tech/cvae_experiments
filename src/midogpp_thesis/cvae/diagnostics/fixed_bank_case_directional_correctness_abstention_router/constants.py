"""Frozen scientific identities and hyperparameters."""

from __future__ import annotations

from ...protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
B_ACTION_ID = "B"
U_ACTION_ID = "U"
OFF_ACTION_ID = "OFF"
A1_PREFIX = "A1::source="
DIRECTION_IDS = ("zero_to_one", "one_to_zero")
SEED_PAIR_COUNT = 9
HARD_THRESHOLD = 0.5

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
CASE_PROXY_WEIGHT_NUMERATOR = 1
CASE_PROXY_WEIGHT_DENOMINATOR = 2
PRIOR_WEIGHT_NUMERATOR = 1
PRIOR_WEIGHT_DENOMINATOR = 2
TIE_TOLERANCE = 1.0e-12
CANDIDATE_FEATURE_PERMUTATION_SEED = 20_260_814
CANDIDATE_FEATURE_PERMUTATION_ALGORITHM = (
    "splitmix64_route_direction_candidate_block_permutation_v1"
)

PRIMARY_METHOD_ID = "CDCA_LOO"
PRE_TERMINAL_METHOD_IDS = (
    B_ACTION_ID,
    U_ACTION_ID,
    PRIMARY_METHOD_ID,
    "G_directional_matched",
    "CDCA_case_proxy_only",
)
TERMINAL_ORACLE_IDS = ("O_directional_static", "O_case_directional")
METHOD_IDS = (*PRE_TERMINAL_METHOD_IDS, *TERMINAL_ORACLE_IDS)
DESCRIPTIVE_METHOD_IDS = ("CDCA_feature_block_permutation_descriptive",)

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
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"


def candidate_sources(target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError(f"Unknown abstention-router target center: {target}.")
    return tuple(center for center in CENTERS if center != target)


def a1_action_id(source: object) -> str:
    value = str(source)
    if value not in CENTERS:
        raise ProtocolError(f"Unknown abstention-router source center: {value}.")
    return f"{A1_PREFIX}{value}"


def source_from_action(action_id: object) -> str:
    action = str(action_id)
    if not action.startswith(A1_PREFIX):
        raise ProtocolError("Abstention-router action does not identify an A1 source.")
    source = action[len(A1_PREFIX) :]
    if source not in CENTERS:
        raise ProtocolError("Abstention-router A1 source is unknown.")
    return source


def physical_action_ids(target_center: object) -> tuple[str, ...]:
    target = str(target_center)
    return (
        B_ACTION_ID,
        U_ACTION_ID,
        *(a1_action_id(source) for source in candidate_sources(target)),
    )


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "a1_action_id",
    "candidate_sources",
    "physical_action_ids",
    "source_from_action",
)
