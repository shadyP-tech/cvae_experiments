"""Frozen constants for the nested donor endpoint-regret diagnostic.

The package is intentionally independent of every other Stage-90 diagnostic.
Only experiment-neutral runtime modules may be imported by the implementation.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "loo_nested_donor_endpoint_regret_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_"
    "loo_nested_donor_endpoint_regret_router_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "loo_nested_donor_endpoint_regret_router_v1"
)

DATASET_FAMILY = "MIDOG++"
EVALUATION_SPLIT = "test"
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_ROLE = (
    "posthoc_fixed_bank_whole_case_loo_nested_donor_endpoint_regret_"
    "router_diagnostic"
)

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTER = "4"
EXPECTED_TOTAL_CASE_COUNT = 218
EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_CASE_COUNTS_BY_CENTER: Mapping[str, int] = MappingProxyType(
    {
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
)

TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
SEED_PAIR_COUNT = 9

B_ROWS_PER_SOURCE_CLASS = 128
U_ROWS_PER_SOURCE_CLASS = 144
A1_SELECTED_ROWS_PER_CLASS = 256
A1_OTHER_ROWS_PER_CLASS = 128
A1_SELECTED_ROW_WEIGHT = 23.0 / 16.0
A1_OTHER_ROW_WEIGHT = 7.0 / 8.0

B_ACTION_ID = "B"
U_ACTION_ID = "U"
A1_ACTION_PREFIX = "A1::source="
DIRECTION_IDS = ("zero_to_one", "one_to_zero")

BASELINE_METHOD_ID = "B"
IDENTIFICATION_METHOD_ID = "I_OPPORTUNITY_GATED"
ROBUST_METHOD_ID = "R_NINE_ARM_ROBUST"
PORTFOLIO_METHOD_ID = "P_PROTECTED"
MODEL_BASED_METHOD_ID = "NDR_MODEL_BASED"
LTT_METHOD_ID = "NDR_CENTER_BLOCK_FEASIBILITY"
ENDPOINT_METHOD_IDS = (
    BASELINE_METHOD_ID,
    IDENTIFICATION_METHOD_ID,
    ROBUST_METHOD_ID,
    PORTFOLIO_METHOD_ID,
)
ALTERNATIVE_METHOD_IDS = (
    BASELINE_METHOD_ID,
    IDENTIFICATION_METHOD_ID,
    ROBUST_METHOD_ID,
)
ENDPOINT_ORDER = MappingProxyType(
    {method: ordinal for ordinal, method in enumerate(ENDPOINT_METHOD_IDS)}
)

HARD_THRESHOLD = 0.5
PORTFOLIO_IDENTIFICATION_WEIGHT = 3.0 / 5.0
PORTFOLIO_ROBUST_WEIGHT = 2.0 / 5.0
K_GRID = (4, 5, 6)
W_GRID = (0.5, 0.6, 0.7)

HELD_FEATURE_NAMES = (
    "directional_flip_rate",
    "baseline_abs_margin_on_directional_flips",
    "candidate_abs_margin_on_directional_flips",
    "directional_probability_shift_on_flips",
    "seed_directional_flip_robustness",
    "candidate_seed_disagreement_on_directional_flips",
)
REGRET_FEATURE_NAMES = (
    "support_regret_sum_pp",
    "same_alternative_voter_fraction",
    "support_voter_dispersion_for_sum_pp",
    "held_case_crossing_rate",
    "held_case_directional_imbalance",
    "held_case_mean_absolute_probability_shift",
    "held_case_mean_signed_probability_shift",
    "portfolio_margin_on_crossings",
    "alternative_margin_on_crossings",
    "alternative_is_B",
    "alternative_is_I",
    "candidate_available",
)

RIDGE_ALPHA = 1.0
CENTER_EFFECT_ALPHA = 8.0
SUPPORT_DISPERSION_MULTIPLIER = 0.5
MIN_DELETE_DONOR_POSITIVE = 7
BACC_REGRET_TOLERANCE = 1.0e-12
PROPER_LOSS_TOLERANCE = 0.0
LOG_LOSS_CLIP_EPSILON = 1.0e-12
PROPER_LOSS_AGGREGATION = "equal_center_of_within_center_sample_mean_log_loss"
PROPER_LOSS_CLASS_WEIGHTING = "unweighted_within_center"
LTT_FAMILYWISE_ALPHA = 0.05
LTT_MAX_CENTER_HARM_RATE = 0.10
SENSITIVITY_DISPERSION_MULTIPLIERS = (0.0, 0.5, 1.0, 1.5, 2.0)
SENSITIVITY_DELETE_COUNTS = (0, 7, 8)

EXPECTED_OUTER_PLAN_COUNT = EXPECTED_TOTAL_CASE_COUNT
EXPECTED_UNORDERED_PAIR_COUNT = sum(
    count * (count - 1) // 2 for count in EXPECTED_CASE_COUNTS_BY_CENTER.values()
)
EXPECTED_ORDERED_VOTER_COUNT = 2 * EXPECTED_UNORDERED_PAIR_COUNT
EXPECTED_ENDPOINT_MODEL_FIT_COUNT = 16 * (
    EXPECTED_OUTER_PLAN_COUNT + EXPECTED_UNORDERED_PAIR_COUNT
)

WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
GPU_DEVICES = ("cuda:0", "cuda:1")
PERSISTENT_GPU_WORKERS = 2
CPU_WORKERS = 4
BLAS_THREADS_PER_CPU_WORKER = 3
SCRATCH_ROOT = "/data/local/fixed_bank_loo_nested_donor_endpoint_regret_router_v1"


def candidate_sources(target_center: object) -> tuple[str, ...]:
    """Return the canonical eight non-target source centers."""

    target = str(target_center)
    if target not in CENTERS:
        raise ValueError(f"Unknown MIDOG++ target center: {target}.")
    return tuple(center for center in CENTERS if center != target)


def a1_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ValueError(f"Unknown MIDOG++ source center: {source}.")
    return f"{A1_ACTION_PREFIX}{source}"


def physical_action_ids(target_center: object) -> tuple[str, ...]:
    return (
        B_ACTION_ID,
        U_ACTION_ID,
        *(a1_action_id(source) for source in candidate_sources(target_center)),
    )


__all__ = tuple(
    name for name in globals() if name.isupper() and not name.startswith("_")
) + ("a1_action_id", "candidate_sources", "physical_action_ids")
