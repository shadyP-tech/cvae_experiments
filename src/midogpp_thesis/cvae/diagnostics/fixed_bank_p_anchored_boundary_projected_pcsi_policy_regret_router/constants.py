"""Frozen constants for the boundary-projected PCSI policy-regret router.

The package is intentionally independent of every other Stage-90 diagnostic.
Only experiment-neutral runtime modules may be imported by the implementation.
The whole MIDOG++ test split is consumed through 218 leave-one-whole-case-out
routes; it is never treated as a fresh routing or downstream-utility surface.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "p_anchored_boundary_projected_pcsi_policy_regret_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_"
    "p_anchored_boundary_projected_pcsi_policy_regret_router_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_test_fixed_bank_"
    "p_anchored_boundary_projected_pcsi_policy_regret_router_v1"
)

DATASET_FAMILY = "MIDOG++"
EVALUATION_SPLIT = "test"
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_ROLE = (
    "posthoc_fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router_"
    "diagnostic"
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
PRIMARY_METHOD_ID = "PCSI_PARC_PROJECTED"
PROJECTED_NO_PARC_METHOD_ID = "PCSI_PARC_PROJECTED_NO_POLICY_REGRET"
UNPROJECTED_PARC_METHOD_ID = "PCSI_PARC_UNPROJECTED"
LEGACY_CONTROL_METHOD_ID = "PCSI_PARC_FRESH_LEGACY_DUAL_VETO"
BLOCKED_CONTROL_METHOD_ID = "PCSI_PARC_BLOCKED_FINGERPRINT"

# Compatibility names used by package-local legacy-control modules.  They do
# not name or import the predecessor experiment.
MODEL_BASED_METHOD_ID = LEGACY_CONTROL_METHOD_ID
BACC_ONLY_METHOD_ID = PROJECTED_NO_PARC_METHOD_ID
FULL_ONLY_METHOD_ID = UNPROJECTED_PARC_METHOD_ID
PERMUTATION_METHOD_ID = BLOCKED_CONTROL_METHOD_ID
COMPOSED_POLICY_IDS = (
    PRIMARY_METHOD_ID,
    PROJECTED_NO_PARC_METHOD_ID,
    UNPROJECTED_PARC_METHOD_ID,
    LEGACY_CONTROL_METHOD_ID,
    BLOCKED_CONTROL_METHOD_ID,
)
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
PROBABILITY_STORAGE_DTYPE = "float32"
PROJECTION_ZERO_TO_ONE_VALUE = 0.5
# The exact value is asserted from numpy.float32 at runtime and persisted as
# little-endian binary32 bytes; the literal is documentation only.
PROJECTION_ONE_TO_ZERO_VALUE = 0.4999999701976776
PROJECTION_GEOMETRY_ID = "BOUNDARY_PROJECTED_BINARY32"
UNPROJECTED_GEOMETRY_ID = "FULL_ENDPOINT_PROBABILITY"
LEGACY_GEOMETRY_ID = "FIXED_SIGN_PRESERVING_SHRINKAGE_0_25"
ACTION_GEOMETRY_IDS = (
    PROJECTION_GEOMETRY_ID,
    UNPROJECTED_GEOMETRY_ID,
    LEGACY_GEOMETRY_ID,
)
PORTFOLIO_IDENTIFICATION_WEIGHT = 3.0 / 5.0
PORTFOLIO_ROBUST_WEIGHT = 2.0 / 5.0
K_GRID = (4, 5, 6)
W_GRID = (0.5, 0.6, 0.7)

# Label-free held-case features used by the independently reconstructed I/R/P
# endpoints.  These remain separate from the router's utility features.
HELD_FEATURE_NAMES = (
    "directional_flip_rate",
    "baseline_abs_margin_on_directional_flips",
    "candidate_abs_margin_on_directional_flips",
    "directional_probability_shift_on_flips",
    "seed_directional_flip_robustness",
    "candidate_seed_disagreement_on_directional_flips",
)

# One complete, label-free row is constructed for every
# case x alternative x direction cell.  Structural no-crossing cells are
# retained with crossing_count=0 so the learner sees abstention evidence.
UTILITY_FEATURE_NAMES = (
    "case_sample_count_log1p",
    "direction_branch_rate",
    "directional_crossing_rate",
    "portfolio_mean_on_branch",
    "alternative_mean_on_branch",
    "portfolio_abs_margin_on_branch",
    "alternative_abs_margin_on_branch",
    "signed_probability_shift_on_crossings",
    "absolute_probability_shift_on_crossings",
    "portfolio_entropy_on_crossings",
    "alternative_entropy_on_crossings",
    "crossing_count_log1p",
)
UTILITY_CELL_IDS = tuple(
    f"{alternative}::{direction}"
    for alternative in ALTERNATIVE_METHOD_IDS
    for direction in DIRECTION_IDS
)

# The route-local posterior sees only this deterministic label-free fingerprint:
# exact-nine mean, exact-nine population SD, and positive-vote fraction for each
# of B, U, and the eight target-excluded A1 actions.
FINGERPRINT_STATISTIC_IDS = (
    "exact_nine_mean",
    "seed_pair_sd",
    "positive_vote_fraction",
)
FINGERPRINT_FEATURE_COUNT = 10 * len(FINGERPRINT_STATISTIC_IDS)
PRIMARY_FINGERPRINT_CONTROL_ID = "IDENTITY"
BLOCKED_FINGERPRINT_CONTROL_ID = "WITHIN_CASE_CYCLIC_SHIFT"

TARGET_POSTERIOR_C = 1.0
TARGET_POSTERIOR_SOLVER = "lbfgs"
TARGET_POSTERIOR_MAX_ITER = 5_000
TARGET_POSTERIOR_RANDOM_STATE = 23
TARGET_POSTERIOR_TOLERANCE = 1.0e-4
TARGET_POSTERIOR_PROBABILITY_CLIP = 1.0e-12

RIDGE_ALPHA = 1.0
UTILITY_RESPONSE_IDS = (
    "bacc_contribution_delta",
    "brier_contribution_delta",
    "log_loss_contribution_delta",
)
# This fixed, label-free shrinkage keeps every selected probability on the
# selected endpoint's side of 0.5 while tempering overconfident probability
# changes.  It is not tuned on the consumed test surface.
SIGN_PRESERVING_SHRINKAGE = 0.25
DELETE_POSITIVE_FRACTION_MIN = 7.0 / 8.0
RESIDUAL_MARGIN_MULTIPLIER = 0.25
UTILITY_ZERO_TOLERANCE = 1.0e-15
LOG_LOSS_CLIP_EPSILON = 1.0e-12

TRANSPORT_FEATURE_NAMES = (
    "sample_count_log1p",
    "portfolio_mean",
    "portfolio_population_sd",
    "portfolio_mean_abs_margin",
    "portfolio_mean_binary_entropy",
    "portfolio_mean_seed_population_sd",
    "B_zero_to_one_crossing_rate",
    "B_one_to_zero_crossing_rate",
    "I_OPPORTUNITY_GATED_zero_to_one_crossing_rate",
    "I_OPPORTUNITY_GATED_one_to_zero_crossing_rate",
    "R_NINE_ARM_ROBUST_zero_to_one_crossing_rate",
    "R_NINE_ARM_ROBUST_one_to_zero_crossing_rate",
)
TRANSPORT_MAD_SCALE = 1.4826
TRANSPORT_SCALE_FLOOR = 1.0e-12
TRANSPORT_MIN_REFERENCE_CENTER_COUNT = 3

EXPECTED_OUTER_PLAN_COUNT = EXPECTED_TOTAL_CASE_COUNT
EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT = 16 * EXPECTED_OUTER_PLAN_COUNT
# The fresh legacy control uses 9 * (1 full + 8 delete-donor) fits.  Each PARC
# geometry additionally performs, for every outer H, one 9-fit target family
# and eight pseudo-target families of 1 full + 7 delete fits:
# 9 * (9 + 8 * 8) = 657.  Projected and full-unprojected PARC are distinct.
EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT = len(CENTERS) * len(CENTERS)
EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY = len(CENTERS) * (
    len(CENTERS) + (len(CENTERS) - 1) * (len(CENTERS) - 1)
)
EXPECTED_UTILITY_MODEL_FIT_COUNT = (
    EXPECTED_LEGACY_UTILITY_MODEL_FIT_COUNT
    + 2 * EXPECTED_PARC_MODEL_FIT_COUNT_PER_GEOMETRY
)
EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT = len(CENTERS) * (len(CENTERS) - 1)
EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY = EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT
EXPECTED_POLICY_REPLAY_COUNT = 2 * EXPECTED_POLICY_REPLAY_COUNT_PER_GEOMETRY
# One primary and one blocked target-local posterior per whole-case route.
EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT = 2 * EXPECTED_OUTER_PLAN_COUNT
# Compatibility name used by reconstructive topology validation.
EXPECTED_CROSSING_MODEL_FIT_COUNT = EXPECTED_UTILITY_MODEL_FIT_COUNT

WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
GPU_DEVICES = ("cuda:0", "cuda:1")
PERSISTENT_GPU_WORKERS = 2
CPU_WORKERS = 4
BLAS_THREADS_PER_CPU_WORKER = 3
UTILITY_CPU_WORKERS = 4
UTILITY_BLAS_THREADS_PER_WORKER = 1
TARGET_POSTERIOR_BLAS_THREADS_PER_WORKER = 1
SCRATCH_ROOT = "/data/local/fixed_bank_p_anchored_boundary_projected_pcsi_policy_regret_router_v1"


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
