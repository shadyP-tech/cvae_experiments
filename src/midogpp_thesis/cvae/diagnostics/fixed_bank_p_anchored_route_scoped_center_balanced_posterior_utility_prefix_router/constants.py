"""Frozen identities and numerical constants for CBPUPR v1.

This package is self-contained.  It consumes only the six original MIDOG++
fixed-bank inputs and never imports another Stage-90 diagnostic.  The whole
test split is used as a leave-one-whole-case-out terminal diagnostic; none of
the resulting policies is promotion eligible.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_"
    "scoped_center_balanced_posterior_utility_prefix_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_test_fixed_bank_p_anchored_route_scoped_center_"
    "balanced_posterior_utility_prefix_router_v1"
)
OUTPUT_ARTIFACT_ID = f"midogpp_output_{EXPERIMENT_NAME}"

DATASET_FAMILY = "MIDOG++"
EVALUATION_SPLIT = "test"
STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "POST_HOC_CONSUMED_TEST_SENSITIVITY"
TERMINAL_DECISION = "TERMINAL_DIAGNOSTIC_ONLY_DO_NOT_PROMOTE"
CLAIM_ROLE = (
    "posthoc_fixed_bank_p_anchored_route_scoped_center_balanced_posterior_"
    "utility_prefix_router_diagnostic"
)

CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTER = "4"
EXPECTED_TOTAL_CASE_COUNT = 218
EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_CASE_COUNTS_BY_CENTER: Mapping[str, int] = MappingProxyType(
    {"0": 23, "1": 20, "2": 24, "3": 39, "5": 23, "6": 23,
     "7": 21, "8": 22, "9": 23}
)

TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
SEED_PAIR_COUNT = 9

# Shared frozen-source and fixed-bank A1 physical-runtime geometry.  These
# values are part of this router's executable contract because the neutral
# materializers validate them independently before doing any GPU or CPU work.
SOURCE_WORKERS_PER_DEVICE = 1
GENERATION_WORKERS_PER_DEVICE = 1
SOURCE_PREFIX_ROWS_PER_CLASS = 270
SOURCE_JOB_COUNT = len(CENTERS) * len(TRAINING_SEEDS)
SOURCE_STREAM_COUNT = SOURCE_JOB_COUNT * len(GENERATION_SEEDS)
TARGET_TASK_COUNT = len(CENTERS) * SEED_PAIR_COUNT
PHYSICAL_ACTION_COUNT_PER_TARGET = 10
TARGET_ACTION_IDENTITY_COUNT = len(CENTERS) * PHYSICAL_ACTION_COUNT_PER_TARGET
TARGET_PROBABILITY_CELL_COUNT = TARGET_ACTION_IDENTITY_COUNT * SEED_PAIR_COUNT
TARGET_UNIQUE_CLASSIFIER_FIT_COUNT = TARGET_PROBABILITY_CELL_COUNT
GENERATED_CACHE_FORMAT = "float32_npy_memmap"

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
PRIMARY_METHOD_ID = "CBPUPR_UNIFIED_PREFIX"
CANDIDATE_ONLY_METHOD_ID = "CBPUPR_UNIFIED_CANDIDATE_ONLY"
OBSERVED_MAX_CONTROL_METHOD_ID = "CBPUPR_OBSERVED_MAX_PREFIX_CONTROL"
BLOCKED_CONTROL_METHOD_ID = "CBPUPR_CYCLIC_FINGERPRINT_PREFIX_CONTROL"
MODEL_BASED_METHOD_ID = PRIMARY_METHOD_ID
COMPOSED_POLICY_IDS = (
    PRIMARY_METHOD_ID,
    CANDIDATE_ONLY_METHOD_ID,
    OBSERVED_MAX_CONTROL_METHOD_ID,
    BLOCKED_CONTROL_METHOD_ID,
)
FIXED_CONTROL_MENU = (PORTFOLIO_METHOD_ID, *COMPOSED_POLICY_IDS)
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
SCIENTIFIC_REDUCTION_DTYPE = "float64"
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
ENDPOINT_IRLS_RIDGE_ALPHA = 1.0
POSTERIOR_FITS_PER_ROUTE_AND_CONTROL = 1
LOG_LOSS_CLIP_EPSILON = 1.0e-12
UTILITY_ZERO_TOLERANCE = 1.0e-12
UTILITY_RESPONSE_IDS = (
    "favorable_bacc_contribution",
    "favorable_brier_contribution",
    "favorable_log_loss_contribution",
)

# Donor centers, not donor cases, are the calibration units.  The resulting
# median correction is descriptive; it is not a confidence or conformal bound.
MIN_SUPPORTED_DONOR_CENTER_COUNT = 6
CALIBRATION_UNIT = "donor_center"
CALIBRATION_SUMMARY = "median_conditional_overprediction_bias"
FINITE_SAMPLE_COVERAGE_CLAIMED = False
CONFIDENCE_BOUND_CLAIMED = False

# Numeric transport is diagnostic only.  Structural lineage remains a gate.
TRANSPORT_MAD_SCALE = 1.4826
TRANSPORT_SCALE_FLOOR = 1.0e-12
TRANSPORT_MIN_REFERENCE_CENTER_COUNT = 3
NUMERIC_TRANSPORT_IS_AUTHORIZATION_GATE = False
STRUCTURAL_TRANSPORT_IS_AUTHORIZATION_GATE = True

EXPECTED_OUTER_PLAN_COUNT = EXPECTED_TOTAL_CASE_COUNT
EXPECTED_DOUBLE_EXCLUSION_PAIR_COUNT = len(CENTERS) * (len(CENTERS) - 1)
EXPECTED_OUTER_ENDPOINT_MODEL_FIT_COUNT = 16 * EXPECTED_OUTER_PLAN_COUNT
# One H-c model is fitted for identity and one for the cyclic-fingerprint
# control.  No inner cross-fit or OOF-reliability layer is part of v1.
EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT = (
    2 * POSTERIOR_FITS_PER_ROUTE_AND_CONTROL * EXPECTED_OUTER_PLAN_COUNT
)
EXPECTED_PSEUDO_ROUTE_COUNT = (len(CENTERS) - 1) * EXPECTED_TOTAL_CASE_COUNT
# Pseudo (H,J,d) routes reuse the already sealed J-d posterior and bind an
# outer-H lineage wrapper.  They never refit a posterior.
EXPECTED_PSEUDO_POSTERIOR_MODEL_FIT_COUNT = 0
EXPECTED_TOTAL_POSTERIOR_MODEL_FIT_COUNT = EXPECTED_TARGET_POSTERIOR_MODEL_FIT_COUNT

WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"
GPU_DEVICES = ("cuda:0", "cuda:1")
PERSISTENT_GPU_WORKERS = 2
CPU_WORKERS = 4
BLAS_THREADS_PER_CPU_WORKER = 3
TARGET_POSTERIOR_BLAS_THREADS_PER_WORKER = 1
SCRATCH_ROOT = (
    "/data/local/fixed_bank_p_anchored_route_scoped_center_balanced_"
    "posterior_utility_prefix_router_v1"
)
RUN_RECOVERY_POLICY = (
    "no_cross_run_recovery_intra_launch_atomic_task_checkpoints_only"
)


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


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "a1_action_id",
    "candidate_sources",
    "physical_action_ids",
)
