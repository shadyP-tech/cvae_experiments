"""Frozen scientific and workstation constants for the flip router."""

from __future__ import annotations

from ...protocol import ProtocolError


CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
BINARY_CLASSES = (0, 1)
TRAINING_SEEDS = (17, 42, 101)
GENERATION_SEEDS = (17, 42, 101)
SEED_PAIR_COUNT = 9

B_ACTION_ID = "B"
U_ACTION_ID = "U"
A1_PREFIX = "A1::source="
METHOD_IDS = (
    "B",
    "U",
    "G_static",
    "S_static",
    "F_G",
    "F_S",
    "F_P",
    "O_static",
    "O_case",
)
PRE_EVALUATION_METHOD_IDS = METHOD_IDS[:-2]
TERMINAL_ORACLE_IDS = METHOD_IDS[-2:]
PRIMARY_METHOD_ID = "F_S"

HARD_THRESHOLD = 0.5
RIDGE_ALPHA = 1.0
VARIANCE_FLOOR = 1.0e-6
SAFE_Z = 1.96
MIN_GAIN = 0.0
RUNNER_UP_MARGIN = 0.0

B_COUNT_PER_SOURCE_CLASS = 128
U_COUNT_PER_SOURCE_CLASS = 144
SELECTED_COUNT_PER_CLASS = 256
OTHER_COUNT_PER_CLASS = 128
A1_SELECTED_SAMPLE_WEIGHT = 23.0 / 16.0
A1_OTHER_SAMPLE_WEIGHT = 7.0 / 8.0
SOURCE_PREFIX_ROWS_PER_CLASS = 270

FEATURE_NAMES = (
    "flip_0_to_1_count",
    "flip_0_to_1_rate",
    "flip_1_to_0_count",
    "flip_1_to_0_rate",
    "zero_flip",
    "baseline_abs_margin_on_flip",
    "candidate_abs_margin_on_flip",
    "signed_probability_delta_on_flip",
    "seed_flip_robustness",
    "candidate_seed_disagreement_on_flip",
    "case_size",
)

OOF_FOLD_COUNT = 5
OOF_FOLD_SEED = 90_902_026
OOF_PARTITION_NAMESPACE = (
    "midogpp_fixed_bank_labeled_support_case_conditional_flip_router_test_folds_v1"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 90_912_030

EXPECTED_TEST_ROW_COUNT = 9_928
EXPECTED_TOTAL_CASE_COUNT = 218
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
TARGET_TASK_COUNT = 81
TARGET_PROBABILITY_CELL_COUNT = 810
SCRATCH_ROOT = "/data/local/fixed_bank_labeled_support_case_conditional_flip_router_v1"
WORKSTATION_PROFILE = "xeon_w2265_12c24t_125gb_2x_rtx_a5000_24gb"


def candidate_sources(target: object) -> tuple[str, ...]:
    center = str(target)
    if center not in CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ target center: {center}.")
    return tuple(value for value in CENTERS if value != center)


def a1_action_id(source: object) -> str:
    center = str(source)
    if center not in CENTERS:
        raise ProtocolError(f"Unknown MIDOG++ source center: {center}.")
    return f"{A1_PREFIX}{center}"


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "a1_action_id",
    "candidate_sources",
)
