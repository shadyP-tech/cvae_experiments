"""Frozen identities and claim boundaries for the consumed Stage-90 diagnostic.

This module deliberately contains no labels, fitted models, or artifact I/O.
It fixes the MIDOG++ geometry and names the terminal diagnostic arms.  In
particular, ``R2`` denotes a two-support-case *diagnostic proposal*, not a
deployable routing policy.
"""

from __future__ import annotations

from itertools import product

from ...expert_bank.uniform_b_v2_promotion.contracts import (
    GENERATION_SEEDS as _BANK_GENERATION_SEEDS,
    TRAINING_SEEDS as _BANK_TRAINING_SEEDS,
)
from ...protocol import ProtocolError


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_validation_"
    "utility_aligned_exact_tail_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_validation_utility_aligned_exact_tail_router_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_"
    "utility_aligned_exact_tail_router_v1"
)
STAGE_ID = "90_oracles_and_diagnostics"
DATASET_FAMILY = "MIDOG++"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
ROUTING_STATUS = "INSUFFICIENT_SUPPORT_FOR_POLICY"

EXPERT_BANK_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_routing_authorized_expert_bank_v1"
)
GENERATION_LOCK_ARTIFACT_ID = "midogpp_output_uniform_b_v2_generation_lock_v1"
EQUAL_UNION_POLICY_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_equal_union_policy_lock_v1"
)
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_utility_aligned_exact_tail_router_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_utility_aligned_exact_tail_router_validation_manifest_v1"
)
METADATA_PROFILE_ARTIFACT_ID = "midogpp_routing_metadata_profiles_v1"
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
    METADATA_PROFILE_ARTIFACT_ID,
)

# Center 4 remains outside the frozen bank.  The order is canonical and is
# used by every hash, action menu, and report table in this experiment.
CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
EXCLUDED_CENTER = "4"
TRAINING_SEEDS = tuple(int(value) for value in _BANK_TRAINING_SEEDS)
GENERATION_SEEDS = tuple(int(value) for value in _BANK_GENERATION_SEEDS)
SEED_PAIR_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)

VALIDATION_SPLIT = "val"
FIXED_SUPPORT_CASE_COUNT_PER_CENTER = 2
MINIMUM_FRESH_POLICY_SUPPORT_CASES = 8
MINIMUM_FRESH_POLICY_BOOTSTRAPS = 32
EXPECTED_TOTAL_CASE_COUNT = 44
EXPECTED_CASE_OOF_FOLD_COUNT = 26
SUPPORT_SPLIT_SEED = 20_260_806
SUPPORT_PARTITION_NAMESPACE = (
    "midogpp_utility_aligned_exact_tail_router_support_v1"
)

INNER_CANDIDATE_COUNT = 7
TARGET_CANDIDATE_COUNT = 8
INNER_BASE_PER_SOURCE_PER_CLASS = 144
INNER_TOPUP_TOTAL_PER_CLASS = 126
INNER_FINAL_TOTAL_PER_CLASS = 1_134
INNER_SELECTED_SOURCE_CAPACITY_PER_CLASS = 270
TARGET_BASE_PER_SOURCE_PER_CLASS = 128
TARGET_TOPUP_TOTAL_PER_CLASS = 128
TARGET_UNIFORM_TOPUP_PER_SOURCE_PER_CLASS = 16
TARGET_BASE_TOTAL_PER_CLASS = 1_024
TARGET_FINAL_TOTAL_PER_CLASS = 1_152
TARGET_SELECTED_SOURCE_CAPACITY_PER_CLASS = 256

EXPECTED_INNER_UTILITY_ROW_COUNT = (
    len(CENTERS)
    * (len(CENTERS) - 1)
    * (len(CENTERS) - 2)
    * SEED_PAIR_COUNT
)
EXPECTED_TARGET_FEATURE_ROW_COUNT = (
    len(CENTERS) * (len(CENTERS) - 1) * SEED_PAIR_COUNT
)
EXPECTED_SOURCE_JOB_COUNT = len(CENTERS) * len(TRAINING_SEEDS)

BASE_ACTION_ID = "B"
UNIFORM_ACTION_ID = "U"
GLOBAL_DELTA_ACTION_ID = "G_delta"
R2_ACTION_ID = "R2"
PERMUTATION_ACTION_ID = "P"
H_X_E_ACTION_PREFIX = "Hxe::"
PRIMARY_DIAGNOSTIC_ACTION_IDS = (
    BASE_ACTION_ID,
    UNIFORM_ACTION_ID,
    GLOBAL_DELTA_ACTION_ID,
    R2_ACTION_ID,
    PERMUTATION_ACTION_ID,
)
EXPECTED_TARGET_ACTION_COUNT = len(PRIMARY_DIAGNOSTIC_ACTION_IDS) + 8
EXPECTED_FROZEN_TARGET_ACTION_COUNT = len(CENTERS) * EXPECTED_TARGET_ACTION_COUNT

# The permutation is fixed before any utility or target label is read.
PERMUTATION_SEED = 90_902_026


def candidate_sources(target_center: object) -> tuple[str, ...]:
    """Return the eight frozen experts available to target ``H``."""

    target = str(target_center)
    if target not in CENTERS:
        raise ProtocolError("Utility-aligned Stage-90 target center is unknown.")
    return tuple(center for center in CENTERS if center != target)


def inner_candidate_sources(
    outer_target: object,
    query_center: object,
) -> tuple[str, ...]:
    """Return the seven legal experts after strict ``H/q/e`` exclusion."""

    target = str(outer_target)
    query = str(query_center)
    if target not in CENTERS or query not in CENTERS or target == query:
        raise ProtocolError("Utility-aligned Stage-90 inner H/q geometry is invalid.")
    return tuple(center for center in CENTERS if center not in {target, query})


def seed_pairs() -> tuple[tuple[int, int], ...]:
    """Return all nine predeclared training/generation seed cells."""

    return tuple(product(TRAINING_SEEDS, GENERATION_SEEDS))


def h_x_e_action_id(source_center: object) -> str:
    source = str(source_center)
    if source not in CENTERS:
        raise ProtocolError("Utility-aligned Hxe source is unknown.")
    return f"{H_X_E_ACTION_PREFIX}{source}"


def h_x_e_source(action_id: object) -> str | None:
    value = str(action_id)
    if not value.startswith(H_X_E_ACTION_PREFIX):
        return None
    source = value.removeprefix(H_X_E_ACTION_PREFIX)
    if source not in CENTERS:
        raise ProtocolError("Utility-aligned Hxe action source is unknown.")
    return source


def expected_target_action_ids(target_center: object) -> tuple[str, ...]:
    return (
        *PRIMARY_DIAGNOSTIC_ACTION_IDS,
        *(h_x_e_action_id(source) for source in candidate_sources(target_center)),
    )


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "candidate_sources",
    "expected_target_action_ids",
    "h_x_e_action_id",
    "h_x_e_source",
    "inner_candidate_sources",
    "seed_pairs",
)
