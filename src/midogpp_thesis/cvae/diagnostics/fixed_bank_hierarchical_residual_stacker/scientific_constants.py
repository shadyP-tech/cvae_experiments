"""Frozen mathematics for the terminal hierarchical residual-stacker audit.

The values in this module are part of the scientific identity.  Runtime
configuration may repeat them for validation, but it must not tune them.
"""

from __future__ import annotations

from .config_payloads import (
    CLUSTER_BOOTSTRAP_SEED,
    CLUSTER_BOOTSTRAP_REPLICATES,
    CONFIDENCE_MULTIPLIER as CONFIG_CONFIDENCE_MULTIPLIER,
    FEATURE_PERMUTATION_SEED,
    LOCAL_RESIDUAL_FEATURE_NAMES,
    LOGIT_CLIP_EPSILON,
    MAXIMUM_LAMBDA,
    MAX_SOURCES_PER_CLASS,
    MIXTURE_TEMPERATURE,
    MODEL_FEATURE_NAMES,
    PRIMARY_INTERACTION_RANK,
    PROBABILITY_THRESHOLD,
    RIDGE_ALPHA_GRID,
    SMOOTH_RESPONSE_TEMPERATURE,
    SUPPORT_INTERCEPT_GRID,
    SUPPORT_LAMBDA_GRID,
    VARIANCE_FLOOR as CONFIG_VARIANCE_FLOOR,
)


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_test_fixed_bank_"
    "hierarchical_residual_stacker.v1"
)
MIDOGPP_CENTERS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
BASELINE_ACTION_ID = "B"
METHOD_IDS = ("B", "B_cal", "G", "R", "P")

PROBABILITY_EPSILON = LOGIT_CLIP_EPSILON
HARD_THRESHOLD = PROBABILITY_THRESHOLD
SMOOTH_TEMPERATURE = SMOOTH_RESPONSE_TEMPERATURE
RIDGE_GRID = RIDGE_ALPHA_GRID
MODEL_RANK = PRIMARY_INTERACTION_RANK
SOFTMAX_TEMPERATURE = MIXTURE_TEMPERATURE
SPARSE_SOURCE_BUDGET = MAX_SOURCES_PER_CLASS
INTERCEPT_GRID = SUPPORT_INTERCEPT_GRID
RESIDUAL_SCALE_GRID = SUPPORT_LAMBDA_GRID
MAX_RESIDUAL_SCALE = MAXIMUM_LAMBDA
CONFIDENCE_MULTIPLIER = CONFIG_CONFIDENCE_MULTIPLIER
VARIANCE_FLOOR = CONFIG_VARIANCE_FLOOR
PERMUTATION_SEED = FEATURE_PERMUTATION_SEED
BOOTSTRAP_SEED = CLUSTER_BOOTSTRAP_SEED
BOOTSTRAP_REPLICATES = CLUSTER_BOOTSTRAP_REPLICATES

PHI_NAMES = LOCAL_RESIDUAL_FEATURE_NAMES
DESIGN_TERMS = MODEL_FEATURE_NAMES


def candidate_sources(target_center: str) -> tuple[str, ...]:
    target = str(target_center)
    if target not in MIDOGPP_CENTERS:
        raise ValueError(f"Unknown MIDOG++ center: {target}")
    return tuple(center for center in MIDOGPP_CENTERS if center != target)


__all__ = tuple(name for name in globals() if name.isupper()) + ("candidate_sources",)
