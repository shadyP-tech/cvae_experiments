"""Frozen identities for the antisymmetric residual-MMD diagnostic."""

from __future__ import annotations

from ..mmd_kmm_router.contracts import (
    CENTERS,
    CLASSIFIER_THREADS_PER_WORKER,
    CLASSIFIER_WORKERS,
    COMMON_FEATURE_DIM,
    COMMON_FRAME_HASH,
    DOWNSTREAM_CLASSIFIER,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    EXPECTED_BANK_LOCK_HASH,
    EXPECTED_EQUAL_UNION_POLICY_LOCK_HASH,
    EXPECTED_GENERATION_LOCK_HASH,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_SOURCE_BLOCK_COUNT,
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_DEVICES,
    GENERATION_LOCK_ARTIFACT_ID,
    GENERATION_SEEDS,
    KERNEL_BATCH_ROWS,
    KERNEL_DEVICES,
    MAX_SOURCE_PREFIX_PER_CLASS,
    MAX_SOURCE_WEIGHT,
    MIN_EFFECTIVE_SOURCES,
    NYSTROEM_COMPONENTS,
    NYSTROEM_GAMMA,
    NYSTROEM_RANDOM_STATE,
    PRIOR_CLASSIFIER,
    PRIOR_PROBABILITY_CLIP,
    PRIOR_SENSITIVITY_POSITIVE_PRIORS,
    PRIOR_TEMPERATURE,
    ROUTER_PREFIX_PER_CLASS,
    SUPPORT_CASE_COUNT,
    SUPPORT_PARTITION_NAMESPACE,
    SUPPORT_SPLIT_SEED,
    TOTAL_PER_CLASS,
    TRAINING_SEEDS,
    VALIDATION_CACHE_REPRESENTATION_ID,
    VALIDATION_CACHE_SEMANTIC_ID,
    WORKSTATION_PROFILE,
    candidate_sources,
    row_identity_hash,
)


EXPERIMENT_ID = (
    "midogpp.oracle.uniform_b_v2_consumed_validation_"
    "antisymmetric_residual_mmd_router.v1"
)
EXPERIMENT_NAME = (
    "uniform_b_v2_consumed_validation_antisymmetric_residual_mmd_router_v1"
)
OUTPUT_ARTIFACT_ID = (
    "midogpp_output_uniform_b_v2_consumed_validation_"
    "antisymmetric_residual_mmd_router_v1"
)
VALIDATION_CACHE_ARTIFACT_ID = (
    "midogpp_stage90_antisymmetric_residual_mmd_router_validation_cache_v1"
)
VALIDATION_MANIFEST_ARTIFACT_ID = (
    "midogpp_stage90_antisymmetric_residual_mmd_router_validation_manifest_v1"
)
INPUT_ARTIFACT_IDS = (
    EXPERT_BANK_ARTIFACT_ID,
    GENERATION_LOCK_ARTIFACT_ID,
    EQUAL_UNION_POLICY_ARTIFACT_ID,
    VALIDATION_CACHE_ARTIFACT_ID,
    VALIDATION_MANIFEST_ARTIFACT_ID,
)

STAGE_ID = "90_oracles_and_diagnostics"
CLAIM_SCOPE = "diagnostic_only"
PUBLICATION_STATUS = "EXPLORATORY_CONSUMED_DATA_ONLY"
ROUTER_MODE = "antisymmetric_class_residual_robust_mmd"
ROUTED_ARM = "antisymmetric_residual_mmd"
CONTROL_ARM = "equal_union_control"
ARM_ROLES = (CONTROL_ARM, ROUTED_ARM)

CROSS_FIT_NAMESPACE = "midogpp_antisymmetric_residual_mmd_crossfit_v1"
CROSS_FIT_MODE = (
    "fixed_two_case_calibration_plus_other_evaluation_cases_"
    "leave_one_evaluation_case_out"
)
EXPECTED_CROSS_FIT_FOLD_COUNT = 26
EXPECTED_SEED_CELL_COUNT = len(TRAINING_SEEDS) * len(GENERATION_SEEDS)
EXPECTED_PREDICTION_CELL_COUNT = (
    EXPECTED_CROSS_FIT_FOLD_COUNT * EXPECTED_SEED_CELL_COUNT * len(ARM_ROLES)
)
# One equal-union fit per target x seed cell plus, at worst, one routed fit per
# held-out case x seed cell.  Composition-hash reuse can only reduce this.
MAXIMUM_UNIQUE_CLASSIFIER_FIT_COUNT = (
    len(CENTERS) * EXPECTED_SEED_CELL_COUNT
    + EXPECTED_CROSS_FIT_FOLD_COUNT * EXPECTED_SEED_CELL_COUNT
)

CLASS_WEIGHTS = (0.5, 0.5)
CONTRAST_WEIGHT = 1.0
RESIDUAL_L1_RADIUS = 0.25
RESIDUAL_L2_REGULARIZATION = 0.10
ROBUST_WORST_VARIANT_PENALTY = 1.0
MINIMUM_ROBUST_PROXY_IMPROVEMENT = 1.0e-6
MINIMUM_SOFT_CLASS_MASS_PER_CASE = 1.0
MINIMUM_SOFT_CLASS_EFFECTIVE_ROWS_PER_CASE = 2.0
SOLVER_TOLERANCE = 1.0e-12
MAX_SOLVER_ITERATIONS = 2000
MINIMUM_WORKSTATION_LOGICAL_CPU_COUNT = 12
MINIMUM_WORKSTATION_RAM_BYTES = 107374182400
MINIMUM_WORKSTATION_DISK_FREE_BYTES = 8589934592
MINIMUM_WORKSTATION_GPU_FREE_MIB = 18000


__all__ = tuple(name for name in globals() if name.isupper()) + (
    "candidate_sources",
    "row_identity_hash",
)
